import logging
import os
import codecs
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict
from filelock import FileLock

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available, RobertaModel, BertPreTrainedModel, XLMRobertaConfig

logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    """
    A single training/test example for semantic role labeling.

    Args:
        guid: `str` Unique id for the example.
        predicate_indicator: `List[int]` The predicate indicator for the examples.
        words: `List[str]` The words of the sequence.
        labels: (Optional) `List[str]` The labels for each word of the sequence. This should be specified for train and dev examples, but not for test examples.

    """
    guid: str
    predicate_indicator: List[int]
    words: List[str]
    tags: Optional[List[str]]

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    labels: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset


    class SRLDataset(Dataset):
        """
        Dataset for reading SRL data.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only real labe ids contribute to loss later.

        def __init__(
            self,
            data: List[Dict],
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
        ):
            # Load data features 
            # NOTE this is kind of hacky, but it works for now.
                examples = read_prediction_input(data)
                self.features = convert_examples_to_append_features(
                        examples, 
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end = bool(model_type in ["xlnet"]), # xlnet has a cls token at the end
                        cls_token = tokenizer.cls_token,
                        cls_token_segment_id = 2 if model_type in ["xlnet"] else 0,
                        sep_token = tokenizer.sep_token,
                        sep_token_extra = False, # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left = bool(tokenizer.padding_side == "left"),
                        pad_token = tokenizer.pad_token_id,
                        pad_token_segment_id = tokenizer.pad_token_type_id,
                        pad_token_label_id = self.pad_token_label_id,
                )
                return
            
        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


def read_prediction_input(data) -> List[InputExample]:
    guid_index = 1
    examples = []
    for entry in data:
        sentence = entry["sentence"] # .strip().split()
        predicate_index = entry["index"]
        if predicate_index not in range(len(sentence)):
            continue
        predicate = [0 if index != predicate_index else 1 for index in range(len(sentence))]
        one_hot_tags = ["O" for _ in sentence]
        one_hot_tags[predicate_index] = "B-V"
        examples.append(InputExample(guid=f"input-{guid_index}", words=sentence, predicate_indicator=predicate, tags=one_hot_tags))
        guid_index += 1
    return examples


def convert_examples_to_append_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end = False,
    cls_token = "[CLS]",
    cls_token_segment_id = 1,
    sep_token =  "[SEP]",
    sep_token_extra = False,
    pad_on_left = False,
    pad_token = 0,
    pad_token_segment_id = 0,
    pad_token_label_id = -100,
    sequence_a_segment_id = 0,
    sequence_b_segment_id = 1,
    mask_padding_with_zero = True,
) -> List[InputFeatures]:
    """
    Loads a list of input examples from read_better_examples_from_file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        tokens = []
        label_ids = []
        predicate_ids = []
        predicate = []
        predicate_label = ""
        for word, label, pred_ind in zip(example.words, example.tags, example.predicate_indicator):
            word_tokens = tokenizer.tokenize(word)
            if pred_ind == 1:
                predicate = word_tokens
                predicate_label = label
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens)-1))
                predicate_ids.extend([pred_ind] * len(word_tokens))

        # Account for [CLS] and [SEP] with "- 2" and "- 3" for RoBERTa then additional for the predicate as the second sentence
        special_tokens_count = tokenizer.num_special_tokens_to_add() + len(predicate) + 1
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length-special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            predicate_ids = predicate_ids[:(max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        predicate_ids += [0]
        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            predicate_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens.extend(predicate)
        label_ids.extend([label_map[predicate_label]] + [pad_token_label_id]*(len(predicate)-1)) # TODO what should the label id for the second sentence (the predicate) be?
        predicate_ids.extend([0] * len(predicate)) # TODO or should it be 1?
        segment_ids.extend([sequence_b_segment_id] * len(predicate))

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        predicate_ids += [0]
        segment_ids += [sequence_b_segment_id]

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
            predicate_ids += [0]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
            predicate_ids = [0] + predicate_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length
        padding_length = max_seq_length - len(input_ids) 
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            predicate_ids = ([0] * padding_length) + predicate_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            predicate_ids += [0] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(predicate_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index % 1000 == 0:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            # logger.info("predicate_ids: %s", " ".join([str(x) for x in predicate_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None
           # predicate_ids = None

        features.append(
            InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids
                )
        )
    return features


def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ['O', 'B-A1', 'I-A1', 'B-A0', 'I-A0', 'B-V', 'I-V']

