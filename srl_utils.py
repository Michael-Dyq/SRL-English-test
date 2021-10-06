import logging
import os
import codecs
import random
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict
# from filelock import FileLock

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
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            labels_file: str,
            labels: List[str],
            predict_input: bool = False,
            max_seq_length: Optional[int] = None,
            overwrite_cache: bool = False,
            metadata: dict = {},
        ):
            # Load data features. Note: all of these readers are customized for certain datasets. 
            
            print('---------------METADATA: ', metadata)
            if predict_input:
                examples = read_prediction_input_from_file(data_path)
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
            elif (os.path.isdir(data_path)) and ("conll-formatted-ontonotes-5.0" in data_path):
                onto_type = "onto_"
                if "3lang" in data_path:
                    onto_type += "3lang_"
                    if "arabic" in data_path:
                        onto_type += "arabic"
                        if "percentage_arabic" in metadata:
                            onto_type += str(metadata["percentage_arabic"])
                        onto_type += "_"
                    elif "english" in data_path:
                        onto_type += "english"
                        if "percentage_english" in metadata:
                            onto_type += str(metadata["percentage_english"])
                        onto_type += "_"
                    elif "chinese" in data_path:
                        onto_type += "chinese"
                        if "percentage_chinese" in metadata:
                            onto_type += str(metadata["percentage_chinese"])
                        onto_type += "_"
                    else: 
                        if "percentage_english" in metadata:
                            onto_type += "{}eng_".format(str(metadata["percentage_english"]))
                        else:
                            onto_type += "1.0eng_"
                        if "percentage_arabic" in metadata:
                            onto_type += "{}arabic_".format(str(metadata["percentage_arabic"]))
                        else:
                            onto_type += "1.0arabic_"
                        if "percentage_chinese" in metadata:
                            onto_type += "{}chinese_".format(str(metadata["percentage_chinese"]))
                        else:
                            onto_type += "1.0chinese_"
                else:
                    onto_type += "regeng_"
                if not labels_file:
                    labels_file = "DefaultLabels"
                onto_type += labels_file
                
                if "train" in data_path:
                    cached_features_file = os.path.join("data", "cached_{}_train_{}_{}".format(onto_type, tokenizer.__class__.__name__, str(max_seq_length)))
                elif "development" in data_path:
                    cached_features_file = os.path.join("data", "cached_{}_development_{}_{}".format(onto_type, tokenizer.__class__.__name__, str(max_seq_length)))
                elif "test" in data_path:
                    cached_features_file = os.path.join("data", "cached_{}_test_{}_{}".format(onto_type, tokenizer.__class__.__name__, str(max_seq_length)))
                else:
                    print("Unsure what type of data being used. Not caching")
                    cached_features_file = None
                    # cached_features_file = os.path.join("data", "cached_{}_{}_{}_[]".format(onto_type, data_path.rsplit('/', 1)[-1], tokenizer.__class__.__name__, str(max_seq_length)))

            else:
                # is BETTER dataset, with only A0 and A1.
                better_type = "better"
                if not labels_file:
                    labels_file = "DefaultLabels"
                better_type += labels_file
                cached_features_file = os.path.join("data", "cached_{}_{}_{}".format(data_path.rsplit('/', 1)[-1], tokenizer.__class__.__name__, str(max_seq_length))) 
            # Make sure only the first process in distributed training processes the dataset, and that others use the cache.  NOTE: this lock keeps hanging, so disabled for now.
            # lock_path = cached_features_file + ".lock"
            # print("ABOUT TO ENTER FILE LOCK: ", lock_path)
            # with FileLock(lock_path):
            #     print("IN FILE LOCK")
            if cached_features_file and (os.path.exists(cached_features_file) and not overwrite_cache):
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                if not cached_features_file:
                    logger.info(f"Creating features from dataset file at {data_path}")
                else:
                    logger.info(f"Creating features from dataset file at {data_path} to cache in {cached_features_file}")
                if "ontonotes" in data_path:
                    examples = read_ontonotes_examples_from_directory(data_path, labels, metadata)
                else:
                    examples = read_better_examples_from_file(data_path, labels)
                # if token_type == "append":
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
                    pad_token_label_id = self.pad_token_label_id
                )
                '''
                elif token_type == "predicate":
                    self.features = convert_examples_to_features(
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
                        pad_token_label_id = self.pad_token_label_id
                    )
                '''
                if cached_features_file:
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)
            
        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

# def conll_rows_to_examples(conll_rows, guid_index, filename, just_a0a1, labels, full_label, keep_v) -> List[InputExample]:
def conll_rows_to_examples(conll_rows, guid_index, filename, labels, short_label) -> List[InputExample]:
    keep_v = "B-V" in labels
    sentence: List[str] = []
    predicates: List[int] = [] # incides of all verbal predicates for this sentence
    span_labels: List[List[str]] = []
    current_span_labels: List[str] = []
    for index, row in enumerate(conll_rows):
        conll_components = row.split()
        word = conll_components[3]
        if not span_labels:
            # This is the first word in the sentence so create empty lists to collect SRL BIO labels.
            span_labels = [[] for _ in conll_components[11:-1]]
            current_span_labels = [None for _ in conll_components[11:-1]]

        # Process span annotation
        for annotation_index, annotation in enumerate(conll_components[11:-1]):
            label = annotation.strip("()*")
            if "(V" in label or label=="V":
                # nested v, V argument dominates
                label = "V" # currently asumes that if V in nested, V will be the shorter.
                if not keep_v:
                    span_labels[annotation_index].append("O")
                    current_span_labels[annotation_index] = None
                    continue
            elif "ARG" in label:
                if short_label:
                    label = label.replace("ARG", "A", 1)
            
            # We only care about labels in our categories
            if (len(label) > 0) and ("B-{}".format(label) not in labels):
                span_labels[annotation_index].append("O")
                current_span_labels[annotation_index] = None
                continue
            
            # Write to BIO format.
            if "(" in annotation:
                # Entering a span
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # We are inside a span
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We are outside a span
                span_labels[annotation_index].append("O")
            if ")" in annotation:
                # Exiting a span, so reset current span label.
                current_span_labels[annotation_index] = None
        
        # If this word is marked as a verbal predicate, record so.
        word_is_predicate = any("(V" in x for x in conll_components[11:-1])
        if word_is_predicate:
            predicates.append(index)

        sentence.append(word)

    examples = []
    for index, srl_frame in enumerate(span_labels):
        # print('WORDS: ', sentence, '; TAGS: ', srl_frame)
        examples.append(InputExample(guid=f"{filename}-{guid_index}", words=sentence, predicate_indicator=[1 if idx==predicates[index] else 0 for idx in range(len(sentence))], tags=srl_frame))
        guid_index += 1

    return examples
        


# def read_ontonotes_examples_from_file(data_file, guid_index, just_a0a1, labels, full_label, keep_v) -> List[InputExample]:
def read_ontonotes_examples_from_file(data_file, guid_index, labels, short_arg) -> List[InputExample]:
    examples = []
    with codecs.open(data_file, "r", encoding="utf8") as open_file:
        conll_rows: List[str] = []
        for line in open_file:
            line = line.strip()
            if line != "" and not line.startswith("#"):
                # Non-empty line. Collect the annotation.
                conll_rows.append(line)
            else:
                # Collect conll_rows into the Ontonotes sentence.
                if conll_rows:
                    examples.extend(conll_rows_to_examples(conll_rows, guid_index, data_file.rsplit('/', 1)[-1], labels, short_arg))
                    conll_rows = []
        # Collect stragglers.
        if conll_rows:
            examples.extend(conll_rows_to_examples(conll_rows, guid_index, data_file.rsplit('/', 1)[-1], labels,short_arg))
            conll_rows = []
    return examples


def read_ontonotes_examples_from_directory(data_path, labels, metadata) -> List[InputExample]:
    logger.info("Reading CONLL sentences from dataset files at %s", data_path)
    examples: List[InputExample] = []
    guid_index = 1
    short_arg = ("B-ARG0" not in labels) and ("B-A0" in labels) # TODO this is a major assumption. Just assumes that if reading Ontonotes but only wanted A0 A1 and V then it uses the default list. Otherrwise it definitely is a lined-up reading and labels list.

    for root, _, files in list(os.walk(data_path)):
        for data_file in files:
            if not data_file.endswith("gold_conll"):
                continue
            new_examples = read_ontonotes_examples_from_file(os.path.join(root, data_file), guid_index, labels, short_arg)
            if "english" in root:
                new_examples = random.sample(new_examples, round(len(new_examples)*(metadata.get("percentage_english") or 1.0)))
            if "arabic" in root:
                new_examples = random.sample(new_examples, round(len(new_examples)*(metadata.get("percentage_arabic") or 1.0)))
            if "chinese" in root:
                new_examples = random.sample(new_examples, round(len(new_examples)*(metadata.get("percentage_chinese") or 1.0)))
            examples.extend(new_examples)
    logger.info("Collected %d examples.", len(examples))
    return examples

def read_better_examples_from_file(data_path, labels) -> List[InputExample]:
    long_arg = ("B-A0" not in labels) and ("B-ARG0" in labels) # TODO this is a major assumption. Just assumes that if wanted long arguments then "ARG0" replaces "A0" and this holds for all #.
    keep_v = "B-V" in labels
    guid_index = 1
    examples = []
    filename = data_path.rsplit('/',1)[-1]
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            str_list = line.strip().split()
            separate_index = str_list.index("|||")
            predicate_index = int(str_list[0])
            sentence = str_list[1:separate_index]
            tags = str_list[separate_index+1:]
            if predicate_index not in range(len(sentence)):
                continue
            if len(sentence) != len(tags):
                continue
            predicate = [0 if index != predicate_index else 1 for index in range(len(sentence))]
            if long_arg:
                tags = [tag.replace("A", "ARG", 1) for tag in tags]
            if not keep_v:
                tags = ["O" if tag in {"B-V", "I-V"} else tag for tag in tags]
            examples.append(InputExample(guid=f"{filename}-{guid_index}", words=sentence, predicate_indicator=predicate, tags=tags))
            guid_index += 1
    return examples
                

def read_prediction_input_from_file(data_path) -> List[InputExample]:
    guid_index = 1
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            str_list = line.strip().split()
            predicate_index = int(str_list[0])
            sentence = str_list[1:]
            if predicate_index not in range(len(sentence)):
                continue
            predicate = [0 if index != predicate_index else 1 for index in range(len(sentence))]
            one_hot_tags = ["O" for _ in sentence]
            one_hot_tags[predicate_index] = "B-V" # TODO consider if we aren't including B-V what happens with this.
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

        features.append(
            InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids
                )
        )
    return features

def convert_examples_to_features(
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
    mask_padding_with_zero = True,
) -> List[InputFeatures]:
    """
    Loads a list of input examples from read_better_examples_from_file into a list of `InputFeatures`

    - `cls_token_segment_id` defines the segment ID associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        
        tokens = []
        label_ids = []
        predicate_ids = []
        for word, label, pred_ind in zip(example.words, example.tags, example.predicate_indicator):
            word_tokens = tokenizer.tokenize(word)

            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens)-1))
                predicate_ids.extend([pred_ind] * len(word_tokens))

        # Account for [CLS] and [SEP] with "- 2" and "- 3" for RoBERTa
        special_tokens_count = tokenizer.num_special_tokens_to_add()
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

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            predicate_ids += [0]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
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
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            predicate_ids += [0] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(predicate_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index % 1000 == 0:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("predicate_ids: %s", " ".join([str(x) for x in predicate_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            predicate_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=predicate_ids, labels=label_ids
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

'''
def get_conll_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        bio_labels = ["O"]
        for label in labels:
            if label != "O":
                # if not full_label:
                #     label = label.replace("ARG", "A")
                bio_labels.append("B-{}".format(label))
                bio_labels.append("I-{}".format(label))
        return bio_labels
    else:
        return get_labels(None)
'''
