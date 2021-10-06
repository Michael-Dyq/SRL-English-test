import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from srl_utils import SRLDataset, get_labels

logger = logging.getLogger(__name__)

"""
TODO list
 - find out way to eliminate output_dir creation
"""


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are fine-tuning from.
    """
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})

@dataclass
class DataTrainingArguments: 
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    input_path: str = field(metadata={"help": "The input data path to predict on."})
    output_path: str = field(metadata={"help": "Test data output file path."})
    labels: Optional[str] = field(default="", metadata={"help": "Path to a file containing all labels. If not specified, default SRL labels are used."})
    max_seq_length: int = field(default=128, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    overwrite_cache: bool = field(default=False, metadata={"help":"Overwrite the cached training and evaluation sets"})
    embedding_dropout: float = field(default=0.1, metadata={"help": "Dropout probability for BERT embeddings during training."})
    hidden_size: int = field(default=768 , metadata={"help": "Size of input to tag projection layer."})
    # use_onto_model: bool = field(default=False, metadata={"help": "Whether or not to be using the ontonotes model."})
    model_metadata: dict = field(default_factory=dict, metadata={"help": "Any extra metadata."})

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If passed only one argument to the script and it's the path to a json file
        # then parse it to get arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
            format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt = "%m%d%Y %H:%M:%S",
            level = logging.INFO if training_args.local_rank in [-1,0] else logging.WARN,
    )
    logger.warning(
            "Process rank: %s, device: %s, n_gpus: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set random seed
    set_seed(training_args.seed)

    # Prepare data task
    labels = get_labels(data_args.labels) 
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer

    # Distributed training: the .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    
    config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=model_args.cache_dir,
            hidden_dropout_prob=data_args.embedding_dropout,
            hidden_size=data_args.hidden_size,
    )
    
    model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast)


    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> List[int]:
        """
        Align predictions and true tags to mask out tags to be ignored.

        Inputs:
        predictions: `np.ndarray`
            Input of size (batch_size, seq_len, num labels) representing output of model.
        label_ids: `np.ndarray`
            Input of size (batch_size, seq_len) representing true tags.

        Outputs:
        preds_list: `List[int]`
            List of predicted label tags.
        out_label_list: `List[int]`
            List of true label tags.
        """
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                # If this label is not masked over, lookup the corresponding tag and append it to outputs.
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list #, out_label_list

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # Predict
    if training_args.do_predict:
        test_dataset = SRLDataset(
            data_path=data_args.input_path,
            tokenizer=tokenizer,
            model_type=config.model_type,
            labels_file=data_args.labels.rsplit('/', 1)[-1],
            labels=labels,
            predict_input=True,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            metadata=data_args.model_metadata,
            )
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list = align_predictions(predictions, label_ids)
        # Save predictions
        if trainer.is_world_master():
            with open(data_args.output_path, "w") as writer:
                with open(data_args.input_path, "r") as f:
                    example_id = 0
                    for line in f:
                        str_list = line.strip().split()
                        sentence = str_list[1:]
                        pred_srl = preds_list[example_id]
                        pred_srl = pred_srl[:-1]
                        pred_srl[int(str_list[0])] = "B-V" # hacky
                        writer.write(" ".join(sentence) + "\n")
                        writer.write(" ".join(pred_srl) + "\n\n")
                        example_id += 1

def _mp_fn(index):
    """
    For xla_spawn (TPUs)
    """
    main()

if __name__ == "__main__":
    main()
