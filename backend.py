import cherrypy
import json
import os
import sys
import spacy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch import nn

from tabular_view import *

from transformers import (
        AutoConfig,
        AutoModelForTokenClassification,
        AutoTokenizer,
        HfArgumentParser,
        Trainer,
        TrainingArguments,
        set_seed
)
from demo_srl_utils import SRLDataset, get_labels

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
    labels: Optional[str] = field(default=None, metadata={"help": "Path to a file containing all labels. If not specified, default SRL labels are used."})
    max_seq_length: int = field(default=128, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
    embedding_dropout: float = field(default=0.1, metadata={"help": "Dropout probability for BERT embeddings during training."})
    hidden_size: int = field(default=768 , metadata={"help": "Size of input to tag projection layer."})
    
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
    raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
    
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
trainer = Trainer(model=model, args=training_args)

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
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    
    preds_list = [[] for _ in range(batch_size)]
    
    for i in range(batch_size):
        for j in range(seq_len):
            # If this label is not masked over, lookup the corresponding tag and append it to outputs.
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                preds_list[i].append(label_map[preds[i][j]])
                
    return preds_list 

class MyWebService(object):

    # spacy_model = 'en_core_web_sm'
    spacy_model = 'es_core_news_sm'
    global tabular_structure
    tabular_structure = TabularView(spacy_model)
    sp = spacy.load(spacy_model)

    @cherrypy.expose
    def index(self):
        return "Transformers Spanish (multilingual) SRL"
        # return open('public/srl.html')

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def info(self, **params):
        return {"status":"online"}

    @cherrypy.expose
    def halt(self, **params):
        cherrypy.engine.exit()

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def annotate(self, sentence=None, index=None):
        try:
            input_json_data = cherrypy.request.json
            sentence = input_json_data["sentence"]
            if "index" in input_json_data:
                index = input_json_data["index"]
        except:
            if sentence is None:
                cherrypy.response.headers['Content-Type'] = 'text/plain'
                input_data = cherrypy.request.body.readline()
                sentence = input_data.decode("utf-8")
        examples = []
        predicate_idx = []
        spacy_sentence = self.sp(sentence)
        tokens = []
        for token in spacy_sentence:
            tokens.append(token.text)
        if index is None:
            for idx,token in enumerate(spacy_sentence):
                if token.pos_ in {"VERB", "NOUN"}:    
                    predicate_idx.append(idx)
                    examples.append({"sentence": tokens, "index": idx})
        else: # Does not currently move index if tokenizer changes things.
            for idx in index:
                if idx <= len(sentence.strip().split()):
                    predicate_idx.append(idx)
                    examples.append({"sentence": tokens, "index": idx})
        output_dict = {"words": sentence.strip().split()}
        output_dict["tokens"] = tokens
        if len(examples) == 0: # This if statement can be removed by accounting for the no-predicate case by assuring dimension changes later.
            output_dict["predicates"] = []
        else:        
            test_data = SRLDataset(
                data=examples,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
            )
            predictions, label_ids, metrics = trainer.predict(test_data)
            preds_list = align_predictions(predictions, label_ids)
            predicates = []
            for idx,pred in enumerate(preds_list):
                predictions = pred[:-1]
                frame = {"predicate_index": predicate_idx[idx], "predicate":spacy_sentence[predicate_idx[idx]].text, "tags":predictions}
                predicates.append(frame)
            output_dict["predicates"] = predicates       
            # output_dict = {"sentence": sentence, "predicate index": predicate_index, "tags": preds_list[0][:-1]}
        print('OUTPUT DICT: ', output_dict)
        tabular_structure.update_sentence(output_dict)
        tabular_structure.update_view("SRL_MULTILINGUAL", output_dict)
        return tabular_structure.get_textannotation()


if __name__ == '__main__':
        print("Starting rest service...")
        cherrypy_config = {'server.socket_host': '0.0.0.0'}
        cherrypy.config.update(cherrypy_config)
        cherrypy.config.update({'server.socket_port': 4038})
        cherrypy.quickstart(MyWebService())
