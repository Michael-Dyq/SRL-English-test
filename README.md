This repository contains the files to run a Huggingface tranformers-based SRL model on the BETTER and Ontonotes datasets.

The design of the models in this repository are based on a BERT + linear layer model used in ['Simple BERT Models for Relation Extraction and Semantic Role Labeling'](https://arxiv.org/pdf/1904.05255.pdf).

# Setup
Setup in a virtual environment, following the instructions on the [huggingface repository](https://github.com/huggingface/transformers#installation).

The GPUs on the CCG machines are CUDA version 10.1, so we set Pytorch back to version 1.4:
```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install appdirs
pip install packaging
pip install ordered_set
pip install transformers==3.0.2
pip install cherrypy
pip install -U spacy
python -m spacy download es_core_news_sm
```

To run configurations faster and be able to use fp16, download [apex](https://github.com/NVIDIA/apex).

# Configure the SRL model
Modify the config.json file to your desired hyperparameters. Currently, the model is configured to train multilingual SRL using Ontonotes-3lang.

## The SRLDataset initialization attributes:

| SRLDataset attribute  | Default value (if any) | Meaning |
| ------------- | --- | ------------- |
| `data_path`  | N/A | Path to the furthest directory/filepath containing the data which you wish to run the model on.  |
| `tokenizer` | N/A | `PreTrainedTokenizer` to use. |
| `model_type` | N/A | string of the model type to use. |
| `labels_file` | N/A | String of labels file name.
| `labels` | N/A | List of strings of all tags used for this model. | 
| `predict_input` | `False` | Boolean indicating whether input `data_path` is an input to be predicted on, rather than trained on. | 
| `max_seq_length` | Optional | Maximum total sequence length after tokenization. | 
| `overwrite_cache` | `False` | Whether or not to overwrite cache. |
| `metadata` | `{}` | Extra configurations used during trainig. Currently only supports `percentage_english`, `percentage_arabic`, and `percentage_chinese` |

## The config file attributes for training:

| Config attribute  | Type | Notes on requirement | Meaning | 
| --- | --- | --- | --- |
| `model_name_or_path` | `str` | Required for all. | Path to pretrained model or model identifier from huggingface.co/models  |
| `config_name` | `str` | Optional. Will default to `model_name_or_path` value. | Pretrained config name or path if not same as `model_name`. |
| `tokenizer_name` | `str` | Optional. Will default to `model_name_or_path` value. | Pretrained tokenizer name or path if not same as `model_name`. |
| `use_fast` | `bool`| Optional. Will default to `False`. | Set this flag to use fast tokenization.  |
| `cache_dir` | `str` | Optional. Will default to `None`. | Where to store the pretrained model downloaded from s3. Recommended on CCG machines to set to `/shared/.cache/transformers` |
| `train_data_path` | `str` | Required. | Furthest path to directory/file of training data. |
| `dev_data_path` | `str` | Required. | Furthest path to directory/file of development data. |
| `test_data_path` | `str` | Required. | Furthest path to directory/file test data. |
| `labels` | `str` | Optional. Defaults to `""`. | Path to file containing all labels. If not provided, defaults to `['O', 'B-ARG1', 'I-ARG1', 'B-ARG0', 'I-ARG0']` |
| `max_seq_length` | `int` | Optional. Defaults to `128`. | The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. |
| `overwrite_cache` | `bool` | Optional. Defaults to `False`. | Whether or not to overwrite cached training and evaluation sets. |
| `embedding_dropout` | `float` | Optional. Defaults to `0.1`. | Dropout probability for BERT embeddings during training. |
| `hidden_size` | `int` | Optional. Defaults to `768`. | Hidden size after BERT layer. |
| `model_metadata` | `dict` | Optional Defaults to `{}`. | Extra metadata used during training. Currently supports `percentage_english`, `percentage_arabic`, and `percentage_chinese`, all `float`s that default to `1.0` and indicate what percentage to keep of each language's data in Ontonotes. |
| `output_dir` | `str` | Required. | The output directory where the model predictions and checkpoints will be written.|
| `overwrite_output_dir` | `bool` | Optional. Defaults to `False`. | If True, overwrite the content of the output directory. Use this to continue training if `output_dir` points to a checkpoint directory. |
| `do_train`, `do_eval`, `do_predict` | `bool` | Optional, default to `False`. | Whether to train, evaluate, and/or predict. | 
| `num_train_epochs` | `float` | Optional. Defaults to `3.0`. | Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training). |
| `per_device_train_batch_size` | `int` | Optional. Defaults to `8`. | The batch size per compute core for training. |
| `learning_rate` | `float` | Optional. Defaults to `5e-5` | The initial learning rate for Adam. |
| Arguments from [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments) | | | Remaining arguments are optional. |


# Train the SRL model
```
. ./set_environment.sh
python run_srl_cleaned.py config.json
```

# Predict with the SRL model
The configuration file for predicting with an SRL model is different from that used to train an SRL model. Modify the `predict_config.json` file as needed, according to the following table. 

## The config file attributes for prediction:

| Config attribute  | Type | Notes on requirement | Meaning | 
| --- | --- | --- | --- |
| `model_name_or_path` | `str` | Required. | Path to pretrained model. |
| `input_path` | `str` | Required. | Path to input file. |
| `output_path` | `str` | Required. | Path to desired output file. |
| `labels` | `str` | Optional. Defaults to `None`. | Path to file containing all labels. If not provided, defaults to `['O', 'B-ARG1', 'I-ARG1', 'B-ARG0', 'I-ARG0']` |
| `output_dir` | `str` | Required. | Not used, but should point to the folder of the model used (same as `model_name_or_path`).|
| `max_seq_length` | `int` | Optional. Defaults to `128`. | The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. |
| `embedding_dropout` | `float` | Optional. Defaults to `0.1`. | Dropout probability for BERT embeddings during training. |
| `hidden_size` | `int` | Optional. Defaults to `768`. | Hidden size after BERT layer. |
| `embedding_dropout` | `float` | Optional. Defaults to `0.1`. | Dropout probability for BERT embeddings during training. |
| `hidden_size` | `int` | Optional. Defaults to `768`. | Hidden size after BERT layer. |
| `overwrite_output_dir` | `bool` | Optional. Defaults to `False`. | If True, overwrite the content of the output directory. Use this to continue training if `output_dir` points to a checkpoint directory. |
| `do_train`, `do_eval`, `do_predict` | `bool` | Optional, default to `False`. | Not used. Remains as legacy. Set to all `False` except set `do_predict=True`. | 
| `num_train_epochs` | `float` | Optional. Defaults to `3.0`. | Not used. Remains aslegacy. |
| `per_device_train_batch_size` | `int` | Optional. Defaults to `8`. | Not used. Remains as legacy. |
| `learning_rate` | `float` | Optional. Defaults to `5e-5`. | Not used. Remains as legacy |


To simply use a pre-trained SRL model, see the table at the end of this section for which models are trained on different data, and how to configure your prediction file to use them.

Input data should be of the form:
```
5 The president of the USA presides from the Oval Office .
2 The girl threw the football all the way to the back of the stadium .
```
Where the first entry is the index of the predicate, and the rest of the line is the sentence. Then, run the command:
```
python predict_srl_cleaned.py predict_config.json
```
The output will be written to the output directory/output file specified in the config file.


Pre-trained models (to go in `model_name_or_path` of predict config files):

| Model folder  | Trained on | Evaluation performance |
| ------------- | --------- | ---------- |
| `xlmr-large-onto3lang-full-cleaned` | Ontonotes 3lang 100% of English, Arabic, Chinese.  | 0.817 F1 |
| `xlmr-large-lr5e5-cleaned` | BETTER Abstract English: train, analysis, devtest. | 0.718 F1 |
| `xlmr-large-onto-ara-eng-a0a1-cleaned` | Ontonotes 3lang English, Arabic A0, A1 | 0.853 F1 |
| `xlmr-large-preonto-finebetter-cleaned` | BETTER Abstract English: train, analysis, devtest after pre-training from `xlmr-large-onto-ara-eng-a0a1-cleaned/epoch-1.99` | 0.723 F1|

(All of these models preside in `/shared/celinel/transformers-srl`.)


# Run the Cherrypy backend (SPANISH SRL)
The cherrypy backend runs the predictor for SRL build off of transformers. Set up the environment and modify the config file and port number as necessary.
```
python backend.py demo_spanish_config.json

```
Then in another terminal window, run the program with any of the following, modifying the port number, sentence, and predicate index number as necessary. The following curl commands are supported:
```
curl -d 'La presidenta de los Estados Unidos tiene mucho poder.' -H "Content-Type: text/plain" -X GET http://localhost:8038/annotate
curl -X GET http://localhost:8038/annotate?sentence=La%20presidenta%20de%20los%20Estados%20Unidos%20tiene%20mucho%20poder.
curl -d '{"sentence": "La presidenta de los Estados Unidos tiene mucho poder."}' -H "Content-Type: application/json" -X POST http://localhost:8038/annotate
```

# The model
The model uses the models for token classification from Huggingface Transformers release 3.0.2. (e.g. [XLMRobertaForTokenClassification](https://github.com/huggingface/transformers/blob/v4.3.0-release/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L141)). The `forward` function of the model can be found [here](https://github.com/huggingface/transformers/blob/v4.3.0-release/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L141): the transformer + dropout + linear.

Some more details about the transforming of the data prior to feeding into the transformer below. The `srl_utils.py` file includes reading of the file data into a format usable by the model. `SRLDataset` inherits from `torch.utils.data.dataset.Dataset`. It takes in a file and processing instructions then calls the corresponding `read_[filetype]_examples_from_directory` function. (Note that these readers are only written for Ontonotes and BETTER input data. If you would like to use this model for another dataset, you will likely need towrite another version of this method for that dataset.) Once examples have been read from `read_[filetype]_examples_from_directory`, they are converted into features using the `convert_examples_to_append_features` function. 
- the `tokens` from which all inputs are processed is the following sequence: tokenized sentence, separator token(s), tokeniezd predicate, separator token(s)
- `input_ids` is a [translation of the `tokens` into their corresponding IDs, according to the tokenizer](https://github.com/CogComp/SRL-Spanish/blob/main/srl_utils.py#L447).
- `attention_mask` is a binary vector the same length as the `input_ids` with [`1` where `input_ids` represents the tokens and `0` elsewhere](https://github.com/CogComp/SRL-Spanish/blob/main/srl_utils.py#L449).
- `token_type_ids` is a vector that distinguishes [which part of the input corresponds to the sentence part of the tokens, which to the predicate, which to the separators, and which to the padding](https://github.com/CogComp/SRL-Spanish/blob/main/srl_utils.py#L424).
- `labels` is a vector of the [corresponding IDs of the labels of the `tokens`](https://github.com/CogComp/SRL-Spanish/blob/main/srl_utils.py#L407).

# Contact
Questions: contact Celine at celine.y.lee@gmail.com
