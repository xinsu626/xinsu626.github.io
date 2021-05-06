---
layout: post
title:  "INFO539: Using the T5 model for clinical multi-task learning"
date:   2021-04-04 22:33:54 -0700
categories: INFO 539
---
* TOC
{:toc}
# Introduction

This tutorial is my final project in INFO 539 Statistical Natural Language Processing at The University of Arizona. In this tutorial, we will use Pytorch library and Hugging Face Transformers library to fine-tune the Text-to-Text Transfer Transformer (T5) ([Raffel et al., 2020](https://arxiv.org/abs/1910.10683)) model in the form of multi-task learning for clinical negation detection and clinical semantic textual similarity (STS) tasks. We will also investigate the different freezing strategies (freeze encoder or decoder during fine-tuning). 

### Tools and Model

The tools we will use are Pytorch 1.8.1, [Hugging Face Transformers](https://github.com/huggingface/transformers) 4.4.2 and a pre-trained T5-base model. [Pytorch](https://pytorch.org/) is a Python library for deep learning that can be used to build and train a variety of different deep learning models. The version of Pytorch that we will use in this tutorial is 1.8.1. Transformers is a Python library from Hugging Face that implements various transformer-based state-of-art natural language processing models. The version we will be using in this tutorial is 4.4.2.

The model we will build and train is the pre-trained T5 model. The T5 model is a transformer-based sequence-to-sequence model (generative model). It is composed of a transformer-based encoder and a transformer-based decoder. Its input and output are both a sequence of words respectively. Encoder encodes the input sequence into some representations, and then decoder will decode these representations into the output sequence. The T5 model has been pre-trained on a large corpus of unlabeled text and has been widely applied to tasks such as machine translation and text summarization. It has achieved state-of-art performance on many natural language processing tasks. By using the T5 model, we aim to transfer the knowledge it learns from pre-training on a large amount of general domain data to our clinical negation detection and STS tasks. We will use multi-task learning to further improve the model's performance on both tasks. In this tutorial, we will use the T5-base implementation from the Hugging Face Transformers library.

### Data and Tasks

**Negation Detection**: Negation detection task is a subtask in [SemEval 2021 Task 10](https://machine-learning-for-medical-language.github.io/source-free-domain-adaptation/). The goal of the negation detection task is to predict whether the clinical event in a sentence is negated by its context. This is a binary sentence classification task. For example, given a clinical event *diarrhea* and the sentence *Has no `<e>`diarrhea`</e>` and no new lumps or masses*, the goal is to predict that *diarrhea* is negated by its context. For the T5 model, the input of this task is a sentence with the entity and the output is negated or not negated. Below are some artificially generated data samples.

```
# Sample 1
Sentence: Has no <e>diarrhea</e> and new lumps or masses
Label: -1 (not negated)

# Sample 1
Sentence: Has no diarrhea and no <e>new lumps or masses</e>
Label: 1 (negated)
```

**Clinical Semantic Textual Similarity (STS) Measurement**: Clinical STS is a subtask of [2019 n2c2 share task](https://n2c2.dbmi.hms.harvard.edu/track1). The goal of Clinical STS is to predict the semantic similarity score of a clinical sentence pair. The similarity score is a real number that takes values in the range 0 to 5. 0 means that the semantics of the two sentences are completely different. 5 means that the semantics of the two sentences are identical. This is a regression task. For the T5 model, the input of this task is a sentence pair and the output is a similarity score in string format. Below are some artificially generated data samples.

```
# Sample 1
Sentence 1: The patient feels headache and stomach pain.
Sentence 2: The patient came in and said she felt a headache and stomach pain.
Similarity Score: 4.5

# Sample 2
Sentence 1: Require the patient to take a vitamin D tablet every two days.
Sentence 2: The patient feels headache and stomach pain.
Similarity Score: 0
```



The data in both tasks are from the clinical institution's electronic medical record (EMR). They are all publicly available research data sets, but access to them requires approval and signing the data use agreements (UDA). They can be obtained through [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/) and their corresponding share tasks. For negation detection, we will use the development set in the share task data and divide it into a training set and a test set at a ratio of 80% : 20%​. For clinical STS, we will use the same data split as in the share task. The models will be trained and tuned on the training set (the training set will be further split into a training set and a development set for each task), and tested on the test set to get the final performance. In the negation detection task, the F1 score will be used as the main evaluation metric. At the same time, precision and recall scores will also be reported. In clinical STS, Pearson correlation coefficient will be used as the main evaluation metric. The size and evaluation metrics of these two data sets are shown in the following table.

| Task               | Train Size | Test Size | Metric                          |
| :----------------- | ---------- | --------- | ------------------------------- |
| Negation Detection | 4436       | 1109      | F1, Precision, Recall           |
| Clinical STS       | 1641       | 441       | Pearson correlation coefficient |

### Methods

We will fine-tune the T5-base model and experiment with the following freezing strategy (MT is multi-task learning and FT is fine-tuning).

- MT (freeze encoder) + FT (freeze decoder): First freeze the encoder for multi-task learning on both tasks to fine tune the decoder. Then freeze the decoder and fine tune the encoder on two tasks separately.
- MT (freeze decoder) + FT (freeze encoder): First freeze decoder for multi-task learning on two tasks to fine tune the encoder. Then freeze the encoder and fine tune the decoder on two tasks separately.
- MT + FT: Perform multi-task learning on two tasks to fine-tune the entire model. Then fine-tune the entire model on both tasks separately.
- MT only: Only perform multi-task learning for both tasks and do not fine-tune for single tasks.
- FT only (baseline models): We will compare the performance of the above four kinds of models with the performance of the T5 model, which is only fine-tuned for the single tasks without any multi-task learning. 

In total, we will train 9 models.

# Implementation and Experiments

### Tools Installation 

The OS environment I am using is `Linux Ubuntu 20.04.2` and the version of my Python is `3.8`.  The following installation commands need to be executed at the command line.

**PIP**

The installation tool we will use is the [pip package manager](https://pip.pypa.io/en/stable/). PIP can be installed with the following commands (other pip installation methods can be found [here](https://pip.pypa.io/en/stable/installing/)).

```bash
# Download the get-pip.py file from web
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Then install the pip for python3.8
python3.8 get-pip.py
```

**Pytorch**

We can install Pytorch with the following command. Note that this tutorial assumes that you have a working Nvidia GPU and that the corresponding CUDA toolkit and driver are already installed. This is because it is very inefficient to fine-tune the T5 model directly on the CPU. The installation commands here work with CUDA 11.1. If you have a different version of CUDA, you can go to [Pytorch's website](https://pytorch.org/) for the appropriate installation commands.

```bash
# Intall the Pytorch 
python3.8 -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**Hugging Face Transformers**

We can install the Hugging Face Transformers library by using the following command.

```bash
# Intsall the Huugingface Transformers library
python3.8 -m pip install transformers==4.4.2
```

**Other Dependencies**

You may also need the following dependencies in this tutorial.

```bash
# Install scikit-learn
python3.8 -m pip install scikit_learn==0.24.2

# Install sentencepiece for T5 tokenizer
python3.8 -m pip install sentencepiece==0.1.95

# Install scipy
python3.8 -m pip install scipy==1.6.3
```

### Project Structure

To better manage our code and avoid unnecessary bugs, we will store the source code in the following structure.

```
src
├── data
│   ├── __init__.py
│   ├── negation_data_provider.py
│   └── sts_data_provider.py
├── experiments
│   ├── __init__.py
│   └── run_t5_model.py
└── utils.py
```

All code used to read and load data from raw text files is stored in `src/data`. All experiment-related code (e.g. training and testing T5 models) is stored in `src/experiments`.  Other helper functions (functions that compute the performance of the models, etc.) will be stored in the `utils.py` file. The `__init__.py` files are just empty Python files.

### Read and Load the Data Sets

**Negation Data**

The raw text data for the Negation task are the following 4 text files in `tsv` format.

- `train.tsv`: It is the training data of negation, where each line is a training example (a sentence with an entity).
- `train_labels.tsv`: It is the labels of the training data of negation, where each row is 1 or -1. 1 means the entity in the sentence is negated by its context. -1 means not negated.
- `test.tsv`：It is the test data of negation task, which has the same format as `train.tsv`.
- `test_labels.tsv`：It is the labels of the test data of the negation task in the same format as `train_labels.tsv`.

Let's write a Python class for reading in the raw text data of the negation task and save it in `src/data/negation_data_provider.py`.

```python
class DataProviderNegation(object):

    def __init__(self, corpus_path, label_path=None) -> None:
        # The path to train.tsv or test.tsv
        self.corpus_path = corpus_path
        # The path to train_labels.tsv or test_labels.tsv
        self.label_path = label_path

    @staticmethod
    def load_text_file(path):
        """Read the tsv file"""
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [i.replace('\n', '') for i in lines]

        return lines

    def get_text_data(self):
        # Load the examples
        examples = self.load_text_file(self.corpus_path)
        outputs = (examples,)

        # Load label_ids
        if self.label_path is not None:
            label_ids = self.load_text_file(self.label_path)

            # Map the label_ids to int
            label_ids = list(map(int, label_ids))

            outputs += (label_ids,)

        return outputs
```

**Clinical STS Data**

The data for the Clinical STS task is stored in the following files.

- `clinicalSTS2019.train.txt`: it is the training data for the clinical STS task, where each row is a training example (a sentence pair) and the corresponding label (similarity score of the sentence pair): `sentence_a \t sentence_b \t similarity score \n`.
- `clinicalSTS2019.test.txt`: It is the test data for the clinical STS task, where each row contains only one sentence pair: `sentence_a \t sentence_b \n`
- `clinicalSTS2019.test.gs.sim.txt`: it is the labels (similarity scores) of the test data of the clinical STS task, where each row is the similarity score of the corresponding sentence pair.

Let's write a Python class for reading in the raw text data of the clinical STS task and save it in `src/data/sts_data_provider.py`.

```python
class STSDataProvider:
    def __init__(self,
                 corpus_path,
                 label_path=None):
        self.corpus_path = corpus_path
        self.label_path = label_path

    def load_train_data(self):
        with open(self.corpus_path, 'r') as f:
            lines = f.readlines()
            examples_a = []
            examples_b = []
            scores = []
            for line in lines:
                split_line = line.replace('\n', '').split('\t')
                examples_a.append(split_line[0])
                examples_b.append(split_line[1])
                scores.append(split_line[2])
        return examples_a, examples_b, scores

    def load_test_data(self):

        # Get the gold scores
        with open(self.label_path, 'r') as f:
            scores = f.readlines()
            scores = [i.replace('\n', '') for i in scores]

        # Get the examples
        with open(self.corpus_path, 'r') as f:
            lines = f.readlines()
            examples_a = []
            examples_b = []
            for line in lines:
                split_line = line.replace('\n', '').split('\t')
                examples_a.append(split_line[0])
                examples_b.append(split_line[1])
        return examples_a, examples_b, scores
```

### Implement Performance Helper Functions

We need to implement functions that calculate the performance of the model in each task. Here all the functions are stored in `src/utils.py`.

In the clinical STS task, we will use the `pearsonr` from `scipy` to calculate the performance of the model.

```python
def sts_performance(preds, labels):
    """
    Calculate the pearson correlation.
    :param preds: a predicted list of similarity scores.
    :param labels: a list of true similarity scores.
    :return: pearson correlation score.
    """
    pearson_corr = pearsonr(preds, labels)[0]

    return pearson_corr
```

In the negation task, we will use the `f1_score`, `precision_score`, and `recall_score` functions from [scikit-learn](https://scikit-learn.org/) to compute the F1, Precision, and Recall scores.

```python
def negation_performance(preds, labels):
    """
    Calculate the F, P, and R scores.
    :param preds: a list of predicted labels.
    :param labels: a list of true labels.
    :return: a dictionary of F1, Precision and recall scores.
    """
    f1 = f1_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)

    performance = {
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    return performance
```

### Implement Reusable Training and Testing Code  

In this section, we will implement the code for fine-tuning and testing the T5 model. We will use Python's `argparse` library to pass in different configuration information from the command line to make our code reusable. What we should avoid is having different Python files for training different configurations of the T5 model, and different Python files for testing different configurations of the T5 model. The risk is that having too many different files to manage may lead to inconsistencies or other bugs between the files. 

All the code for this section will be stored in `src/run_t5_model.py`. The following is a step-by-step guide.

**Step 1**: import the Python libraries we need to use.

```python
import argparse
import os
from datetime import datetime
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import \
    (T5ForConditionalGeneration,
     T5Tokenizer,
     Seq2SeqTrainer,
     Seq2SeqTrainingArguments,
     PreTrainedTokenizer,
     EarlyStoppingCallback)
from sklearn.model_selection import train_test_split

from src.data.negation_data_provider import DataProviderNegation
from src.data.sts_data_provider import STSDataProvider
from src.utils import negation_performance, sts_performance
```

**Step 2**: Set the command line arguments that we will use and their default values and data types. The arguments we need to set in this section are:

- Paths to raw text data for different tasks: all arguments with `corpus_path` and `label_path` as suffixes.
- The path to the directory used to save the output: `output_path`.
- Local path or name of the model: `model_name`, `tokenizer_name`.
- Hyperparameters of the T5 model: arguments that are labeled as Training arguments.
- Some other arguments that are used to control the behavior of our code: the arguments that are labeled as MISC.

```python
parser = argparse.ArgumentParser()

# Negation data args
parser.add_argument("--negation_train_corpus_path", type=str)
parser.add_argument("--negation_train_label_path", type=str)
parser.add_argument("--negation_test_corpus_path", type=str)
parser.add_argument("--negation_test_label_path", type=str)

# STS data args
parser.add_argument("--sts_train_corpus_path", type=str)
parser.add_argument("--sts_test_corpus_path", type=str)
parser.add_argument("--sts_test_label_path", type=str)

# Directory to save all outputs
parser.add_argument("--output_path", type=str)

# Model args
parser.add_argument("--model_name", type=str, default="t5-base")
parser.add_argument("--tokenizer_name", type=str, default="t5-base")

# MISC
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--do_train', type=str, default='True')
parser.add_argument("--parts_to_freeze", type=str, default="none")  # none, encoder, decoder
parser.add_argument("--tasks_to_train", type=str, default='all')  # all, sts, negation, none
parser.add_argument("--tasks_to_eval", type=str, default='all')  # all, sts, negation, none

# Training argument
parser.add_argument("--max_source_length", type=int, default=128)
parser.add_argument("--max_target_length", type=int, default=5)
parser.add_argument('--per_gpu_train_batch_size', type=int, default=8)
parser.add_argument('--per_gpu_eval_batch_size', type=int, default=16)
parser.add_argument('--grad_accum_steps', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--warmup_steps', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_beams', type=int, default=3)
```

**Step 3**: We will use Hugging Face's `Seq2SeqTrainer` API to train and test our model. It accepts passed in data in the form of `torch.utils.data.dataset.Dataset`. So we need to implement a [Pytorch map-style dataset object](https://pytorch.org/docs/stable/data.html#map-style-datasets) for storing our training and test data, and it can be indexed (implement `__getitem__()`  and `__len()__` protocols).

```python
class T5Dataset(Dataset):
    def __init__(self,
                 pre_trained_tokenizer: PreTrainedTokenizer,
                 max_source_length: int,
                 max_target_length: int,
                 examples: List[str],
                 labels: List[str] = None):
        # T5 tokenizer
        self.tokenizer = pre_trained_tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.examples = examples
        self.labels = labels

    def __getitem__(self, idx):
        source = self.tokenizer(
            self.examples[idx],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        target = self.tokenizer(
            self.labels[idx],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')

        input_ids = source.input_ids.squeeze()
        attention_mask = source.attention_mask.squeeze()
        decoder_input_ids = target.input_ids.squeeze()
        decoder_attention_mask = target.attention_mask.squeeze()

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=decoder_input_ids)

    def __len__(self):
        return len(self.examples)
```

**Step 4**: We will parse the command line arguments and update our output_path arguments based on the current time. Since we may need to run the model with the same configuration multiple times, we want to store them in a directory named after the date for future use.

```python
# Parse the command line arguments.
args = parser.parse_args()

# Updates the output path and make the directory to save the outputs
args.output_path = os.path.join(args.output_path, datetime.now().strftime('%Y-%m-%d-%H-%M'))
os.makedirs(args.output_path, exist_ok=True)
```

**Step 5**: Probe the GPUs available in the current machine to make sure our code is able to see the GPUs.

```python
# Count the available number of GPUs and save the number to args.
print("-" * 80)
args.num_gpus = torch.cuda.device_count()
print("Number of GPUs = {}".format(args.num_gpus))
```

**Step 6**: Load the model stored locally or via the Hugging Face model hub with the given model's local path or model name.

```python
# Load the models
print('-' * 80)
print(f'Load model: {args.model_name}')
tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(args.model_name)
```

**Step 7**: Based on the input arguments to determine whether the encoder or decoder of the T5 model needs to be frozen (without updating the values of the parameters during fine-tuning). After freezing, we need to print out the total number of parameters and the total number of trainable parameters to ensure that our code is successfully freezing the part we want to freeze.

```python
# Freeze the model
if args.parts_to_freeze == "encoder":
    print('Freeze the encoder.')
    for p in model.encoder.parameters():
    p.requires_grad = False

if args.parts_to_freeze == "decoder":
    print('Freeze the decoder')
    for p in model.decoder.parameters():
    p.requires_grad = False

# Check the number of parameters could be trained
total_params = sum(p.numel() for p in model.parameters())
total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters = {total_params / 10 ** 7:.1f} M")
print(f"Total number of trainable parameters = {total_train_params / 10 ** 7:.1f} M")
```

**Step 8**: Load the data of the negation task and do the following processing works based on the convention of the T5 model input and output formats.

- Converts -1 and 1 in the negation labels to not negated and negated, respectively.
- Prefix each input example with the prefix `negation detection:`.

```python
# Negation
train_negation_data_provider = DataProviderNegation(corpus_path=args.negation_train_corpus_path,
                                                    label_path=args.negation_train_label_path)
test_negation_data_provider = DataProviderNegation(corpus_path=args.negation_test_corpus_path,
                                                   label_path=args.negation_test_label_path)
train_negation_examples, train_negation_label_ids = train_negation_data_provider.get_text_data()
test_negation_examples, test_negation_label_ids = test_negation_data_provider.get_text_data()

# Convert the numeric label ids -1 and 1 to not negated and negated
train_negation_labels = ['negated' if i == 1 else "not negated" for i in train_negation_label_ids]
test_negation_labels = ['negated' if i == 1 else "not negated" for i in test_negation_label_ids]

# Add task specific prefix to the examples
train_negation_examples = ['negation detection: ' + i for i in train_negation_examples]
test_negation_examples = ['negation detection: ' + i for i in test_negation_examples]
```

**Step 9**: Load the clinical STS data and convert the input to `clinical sts: sentence1: {words in sentence 1} sentence2: {words in sentence 2}` based on the convention of the T5 model input format.

```python
# STS
train_sts_data_provider = STSDataProvider(corpus_path=args.sts_train_corpus_path)
test_sts_data_provider = STSDataProvider(corpus_path=args.sts_test_corpus_path,
label_path=args.sts_test_label_path)
train_sts_examples_a, train_sts_examples_b, train_sts_scores = train_sts_data_provider.load_train_data()
test_sts_examples_a, test_sts_examples_b, test_sts_scores = test_sts_data_provider.load_test_data()

# Add task specific prefix to the examples and combine them together
train_sts_examples = []
test_sts_examples = []
for a, b in zip(train_sts_examples_a, train_sts_examples_b):
train_sts_examples.append(f"clinical sts: sentence1: {a} sentence2: {b}")
for a, b in zip(test_sts_examples_a, test_sts_examples_b):
test_sts_examples.append(f"clinical sts: sentence1: {a} sentence2: {b}")
```

**Step 10**: Split the training dataset of negation task and clinical STS task into a training set and a development set in the ratio of 80% : 20%. We will use the development set to tune the hyperparameters. Specifically, in this tutorial, we will only tune the number of training epochs.

```python
# Split the training sets into train and test use 80% and 20% ratio
# negation
train_negation_examples, dev_negation_examples, train_negation_labels, dev_negation_labels = train_test_split(
    train_negation_examples,
    train_negation_labels,
    test_size=0.2,
    random_state=args.seed
)

# sts
train_sts_examples, dev_sts_examples, train_sts_scores, dev_sts_scores = train_test_split(
    train_sts_examples,
    train_sts_scores,
    test_size=0.2,
    random_state=args.seed
)

print(f"Number of negation train examples = {len(train_negation_label_ids)}")
print(f"Number of negation test examples = {len(test_negation_label_ids)}")
print(f"Number of STS train examples = {len(train_sts_scores)}")
print(f"Number of STS test examples = {len(test_sts_scores)}")
```

**Step 11**: Build the final training and development sets for fine-tuning the T5 model based on the arguments of the common line. Here we have three options:

1. Only the training and development set data of the negation task are used.
2. Only the training and development set data from the clinical STS task are used.
3. The training and development sets of both tasks are combined together to obtain the final training and development sets (multi-task learning).

```python
final_train_examples = []
final_train_labels = []
final_dev_examples = []
final_dev_labels = []

if args.tasks_to_train == "negation":
	final_train_examples = train_negation_examples
	final_train_labels = train_negation_labels
	final_dev_examples = dev_negation_examples
	final_dev_labels = dev_negation_labels
elif args.tasks_to_train == "sts":
	final_train_examples = train_sts_examples
	final_train_labels = train_sts_scores
	final_dev_examples = dev_sts_examples
    final_dev_labels = dev_sts_scores
elif args.tasks_to_train == "all":
	final_train_examples = train_negation_examples + train_sts_examples
    final_train_labels = train_negation_labels + train_sts_scores
	final_dev_examples = dev_negation_examples + dev_sts_examples
	final_dev_labels = dev_negation_labels + dev_sts_scores
else:
	raise ValueError("The task_to_train arg have to be specified: {all, negation, sts}")
```

**Step 12**: Use the `T5Dataset` class we implemented in step 3 to build training, development and test datasets.

```python
# Build the train dataset and dev dataset.
train_dataset = T5Dataset(pre_trained_tokenizer=tokenizer,
                          examples=final_train_examples,
                          labels=final_train_labels,
                          max_source_length=args.max_source_length,
                          max_target_length=args.max_target_length)
dev_dataset = T5Dataset(pre_trained_tokenizer=tokenizer,
                        examples=final_dev_examples,
                        labels=final_dev_labels,
                        max_source_length=args.max_source_length,
                        max_target_length=args.max_target_length)

# Prepare the test sets for these two tasks
negation_test_dataset = T5Dataset(pre_trained_tokenizer=tokenizer,
                                  examples=test_negation_examples,
                                  labels=test_negation_labels,
                                  max_source_length=args.max_source_length,
                                  max_target_length=args.max_target_length)
sts_test_dataset = T5Dataset(pre_trained_tokenizer=tokenizer,
                             examples=test_sts_examples,
                             labels=test_sts_scores,
                             max_source_length=args.max_source_length,
                             max_target_length=args.max_target_length)
```

**Step 13:** Initialize the `Seq2SeqTrainingArguments` and `Seq2SeqTrainer` objects from Hugging Face Transformers. We will use the `Seq2SeqTrainer` API to fine tune the T5 model. The `Seq2SeqTrainingArguments` is used to store the parameters that will be used when training and testing the model. Here we will also use the `EarlyStoppingCallback` to tune our optimal training epochs on the development set. The basic idea of `EarlyStoppingCallback` is to test the performance of the model on the development set after each training epoch ( here the value of the loss function is calculated), and if the model does not perform as well as the current best for 3 consecutive epochs, then stop training and return the best model.

```python
# Setup the trainer
training_args = Seq2SeqTrainingArguments(output_dir=args.output_path,
                                         per_device_train_batch_size=args.per_gpu_train_batch_size,
                                         per_device_eval_batch_size=args.per_gpu_eval_batch_size,
                                         gradient_accumulation_steps=args.grad_accum_steps,
                                         learning_rate=args.learning_rate,
                                         num_train_epochs=args.epochs,
                                         warmup_steps=args.warmup_steps,
                                         save_total_limit=1,
                                         logging_steps=100,
                                         seed=args.seed,
                                         disable_tqdm=True,
                                         save_steps=10000,
                                         evaluation_strategy="epoch",
                                         predict_with_generate=True,
                                         load_best_model_at_end=True)
trainer = Seq2SeqTrainer(model=model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=dev_dataset,
                         tokenizer=tokenizer,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

```

**Step 14**: Print out some key information to make sure our settings are as expected, and then start training. After the training is done, save the model to `output_path`.

```python
# Train the model
if eval(args.do_train):
    print("-" * 80)
    print("Start training the model")
    print('=' * 80)
    print({'num_examples': len(trainer.train_dataset),
           'batch': trainer.args.per_device_train_batch_size,
           'epochs': trainer.args.num_train_epochs,
           'grad_accum': trainer.args.gradient_accumulation_steps,
           'warmup': trainer.args.warmup_steps}.__str__())
    trainer.train()
    print('=' * 80)
    print("Saving the model ...")
    trainer.model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
```

**Step 15**: Test the final performance of the model on the test set of negation. First, we will use the trained model to do inference on the test set, and decode the results into string format. Finally, we will do the following processing on the predicted labels to get the final performance.

- If the model predicts a `not negated` label then convert to -1.
- If the model predicts a label that is `negated` then convert to 1.
- If the predicted label is neither `negated` nor `not negated`, then it is converted to -1.

```python
if args.tasks_to_eval == "negation" or args.tasks_to_eval == "all":
    print('-' * 80)
    print("Start running negation evaluation.")
    negation_results = trainer.predict(test_dataset=negation_test_dataset,
                                       max_length=args.max_target_length,
                                       num_beams=args.num_beams)
    # decode it
    negation_predictions = tokenizer.batch_decode(negation_results.predictions,
                                                  skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
    negation_predictions = [pred.strip() for pred in negation_predictions]

    # Convert back to numeric values
    negation_predictions_ids = []
    for p in negation_predictions:
        if p == 'not negated':
            negation_predictions_ids.append(-1)
        elif p == 'negated':
			negation_predictions_ids.append(1)
        else:
			negation_predictions_ids.append(-1)

	# Evaluate the performance
    print(negation_performance(preds=negation_predictions_ids,
                               labels=test_negation_label_ids))
```

**Step 16**: Test the final performance of the model on the test set of clinical STS. As above, we will also use the trained model to do inference on the test set, and decode the results into string format. Finally, all the predictions are converted to float data type, or set to 0 if they cannot be converted.

```python
if args.tasks_to_eval == "sts" or args.tasks_to_eval == "all":
    print('-' * 80)
    print('Start running sts evaluation')
    sts_results = trainer.predict(test_dataset=sts_test_dataset,
                                  max_length=args.max_target_length,
                                  num_beams=args.num_beams)

    # decode it
    sts_predictions = tokenizer.batch_decode(sts_results.predictions,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
    sts_predictions = [pred.strip() for pred in sts_predictions]

    # convert back to numeric values
    sts_predictions_numeric_values = []
    for p in sts_predictions:
        try:
            sts_predictions_numeric_values.append(float(p))
		except ValueError:
			sts_predictions_numeric_values.append(0)
			test_sts_scores = [float(i) for i in test_sts_scores]

	print(sts_performance(preds=sts_predictions_numeric_values, labels=test_sts_scores))
```

### Run experiments

Up to now, we have implemented all the Python code we need, and now we will train and test the model from the command line.

#### Negation FT

We use the following commands to directly fine-tune T5 model on negation training set and test on test set.

```bash
# Set the python PATH
export  PYTHONPATH=/path/to/technical-tutorial-xinsu626:${PYTHONPATH}
export PYTHONPATH=/path/to/technical-tutorial-xinsu626/src:${PYTHONPATH}
export PYTHONPATH=/path/to/technical-tutorial-xinsu626/src/data:${PYTHONPATH}
export PYTHONPATH=/path/to/technical-tutorial-xinsu626/src/experiments:${PYTHONPATH}

# Run
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/negation-baseline \
--model_name t5-base \
--do_train True \
--parts_to_freeze none \
--tasks_to_train negation \
--tasks_to_eval negation
```

#### Clinical STS FT

We use the following commands to directly fine-tune T5 model on clinical STS training set and test on test set.

```bash
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/sts-baseline \
--model_name t5-base \
--do_train True \
--parts_to_freeze none \
--tasks_to_train sts \
--tasks_to_eval sts
```

#### MT (freeze encoder) + FT (freeze decoder)

 ```bash
#Do multi-task learning: freeze the encoder and do multi-task learning
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/freeze_encoder \
--model_name t5-base \
--do_train True \
--parts_to_freeze encoder \
--tasks_to_train all \
--tasks_to_eval none
 ```

```bash
# Then fine-tuning for negation: unfreeze the encoder and freeze decoder and fine-tune on negation
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/unfeerze_encoder_freeze_decoder_negation \
--model_name/path/to/freeze_encoder/checkpoint \
--do_train True \
--parts_to_freeze decoder \
--tasks_to_train negation \
--tasks_to_eval negation
```

```bash
# Then fine-tuning for STS:unfreeze the encoder and freeze decoder and fine-tune on clinical sts
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/unfeerze_encoder_freeze_decoder_sts \
--model_name /path/to/freeze_encoder/checkpoint \
--do_train True \
--parts_to_freeze decoder \
--tasks_to_train sts \
--tasks_to_eval sts
```

#### MT (freeze decoder) + FT (freeze encoder)

```bash
# Multi-task learning: freeze the encoder and do multi-task learning
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/freeze_decoder \
--model_name t5-base \
--do_train True \
--parts_to_freeze decoder \
--tasks_to_train all \
--tasks_to_eval none
```

```bash
# Then fine-tuning for negation: unfreeze the decoder and freeze encoder and fine-tune on negation
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/unfeerze_decoder_freeze_encoder_negation \
--model_name/path/to/freeze_decoder/checkpoint \
--do_train True \
--parts_to_freeze encoder \
--tasks_to_train negation \
--tasks_to_eval negation
```

```bash
# Then fine-tuning of STS: unfreeze the decoder and freeze encoder and fine-tune on sts
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/unfeerze_decoder_freeze_encoder_sts \
--model_name/path/to/freeze_decoder/checkpoint \
--do_train True \
--parts_to_freeze encoder \
--tasks_to_train sts \
--tasks_to_eval sts
```

#### MT

```bash
# Multi-task learning without freezing
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/multi-task-learning \
--model_name t5-base \
--do_train True \
--parts_to_freeze none \
--tasks_to_train all \
--tasks_to_eval all
```

#### MT + FT

```bash
# Use the fine-tuned model in multi-task learning and fine-tune it on negation
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/multi-task-learning-and-fine-tune-on-negation \
--model_name /path/to/multi-task-learning/checkpoint \
--do_train True \
--parts_to_freeze none \
--tasks_to_train negation \
--tasks_to_eval negation
```

```bash
# Use the fine-tuned model in multi-task learning and fine-tune it on negation
python3.8 /path/to/run_t5_model.py \
--negation_train_corpus_path /path/to/negation-data/train.tsv \
--negation_train_label_path /path/to/train_labels.tsv \
--negation_test_corpus_path /path/to/test.tsv \
--negation_test_label_path /path/to/test_labels.tsv \
--sts_train_corpus_path /path/to/clinicalSTS2019.train.txt \
--sts_test_corpus_path /path/to/clinicalSTS2019.test.txt \
--sts_test_label_path /path/to/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /path/to/multi-task-learning-and-fine-tune-on-sts \
--model_name /path/to/multi-task-learning/checkpoint \
--do_train True \
--parts_to_freeze none \
--tasks_to_train sts \
--tasks_to_eval sts
```

# Results

I summarized the results of training and testing the above model on my machine in the following table (FT is fine-tuning and MT is multi-task learning).

### Negation

|                                            | F         | P     | R     |
| ------------------------------------------ | --------- | ----- | ----- |
| FT only (baseline)                         | 0.922     | 0.941 | 0.905 |
| MT  (freeze encoder) + FT (freeze decoder) | 0.914     | 0.973 | 0.862 |
| MT (freeze decoder) + FT (freeze encoder)  | **0.928** | 0.964 | 0.895 |
| MT only                                    | 0.921     | 0.915 | 0.929 |
| MT + FT                                    | 0.919     | 0.949 | 0.890 |

### Clinical STS

|                                            | Pearson Correlation |
| ------------------------------------------ | ------------------- |
| FT only (baseline)                         | 0.691               |
| MT  (freeze encoder) + FT (freeze decoder) | 0.706               |
| MT (freeze decoder) + FT (freeze encoder)  | **0.720**           |
| MT only                                    | 0.625               |
| MT + FT                                    | 0.676               |

As we can see, the best-performing strategy on both tasks is MT (freeze decoder) + FT (freeze encoder): first freeze the decoder for multi-task learning and fine-tune only the encoder, then freeze the encoder and fine-tune only the decoder on both tasks separately.

# Using Containerized Code

All the code in this tutorial is containerized using [Docker](https://www.docker.com/) (The tutorial for installing docker can be found [here](https://docs.docker.com/engine/install/ubuntu/)). To replicate the experiments, you can pull a docker image from my docker hub and train the model in the container by connecting to the docker's interactive session. Below I will provide an example of how to train the STS baseline model using this docker image. You can also reproduce other models based on this template.

Since we will use GPUs to train the models, we first need to install the Nvidia docker toolkit so that we can use the GPUs on the host machine inside the container. This installation command is from [Nvidia](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).

```bash
# Setup the stable repository and the GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
# Install the nvidia-docker2 package (and dependencies) after updating the package listing
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon 
sudo systemctl restart docker
```

We need to run the following command to pull the image from the docker hub.

```bash
docker pull xinsu1/teachnical-tutorial-xinsu626:latest
```

Now we will run the container interactively based on the image we just pulled. Also, we need to specify in the command the data directory we want to mount to the container and specify that we want to use all the GPUs on the host machine.

```bash
docker run -it --rm --gpus all -v /path/to/raw-data/directory/on/host:/technical-tutorial-xinsu626/raw-data xinsu1/teachnical-tutorial-xinsu626
```

Then we run the following command to train the baseline model of clinical STS in the container.

```bash
# Go to the source code
cd src/experiments/

# Set the PYTHONPATH
export PYTHONPATH=/technical-tutorial-xinsu626:${PYTHONPATH}
export PYTHONPATH=/technical-tutorial-xinsu626/src:${PYTHONPATH}
export PYTHONPATH=/technical-tutorial-xinsu626/src/data:${PYTHONPATH}
export PYTHONPATH=/technical-tutorial-xinsu626/src/experiments:${PYTHONPATH}

# Train the model
python3.8 /technical-tutorial-xinsu626/src/experiments/run_t5_model.py \
--negation_train_corpus_path /technical-tutorial-xinsu626/raw-data/negation-data/train.tsv \
--negation_train_label_path /technical-tutorial-xinsu626/raw-data/negation-data/train_labels.tsv \
--negation_test_corpus_path /technical-tutorial-xinsu626/raw-data/negation-data/test.tsv \
--negation_test_label_path /technical-tutorial-xinsu626/raw-data/negation-data/test_labels.tsv \
--sts_train_corpus_path /technical-tutorial-xinsu626/raw-data/n2c2-2019-track1-data/clinicalSTS2019.train.txt \
--sts_test_corpus_path /technical-tutorial-xinsu626/raw-data/n2c2-2019-track1-data/clinicalSTS2019.test.txt \
--sts_test_label_path /technical-tutorial-xinsu626/raw-data/n2c2-2019-track1-data/clinicalSTS2019.test.gs.sim.txt \
--output_path /technical-tutorial-xinsu626/sts-baseline \
--model_name t5-base \
--do_train True \
--parts_to_freeze none \
--tasks_to_train sts \
--tasks_to_eval sts
```

# References

Colin  Raffel,  Noam  Shazeer,  Adam  Roberts,  Katherine  Lee,  Sharan  Narang,  Michael  Matena,  Yanqi Zhou,  Wei  Li,  and  Peter  J.  Liu.  2020.   Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Re-search, 21(140):1–67.

