# Natural Language Inference

Team: 26

## Directory Structure

```
.
├── bert
│   ├── multi_nli
│   │   ├── bert_test.py
│   │   ├── bert_train.py
    |
│   └── snli
│       ├── bert_test.py
│       ├── bert_train.py
│  
├── bigru
│   ├── multi_nli
│   │   ├── bigru_test.py
│   │   ├── bigru_train.py
│   │  
│   └── snli
│       ├── bigru_test.py
│       ├── bigru_train.py
│       └── __pycache__
│           ├── bigru_test.cpython-38.pyc
│           └── bigru_train.cpython-38.pyc
├── bilstm
│   ├── multi_nli
│   │   ├── bilstm_test.py
│   │   ├── bilstm_train.py
│   │ 
│   └── snli
│       ├── bilstm_test.py
│       ├── bilstm_train.py
│  
├── glove.6B
│   └── glove.6B.300d.txt
├── logistic_regression
│   ├── multi_nli
│   │   ├── lr_test.py
│   │   ├── lr_train.py
│   │  
│   └── snli
│       ├── lr_test.py
│       ├── lr_train.py
│  
├── main.py
├── model
│   ├── bert
│   │   ├── multi_nli
│   │   │   ├── bert-nli.pt
│   │   │   ├── dev_data_mnli.csv
│   │   │   ├── test_data_mnli.csv
│   │   │   ├── tokenizer.pkl
│   │   │   └── train_data_mnli.csv
│   │   └── snli
│   │       ├── bert-nli.pt
│   │       ├── snli_1.0_dev.csv
│   │       ├── snli_1.0_test.csv
│   │       ├── snli_1.0_train.csv
│   │       ├── tokenizer.pkl
│   │       └── traindata.pkl
│   ├── bigru
│   │   ├── bi_gru_model.png
│   │   ├── multi_nli
│   │   │   ├── BiGRU.h5
│   │   │   ├── test_data.txt
│   │   │   └── tokenizer.pickle
│   │   └── snli
│   │       ├── BiGRU.h5
│   │       ├── test_data.pickle
│   │       └── tokenizer.pickle
│   ├── bilstm
│   │   ├── bi_lstm_model.png
│   │   ├── multi_nli
│   │   │   ├── BiLSTM.h5
│   │   │   ├── test_data.txt
│   │   │   └── tokenizer.pickle
│   │   └── snli
│   │       ├── BiLSTM.h5
│   │       ├── test_data.txt
│   │       └── tokenizer.pickle
│   └── logistic_regression
│       ├── multi_nli
│       │   ├── best_lr.pickle
│       │   ├── label_encoder.pickle
│       │   └── vectorizer.pickle
│       └── snli
│           ├── best_lr.pickle
│           ├── label_encoder.pickle
│           └── vectorizer.pickle
├── multinli_0.9
│   ├── multinli_0.9_dev.txt
│   └── multinli_0.9_test.txt
└── snli_1.0
    ├── snli_1.0_dev.txt
    ├── snli_1.0_test.txt
    └── snli_1.0_train.txt
```

- `model`: This directory has all the model and data files.
- `bert|bigru|bisltm|lr|elmo`: All these directories have 2 files inside them, one is for training, one is for evaluating the models.
- `main.py`: This is the main file to run as shown below

## Steps to execute

- Run the below commands

```sh
$ python main.py --test --dataset=snli bilstm

$ python main.py --train --dataset=multi_nli bert
```

> Due to dependencies issues, elmo could not be integrated into the `main.py` file. Thus has to run separately.

```sh
$ cd elmp
$ python main.py
```

> There is one python notebook that shows **Sentence Embedding** work that we have done using NLI datasets.

---

# Project Objective

The objective of the project is to understand the task of Natural Language Inference in the field of Natural Language Processing. Natural Language Inference deals with understanding the relationship between two given sentences. It tries to identify whether a given sentence, called the "Hypothesis," can be derived, inferred, or deduced from another given sentence, called the "Premise," or not. The relationship can be classified into three different classes/labels:

- **Contradiction** – refers to a situation when both premise and hypothesis cannot be true at the same time.
- **Entailment** - refers to a situation where hypotheses can be derived or inferred from the given premise.
- **Neutral** – refers to a situation where there is not enough information in the premise to infer or derive the given hypothesis.

# About Datasets

For this task, we have explored two famous available datasets – SNLI and MultiNLI.

## SNLI

SNLI stands for Stanford Natural Language Inference. It is a benchmark dataset for natural language inference tasks.

- All the premises are the image captions from the Flicker30 dataset, making SNLI somewhat genre-restricted.
- All the hypotheses are written by Crowd-workers, corresponding to a premise, crowd workers will write three sentences, one for each class.
- Total of 550,152 training examples with 10,000 as dev samples and 10,000 as test samples, balanced equally across three classes.
- Mean token length in SNLI dataset:
  - For premise – 12.9
  - For hypothesis – 7.4
- Approximately 56,951 examples are validated by four additional annotators with 91.2% of gold labels matched with the author's labels.
- For SNLI dataset, the overall Fleiss' kappa is 0.70, which is defined as the degree of agreement between the annotators, based on both categorical labels and similarity matrix.
- Progress on SNLI dataset for Natural Language Inference task.

## MultiNLI

MultiNLI stands for Multi-Genre Natural Language Inference. It is also another benchmark dataset for natural language inference tasks. It is an extension of the SNLI dataset, with a more diverse set of genres and sources, making it a more challenging dataset to train and evaluate NLI models.

- Train Premises in MultiNLI are contributed from five genres:
  - Fiction: works from 1912 – 2010, spanned across many genres.
  - Govt. information – available public reports, letters, speeches, Govt. websites, etc.
  - The Slate website.
  - The Switchboard corpus – Telephonic conversation.
  - Bertlitz travel guide.
- In addition to the above genres, premises from some other genres are also present in dev and test datasets like:
  - From 9/11 reports.
  - From fundraising letters.
  - Nonfiction from Oxford University Press.
  - Verbatim – articles about linguistics.
- Total of approximately 392,702 training examples with 20,000 as dev samples and 20,000 as test samples.
- Approximately 19,647 samples are validated by four additional annotators with 92.6% of gold labels matched with the author's labels.
- Test dataset of MultiNLI is only available on Kaggle and in the form of competition.
- Progress on SNLI dataset for Natural Language Inference task.
- MultiNLI dataset is filled with distributed annotations that help to perform out-of-the-box error analysis.

# Exploratory Data Analysis

To get the look and feel of the data, some basic data analysis is done. It helps us to understand and gain insights into our data before starting any model or making decisions based on it.

Visualization of the distribution of each gold label (Contradiction, Neutral, Entailment) has been done to know the distribution of each gold label in the dataset or to know whether there is some bias towards any label is present or not.

Clearly, it can be seen that the dataset contains almost an equal proportion of Entailment, Contradiction, and Neutral statements.

We have also analyzed the various statistical parameters of premise and hypothesis that helped to better understand SNLI & MultiNLI dataset.

Also, the most frequently occurred 20 words in premise sentences are identified and viewed.

# Techniques Used

We have utilized 4 techniques to address the challenge of Natural Language Inference, namely:

- Logistic Regression
- Bi-directional LSTM
- Bi-directional GRU
- ELMo
- BERT

## Logistic Regression

First things first, Logistic Regression has been applied on both SNLI and MultiNLI datasets.

- Data pre-processing.
- Encoding gold labels to categories.
- Using TF-IDF (Term Frequency and Inverse Document Frequency) vectorizer for text data.
- GridSearchCV for hyperparameter tuning.
- The trained best logistic regression model.
- Classification report and confusion matrix calculated over test data using this Hyperparameter tuned Logistic Regression model.

## Bi-directional LSTM

Bi-directional LSTM is also implemented on both SNLI and MultiNLI datasets.

- Data pre-processing.
- GloVe word embeddings.
- Model architecture.
- Model training and hyperparameter tuning.
- Classification report and confusion matrix for test data.

## Bi-directional GRU

Bi-directional GRU (Gated Recurrent Unit) architecture-based model is also trained on both the SNLI and MultiNLI datasets.

- Data pre-processing.
- GloVe word embeddings.
- Model architecture.
- Model training and hyperparameter tuning.
- Classification report and confusion matrix for test data.

## BERT

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained natural language processing (NLP) model developed by Google.

- Introduction to BERT.
- Fine-tuning BERT for Natural Language Inference.
- Training details.
- Classification report and confusion matrix for test data.

## ELMo

ELMo (Embeddings from Language Models) is a deep contextualized word embedding model.

- Introduction to ELMo.
- Model architecture and training.
- Classification report and confusion matrix for test data.

# Conclusion

To address the task of Natural Language Inference, which involves determining whether a given sentence (hypothesis) can be inferred from another sentence (premise) or not, we have implemented various machine learning models. The accuracy achieved with all the models is as follows:

- Logistic Regression: 63%
- Bi-directional LSTM: 76%
- Bi-directional GRU: 77%
- BERT: 91%
- ELMo: Results not provided.

In summary, transformer-based techniques, such as BERT, outperform traditional RNN-based techniques for Natural Language Inference tasks. Data preprocessing and choosing the correct hyperparameters play a significant role in achieving accurate results.

# Limitations

- Limited exploration of language modeling techniques.
- Training BERT for a limited number of epochs due to resource limitations.
- Results may not be generalizable to other datasets.
