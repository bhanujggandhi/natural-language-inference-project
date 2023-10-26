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

[Link for the files](https://drive.google.com/drive/folders/1SOKwwFvbbDyKMlf1WFsUIY3-eevIaMua?usp=sharing)
