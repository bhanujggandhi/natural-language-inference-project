import argparse

# Logistic Regression
from logistic_regression.lr_train import lr_train
from logistic_regression.lr_test import lr_test

# BiLSTM
from bilstm.bilstm_train import bilstm_train
from bilstm.bilstm_test import bilstm_test

# BiGRU
from bigru.bigru_train import bigru_train
from bigru.bigru_test import bigru_test

# BiGRU
from bert.bert_train import bert_train
from bert.bert_test import bert_test

# Define the available models
models = {
    "lr": {"train": lr_train, "test": lr_test},
    "bilstm": {"train": bilstm_train, "test": bilstm_test},
    "bigru": {"train": bigru_train, "test": bigru_test},
    "bert": {"train": bert_train, "test": bert_test},
}

# Set up the CLI
parser = argparse.ArgumentParser(description="Train and test NLP models")
parser.add_argument("model", choices=models.keys(), help="The name of the model to run")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--test", action="store_true", help="Test the model")
args = parser.parse_args()

# Get the selected model
model_name = args.model
model = models[model_name]

# Train or test the model as requested
if args.train:
    model["train"]()
if args.test:
    model["test"]()
