import argparse

# Logistic Regression
from logistic_regression.lr_train import lr_train
from logistic_regression.lr_test import lr_test

# BiLSTM
from bilstm.bilstm_train import bilstm_train
from bilstm.bilstm_test import bilstm_test

# BiGRU
from model_2 import train_model_2, test_model_2
from model_3 import train_model_3, test_model_3

# Define the available models
models = {
    "logistic_regression": {"train": train_model_1, "test": test_model_1},
    "bilstm": {"train": train_model_2, "test": test_model_2},
    "bigru": {"train": train_model_3, "test": test_model_3},
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
