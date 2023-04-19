import argparse

# Bert
from bert.bert_test import bert_test
from bert.bert_train import bert_train

# BiGRU
from bigru.bigru_test import bigru_test
from bigru.bigru_train import bigru_train

# BiLSTM
from bilstm.bilstm_test import bilstm_test
from bilstm.bilstm_train import bilstm_train

# Logistic Regression
from logistic_regression.multi_nli.lr_test import lr_test as lr_test_multinli
from logistic_regression.multi_nli.lr_train import lr_train as lr_train_multinli
from logistic_regression.snli.lr_test import lr_test as lr_test_snli
from logistic_regression.snli.lr_train import lr_train as lr_train_snli

# Define the available models
models = {
    "lr_snli": {"train": lr_train_snli, "test": lr_test_snli},
    "bilstm_snli": {"train": bilstm_train, "test": bilstm_test},
    "bigru_snli": {"train": bigru_train, "test": bigru_test},
    "elmo_snli": {"train": print, "test": print},
    "bert_snli": {"train": bert_train, "test": bert_test},
    "lr_multi_nli": {"train": lr_train_multinli, "test": lr_test_multinli},
    "bilstm_multi_nli": {"train": bilstm_train, "test": bilstm_test},
    "bigru_multi_nli": {"train": bigru_train, "test": bigru_test},
    "elmo_multi_nli": {"train": print, "test": print},
    "bert_multi_nli": {"train": bert_train, "test": bert_test},
}

# Set up the CLI
parser = argparse.ArgumentParser(description="Train and test NLI models")
parser.add_argument(
    "model", choices=["lr", "bilstm", "bigru", "elmo", "bert"], help="The name of the model to run", default="lr"
)
parser.add_argument("--train", action="store_true", help="Train the model", default=False)
parser.add_argument("--test", action="store_true", help="Test the model", default=False)
parser.add_argument("--dataset", choices=["snli", "multi_nli"], help="The name of the dataset to use", default="snli")
args = parser.parse_args()

# Get the selected model
model_name = args.model + "_" + args.dataset
model = models[model_name]


# Train or test the model as requested
if args.train:
    model["train"]()
if args.test:
    model["test"]()
