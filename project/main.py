import argparse

# Bert
from bert.multi_nli.bert_test import bert_test as bert_test_multi_nli
from bert.multi_nli.bert_train import bert_train as bert_train_multi_nli
from bert.snli.bert_test import bert_test as bert_test_snli
from bert.snli.bert_train import bert_train as bert_train_snli

# BiGRU
from bigru.multi_nli.bigru_test import bigru_test as bigru_test_multi_nli
from bigru.multi_nli.bigru_train import bigru_train as bigru_train_multi_nli
from bigru.snli.bigru_test import bigru_test as bigru_test_snli
from bigru.snli.bigru_train import bigru_train as bigru_train_snli

# BiLSTM
from bilstm.multi_nli.bilstm_test import bilstm_test as bilstm_test_multinli
from bilstm.multi_nli.bilstm_train import bilstm_train as bilstm_train_multinli
from bilstm.snli.bilstm_test import bilstm_test as bilstm_test_snli
from bilstm.snli.bilstm_train import bilstm_train as bilstm_train_snli

# Logistic Regression
from logistic_regression.multi_nli.lr_test import lr_test as lr_test_multinli
from logistic_regression.multi_nli.lr_train import lr_train as lr_train_multinli
from logistic_regression.snli.lr_test import lr_test as lr_test_snli
from logistic_regression.snli.lr_train import lr_train as lr_train_snli

# Define the available models
models = {
    "lr_snli": {"train": lr_train_snli, "test": lr_test_snli},
    "bilstm_snli": {"train": bilstm_train_snli, "test": bilstm_test_snli},
    "bigru_snli": {"train": bigru_train_snli, "test": bigru_test_snli},
    "elmo_snli": {"train": print, "test": print},
    "bert_snli": {"train": bert_train_snli, "test": bert_test_snli},
    "lr_multi_nli": {"train": lr_train_multinli, "test": lr_test_multinli},
    "bilstm_multi_nli": {"train": bilstm_train_multinli, "test": bilstm_test_multinli},
    "bigru_multi_nli": {"train": bigru_train_multi_nli, "test": bigru_test_multi_nli},
    "elmo_multi_nli": {"train": print, "test": print},
    "bert_multi_nli": {"train": bert_train_multi_nli, "test": bert_test_multi_nli},
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
