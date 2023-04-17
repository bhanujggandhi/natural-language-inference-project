# ===============
# Imports
# ===============

import torch

SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from transformers import BertTokenizer

# =====================
# Load Datasets
# =====================
df_train = pd.read_csv("snli_1.0/snli_1.0_train.txt", sep="\t", on_bad_lines="skip")
df_dev = pd.read_csv("snli_1.0/snli_1.0_dev.txt", sep="\t", on_bad_lines="skip")
df_test = pd.read_csv("snli_1.0/snli_1.0_test.txt", sep="\t", on_bad_lines="skip")


# =====================
# Preprocess Data
# =====================
train_df.drop(
    columns=[
        "sentence1_binary_parse",
        "sentence2_binary_parse",
        "sentence1_parse",
        "sentence2_parse",
        "captionID",
        "pairID",
        "label1",
        "label2",
        "label3",
        "label4",
        "label5",
    ],
    inplace=True,
)
val_df.drop(
    columns=[
        "sentence1_binary_parse",
        "sentence2_binary_parse",
        "sentence1_parse",
        "sentence2_parse",
        "captionID",
        "pairID",
        "label1",
        "label2",
        "label3",
        "label4",
        "label5",
    ],
    inplace=True,
)
test_df.drop(
    columns=[
        "sentence1_binary_parse",
        "sentence2_binary_parse",
        "sentence1_parse",
        "sentence2_parse",
        "captionID",
        "pairID",
        "label1",
        "label2",
        "label3",
        "label4",
        "label5",
    ],
    inplace=True,
)

# Subset of the data for computation
df_train = df_train[:200000]
df_dev = df_dev[:8000]
df_test = df_test[:8000]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def trim_sent(sent):
    try:
        sent = sent.split()
        sent = sent[:128]
        return " ".join(sent)
    except:
        return sent


# Trimming the sentence
df_train["sentence1"] = df_train["sentence1"].apply(trim_sent)
df_train["sentence2"] = df_train["sentence2"].apply(trim_sent)
df_dev["sentence1"] = df_dev["sentence1"].apply(trim_sent)
df_dev["sentence2"] = df_dev["sentence2"].apply(trim_sent)
df_test["sentence1"] = df_test["sentence1"].apply(trim_sent)
df_test["sentence2"] = df_test["sentence2"].apply(trim_sent)

# Start and End tokens
df_train["sentence1"] = "[CLS] " + df_train["sentence1"] + " [SEP] "
df_train["sentence2"] = df_train["sentence2"] + " [SEP]"
df_dev["sentence1"] = "[CLS] " + df_dev["sentence1"] + " [SEP] "
df_dev["sentence2"] = df_dev["sentence2"] + " [SEP]"
df_test["sentence1"] = "[CLS] " + df_test["sentence1"] + " [SEP] "
df_test["sentence2"] = df_test["sentence2"] + " [SEP]"

df_train = df_train.dropna()
df_dev = df_dev.dropna()
df_test = df_test.dropna()

max_input_length = 256


def tokenize(sentence):
    tokens = tokenizer.tokenize(sentence)
    return tokens


df_train["sent1_tokens"] = df_train["sentence1"].apply(tokenize)
df_train["sent2_tokens"] = df_train["sentence2"].apply(tokenize)
df_dev["sent1_tokens"] = df_dev["sentence1"].apply(tokenize)
df_dev["sent2_tokens"] = df_dev["sentence2"].apply(tokenize)
df_test["sent1_tokens"] = df_test["sentence1"].apply(tokenize)
df_test["sent2_tokens"] = df_test["sentence2"].apply(tokenize)


def split_and_cut(sentence):
    tokens = sentence.strip().split(" ")
    tokens = tokens[:max_input_length]
    return tokens


def get_sent1_token_type(sent):
    try:
        return [0] * len(sent)
    except:
        return []


def get_sent2_token_type(sent):
    try:
        return [1] * len(sent)
    except:
        return []


df_train["sent1_token_type"] = df_train["sent1_t"].apply(lambda x: [0] * len(x))
df_train["sent2_token_type"] = df_train["sent2_t"].apply(get_sent2_token_type)
df_dev["sent1_token_type"] = df_dev["sent1_t"].apply(get_sent1_token_type)
df_dev["sent2_token_type"] = df_dev["sent2_t"].apply(get_sent2_token_type)
df_test["sent1_token_type"] = df_test["sent1_t"].apply(get_sent1_token_type)
df_test["sent2_token_type"] = df_test["sent2_t"].apply(get_sent2_token_type)


def combine_seq(seq):
    return " ".join(seq)


def combine_mask(mask):
    mask = [str(m) for m in mask]
    return " ".join(mask)
