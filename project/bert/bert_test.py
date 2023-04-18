# ===============
# Imports
# ===============
import pickle

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torchtext.legacy import data
from transformers import BertModel


# ===============
# Hyperparameters
# ===============
SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
MAX_INPUT_LENGTH = 256
BATCH_SIZE = 64
HIDDEN_DIM = 512
OUTPUT_DIM = 3


# ===============
# Utilities
# ===============
def word_to_idx(idx):
    idx = [int(x) for x in idx]
    return idx


def trim_sentence(sentence):
    tokens = sentence.strip().split(" ")
    tokens = tokens[:MAX_INPUT_LENGTH]
    return tokens


def categorical_accuracy(y_pred, y_test):
    y_pred = y_pred.argmax(dim=1, keepdim=True)
    correct = (y_pred.squeeze(1) == y_test).float()

    return correct.sum() / len(y_test)


class BERTMODEL(nn.Module):
    def __init__(self, bert, HIDDEN_DIM, OUTPUT_DIM):
        super().__init__()
        self.bert = bert

        embedding_dim = bert.config.to_dict()["hidden_size"]
        self.out = nn.Linear(embedding_dim, OUTPUT_DIM)

    def forward(self, sequence, attention_mask, token_type):
        embedded = self.bert(input_ids=sequence, attention_mask=attention_mask, token_type_ids=token_type)[1]
        output = self.out(embedded)
        return output


true_labels = []
predicted_labels = []


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            sequence = batch.sequence
            attn_mask = batch.attention_mask
            token_type = batch.token_type
            labels = batch.label
            predictions = model(sequence, attn_mask, token_type)
            predicted_labels.extend(torch.argmax(predictions, dim=1).tolist())
            true_labels.extend(labels.tolist())
            loss = criterion(predictions, labels)
            acc = categorical_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def bert_test():
    # ===============
    # Load Data
    # ===============
    with open("model/bert/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    text_field = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=trim_sentence,
        preprocessing=tokenizer.convert_tokens_to_ids,
        pad_token=tokenizer.pad_token_id,
        unk_token=tokenizer.unk_token_id,
    )

    label_field = data.LabelField()

    attention_mask_field = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=trim_sentence,
        preprocessing=word_to_idx,
        pad_token=tokenizer.pad_token_id,
    )

    token_type_field = data.Field(
        batch_first=True, use_vocab=False, tokenize=trim_sentence, preprocessing=word_to_idx, pad_token=1
    )

    fields = [
        ("label", label_field),
        ("sequence", text_field),
        ("attention_mask", attention_mask_field),
        ("token_type", token_type_field),
    ]

    train_data, valid_data, test_data = data.TabularDataset.splits(
        path="model/bert",
        train="snli_1.0_train.csv",
        validation="snli_1.0_dev.csv",
        test="snli_1.0_test.csv",
        format="csv",
        fields=fields,
        skip_header=True,
    )

    # ===============
    # Prepare Data
    # ===============

    label_field.build_vocab(train_data)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.sequence),
        sort_within_batch=False,
        device=device,
    )

    # ===============
    # Model Deifinition
    # ===============

    bert_model = BertModel.from_pretrained("bert-base-uncased")

    model = BERTMODEL(bert_model, HIDDEN_DIM, OUTPUT_DIM).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    # ===============
    # Test Model
    # ===============

    model.load_state_dict(torch.load("model/bert/bert-nli.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")

    target_names = label_field.vocab.itos
    print(classification_report(true_labels, predicted_labels, target_names=target_names))
