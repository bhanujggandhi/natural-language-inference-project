# ===============
# Imports
# ===============

import math
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data
from transformers import AdamW, BertModel, BertTokenizer, get_constant_schedule_with_warmup


def bert_train():
    # =====================
    # Hyperparameters
    # =====================
    SEED = 1111
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    max_input_length = 256
    max_words = 128
    BATCH_SIZE = 64
    HIDDEN_DIM = 512
    OUTPUT_DIM = 3
    N_EPOCHS = 2
    warmup_percent = 0.2

    # =====================
    # Load Datasets
    # =====================
    train_df = pd.read_csv("snli_1.0/snli_1.0_train.txt", sep="\t", on_bad_lines="skip")
    dev_df = pd.read_csv("snli_1.0/snli_1.0_dev.txt", sep="\t", on_bad_lines="skip")
    test_df = pd.read_csv("snli_1.0/snli_1.0_test.txt", sep="\t", on_bad_lines="skip")

    # =====================
    # Preprocess Data
    # =====================
    train_df = train_df[["gold_label", "sentence1", "sentence2"]]
    dev_df = dev_df[["gold_label", "sentence1", "sentence2"]]
    test_df = test_df[["gold_label", "sentence1", "sentence2"]]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # ===============
    # Utilities
    # ===============

    def trim_sentence(sentence):
        tokens = sentence.strip().split(" ")
        tokens = tokens[:max_input_length]
        return tokens

    def to_int(tok_ids):
        tok_ids = [int(x) for x in tok_ids]
        return tok_ids

    def categorical_accuracy(preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)

        correct = (max_preds.squeeze(1) == y).float()

        return correct.sum() / len(y)

    textfield = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=trim_sentence,
        preprocessing=tokenizer.convert_tokens_to_ids,
        pad_token=tokenizer.pad_token_id,
        unk_token=tokenizer.unk_token_id,
    )

    labelfield = data.LabelField()

    attentionfield = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=trim_sentence,
        preprocessing=to_int,
        pad_token=tokenizer.pad_token_id,
    )

    token_type = data.Field(
        batch_first=True, use_vocab=False, tokenize=trim_sentence, preprocessing=to_int, pad_token=1
    )

    fields = [
        ("label", labelfield),
        ("sequence", textfield),
        ("attention_mask", attentionfield),
        ("token_type", token_type),
    ]

    train_data, valid_data, test_data = data.TabularDataset.splits(
        path="model/bert/",
        train="snli_1.0_train.csv",
        validation="snli_1.0_dev.csv",
        test="snli_1.0_test.csv",
        format="csv",
        fields=fields,
        skip_header=True,
    )

    print(f"Train data {len(train_data)}")
    print(f"Validation data {len(valid_data)}")
    print(f"Test data {len(test_data)}")

    labelfield.build_vocab(train_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.sequence),
        sort_within_batch=False,
        device=device,
    )

    bert_model = BertModel.from_pretrained("bert-base-uncased")

    class BERTNLIModel(nn.Module):
        def __init__(self, bert_model, hidden_dim, output_dim):
            super().__init__()
            self.bert = bert_model

            embedding_dim = bert_model.config.to_dict()["hidden_size"]
            self.out = nn.Linear(embedding_dim, output_dim)

        def forward(self, sequence, attn_mask, token_type):
            embedded = self.bert(input_ids=sequence, attention_mask=attn_mask, token_type_ids=token_type)[1]
            output = self.out(embedded)
            return output

    model = BERTNLIModel(bert_model, HIDDEN_DIM, OUTPUT_DIM).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)

    def get_scheduler(optimizer, warmup_steps):
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        return scheduler

    criterion = nn.CrossEntropyLoss().to(device)

    max_grad_norm = 1

    def train(model, iterator, optimizer, criterion, scheduler):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for batch in iterator:
            optimizer.zero_grad()  # clear gradients first
            torch.cuda.empty_cache()  # releases all unoccupied cached memory
            sequence = batch.sequence
            attn_mask = batch.attention_mask
            token_type = batch.token_type
            label = batch.label
            predictions = model(sequence, attn_mask, token_type)
            loss = criterion(predictions, label)
            acc = categorical_accuracy(predictions, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

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
                loss = criterion(predictions, labels)
                acc = categorical_accuracy(predictions, labels)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    total_steps = math.ceil(N_EPOCHS * len(train_iterator) * 1.0 / BATCH_SIZE)
    warmup_steps = int(total_steps * warmup_percent)
    scheduler = get_scheduler(optimizer, warmup_steps)
    best_valid_loss = float("inf")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, scheduler)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "bert-nli.pt")
        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%")

    model.load_state_dict(torch.load("bert-nli.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    bert_train()
