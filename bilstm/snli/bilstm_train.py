# ========================
# Imports
# ========================

import pickle
import tempfile

import contractions
import numpy as np
import pandas as pd
import spacy
import unidecode
from bs4 import BeautifulSoup
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Input,
    TimeDistributed,
    concatenate,
)
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.utils.vis_utils import plot_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from word2number import w2n


def bilstm_train():
    # ========================
    # Load dataset
    # ========================
    train_df = pd.read_csv("snli_1.0/snli_1.0_train.txt", delimiter="\t", encoding="utf-8", on_bad_lines="skip")
    val_df = pd.read_csv("snli_1.0/snli_1.0_dev.txt", delimiter="\t", encoding="utf-8", on_bad_lines="skip")
    test_df = pd.read_csv("snli_1.0/snli_1.0_test.txt", delimiter="\t", encoding="utf-8", on_bad_lines="skip")

    
    train_df = train_df[["gold_label", "sentence1", "sentence2"]]
    val_df = val_df[["gold_label", "sentence1", "sentence2"]]
    test_df = test_df[["gold_label", "sentence1", "sentence2"]]

    # ========================
    # Preprocess the data
    # ========================

    train_df = pd.concat([train_df, val_df], ignore_index=True)

    train_df.dropna(inplace=True)
    train_df.drop(train_df.loc[train_df["gold_label"] == "-"].index, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df.dropna(inplace=True)
    test_df.drop(test_df.loc[test_df["gold_label"] == "-"].index, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    def preprocess_sentence(sent: str) -> str:
        # Remove HTML
        soup = BeautifulSoup(sent, "html.parser")
        sent = soup.get_text(separator=" ")

        # Remove whitespaces
        sent = sent.strip()
        sent = " ".join(sent.split())

        # Lowercase
        sent = sent.lower()

        # Remove accent characters
        sent = unidecode.unidecode(sent)

        # Expand the contractions
        sent = contractions.fix(sent)

        return sent

    # Train set
    train_df["sentence1"] = train_df["sentence1"].apply(lambda x: preprocess_sentence(x))
    train_df["sentence2"] = train_df["sentence2"].apply(lambda x: preprocess_sentence(x))

    # Test set
    test_df["sentence1"] = test_df["sentence1"].apply(lambda x: preprocess_sentence(x))
    test_df["sentence2"] = test_df["sentence2"].apply(lambda x: preprocess_sentence(x))

    # Encode the labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["gold_label"])
    num_classes = len(label_encoder.classes_)

    y_train = label_encoder.transform(train_df["gold_label"])
    y_test = label_encoder.transform(test_df["gold_label"])

    train_sent1 = train_df["sentence1"].to_numpy()
    train_sent2 = train_df["sentence2"].to_numpy()

    test_sent1 = test_df["sentence1"].to_numpy()
    test_sent2 = test_df["sentence2"].to_numpy()

    train_corpus = [train_sent1[ind] + " " + train_sent2[ind] for ind in range(len(y_train))]

    embedding_dict = {}

    with open("glove.6B/glove.6B.300d.txt", "r") as f:
        for line in f:
            line_list = line.split()
            word = line_list[0]
            embeddings = np.asarray(line_list[1:], dtype=float)

            embedding_dict[word] = embeddings

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(train_corpus)

    word_index = tokenizer.word_index
    embed_matrix = np.zeros((len(word_index) + 1, 300))
    for word, ind in word_index.items():
        embedding_vector = embedding_dict.get(word)

        if embedding_vector is not None:
            embed_matrix[ind] = embedding_vector

    sequence = lambda sentence: pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=42)
    process = lambda item: (sequence(item[0]), sequence(item[1]), to_categorical(item[2]))

    train_process_data = [train_sent1, train_sent2, y_train]
    test_process_data = [test_sent1, test_sent2, y_test]
    training_data = process(train_process_data)
    test_data = process(test_process_data)

    with open("model/bilstm/snli/test_data.txt", "wb") as f:
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("model/bilstm/snli/tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ========================
    # Model Layers
    # ========================

    # Pre-trained glove embeddings
    embeddings = Embedding(
        input_dim=embed_matrix.shape[0],
        output_dim=embed_matrix.shape[1],
        weights=[embed_matrix],
        input_length=42,
        trainable=False,
    )

    # BiLSTM Declaration
    BiLSTM = Bidirectional(LSTM(64))

    # Input Declaration
    premise_input = Input(shape=(42,))
    hypothesis_input = Input(shape=(42,))

    # embedded input
    premise_embedded = embeddings(premise_input)
    hypothesis_embedded = embeddings(hypothesis_input)

    # LSTM Layers
    premise_BiLSTM = BiLSTM(premise_embedded)
    hypothesis_BiLSTM = BiLSTM(hypothesis_embedded)

    # Batch Normalization
    premise_normalized = BatchNormalization()(premise_BiLSTM)
    hypothesis_normalized = BatchNormalization()(hypothesis_BiLSTM)

    # Concatenate the output
    train_input = concatenate([premise_normalized, hypothesis_normalized])
    train_input = Dropout(0.2)(train_input)

    # Dense Layer1 + Dropout + Normalization
    train_input = Dense(2 * 300, activation="relu", kernel_regularizer=l2(4e-6))(train_input)
    train_input = Dropout(0.2)(train_input)
    train_input = BatchNormalization()(train_input)

    # Dense Layer1 + Dropout + Normalization
    train_input = Dense(2 * 300, activation="relu", kernel_regularizer=l2(4e-6))(train_input)
    train_input = Dropout(0.2)(train_input)
    train_input = BatchNormalization()(train_input)

    # Dense Layer1 + Dropout + Normalization
    train_input = Dense(2 * 300, activation="relu", kernel_regularizer=l2(4e-6))(train_input)
    train_input = Dropout(0.2)(train_input)
    train_input = BatchNormalization()(train_input)

    # Softmax layer
    prediction = Dense(3, activation="softmax")(train_input)

    model = Model(inputs=[premise_input, hypothesis_input], outputs=prediction)

    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())

    _, tmpfn = tempfile.mkstemp()
    model_checkpoint = ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=4)

    callbacks = [early_stopping, model_checkpoint]

    print("Training model")
    history = model.fit(
        x=[training_data[0], training_data[1]],
        y=training_data[2],
        batch_size=512,
        epochs=50,
        validation_split=0.02,
        callbacks=callbacks,
    )

    model.load_weights(tmpfn)
    model.save("model/snli/bilstm/BiLSTM.h5")

    # =====================
    # Load Model
    # =====================

    model2 = load_model("model/snli/bilstm/BiLSTM.h5")
    y_pred = model2.predict([test_data[0], test_data[1]])
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(test_data[2], axis=1)
    print(classification_report(y_pred, y_test))


if __name__ == "__main__":
    bilstm_train()
