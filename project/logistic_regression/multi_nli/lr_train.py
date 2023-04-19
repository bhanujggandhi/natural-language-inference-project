# ===============
# Imports
# ===============

import pickle

import numpy as np
import pandas as pd
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


def lr_train():
    # ===============
    # Hyperparameters
    # ===============

    MAX_ITERATIONS = 10000

    # ===============
    # Read file
    # ===============
    train_df = pd.read_csv("multinli_0.9/multinli_0.9_train.txt", delimiter="\t", encoding="utf-8", on_bad_lines="skip")
    test_df = pd.read_csv("multinli_0.9/multinli_0.9_dev.txt", delimiter="\t", encoding="utf-8", on_bad_lines="skip")

    train_df = train_df[["gold_label", "sentence1", "sentence2"]]

    train_df.drop(train_df.loc[train_df["gold_label"] == "-"].index, inplace=True)
    train_df.dropna(inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df = test_df[["gold_label", "sentence1", "sentence2"]]

    test_df.drop(test_df.loc[test_df["gold_label"] == "-"].index, inplace=True)
    test_df.dropna(inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # ===============
    # Vectorize data
    # ===============

    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    corpus = tfidf_vectorizer.fit_transform(train_df["sentence1"] + " " + train_df["sentence2"])

    with open("model/logistic_regression/vectorizer.pickle", "wb") as f:
        pickle.dump(tfidf_vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    X_train1 = tfidf_vectorizer.transform(train_df["sentence1"])
    X_train2 = tfidf_vectorizer.transform(train_df["sentence2"])
    X_train = scipy.sparse.hstack((X_train1, X_train2))

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["gold_label"])
    num_classes = len(label_encoder.classes_)
    y_train = label_encoder.transform(train_df["gold_label"])

    with open("model/logistic_regression/label_encoder.pickle", "wb") as f:
        pickle.dump(label_encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

    X_test1 = tfidf_vectorizer.transform(test_df["sentence1"])
    X_test2 = tfidf_vectorizer.transform(test_df["sentence2"])
    X_test = scipy.sparse.hstack((X_test1, X_test2))
    y_test = label_encoder.transform(test_df["gold_label"])

    # ===============
    # Defining Model
    # ===============
    lr = LogisticRegression(max_iter=MAX_ITERATIONS)

    # ===============
    # Hyperparameter tuning
    # ===============
    param_grid = {"C": [0.1, 1, 10, 50]}
    grid_search = GridSearchCV(lr, param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    with open("model/logistic_regression/best_lr.pickle", "wb") as f:
        pickle.dump(grid_search, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(grid_search.best_params_)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    lr_train()
