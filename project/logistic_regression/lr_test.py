# ===============
# Imports
# ===============


import pickle

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import classification_report


def lr_test():
    # ===============
    # Read file
    # ===============
    test_df = pd.read_csv("../snli_1.0/snli_1.0_test.txt", delimiter="\t", encoding="utf-8", on_bad_lines="skip")

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

    test_df.drop(test_df.loc[test_df["gold_label"] == "-"].index, inplace=True)
    test_df.dropna(inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # ===============
    # Vectorize data
    # ===============

    with open("../model/logistic_regression/vectorizer.pickle", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open("../model/logistic_regression/label_encoder.pickle", "rb") as f:
        label_encoder = pickle.load(f)

    X_test1 = tfidf_vectorizer.transform(test_df["sentence1"])
    X_test2 = tfidf_vectorizer.transform(test_df["sentence2"])
    X_test = scipy.sparse.hstack((X_test1, X_test2))
    y_test = label_encoder.transform(test_df["gold_label"])

    # ===============
    # Load Model
    # ===============
    with open("../model/logistic_regression/best_lr.pickle", "rb") as f:
        grid_search = pickle.load(f)

    y_pred = grid_search.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    lr_test()
