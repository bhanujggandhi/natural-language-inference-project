# ==================
# Imports
# ==================

import pickle

import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model


def bilstm_test():
    # ==================
    # Load Data
    # ==================
    with open("model/bilstm/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    with open("model/bilstm/test_data.txt", "rb") as f:
        test_data = pickle.load(f)

    # ==================
    # Load model
    # ==================
    model = load_model("model/bilstm/BiLSTM.h5")

    # ==================
    # Predict
    # ==================
    y_pred = model.predict([test_data[0], test_data[1]])
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(test_data[2], axis=1)
    print(classification_report(y_pred, y_test))


if __name__ == "__main__":
    bilstm_test()
