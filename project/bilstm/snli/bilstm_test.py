# ==================
# Imports
# ==================

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.models import load_model


def bilstm_test():
    # ==================
    # Load Data
    # ==================
    with open("model/bilstm/snli/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    with open("model/bilstm/snli/test_data.txt", "rb") as f:
        test_data = pickle.load(f)

    # ==================
    # Load model
    # ==================
    model = load_model("model/bilstm/snli/BiLSTM.h5")

    # ==================
    # Predict
    # ==================
    y_pred = model.predict([test_data[0], test_data[1]])
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(test_data[2], axis=1)
    print(classification_report(y_pred, y_test))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.show()


if __name__ == "__main__":
    bilstm_test()
