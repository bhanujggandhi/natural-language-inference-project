# ==================
# Imports
# ==================

import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow.keras.saving import load_model


def bigru_test():
    # ==================
    # Load Data
    # ==================
    with open("model/bigru/snli/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    with open("model/bigru/snli/test_data.pickle", "rb") as f:
        test_data = pickle.load(f)

    print(test_data[0][0])
    # ==================
    # Load model
    # ==================
    model = load_model("model/bigru/snli/BiGRU.h5")

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
    bigru_test()
