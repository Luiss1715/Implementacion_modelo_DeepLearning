import os, numpy as np, itertools
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATA_DIR = "data/processed/pets/test"
MODEL_PATH = "runs/best.keras"
IMG_SIZE = (224,224)

def load_all(ds):
    Xs, Ys = [], []
    for x,y in ds:
        Xs.append(x.numpy()); Ys.append(y.numpy())
    return np.vstack(Xs), np.vstack(Ys)

def plot_cm(cm, classes, out="reports/figures/confusion_matrix.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90)
    plt.yticks(ticks, classes)
    thresh = cm.max()/2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    plt.ylabel("True"); plt.xlabel("Pred")
    plt.savefig(out, dpi=160)

if __name__ == "__main__":
    ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, image_size=IMG_SIZE, batch_size=32,
        label_mode="categorical", shuffle=False
    )
    class_names = ds.class_names
    X, Y = load_all(ds)
    model = keras.models.load_model(MODEL_PATH)
    P = model.predict(X, verbose=0)
    y_true = Y.argmax(1); y_pred = P.argmax(1)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, class_names)
    print("Guardada matriz en reports/figures/confusion_matrix.png")
