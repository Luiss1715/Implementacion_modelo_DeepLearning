# predict.py
import argparse, numpy as np, os
from tensorflow import keras
from PIL import Image

MODEL_PATH = "runs/best.keras"
IMG_SIZE = (224,224)
CLASS_DIR = "data/processed/pets/train"  # lee clases del directorio de train

def class_names_from_dir(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def load_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")[None, ...]
    arr = keras.applications.mobilenet_v3.preprocess_input(arr)
    return arr

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    names = class_names_from_dir(CLASS_DIR)
    model = keras.models.load_model(MODEL_PATH)
    x = load_image(args.image)
    probs = model.predict(x, verbose=0)[0]

    topk = probs.argsort()[-5:][::-1]
    print("\nTop-5 predicciones:")
    for i in topk:
        print(f"{names[i]:25s}  prob={probs[i]:.3f}")
