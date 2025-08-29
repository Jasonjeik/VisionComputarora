# make_dummy_model.py
import os
os.makedirs("models", exist_ok=True)

import tensorflow as tf
from tensorflow.keras import layers, models

INPUT_SHAPE = (256, 256, 3)

def build_dummy_seg_model():
    x = inp = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = models.Model(inp, out, name="dummy_seg_model")
    return model

if __name__ == "__main__":
    model = build_dummy_seg_model()
    model.save("models/CNNWater_best_model.keras")
    print("Modelo dummy guardado en models/CNNWater_best_model.keras")
    model.summary()
