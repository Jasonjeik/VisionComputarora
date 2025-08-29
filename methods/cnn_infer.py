# methods/cnn_infer.py
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

_MODEL = None
_INPUT_SIZE = (256, 256)

def get_model(path: str = "models/CNNWater_best_model.keras"):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if load_model is None:
        return None
    if not os.path.isfile(path):
        return None
    _MODEL = load_model(path, compile=False)
    return _MODEL

def _preprocess_bgr(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, _INPUT_SIZE, interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def _postprocess_mask(prob_small, out_shape):
    prob = prob_small[0, ..., 0]
    prob = cv2.resize(prob, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
    prob = np.clip(prob, 0, 1).astype(np.float32)
    return prob

def cnn_predict_tile(bgr_image):
    mdl = get_model()
    H, W = bgr_image.shape[:2]
    if mdl is not None:
        x = _preprocess_bgr(bgr_image)
        prob_small = mdl.predict(x, verbose=0)
        prob = _postprocess_mask(prob_small, (H, W))
        hay = bool(prob.mean() > 0.02)
        return prob, hay

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    prob = (th / 255.0).astype(np.float32)
    hay = bool(prob.mean() > 0.02)
    return prob, hay

def overlay_water(bgr_image, mask_prob, alpha=0.45, thr=0.5):
    water = (mask_prob > thr).astype(np.uint8)
    overlay = bgr_image.copy()
    blue = np.zeros_like(bgr_image, dtype=np.uint8)
    blue[:, :, 0] = 255
    overlay = np.where(
        water[..., None] == 1,
        cv2.addWeighted(blue, float(alpha), overlay, 1.0 - float(alpha), 0),
        overlay,
    )
    return overlay
