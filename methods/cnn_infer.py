# methods/cnn_infer.py
# -*- coding: utf-8 -*-
"""
Inferencia para tu modelo original:
- Carga: models/Red_best_model.keras
- Tamaño de entrada: se detecta automáticamente desde model.input_shape (por defecto 128x128x3)
- API compatible con tu app: get_model(), cnn_predict_tile(bgr_image), overlay_water(...)
"""
import os
import cv2
import numpy as np

try:
    from tensorflow.keras.models import load_model
except Exception:  # TensorFlow no disponible
    load_model = None

_MODEL = None
_INPUT_SIZE = (128, 128)  # (H, W) por defecto; se ajusta al cargar si el modelo indica otro tamaño


def _infer_input_size_from_model(model):
    """
    Intenta leer (H, W) desde model.input_shape. Soporta lista o tupla.
    input_shape típico: (None, H, W, C)
    """
    try:
        shape = model.input_shape
        if isinstance(shape, (list, tuple)) and isinstance(shape[0], (list, tuple)):
            shape = shape[0]  # por si es una lista de entradas
        # Esperamos (None, H, W, C)
        if len(shape) >= 4 and shape[1] is not None and shape[2] is not None:
            H, W = int(shape[1]), int(shape[2])
            return (H, W)
    except Exception:
        pass
    return None


def get_model(path: str = "models/Red_best_model.keras"):
    """
    Carga y memoriza el modelo Keras desde models/Red_best_model.keras.
    Ajusta _INPUT_SIZE si el modelo expone el tamaño de entrada.
    """
    global _MODEL, _INPUT_SIZE
    if _MODEL is not None:
        return _MODEL
    if load_model is None:
        return None
    if not os.path.isfile(path):
        return None
    _MODEL = load_model(path, compile=False)
    inferred = _infer_input_size_from_model(_MODEL)
    if inferred is not None:
        _INPUT_SIZE = inferred  # (H, W) detectado
    return _MODEL


def _preprocess_bgr(bgr):
    """BGR -> RGB, resize a (_INPUT_SIZE), escala [0,1], shape (1,H,W,3)."""
    Ht, Wt = _INPUT_SIZE
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_res = cv2.resize(rgb, (Wt, Ht), interpolation=cv2.INTER_AREA)
    x = np.expand_dims(rgb_res, axis=0)
    return x


def _postprocess_to_full(pred_small, out_shape):
    """(h,w) -> (Horig,Worig) en [0,1]"""
    H, W = out_shape
    pred_full = cv2.resize(pred_small, (W, H), interpolation=cv2.INTER_LINEAR)
    pred_full = np.clip(pred_full, 0, 1).astype(np.float32)
    return pred_full


def cnn_predict_tile(bgr_image, thr: float = 0.5):
    """
    Devuelve (mask_prob[0..1], hay_agua: bool) usando tu Red_best_model.keras.
    Si no hay modelo disponible, aplica un Otsu invertido como fallback.
    """
    mdl = get_model()
    H, W = bgr_image.shape[:2]

    if mdl is not None:
        x = _preprocess_bgr(bgr_image)                    # (1,h,w,3)
        pred = mdl.predict(x, verbose=0)                  # (1,h,w,1) o (1,h,w)
        if pred.ndim == 4 and pred.shape[-1] == 1:
            pred = pred[0, ..., 0]                        # (h,w)
        else:
            pred = np.squeeze(pred, axis=0)               # (h,w,?) -> (h,w) si aplica
            if pred.ndim == 3:
                pred = pred[..., 0]
        prob = _postprocess_to_full(pred, (H, W))         # (H,W)
        hay_agua = bool((prob > float(thr)).mean() > 0.001)
        return prob, hay_agua

    # ---- Fallback sin modelo ----
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    prob = (th / 255.0).astype(np.float32)
    hay = bool(prob.mean() > 0.02)
    return prob, hay


def overlay_water(bgr_image, mask_prob, alpha=0.45, thr=0.5):
    """
    Sombrea en azul (canal B) las zonas con probabilidad > thr.
    """
    water = (mask_prob > thr).astype(np.uint8)
    overlay = bgr_image.copy()
    blue = np.zeros_like(bgr_image, dtype=np.uint8)
    blue[:, :, 0] = 255  # canal B
    overlay = np.where(
        water[..., None] == 1,
        cv2.addWeighted(blue, float(alpha), overlay, 1.0 - float(alpha), 0),
        overlay,
    )
    return overlay
