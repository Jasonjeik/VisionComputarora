# -*- coding: utf-8 -*-
"""Watershed para cuerpos de agua.
Extraído de: Actividad 5 (1).ipynb
Requiere: numpy, opencv-python (cv2)
"""
import numpy as np
import cv2
from typing import Dict, Tuple

def prep_red_clahe(
    im_bgr: np.ndarray,
    cliplimit: float = 4.0,
    tile: tuple = (3,3),
    invert_otsu: bool = True,
    dist_ratio: float = 0.5,
    morph_kernel: tuple = (3,3),
    plot: bool = False
):
    """
    Pre: imagen BGR (uint8).
    Hace:
      - CLAHE SOLO en R -> Rc (gris de trabajo)
      - Otsu (invertido por defecto, con fallback)
      - Apertura morfológica
      - Distance transform + marcadores
      - im_enh: BGR con R <- Rc

    Retorna dict con:
      {Rc, im_enh, gray, th, opening, dist, markers, (opcionales para plot)}
    """
    B, G, R = cv2.split(im_bgr)

    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tile)
    img_blue = clahe.apply(B)
    img_grn  = clahe.apply(G)
    Rc       = clahe.apply(R)   # gris de trabajo
    img_red  = Rc               # para panel

    # imagen mejorada para watershed
    im_enh = cv2.merge([B, G, Rc])

    # --- pre ---
    gray = cv2.medianBlur(Rc, 5)

    mode = cv2.THRESH_BINARY_INV if invert_otsu else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    if (th.mean()/255.0) < 0.01:  # fallback si queda casi vacía
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # sólo para mostrar (umbral directo sobre Rc)
    thr_val, th_red = cv2.threshold(Rc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)

    sure_bg = cv2.dilate(opening, k, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if plot:
        # canales + clahe + hists + umbral opcional
        histB = cv2.calcHist([B],[0],None,[256],[0,256])
        histG = cv2.calcHist([G],[0],None,[256],[0,256])
        histR = cv2.calcHist([R],[0],None,[256],[0,256])

        plt.figure(figsize=(15,15))
        plt.subplot(331); plt.imshow(B, cmap='gray'); plt.title("Blue"); plt.axis('off')
        plt.subplot(332); plt.imshow(G, cmap='gray'); plt.title("Green"); plt.axis('off')
        plt.subplot(333); plt.imshow(R, cmap='gray'); plt.title("Red");   plt.axis('off')

        plt.subplot(334); plt.imshow(img_blue, cmap='gray'); plt.title("Blue CLAHE"); plt.axis('off')
        plt.subplot(335); plt.imshow(img_grn,  cmap='gray'); plt.title("Green CLAHE"); plt.axis('off')
        plt.subplot(336); plt.imshow(img_red,  cmap='gray'); plt.title("Red CLAHE (Rc)"); plt.axis('off')

        plt.subplot(337); plt.plot(histB); plt.title("Hist Blue")
        plt.subplot(338); plt.plot(histG); plt.title("Hist Green")
        plt.subplot(339); plt.plot(histR); plt.title("Hist Red")
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(15,4))
        plt.subplot(131); plt.imshow(th_red, cmap='gray'); plt.title(f"Otsu sobre Rc (thr={thr_val:.0f})"); plt.axis('off')
        plt.subplot(132); plt.imshow(opening, cmap='gray'); plt.title("Umbral + apertura"); plt.axis('off')
        plt.subplot(133); plt.imshow(dist); plt.title("Distance transform"); plt.axis('off')
        plt.tight_layout(); plt.show()

    return {
        "Rc": Rc,
        "im_enh": im_enh,
        "gray": gray,
        "th": th,
        "opening": opening,
        "dist": dist,
        "markers": markers,
        # útiles si quieres diagnosticar
        "sure_fg": sure_fg,
        "sure_bg": sure_bg,
        "unknown": unknown
    }



def watershed_segment(prep: dict, original_bgr: np.ndarray | None = None, plot: bool = False) -> dict:
    """
    Ejecuta watershed a partir del dict devuelto por prep_red_clahe.
    Si plot=True, muestra un panel 2x3:
       [Original, Gray, Otsu(th), Distance, Markers, Bordes sobre Original]
    """
    im_enh  = prep["im_enh"]
    gray    = prep["gray"]
    th      = prep["th"]          # binaria (Otsu con posible inversión)
    dist    = prep["dist"]
    markers = prep["markers"]

    # Watershed
    ws = cv2.watershed(im_enh, markers.copy())

    # Máscara principal
    mask = np.zeros(ws.shape, np.uint8)
    uniq, counts = np.unique(ws, return_counts=True)
    candidatos = [(l, c) for l, c in zip(uniq, counts) if l > 1]
    if candidatos:
        main_label = max(candidatos, key=lambda t: t[1])[0]
        mask[ws == main_label] = 255

    # Bordes en rojo sobre original o sobre im_enh
    base_bgr = original_bgr if original_bgr is not None else im_enh
    overlay = base_bgr.copy()
    overlay[ws == -1] = (0, 0, 255)

    # Plot opcional
    if plot:
        fig, ax = plt.subplots(2, 3, figsize=(14, 9))
        ax[0,0].imshow(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
        ax[0,0].set_title("Original"); ax[0,0].axis("off")
        ax[0,1].imshow(gray, cmap="gray")
        ax[0,1].set_title("Gris preprocesada"); ax[0,1].axis("off")
        ax[0,2].imshow(th, cmap="gray")
        ax[0,2].set_title("Otsu (th)"); ax[0,2].axis("off")
        ax[1,0].imshow(dist, cmap="magma")
        ax[1,0].set_title("Distance transform"); ax[1,0].axis("off")

        mk_rgb = colorize_markers(markers)
        ax[1,1].imshow(mk_rgb)
        ax[1,1].set_title("Marcadores"); ax[1,1].axis("off")

        ax[1,2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax[1,2].set_title("Bordes (watershed)"); ax[1,2].axis("off")

        plt.tight_layout()
        plt.show()

    return {"markers": ws, "mask": mask, "overlay": overlay}

