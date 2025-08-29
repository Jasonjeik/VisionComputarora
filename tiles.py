# -*- coding: utf-8 -*-

"""Utilidad para descargar mosaicos (tiles) de imágenes satelitales y ensamblarlos en una sola imagen.

Fuente: Esri World Imagery (solo para demostración). Revisa términos de uso antes de usos productivos.
"""
import math
import io
import requests
from PIL import Image
import numpy as np
import mercantile

ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def _latlon_to_tile_bounds(bounds, zoom):
    """bounds: (south, west, north, east) en grados"""
    south, west, north, east = bounds
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    return tiles

def _tile_to_img(z, x, y):
    url = ESRI_URL.format(z=z, x=x, y=y)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def stitch_from_bounds(bounds, zoom=16, max_tiles=25):
    """Devuelve PIL Image con el mosaico de los tiles que cubren bounds.
    bounds = (south, west, north, east). Limita a max_tiles para evitar descargas excesivas.
    """
    tiles = _latlon_to_tile_bounds(bounds, zoom)
    if len(tiles) > max_tiles:
        raise RuntimeError(f"Demasiados tiles ({len(tiles)}). Reduce el área o el zoom.")
    xs = sorted(set(t.x for t in tiles))
    ys = sorted(set(t.y for t in tiles))
    tile_imgs = {}
    for t in tiles:
        tile_imgs[(t.x, t.y)] = _tile_to_img(t.z, t.x, t.y)
    w, h = next(iter(tile_imgs.values())).size
    canvas = Image.new("RGB", (len(xs)*w, len(ys)*h))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            canvas.paste(tile_imgs[(x,y)], (ix*w, iy*h))
    return canvas
