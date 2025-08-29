# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, os
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw, LocateControl

from tiles import stitch_from_bounds
from methods.watershed import prep_red_clahe, watershed_segment
from methods.cnn_infer import get_model, cnn_predict_tile, overlay_water


st.set_page_config(page_title="Detector de Agua Satelital", layout="wide")

st.title("🌊 Detector simple de cuerpos de agua (demo)")

with st.sidebar:
    st.header("Ajustes")
    metodo = st.radio("Método de identificación", ["Watershed", "Red Neuronal Convolucional"])
    zoom = st.slider("Zoom de captura (XYZ)", min_value=13, max_value=18, value=16)
    max_tiles = st.slider("Máx. tiles a descargar", min_value=4, max_value=64, value=25, help="Evita capturas enormes.")
    pedir_u = st.checkbox("Intentar ubicarme (geolocalización del navegador)", value=True)

st.markdown("""
1. Panoramea el mapa con la vista satelital hasta ubicar tu zona de interés.
2. (Opcional) Pulsa el botón de ubicación para centrar en tu posición.
3. Ajusta el **zoom** para decidir el nivel de detalle.
4. Pulsa **Capturar imagen**.
5. Elige el método y ejecuta la **detección**.
""")

# ----- Mapa -----
m = folium.Map(location=[4.711, -74.072], zoom_start=13, tiles=None, control_scale=True)
folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attr="Esri World Imagery", name="Satélite").add_to(m)
LocateControl(auto_start=False, flyTo=True).add_to(m)
Draw(export=False).add_to(m)
folium.LayerControl().add_to(m)

map_state = st_folium(m, width=None, height=500, returned_objects=["last_active_drawing", "bounds", "center", "zoom"])

bounds = map_state.get("bounds")
center = map_state.get("center")
zoom_from_map = map_state.get("zoom")

if bounds:
    # Bounds: {'_southWest': {'lat': .., 'lng': ..}, '_northEast': {'lat': .., 'lng': ..}}
    south = bounds["_southWest"]["lat"]
    west  = bounds["_southWest"]["lng"]
    north = bounds["_northEast"]["lat"]
    east  = bounds["_northEast"]["lng"]
    st.caption(f"Área actual (S,W,N,E): {south:.5f}, {west:.5f}, {north:.5f}, {east:.5f}")
else:
    # Fallback a un área pequeña alrededor del centro
    if center:
        c_lat, c_lon = center["lat"], center["lng"]
    else:
        c_lat, c_lon = 4.711, -74.072
    delta = 0.01
    south, west, north, east = c_lat - delta, c_lon - delta, c_lat + delta, c_lon + delta

col1, col2 = st.columns([1,1], gap="large")

with col1:
    if st.button("📸 Capturar imagen de la vista actual"):
        try:
            img = stitch_from_bounds((south, west, north, east), zoom=zoom, max_tiles=max_tiles)
            st.session_state["captured_pil"] = img
            st.success("Imagen capturada.")
        except Exception as e:
            st.error(f"No fue posible capturar: {e}")

# Mostrar imagen capturada
pil_img = st.session_state.get("captured_pil")
if pil_img:
    st.image(pil_img, caption="Imagen satelital capturada", use_container_width=True)

    # Convertir a BGR para procesamiento
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with col2:
        if metodo == "Watershed":
            if st.button("🏁 Detectar con Watershed"):
                prep = prep_red_clahe(bgr, plot=False)
                seg  = watershed_segment(prep, original_bgr=bgr, plot=False)
                prob = seg.get("prob", (seg["mask"]/255.0).astype(np.float32))
                hay  = bool(seg.get("hay_agua", (prob.mean()>0.02)))
                overlay = overlay_water(bgr, prob, alpha=0.45, thr=0.5)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Resultado Watershed – ¿Hay agua? {'Sí' if hay else 'No'}", use_container_width=True)

        else:
            # CNN
            mdl = get_model('models/Red_best_model.keras')
            if mdl is None:
                st.warning("Modelo CNN no disponible. Se usará un umbral simple como fallback.")
            if st.button("🤖 Detectar con Red Neuronal"):
                prob, hay = cnn_predict_tile(bgr)
                overlay = overlay_water(bgr, prob, alpha=0.45, thr=0.5)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Resultado CNN – ¿Hay agua? {'Sí' if hay else 'No'}", use_container_width=True)

else:
    st.info("Mueve el mapa y pulsa **Capturar imagen** para continuar.")

st.divider()
with st.expander("ℹ️ Notas y responsabilidades"):
    st.markdown("""
- Esta app es una **demostración**. Las imágenes provienen de *Esri World Imagery*. Verifica los **términos de uso/licencia** antes de usos productivos.
- El método *Watershed* puede fallar en ambientes con sombras/niebla o ríos turbios. Ajusta parámetros si es necesario.
- La opción *Red Neuronal* espera un archivo `models/CNNWater_best_model.keras`. Si no existe, se usa un método de umbral simple como reserva.
""")
