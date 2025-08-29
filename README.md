# Demo Streamlit – Detector de cuerpos de agua

## Estructura
```
streamlit_water_app/
├─ app.py
├─ tiles.py
├─ methods/
│  ├─ watershed.py
│  └─ cnn_infer.py
├─ models/
│  └─ CNNWater_best_model.keras   # (coloca aquí tu modelo, opcional)
└─ requirements.txt
```

## Uso
1. Crea un ambiente e instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta:
   ```bash
   streamlit run app.py
   ```
3. En el panel lateral selecciona el método, ajusta zoom, **captura** la vista y luego **detecta**.
