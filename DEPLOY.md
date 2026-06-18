# Desplegar el dashboard en Streamlit Community Cloud

El dashboard `dashboard_bomba.py` carga `resultados.pkl` (resultados reales del modelo,
ya versionado en el repo) y muestra el semáforo de alerta + el gráfico predicción vs
estado real, con un slider de umbral.

## Pasos

1. Entrar a https://share.streamlit.io con la cuenta de GitHub.
2. **New app** → seleccionar el repo `nakato156/Deep-Learning-course`, branch `main`.
3. **Main file path**: `dashboard_bomba.py`.
4. **Deploy**. Streamlit Cloud instala lo de `requirements.txt` y publica una URL pública.

## Archivos que el deploy necesita (ya en el repo)

- `dashboard_bomba.py` — la app.
- `requirements.txt` — `streamlit`, `pandas`, `numpy>=2.0`, `plotly`.
- `resultados.pkl` — datos reales del modelo (`y_true`, `y_pred`, `y_prob`, `timestamps`, `recall`).

## Local

```bash
pip install -r requirements.txt
streamlit run dashboard_bomba.py
```
