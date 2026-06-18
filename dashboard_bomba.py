
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuracion de la pagina
st.set_page_config(
    page_title="Monitor de Bomba — Mantenimiento Predictivo",
    layout="wide"
)

st.title("Sistema de Alerta Temprana — Bomba Industrial")
st.caption("Modelo LSTM entrenado sobre datos reales (Kaggle: pump_sensor_data)")

# ---- Sidebar ----
st.sidebar.header("Configuracion del operador")
umbral_alerta = st.sidebar.slider(
    "Umbral de probabilidad para alerta",
    min_value=0.30, max_value=0.90, value=0.50, step=0.05
)
ventana_viz = st.sidebar.slider(
    "Ventana de visualizacion (pasos)",
    min_value=100, max_value=2000, value=500
)

# ---- Cargar resultados reales del notebook (resultados.pkl) ----
if os.path.exists("resultados.pkl"):
    with open("resultados.pkl", "rb") as f:
        R = pickle.load(f)
    y_true_full     = np.asarray(R["y_true"])
    y_prob_full     = np.asarray(R["y_prob"])
    ts_full         = pd.to_datetime(R["timestamps"])
    recall_anomalia = float(R["recall"])
    st.sidebar.success("Datos reales del modelo cargados (resultados.pkl)")
else:
    # Respaldo de demostracion si no se exporto resultados.pkl
    n = 2000
    y_true_full     = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
    y_prob_full     = np.column_stack([1 - y_true_full * 0.8, y_true_full * 0.8])
    ts_full         = pd.date_range("2018-04-01", periods=n, freq="1min")
    recall_anomalia = 0.74
    st.sidebar.warning("resultados.pkl no encontrado: usando datos de demostracion")

# ---- Seleccionar una ventana centrada en la zona mas anomala ----
centro = int(np.argmax(y_prob_full[:, 1])) if len(y_prob_full) else 0
ini = max(0, centro - ventana_viz // 2)
fin = min(len(y_true_full), ini + ventana_viz)
ini = max(0, fin - ventana_viz)

y_true_test     = y_true_full[ini:fin]
y_prob_test     = y_prob_full[ini:fin]
timestamps_test = ts_full[ini:fin]
y_pred_test     = (y_prob_test[:, 1] >= umbral_alerta).astype(int)

# ---- Indicadores principales ----
ultimo_estado = "ANOMALIA" if y_pred_test[-1] == 1 else "NORMAL"
ultima_prob   = float(y_prob_test[-1, 1])
n_alertas     = int((y_pred_test == 1).sum())

col1, col2, col3 = st.columns(3)

with col1:
    color = "#d32f2f" if ultimo_estado == "ANOMALIA" else "#388e3c"
    st.markdown(
        f"<div style='background:{color};padding:18px;border-radius:8px;"
        f"text-align:center;color:white;font-size:18px;font-weight:bold;'>"
        f"Estado actual: {ultimo_estado}</div>",
        unsafe_allow_html=True
    )

with col2:
    st.metric("Probabilidad de anomalia", f"{ultima_prob:.1%}")
    st.progress(float(min(max(ultima_prob, 0.0), 1.0)))

with col3:
    st.metric(
        "Recall del modelo", f"{recall_anomalia:.1%}",
        help="De cada 10 fallas reales, el modelo detecta este porcentaje"
    )
    st.metric("Alertas generadas (ventana)", n_alertas)

st.divider()

# ---- Grafico principal ----
st.subheader("Prediccion del modelo vs estado real")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=timestamps_test, y=y_true_test,
    mode="lines", name="Estado real",
    line=dict(color="steelblue", width=1.5)
))
fig.add_trace(go.Scatter(
    x=timestamps_test, y=y_pred_test,
    mode="lines", name="Prediccion modelo",
    line=dict(color="darkorange", width=1.5, dash="dot")
))
fig.add_trace(go.Scatter(
    x=timestamps_test, y=y_prob_test[:, 1],
    mode="lines", name="Probabilidad anomalia",
    line=dict(color="red", width=1, dash="dash"),
    opacity=0.5
))
fig.add_hline(
    y=umbral_alerta, line_dash="dot",
    line_color="gray", annotation_text=f"Umbral: {umbral_alerta:.2f}"
)
fig.update_layout(
    xaxis_title="Tiempo",
    yaxis=dict(tickvals=[0, 1], ticktext=["NORMAL", "ANOMALIA"]),
    legend=dict(orientation="h", y=-0.2),
    height=350,
    margin=dict(l=40, r=20, t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

st.caption("Prototipo academico. No usar en produccion sin validacion adicional.")
