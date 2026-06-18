import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Monitor de Bomba — Mantenimiento Predictivo",
    page_icon="\U0001F6A6", layout="wide"
)

# ============================================================
# Carga de datos (cacheada) + utilidades
# ============================================================
@st.cache_data
def cargar_datos():
    if os.path.exists("resultados.pkl"):
        with open("resultados.pkl", "rb") as f:
            R = pickle.load(f)
        y_true = np.asarray(R["y_true"]).astype(int)
        y_prob = np.asarray(R["y_prob"])[:, 1].astype(float)
        ts = pd.to_datetime(R["timestamps"])
        return y_true, y_prob, ts, True
    # Respaldo de demostracion si no existe resultados.pkl
    n = 4000
    rng = np.random.default_rng(0)
    y_true = np.zeros(n, dtype=int)
    y_true[1500:1850] = 1
    y_true[3000:3120] = 1
    y_prob = np.clip(rng.uniform(0, 0.2, n) + y_true * rng.uniform(0.6, 0.9, n), 0, 1)
    ts = pd.date_range("2018-07-01", periods=n, freq="1min")
    return y_true, y_prob, ts, False


def metricas(yt, prob, thr):
    pred = prob >= thr
    tp = int((pred & (yt == 1)).sum())
    fp = int((pred & (yt == 0)).sum())
    fn = int((~pred & (yt == 1)).sum())
    tn = int((~pred & (yt == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(yt) if len(yt) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=prec, rec=rec, f1=f1, acc=acc)


def episodios(yt):
    idx = np.where(yt == 1)[0]
    if len(idx) == 0:
        return []
    brk = np.where(np.diff(idx) > 1)[0]
    ini = np.concatenate([[idx[0]], idx[brk + 1]])
    fin = np.concatenate([idx[brk], [idx[-1]]])
    return list(zip(ini.tolist(), fin.tolist()))


def runs_anomalia(yt):
    """Tramos contiguos donde yt==1 (para sombrear el grafico)."""
    out, i, n = [], 0, len(yt)
    while i < n:
        if yt[i] == 1:
            j = i
            while j < n and yt[j] == 1:
                j += 1
            out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


y_true, y_prob, ts, datos_reales = cargar_datos()
N = len(y_true)
eps = episodios(y_true)
m = None  # se calcula tras leer el umbral

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("⚙️ Configuración del operador")
umbral = st.sidebar.slider("Umbral de probabilidad para alerta", 0.05, 0.95, 0.50, 0.05)
if datos_reales:
    st.sidebar.success("Datos reales del modelo (resultados.pkl)")
else:
    st.sidebar.warning("resultados.pkl no encontrado: datos de demostración")

opciones = ["Zona más crítica"] + [
    f"Episodio {i+1} — {ts[s].strftime('%d/%b %H:%M')} ({e - s + 1} min)"
    for i, (s, e) in enumerate(eps)
]
sel = st.sidebar.selectbox("Ver episodio de falla", opciones)
ventana = st.sidebar.slider("Ancho de ventana (pasos = minutos)", 200, 4000, 1500, 100)

if sel == "Zona más crítica" or not eps:
    centro = int(np.argmax(y_prob)) if N else 0
else:
    s, e = eps[opciones.index(sel) - 1]
    centro = (s + e) // 2
ini = max(0, centro - ventana // 2)
fin = min(N, ini + ventana)
ini = max(0, fin - ventana)

# ============================================================
# Encabezado + KPIs (reaccionan al umbral)
# ============================================================
st.title("\U0001F6A6 Sistema de Alerta Temprana — Bomba Industrial")
st.caption(
    "Modelo LSTM · datos reales (Kaggle: pump_sensor_data) · "
    f"{N:,} ventanas de prueba, {int(y_true.sum()):,} de falla"
)

m = metricas(y_true, y_prob, umbral)
m50 = metricas(y_true, y_prob, 0.50)
estado = "ANOMALÍA" if (N and y_prob[fin - 1] >= umbral) else "NORMAL"
color = "#d32f2f" if estado == "ANOMALÍA" else "#2e7d32"

c0, c1, c2, c3 = st.columns([1.4, 1, 1, 1])
with c0:
    st.markdown(
        f"<div style='background:{color};padding:18px;border-radius:10px;"
        f"text-align:center;color:#fff;font-size:19px;font-weight:700;'>"
        f"Estado actual<br>{estado}</div>",
        unsafe_allow_html=True,
    )
    if N:
        st.caption(f"Prob. anomalía (último punto de la ventana): {y_prob[fin-1]:.0%}")
c1.metric("Recall — fallas detectadas", f"{m['rec']:.1%}",
          f"{(m['rec']-m50['rec'])*100:+.1f} pp vs 0.50",
          help="De todas las fallas reales, cuántas detecta. Lo más crítico en mantenimiento predictivo.")
c2.metric("Precisión — alertas acertadas", f"{m['prec']:.1%}",
          f"{(m['prec']-m50['prec'])*100:+.1f} pp vs 0.50",
          help="De todas las alertas que emite, cuántas eran fallas reales.")
c3.metric("Fallas NO detectadas", f"{m['fn']:,}",
          f"{m['fn']-m50['fn']:+d} vs 0.50", delta_color="inverse",
          help="Falsos negativos: el peor caso operativo.")

st.divider()

tab_mon, tab_perf = st.tabs(["\U0001F4CA Monitor en vivo", "\U0001F4C8 Rendimiento del modelo"])

# ============================================================
# Tab 1 — Monitor en vivo
# ============================================================
with tab_mon:
    wt = ts[ini:fin]
    wy = y_true[ini:fin]
    wp = y_prob[ini:fin]
    wpred = (wp >= umbral).astype(int)

    fig = go.Figure()
    # sombrear episodios reales de falla dentro de la ventana
    for a, b in runs_anomalia(wy):
        fig.add_vrect(x0=wt[a], x1=wt[b], fillcolor="#d32f2f", opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=wt, y=wy, mode="lines", name="Estado real",
                             line=dict(color="steelblue", width=1.8)))
    fig.add_trace(go.Scatter(x=wt, y=wpred, mode="lines", name="Predicción modelo",
                             line=dict(color="darkorange", width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=wt, y=wp, mode="lines", name="Prob. anomalía",
                             line=dict(color="#c62828", width=1), opacity=0.55))
    fig.add_hline(y=umbral, line_dash="dot", line_color="gray",
                  annotation_text=f"Umbral {umbral:.2f}")
    fig.update_layout(
        xaxis_title="Tiempo",
        yaxis=dict(tickvals=[0, 1], ticktext=["NORMAL", "ANOMALÍA"], range=[-0.05, 1.1]),
        legend=dict(orientation="h", y=-0.25), height=380,
        margin=dict(l=40, r=20, t=10, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    a, b, c = st.columns(3)
    a.metric("Alertas en la ventana", int(wpred.sum()))
    b.metric("Falsas alarmas en la ventana", int(((wpred == 1) & (wy == 0)).sum()))
    c.metric("Minutos de falla reales en la ventana", int(wy.sum()))
    st.caption("Franja roja = episodio de falla real. La predicción y las métricas de arriba "
               "cambian al mover el umbral en la barra lateral.")

# ============================================================
# Tab 2 — Rendimiento del modelo (sobre TODO el test)
# ============================================================
with tab_perf:
    izq, der = st.columns([1, 1.2])

    with izq:
        st.markdown("**Matriz de confusión** (% por clase real)")
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
        fila = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, fila, out=np.zeros_like(cm), where=fila != 0) * 100
        txt = [[f"{cm_pct[i, j]:.1f}%<br>({int(cm[i, j]):,})" for j in range(2)]
               for i in range(2)]
        figcm = go.Figure(go.Heatmap(
            z=cm_pct, x=["Pred. NORMAL", "Pred. ANOMALÍA"],
            y=["Real NORMAL", "Real ANOMALÍA"],
            text=txt, texttemplate="%{text}", zmin=0, zmax=100,
            colorscale="Blues", showscale=False, hoverinfo="skip"))
        figcm.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(figcm, width="stretch")
        st.caption("Cada fila suma 100% (porcentaje de cada clase real); entre paréntesis, "
                   "el conteo. La celda inferior derecha es el recall.")
        st.metric("Accuracy global", f"{m['acc']:.2%}")
        st.metric("F1-score (anomalía)", f"{m['f1']:.3f}")

    with der:
        st.markdown("**Precisión y Recall según el umbral**")
        ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
        precs, recs = [], []
        for t in ths:
            mm = metricas(y_true, y_prob, float(t))
            precs.append(mm["prec"]); recs.append(mm["rec"])
        figpr = go.Figure()
        figpr.add_trace(go.Scatter(x=ths, y=recs, name="Recall",
                                   line=dict(color="#1565c0", width=2)))
        figpr.add_trace(go.Scatter(x=ths, y=precs, name="Precisión",
                                   line=dict(color="#ff8c00", width=2)))
        figpr.add_vline(x=umbral, line_dash="dot", line_color="gray",
                        annotation_text=f"{umbral:.2f}")
        figpr.update_layout(xaxis_title="Umbral", yaxis_title="Valor",
                            yaxis_range=[0, 1.02], height=300,
                            legend=dict(orientation="h", y=-0.3),
                            margin=dict(l=40, r=20, t=10, b=20))
        st.plotly_chart(figpr, width="stretch")
        st.caption("Subir el umbral aumenta la precisión pero reduce el recall (menos falsas "
                   "alarmas, pero más fallas podrían escaparse). El punto de operación se elige "
                   "con el área de confiabilidad según el costo de cada error.")

st.divider()
st.caption("Prototipo académico (examen DL_TA4). No usar en producción sin validación adicional.")
