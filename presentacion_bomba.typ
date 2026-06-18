// ============================================================
// Presentacion ejecutiva — Fase 4 (DL_TA4)
// Compilar:  typst compile presentacion_bomba.typ
// ============================================================
#set document(title: "Sistema de Alerta Temprana — Bomba Industrial",
              author: "Grupo DL_TA4")
#set page(paper: "presentation-16-9",
          margin: (x: 2.2cm, top: 1.7cm, bottom: 1.3cm),
          fill: white)
#set text(size: 22pt)
#set par(spacing: 1.0em)

#let accent = rgb("#1565c0")
#let danger = rgb("#c62828")
#let ok = rgb("#2e7d32")

#let slide(title, body) = {
  pagebreak()
  block(spacing: 0pt)[
    #text(size: 30pt, weight: "bold", fill: accent)[#title]
    #v(4pt)
    #line(length: 100%, stroke: 2pt + accent)
    #v(14pt)
  ]
  set text(size: 21pt)
  body
}

// ---------- Portada ----------
#align(center + horizon)[
  #text(size: 38pt, weight: "bold", fill: accent)[
    Sistema de Alerta Temprana \ para una Bomba Industrial
  ]
  #v(12pt)
  #text(size: 24pt)[Mantenimiento predictivo con redes recurrentes (LSTM)]
  #v(26pt)
  #text(size: 18pt, fill: rgb("#555"))[
    Examen DL_TA4 — Modelado secuencial · Presentación ejecutiva
  ]
  #v(8pt)
  #text(size: 15pt, fill: rgb("#888"))[
    Grupo: Vilchez Rody · Velasquez Christian · Lazaro Luis \
    Sección: 18522 · Fecha: 17/06/2026
  ]
]

// ---------- Diapositiva 1 ----------
#slide("1 · El problema en números")[
  - *7 paradas por falla* en \~5 meses de operación (abr–ago 2018, registros cada minuto).
  - La bomba estuvo en estado de anomalía el *6.6 % del tiempo*.
  - Cada falla no anticipada = *parada correctiva* (no planificada): mucho más cara que una intervención *preventiva* programada.
  #v(6pt)
  #block(fill: rgb("#fff3e0"), inset: 12pt, radius: 6pt, width: 100%)[
    *Ejemplo de costo:* si 1 h de parada ≈ US\$ 5 000 y una falla implica \~6 h fuera de
    servicio → *\~US\$ 30 000 por evento*. Anticiparla lo reduce a una intervención
    planificada de pocas horas.
  ]
]

// ---------- Diapositiva 2 ----------
#slide("2 · Cómo funciona el modelo")[
  - El modelo observa los *últimos 30 minutos* de los 49 sensores antes de cada instante.
  - Analogía: no juzga por una sola foto, sino por la *evolución reciente* de los signos
    vitales del equipo.
  - Aprende a reconocer la *deriva temprana* de las señales —vibración, temperatura o
    presión que empiezan a desviarse— *antes* de que la bomba falle.
  #v(10pt)
  #align(center)[
    #block(fill: rgb("#e3f2fd"), inset: 14pt, radius: 6pt)[
      #text(size: 20pt)[ últimos 30 min de sensores  →  *modelo LSTM*  →  ¿falla en camino? ]
    ]
  ]
]

// ---------- Diapositiva 3 ----------
#slide("3 · Qué tan confiable es")[
  - Detecta el *99.6 % de las fallas* reales (4 311 de 4 327; solo *16 sin detectar*).
  - De cada 100 alertas, *\~88 son fallas reales* (precisión 88 %); el resto son falsas
    alarmas que se resuelven con una inspección rápida.
  #v(8pt)
  #align(center)[
    #image("results/fase2_pred_vs_real.png", width: 86%)
    #text(size: 14pt, fill: rgb("#777"))[
      Predicción del modelo (naranja) vs estado real (azul) en un episodio de falla del periodo de prueba.
    ]
  ]
]

// ---------- Diapositiva 4 ----------
#slide("4 · Qué falta antes de producción")[
  + *Una sola bomba y pocos episodios* (7 fallas): validar con más equipos y más eventos.
  + *Umbral de alerta por calibrar* con operaciones (el slider del dashboard lo simula en vivo).
  + *Posible cambio de comportamiento* en el tiempo (concept drift): requiere reentrenamiento periódico y monitoreo.
  #v(10pt)
  #block(fill: rgb("#e8f5e9"), inset: 12pt, radius: 6pt, width: 100%)[
    *Siguiente paso:* piloto de 1–2 meses en _shadow mode_ (en paralelo a la operación
    actual), midiendo cuántas fallas habría anticipado y con cuánto margen, antes de
    conectarlo a alertas reales.
  ]
]
