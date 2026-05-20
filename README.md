# Fatigue Risk Predictor ✈️

Predicción en tiempo real de riesgo de fatiga en tripulaciones de vuelo, basada en datos reales del **NASA Aviation Safety Reporting System (ASRS)** y entrenada con **Random Forest**.

## Stack

- **Python** — pandas, numpy, scikit-learn, streamlit
- **Modelo:** Random Forest (200 trees, max_depth=8, class_weight='balanced')
- **Dataset:** ~200 reportes reales ASRS — Air Carrier FAR 121 Fatigue Reports

## Regulaciones

- FAA 14 CFR Part 117 — Flight & Duty Limitations
- ICAO Annex 6 — Operation of Aircraft
- EASA ORO.FTL — Flight Time Limitations

## Uso

```bash
pip install streamlit pandas numpy scikit-learn
streamlit run app.py
```

## Inputs del usuario

- Período circadiano (WOCL, Morning, Afternoon, Night)
- Fase de vuelo (crítica vs no crítica)
- Fatiga reportada por capitán y/o primer oficial
- Factores humanos concurrentes (workload, distraction, situational awareness, comm breakdown, physiological)
- Severidad de anomalía y menciones de fatiga en narrativa

## Salida

- Nivel de riesgo: 🟢 Low / 🟡 Medium / 🔴 High
- Probabilidades del modelo (porcentaje por clase)
- Alerta regulatoria según FAA Part 117 (§117.23 / §117.25)

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Streamlit app de predicción en tiempo real |
| `fatigue_analysis-checkpoint.py` | Análisis completo del dataset ASRS + dashboard |
| `data.csv` | Dataset original NASA ASRS |
| `fatigue_asrs_dashboard.png` | Dashboard generado con 6 gráficos |

## Fuente

NASA Aviation Safety Reporting System — [asrs.arc.nasa.gov](https://asrs.arc.nasa.gov)
