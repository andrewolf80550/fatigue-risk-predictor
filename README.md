# Fatigue Risk Predictor ✈️

Predicción en tiempo real de riesgo de fatiga en tripulaciones de vuelo, basada en datos reales del **NASA Aviation Safety Reporting System (ASRS)** y entrenada con **Random Forest**.

## Stack

- **Python** — pandas, numpy, scikit-learn, streamlit
- **Modelo:** Random Forest (200 trees, max_depth=8, class_weight='balanced')
- **Dataset:** 208 reportes reales ASRS — Air Carrier FAR 121 Fatigue Reports

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

---

## 📊 Lo que nos dicen los gráficos

### No existe la fatiga "suave"
El 78% de los reportes (163 de 208) eran de alto riesgo. El resto, riesgo medio. Cuando un piloto reporta fatiga, la situación casi siempre es grave.

### La madrugada es el enemigo
Los peores incidentes se concentran entre las 00:01 y las 06:00. Biología pura. Es la Ventana de Mínimo Rendimiento Circadiano (WOCL). El cuerpo simplemente rinde menos.

### El peor momento posible
La mayoría de los incidentes ocurren durante la aproximación final y el aterrizaje. Justo cuando más concentración se necesita y cuando la fatiga acumulada del vuelo hace su trabajo.

### La fatiga nunca viaja sola
En el 85% de los casos venía acompañada de sobrecarga de trabajo, pérdida de conciencia situacional o fallas de comunicación. Como dice la OACI: la fatiga no causa el accidente sola, los amplifica todos.

### Las palabras del peligro
Al analizar el texto libre de los reportes de alto riesgo, las palabras más frecuentes fueron: *aircraft, approach, runway, landing, crew, captain*. Aviones aterrizando con tripulaciones agotadas.

---

## 🤖 ¿Por qué Random Forest?

Si quieres decidir si un vuelo es riesgoso, podrías preguntarle a un solo experto. Pero ese experto puede tener sesgos.

El Bosque Aleatorio hace algo más inteligente: consulta a **200 expertos al mismo tiempo**. Cada uno mira el problema desde un ángulo diferente. Luego, toma la decisión que la mayoría recomienda.

Lo elegimos por 3 razones clave para este proyecto:

- **Robusto con pocos datos:** Solo teníamos 208 reportes. Muchos algoritmos necesitan miles para funcionar; este modelo es fuerte con muestras pequeñas.
- **Es explicable:** No es una caja negra. Nos dice exactamente qué variables fueron decisivas, algo fundamental en aviación para justificar decisiones ante los reguladores.
- **No colapsa:** Si un reporte tiene datos raros o incompletos, el consenso de los "expertos" (árboles) salva el resultado.

**Resultado:** 90% de exactitud general y detectó el **98% de los casos de alto riesgo reales**. En seguridad operacional, esto significa que el modelo casi no deja pasar ninguna situación crítica sin levantar una alerta.

---

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `app.py` | Streamlit app de predicción en tiempo real |
| `fatigue_analysis-checkpoint.py` | Análisis completo del dataset ASRS + dashboard |
| `data.csv` | Dataset original NASA ASRS (208 reportes) |
| `fatigue_asrs_dashboard.png` | Dashboard generado con 6 gráficos |

## Fuente

NASA Aviation Safety Reporting System — [asrs.arc.nasa.gov](https://asrs.arc.nasa.gov)
