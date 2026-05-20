import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fatigue Risk Predictor — ASRS",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_train():
    df = pd.read_csv('data.csv', header=[0, 1], skiprows=[2])
    df.columns = ['__'.join(c).strip() for c in df.columns]
    rename = {
        ' __ACN': 'ACN',
        'Time__Local Time Of Day': 'Time_Of_Day',
        'Aircraft 1__Flight Phase': 'Flight_Phase',
        'Aircraft 1__Operating Under FAR Part': 'FAR_Part',
        'Environment__Flight Conditions': 'Flight_Conditions',
        'Environment__Light': 'Light_Condition',
        'Person 1__Human Factors': 'HF_Person1',
        'Person 2__Human Factors': 'HF_Person2',
        'Events__Anomaly': 'Anomaly',
        'Events__Result': 'Result',
        'Assessments__Primary Problem': 'Primary_Problem',
        'Report 1__Narrative': 'Narrative',
        'Report 1__Synopsis': 'Synopsis',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)

    def has_factor(text, kw):
        return 0 if pd.isna(text) else (1 if kw.lower() in str(text).lower() else 0)

    df['fatigue_p1'] = df['HF_Person1'].apply(lambda x: has_factor(x, 'fatigue'))
    df['fatigue_p2'] = df['HF_Person2'].apply(lambda x: has_factor(x, 'fatigue'))
    df['fatigue_either'] = ((df['fatigue_p1'] == 1) | (df['fatigue_p2'] == 1)).astype(int)

    for kw in ['Workload', 'Distraction', 'Situational Awareness', 'Communication Breakdown', 'Physiological']:
        col = kw.replace(' ', '_').replace('/', '_')
        df[f'hf_{col}'] = (
            df['HF_Person1'].apply(lambda x: has_factor(x, kw)) |
            df['HF_Person2'].apply(lambda x: has_factor(x, kw))
        ).astype(int)

    hf_cols = [c for c in df.columns if c.startswith('hf_')]
    df['total_hf_factors'] = df[hf_cols].sum(axis=1) + df['fatigue_either']

    time_map = {'0001-0600': 'WOCL', '0601-1200': 'Morning', '1201-1800': 'Afternoon', '1801-2400': 'Night'}
    df['Time_Period'] = df['Time_Of_Day'].map(time_map).fillna('Unknown')
    df['is_WOCL'] = (df['Time_Period'] == 'WOCL').astype(int)
    df['is_Night'] = df['Time_Period'].isin(['WOCL', 'Night']).astype(int)

    critical_phases = ['Final Approach', 'Initial Approach', 'Takeoff / Launch', 'Landing', 'Initial Climb; Climb']
    df['critical_phase'] = df['Flight_Phase'].apply(lambda x: 1 if str(x) in critical_phases else 0)

    fatigue_kw = ['fatigue', 'tired', 'exhausted', 'sleep', 'rest', 'duty time', 'duty period', 'circadian', 'wocl']
    def nlp_score(text):
        if pd.isna(text): return 0
        txt = str(text).lower()
        return sum(1 for kw in fatigue_kw if kw in txt)

    df['narrative_fatigue_mentions'] = df['Narrative'].apply(nlp_score)
    df['synopsis_fatigue_mentions'] = df['Synopsis'].apply(nlp_score)

    severe_kw = ['Critical', 'Serious', 'CFIT', 'Conflict Airborne']
    df['anomaly_severity'] = df['Anomaly'].apply(lambda x: 0 if pd.isna(x) else sum(1 for s in severe_kw if s.lower() in str(x).lower()))
    df['primary_human'] = (df['Primary_Problem'] == 'Human Factors').astype(int)

    def classify_risk(row):
        score = row['fatigue_either'] * 3 + row['is_WOCL'] * 3 + row['is_Night'] * 1 + \
                row['critical_phase'] * 2 + row['total_hf_factors'] + \
                min(row['narrative_fatigue_mentions'], 3) + row['anomaly_severity'] * 2 + row['primary_human'] * 1
        return 'High' if score >= 9 else ('Medium' if score >= 5 else 'Low')

    df['Risk_Level'] = df.apply(classify_risk, axis=1)

    features = [
        'fatigue_p1', 'fatigue_p2', 'is_WOCL', 'is_Night', 'critical_phase',
        'total_hf_factors', 'narrative_fatigue_mentions', 'synopsis_fatigue_mentions',
        'anomaly_severity', 'primary_human', 'hf_Workload', 'hf_Distraction',
        'hf_Situational_Awareness', 'hf_Communication_Breakdown', 'hf_Physiological'
    ]

    X = df[features].fillna(0)
    y = df['Risk_Level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    return model, features, df

model, features, df = load_and_train()

st.title("✈️ Fatigue Risk Predictor — Tiempo Real")
st.markdown("### NASA ASRS · FAA Part 117 · ICAO Annex 6 · EASA ORO.FTL")

col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

with col1:
    st.subheader("⏰ Período del Día")
    time_period = st.selectbox(
        "Período Circadiano (FAA Part 117)",
        ["WOCL (00:01–06:00)", "Morning (06:01–12:00)", "Afternoon (12:01–18:00)", "Night (18:01–24:00)"],
        help="WOCL = Window of Circadian Low, mayor riesgo de fatiga"
    )
    is_wocl = 1 if "WOCL" in time_period else 0
    is_night = 1 if ("WOCL" in time_period or "Night" in time_period) else 0

    flight_phase = st.selectbox(
        "Fase de Vuelo",
        ["Cruise", "Final Approach", "Landing", "Initial Approach", "Takeoff / Launch",
         "Climb", "Descent", "Initial Climb; Climb", "Taxi", "Parked"]
    )
    critical_phases_list = ['Final Approach', 'Initial Approach', 'Takeoff / Launch', 'Landing', 'Initial Climb; Climb']
    critical_phase = 1 if flight_phase in critical_phases_list else 0

    st.subheader("👥 Estado de la Tripulación")
    fatigue_p1 = st.checkbox("Capitán reporta fatiga", value=False)
    fatigue_p2 = st.checkbox("Primer Oficial reporta fatiga", value=False)

with col2:
    st.subheader("⚙️ Factores Humanos")
    hf_workload = st.checkbox("Workload (Alta carga de trabajo)", value=False, help="Presión operacional elevada")
    hf_distraction = st.checkbox("Distraction", value=False, help="Factores de distracción presentes")
    hf_sa = st.checkbox("Situational Awareness", value=False, help="Pérdida de conciencia situacional")
    hf_comm = st.checkbox("Communication Breakdown", value=False, help="Falla en comunicación")
    hf_physio = st.checkbox("Physiological", value=False, help="Factores fisiológicos (hambre, sed, etc.)")

    st.subheader("📊 Severidad")
    narrative_mentions = st.slider("Menciones de fatiga en reporte", 0, 5, 1, help="Frecuencia de términos de fatiga en narrativa")
    synopsis_mentions = st.slider("Menciones en sinopsis", 0, 5, 0)
    anomaly_severity = st.slider("Severidad de anomalía", 0, 3, 0, help="0=Ninguna, 3=Crítica (CFIT/Conflict)")

with col3:
    st.subheader("🎯 Resultado de la Predicción")

    input_data = {
        'fatigue_p1': int(fatigue_p1), 'fatigue_p2': int(fatigue_p2),
        'is_WOCL': is_wocl, 'is_Night': is_night,
        'critical_phase': critical_phase,
        'total_hf_factors': int(fatigue_p1) + int(fatigue_p2) + int(hf_workload) + int(hf_distraction) + int(hf_sa) + int(hf_comm) + int(hf_physio),
        'narrative_fatigue_mentions': narrative_mentions,
        'synopsis_fatigue_mentions': synopsis_mentions,
        'anomaly_severity': anomaly_severity,
        'primary_human': 1 if (hf_workload or hf_distraction or hf_sa or hf_comm or hf_physio) else 0,
        'hf_Workload': int(hf_workload), 'hf_Distraction': int(hf_distraction),
        'hf_Situational_Awareness': int(hf_sa),
        'hf_Communication_Breakdown': int(hf_comm), 'hf_Physiological': int(hf_physio)
    }

    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    color_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    emoji_map = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
    msg_map = {
        'Low': 'Riesgo aceptable. Monitoreo estándar.',
        'Medium': 'Monitoreo activo requerido. Verificar horas acumuladas (§117.23).',
        'High': 'ALERTA CRÍTICA. Considerar relevo de tripulación. Revisar descanso (§117.25) y notificar Safety Officer.'
    }

    st.markdown(
        f"<div style='background:{color_map[pred]}20; padding:25px; border-radius:15px; "
        f"border:3px solid {color_map[pred]}; text-align:center;'>"
        f"<h1 style='font-size:72px; margin:0;'>{emoji_map[pred]}</h1>"
        f"<h2 style='color:{color_map[pred]}; font-weight:bold;'>RIESGO {pred.upper()}</h2>"
        f"<p style='font-size:16px;'>{msg_map[pred]}</p>"
        f"</div>", unsafe_allow_html=True
    )

    st.markdown("### 📊 Probabilidades")
    classes = model.classes_
    for label in ['Low', 'Medium', 'High']:
        if label in classes:
            idx = list(classes).index(label)
            pct = proba[idx]
            bar_color = color_map[label]
            st.markdown(
                f"<div style='display:flex; align-items:center; margin:6px 0;'>"
                f"<span style='width:70px; font-weight:bold;'>{label}</span>"
                f"<div style='flex:1; height:24px; background:#eee; border-radius:12px; margin:0 10px;'>"
                f"<div style='width:{pct*100:.1f}%; height:24px; background:{bar_color}; "
                f"border-radius:12px; text-align:center; line-height:24px; color:white; font-size:13px;'>"
                f"{pct:.1%}</div></div></div>", unsafe_allow_html=True
            )

    if pred == 'High':
        st.error("⚠️ ACTIVAR PROTOCOLO DE FATIGA — Relevo inmediato recomendado")
    elif pred == 'Medium':
        st.warning("⚡ Monitorear — Revisar horas de servicio §117.23")
    else:
        st.success("✅ Operación dentro de parámetros normales")

st.markdown("---")
with st.expander("📋 Resumen de Factores Ingresados"):
    st.json(input_data)

with st.expander("📚 Base Regulatoria"):
    st.markdown("""
    - **FAA 14 CFR Part 117** — Flight & Duty Limitations
    - **ICAO Annex 6** — Operation of Aircraft
    - **EASA ORO.FTL** — Flight Time Limitations
    - **NASA ASRS** — Aviation Safety Reporting System
    - **Modelo:** Random Forest (200 trees, max_depth=8) entrenado con datos ASRS reales
    """)

st.caption("Fuente: NASA Aviation Safety Reporting System (ASRS) · Air Carrier FAR 121 Fatigue Reports")
