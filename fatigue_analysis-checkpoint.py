"""
ANÁLISIS PREDICTIVO DE FATIGA EN TRIPULACIONES DE VUELO
Dataset Real: NASA Aviation Safety Reporting System (ASRS)
Air Carrier (FAR 121) Flight Crew Fatigue Reports

Autor: Raúl Andrés Fajardo Murillo
Regulaciones: FAA Part 117, ICAO Annex 6, EASA ORO.FTL

Fuente de datos: https://asrs.arc.nasa.gov/search/reportsets.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# ─────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA DEL DATASET REAL
# ─────────────────────────────────────────────

def load_asrs_data(filepath):
    """Carga el CSV real de NASA ASRS (2 filas de encabezado)."""
    print("=" * 70)
    print("CARGA DEL DATASET NASA ASRS — FATIGUE REPORTS (FAR 121)")
    print("=" * 70)

    df = pd.read_csv(filepath, header=[0, 1], skiprows=[2])
    df.columns = ['__'.join(c).strip() for c in df.columns]

    rename = {
        ' __ACN':                                         'ACN',
        'Time__Date':                                     'Date',
        'Time__Local Time Of Day':                        'Time_Of_Day',
        'Aircraft 1__Flight Phase':                       'Flight_Phase',
        'Aircraft 1__Operating Under FAR Part':           'FAR_Part',
        'Aircraft 1__Make Model Name':                    'Aircraft_Model',
        'Aircraft 1__Mission':                            'Mission',
        'Environment__Flight Conditions':                 'Flight_Conditions',
        'Environment__Light':                             'Light_Condition',
        'Person 1__Human Factors':                        'HF_Person1',
        'Person 1__Function':                             'Function_P1',
        'Person 1__Qualification':                        'Qualification_P1',
        'Person 2__Human Factors':                        'HF_Person2',
        'Person 2__Function':                             'Function_P2',
        'Events__Anomaly':                                'Anomaly',
        'Events__When Detected':                          'When_Detected',
        'Events__Result':                                 'Result',
        'Assessments__Contributing Factors / Situations': 'Contributing_Factors',
        'Assessments__Primary Problem':                   'Primary_Problem',
        'Report 1__Narrative':                            'Narrative',
        'Report 1__Synopsis':                             'Synopsis',
        'Place__State Reference':                         'State',
    }
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns},
              inplace=True)

    print(f"\n✅ Dataset cargado: {len(df)} reportes reales de incidentes")
    print(f"   Período: {df['Date'].dropna().astype(str).min()[:4]} – "
          f"{df['Date'].dropna().astype(str).max()[:4]}")
    return df


# ─────────────────────────────────────────────
# 2. INGENIERÍA DE FEATURES
# ─────────────────────────────────────────────

def engineer_features(df):
    """Extrae y construye features desde los campos del ASRS."""

    def has_factor(text, keyword):
        if pd.isna(text):
            return 0
        return 1 if keyword.lower() in str(text).lower() else 0

    # Fatiga explícita por tripulante
    df['fatigue_p1']     = df['HF_Person1'].apply(lambda x: has_factor(x, 'fatigue'))
    df['fatigue_p2']     = df['HF_Person2'].apply(lambda x: has_factor(x, 'fatigue'))
    df['fatigue_either'] = ((df['fatigue_p1'] == 1) | (df['fatigue_p2'] == 1)).astype(int)

    # Otros factores humanos
    hf_keywords = ['Workload', 'Distraction', 'Situational Awareness',
                   'Communication Breakdown', 'Physiological']
    for kw in hf_keywords:
        col = kw.replace(' ', '_').replace('/', '_')
        df[f'hf_{col}'] = (
            df['HF_Person1'].apply(lambda x: has_factor(x, kw)) |
            df['HF_Person2'].apply(lambda x: has_factor(x, kw))
        )

    # Total de factores humanos por reporte
    hf_cols = [c for c in df.columns if c.startswith('hf_')]
    df['total_hf_factors'] = df[hf_cols].sum(axis=1) + df['fatigue_either']

    # Período circadiano (FAA Part 117 — WOCL: 00:01–05:59)
    time_map = {
        '0001-0600': 'WOCL',
        '0601-1200': 'Morning',
        '1201-1800': 'Afternoon',
        '1801-2400': 'Night',
    }
    df['Time_Period'] = df['Time_Of_Day'].map(time_map).fillna('Unknown')
    df['is_WOCL']     = (df['Time_Period'] == 'WOCL').astype(int)
    df['is_Night']    = df['Time_Period'].isin(['WOCL', 'Night']).astype(int)

    # Fases críticas de vuelo (ICAO Annex 6 — Sterile Cockpit)
    critical_phases = ['Final Approach', 'Initial Approach',
                       'Takeoff / Launch', 'Landing', 'Initial Climb; Climb']
    df['critical_phase']      = df['Flight_Phase'].apply(
        lambda x: 1 if str(x) in critical_phases else 0)
    df['Flight_Phase_Clean']  = df['Flight_Phase'].fillna('Unknown')

    # NLP básico sobre narrativas — menciones de fatiga
    fatigue_keywords = ['fatigue', 'tired', 'exhausted', 'sleep', 'rest',
                        'duty time', 'duty period', 'circadian', 'wocl']
    def narrative_fatigue_score(text):
        if pd.isna(text):
            return 0
        txt = str(text).lower()
        return sum(1 for kw in fatigue_keywords if kw in txt)

    df['narrative_fatigue_mentions'] = df['Narrative'].apply(narrative_fatigue_score)
    df['synopsis_fatigue_mentions']  = df['Synopsis'].apply(narrative_fatigue_score)

    # Severidad del incidente
    severe_keywords = ['Critical', 'Serious', 'CFIT', 'Conflict Airborne']
    def anomaly_severity(text):
        if pd.isna(text):
            return 0
        return sum(1 for s in severe_keywords if s.lower() in str(text).lower())

    df['anomaly_severity'] = df['Anomaly'].apply(anomaly_severity)
    df['primary_human']    = (df['Primary_Problem'] == 'Human Factors').astype(int)

    # Etiqueta de riesgo (basada en factores reales del reporte)
    def classify_risk(row):
        score  = row['fatigue_either']        * 3
        score += row['is_WOCL']               * 3
        score += row['is_Night']              * 1
        score += row['critical_phase']        * 2
        score += row['total_hf_factors']
        score += min(row['narrative_fatigue_mentions'], 3)
        score += row['anomaly_severity']      * 2
        score += row['primary_human']         * 1
        if score >= 9:
            return 'High'
        elif score >= 5:
            return 'Medium'
        else:
            return 'Low'

    df['Risk_Level'] = df.apply(classify_risk, axis=1)

    print("\n📊 Distribución de Riesgo (datos reales ASRS):")
    print(df['Risk_Level'].value_counts())
    print(f"\n📊 Reportes con fatiga explícita: {df['fatigue_either'].sum()} "
          f"({df['fatigue_either'].mean()*100:.1f}%)")
    print(f"📊 Reportes en WOCL (00:01-06:00): {df['is_WOCL'].sum()} "
          f"({df['is_WOCL'].mean()*100:.1f}%)")
    print(f"📊 Fase crítica de vuelo: {df['critical_phase'].sum()} "
          f"({df['critical_phase'].mean()*100:.1f}%)")
    return df


# ─────────────────────────────────────────────
# 3. MODELO RANDOM FOREST
# ─────────────────────────────────────────────

def train_model(df):
    """Entrena Random Forest con los features del dataset real."""
    features = [
        'fatigue_p1', 'fatigue_p2', 'is_WOCL', 'is_Night',
        'critical_phase', 'total_hf_factors',
        'narrative_fatigue_mentions', 'synopsis_fatigue_mentions',
        'anomaly_severity', 'primary_human',
        'hf_Workload', 'hf_Distraction', 'hf_Situational_Awareness',
        'hf_Communication_Breakdown', 'hf_Physiological'
    ]

    X = df[features].fillna(0)
    y = df['Risk_Level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 70)
    print("REPORTE DE CLASIFICACIÓN — RANDOM FOREST (Datos ASRS Reales)")
    print("=" * 70)
    print(classification_report(y_test, y_pred))

    feat_imp = pd.DataFrame({
        'feature':    features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, X_test, y_test, y_pred, feat_imp, features


# ─────────────────────────────────────────────
# 4. ANÁLISIS NLP
# ─────────────────────────────────────────────

def analyze_narratives(df):
    """Top keywords en narrativas de alto riesgo."""
    high_risk_texts = df[df['Risk_Level'] == 'High']['Narrative'].dropna()

    stopwords = {
        'the','a','an','and','or','but','in','on','at','to','for','of',
        'is','was','were','had','has','have','been','be','are','with',
        'from','that','this','it','we','i','my','our','their','as','by',
        'not','so','if','he','she','they','his','her','which','who',
        'when','what','how','would','could','should','did','do','will',
        'then','than','also','into','after','before','during','while',
        'about','up','out','no','more','very','just','other','one','two',
        'three','back','get','got','us','me','him','its','there','each',
        'any','all','some','them','such','am','via','per','upon'
    }

    words = []
    for t in high_risk_texts:
        for w in re.findall(r'\b[a-z]{4,}\b', str(t).lower()):
            if w not in stopwords:
                words.append(w)
    return Counter(words).most_common(20)


# ─────────────────────────────────────────────
# 5. VISUALIZACIONES
# ─────────────────────────────────────────────

def create_visualizations(df, feat_imp, X_test, y_test, y_pred, hr_words):
    """Dashboard con 6 gráficos sobre datos reales ASRS."""

    colors_risk = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    risk_order  = ['Low', 'Medium', 'High']

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        'Análisis Predictivo de Fatiga — NASA ASRS (FAR 121 Real Reports)\n'
        'FAA Part 117 | ICAO Annex 6 | EASA ORO.FTL',
        fontsize=14, fontweight='bold', y=0.98)

    # ── 1. Distribución de riesgo ─────────────────────────────
    ax1 = plt.subplot(2, 3, 1)
    risk_counts = df['Risk_Level'].value_counts().reindex(
        [r for r in risk_order if r in df['Risk_Level'].values])
    bars = ax1.bar(risk_counts.index, risk_counts.values,
                   color=[colors_risk[r] for r in risk_counts.index],
                   edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, risk_counts.values):
        pct = val / len(df) * 100
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
    ax1.set_title('Distribución de Riesgo de Fatiga\n(208 Reportes Reales ASRS)',
                  fontweight='bold')
    ax1.set_ylabel('Número de Reportes')
    ax1.set_ylim(0, risk_counts.max() * 1.2)
    ax1.grid(axis='y', alpha=0.3)

    # ── 2. Feature Importance ─────────────────────────────────
    ax2 = plt.subplot(2, 3, 2)
    label_map = {
        'fatigue_p1':                  'Fatiga P1 (explícita)',
        'fatigue_p2':                  'Fatiga P2 (explícita)',
        'is_WOCL':                     'Período WOCL (00–06h)',
        'is_Night':                    'Operación Nocturna',
        'critical_phase':              'Fase Crítica de Vuelo',
        'total_hf_factors':            'Total Factores Humanos',
        'narrative_fatigue_mentions':  'Menciones en Narrativa',
        'synopsis_fatigue_mentions':   'Menciones en Synopsis',
        'anomaly_severity':            'Severidad Anomalía',
        'primary_human':               'Problema: Human Factors',
        'hf_Workload':                 'Workload',
        'hf_Distraction':              'Distracción',
        'hf_Situational_Awareness':    'Situational Awareness',
        'hf_Communication_Breakdown':  'Comm. Breakdown',
        'hf_Physiological':            'Fisiológico',
    }
    top     = feat_imp.head(10)
    labels  = [label_map.get(f, f) for f in top['feature']]
    ax2.barh(labels, top['importance'], color='steelblue', edgecolor='white')
    ax2.set_xlabel('Importancia (Gini)')
    ax2.set_title('Factores Más Influyentes\n(Random Forest Model)', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # ── 3. Riesgo por Período del Día ─────────────────────────
    ax3 = plt.subplot(2, 3, 3)
    time_risk  = df.groupby(['Time_Period', 'Risk_Level']).size().unstack(fill_value=0)
    time_order = ['WOCL', 'Morning', 'Afternoon', 'Night', 'Unknown']
    time_risk  = time_risk.reindex([t for t in time_order if t in time_risk.index])
    time_risk[[r for r in risk_order if r in time_risk.columns]].plot(
        kind='bar', ax=ax3,
        color=[colors_risk[r] for r in risk_order if r in time_risk.columns],
        edgecolor='white')
    ax3.set_title('Riesgo por Período del Día\n(FAA Part 117 — WOCL Compliance)',
                  fontweight='bold')
    ax3.set_xlabel('Período del Día')
    ax3.set_ylabel('Reportes')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha='right')
    ax3.legend(title='Riesgo')
    ax3.grid(axis='y', alpha=0.3)
    if 'WOCL' in time_risk.index:
        ax3.axvspan(-0.5, 0.5, alpha=0.08, color='red')
        ax3.text(0, ax3.get_ylim()[1]*0.92, 'WOCL\n⚠️',
                 ha='center', fontsize=8, color='red')

    # ── 4. Riesgo por Fase de Vuelo ───────────────────────────
    ax4 = plt.subplot(2, 3, 4)
    phase_risk = (df.groupby(['Flight_Phase_Clean', 'Risk_Level'])
                    .size().unstack(fill_value=0))
    phase_risk['total'] = phase_risk.sum(axis=1)
    phase_risk = phase_risk.sort_values('total', ascending=False).head(8)
    phase_risk[[r for r in risk_order if r in phase_risk.columns]].plot(
        kind='barh', ax=ax4, stacked=True,
        color=[colors_risk[r] for r in risk_order if r in phase_risk.columns],
        edgecolor='white')
    ax4.set_title('Fatiga por Fase de Vuelo\n(ICAO Annex 6 — Critical Phases)',
                  fontweight='bold')
    ax4.set_xlabel('Número de Reportes')
    ax4.legend(title='Riesgo')
    ax4.grid(axis='x', alpha=0.3)

    # ── 5. Factores Humanos Co-ocurrentes con Fatiga ──────────
    ax5 = plt.subplot(2, 3, 5)
    fatigue_df = df[df['fatigue_either'] == 1]
    hf_cols    = ['hf_Workload', 'hf_Distraction', 'hf_Situational_Awareness',
                  'hf_Communication_Breakdown', 'hf_Physiological']
    hf_labels  = ['Workload', 'Distraction', 'Sit. Awareness',
                  'Comm. Breakdown', 'Physiological']
    hf_counts  = [fatigue_df[c].sum() for c in hf_cols]
    bars5 = ax5.barh(hf_labels, hf_counts,
                     color=['#3498db','#9b59b6','#e67e22','#1abc9c','#e74c3c'],
                     edgecolor='white')
    for bar, val in zip(bars5, hf_counts):
        ax5.text(bar.get_width() + 0.3,
                 bar.get_y() + bar.get_height()/2,
                 str(val), va='center', fontsize=10)
    ax5.set_title(f'Factores Co-ocurrentes con Fatiga\n'
                  f'(Base: {len(fatigue_df)} reportes con fatiga explícita)',
                  fontweight='bold')
    ax5.set_xlabel('Número de Reportes')
    ax5.invert_yaxis()
    ax5.grid(axis='x', alpha=0.3)

    # ── 6. Top Keywords en Reportes de Alto Riesgo ───────────
    ax6 = plt.subplot(2, 3, 6)
    words, counts = zip(*hr_words[:15])
    colors6 = plt.cm.Reds_r(np.linspace(0.2, 0.8, len(words)))
    ax6.barh(list(words), list(counts), color=colors6, edgecolor='white')
    ax6.set_title('Top Keywords — Reportes Alto Riesgo\n(NLP sobre Narrativas ASRS)',
                  fontweight='bold')
    ax6.set_xlabel('Frecuencia')
    ax6.invert_yaxis()
    ax6.grid(axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'fatigue_asrs_dashboard.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"\n✅ Dashboard guardado: {out}")


# ─────────────────────────────────────────────
# 6. REPORTE DE CUMPLIMIENTO
# ─────────────────────────────────────────────

def compliance_report(df):
    print("\n" + "=" * 70)
    print("REPORTE DE CUMPLIMIENTO FAA PART 117 — DATOS REALES ASRS")
    print("=" * 70)
    total = len(df)
    print(f"\n📋 TOTAL DE REPORTES ANALIZADOS: {total}")
    print(f"\n🔴 INDICADORES DE RIESGO:")
    print(f"   • Fatiga explícita:            "
          f"{df['fatigue_either'].sum():>4}  ({df['fatigue_either'].mean()*100:.1f}%)")
    print(f"   • Operación en WOCL:           "
          f"{df['is_WOCL'].sum():>4}  ({df['is_WOCL'].mean()*100:.1f}%)")
    print(f"   • Fase crítica de vuelo:       "
          f"{df['critical_phase'].sum():>4}  ({df['critical_phase'].mean()*100:.1f}%)")
    print(f"   • Human Factors como causa:    "
          f"{df['primary_human'].sum():>4}  ({df['primary_human'].mean()*100:.1f}%)")

    print(f"\n📊 DISTRIBUCIÓN DE RIESGO:")
    for risk in ['High', 'Medium', 'Low']:
        n = (df['Risk_Level'] == risk).sum()
        print(f"   • {risk:6}: {n:>4} reportes  ({n/total*100:.1f}%)")

    danger = df[(df['fatigue_either']==1) &
                (df['is_WOCL']==1) &
                (df['critical_phase']==1)]
    print(f"\n⚠️  COMBINACIÓN CRÍTICA (fatiga + WOCL + fase crítica):")
    print(f"   → {len(danger)} reportes cumplen los 3 criterios")
    if len(danger) > 0:
        print(f"   → Fases: {danger['Flight_Phase_Clean'].value_counts().to_dict()}")

    print(f"\n📚 REFERENCIAS REGULATORIAS:")
    print(f"   • FAA 14 CFR Part 117  — Flight & Duty Limitations")
    print(f"   • ICAO Annex 6         — Operation of Aircraft")
    print(f"   • EASA ORO.FTL         — Flight Time Limitations")
    print(f"   • NASA ASRS            — Aviation Safety Reporting System")
    print("=" * 70)


# ─────────────────────────────────────────────
# 7. PREDICCIÓN EN TIEMPO REAL
# ─────────────────────────────────────────────

def predict_scenario(model, features):
    print("\n" + "=" * 70)
    print("PREDICCIÓN EN TIEMPO REAL — ESCENARIO REAL")
    print("=" * 70)

    scenario = pd.DataFrame([{
        'fatigue_p1': 1, 'fatigue_p2': 1,
        'is_WOCL': 1, 'is_Night': 1,
        'critical_phase': 1, 'total_hf_factors': 4,
        'narrative_fatigue_mentions': 3,
        'synopsis_fatigue_mentions': 2,
        'anomaly_severity': 1, 'primary_human': 1,
        'hf_Workload': 1, 'hf_Distraction': 0,
        'hf_Situational_Awareness': 1,
        'hf_Communication_Breakdown': 1,
        'hf_Physiological': 1,
    }])

    pred  = model.predict(scenario)[0]
    proba = model.predict_proba(scenario)[0]

    print("\nESCENARIO:")
    print("  • Hora local: 02:30 (zona WOCL — FAA Part 117)")
    print("  • Fase: Final Approach (fase crítica)")
    print("  • Fatiga reportada por ambos tripulantes")
    print("  • Factores: Workload, Sit. Awareness, Comm. Breakdown")

    print(f"\n🎯 PREDICCIÓN: Riesgo → {pred}")
    for label, p in sorted(zip(model.classes_, proba), key=lambda x: -x[1]):
        bar = '█' * int(p * 30)
        print(f"   {label:6}: {bar:<30} {p:.1%}")

    if pred == 'High':
        print("\n🔴 ALERTA CRÍTICA — Considerar relevo de tripulación")
        print("   → Revisar descanso (§117.25) y notificar Safety Officer")
    elif pred == 'Medium':
        print("\n🟡 MONITOREO ACTIVO — Verificar horas acumuladas (§117.23)")
    else:
        print("\n🟢 RIESGO ACEPTABLE — Monitoreo estándar")
    print("=" * 70)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Cambia esta ruta por la ubicación de tu archivo CSV ──
    CSV_PATH = 'data.csv'

    df      = load_asrs_data(CSV_PATH)
    df      = engineer_features(df)
    model, X_test, y_test, y_pred, feat_imp, features = train_model(df)

    print("\n🔤 Analizando narrativas con NLP...")
    hr_words = analyze_narratives(df)
    print("   Top keywords:", [w for w, _ in hr_words[:10]])

    print("\n📈 Generando dashboard...")
    create_visualizations(df, feat_imp, X_test, y_test, y_pred, hr_words)

    compliance_report(df)
    predict_scenario(model, features)

    print("\n✅ ANÁLISIS COMPLETADO")
    print("📁 Generado: fatigue_asrs_dashboard.png")

if __name__ == "__main__":
    main()