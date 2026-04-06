"""
Streamlit Dashboard pour Maintenance Prédictive
Lit CSV, interactif, sans DB.
Installe : pip install streamlit pandas plotly
Lancer : streamlit run dashboard_streamlit.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time

# Chargement données (cache avec TTL pour refresh)
@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv('data/04_rul_results.csv', parse_dates=['timestamp'])
    df_rul = pd.read_csv('data/rul_summary.csv')
    return df, df_rul

df, df_rul = load_data()

st.title("Dashboard Maintenance Prédictive IA/ML")

# Sidebar pour sélection moteur
motor_id = st.sidebar.selectbox("Sélectionner Moteur", df['motor_id'].unique())

# Section 1: Résumé Risques
st.header("Résumé des Risques")
st.dataframe(df_rul.style.highlight_max(subset=['current_di'], color='red'))

# Section 2: Graphique RUL (corrigé avec colonnes réelles)
st.header("RUL par Moteur")
fig_rul = px.bar(
    df_rul,
    x='motor_id',
    y='rul_days',
    color='risk_level',
    error_y='rul_high',
    error_y_minus='rul_low',
    title="RUL par Moteur (avec Intervalle de Confiance)"
)
st.plotly_chart(fig_rul)

# Section 3: Détails Moteur Sélectionné
st.header(f"Détails Moteur {motor_id}")
df_motor = df[df['motor_id'] == motor_id]
fig_vib = px.line(df_motor, x='timestamp', y='vibration', title="Évolution Vibration")
st.plotly_chart(fig_vib)

# Mode Temps Réel (simulé, refresh auto)
if st.checkbox("Mode Temps Réel"):
    st.write("Mise à jour automatique en cours...")
    time.sleep(5)  # Simule attente pour nouvelles données
    st.experimental_rerun()

# Section Grafana (iframe pour futur monitoring)
st.header("Monitoring Grafana")
st.markdown('<iframe src="http://localhost:3000" width="100%" height="600"></iframe>', unsafe_allow_html=True)