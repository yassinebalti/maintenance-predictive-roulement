"""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 2 — DASHBOARD STREAMLIT TEMPS RÉEL                        ║
║  Maintenance Prédictive — Sans base de données                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Lit directement les fichiers JSON produits par le consumer      ║
║  Rafraîchissement automatique toutes les 3 secondes              ║
║                                                                  ║
║  Pages :                                                         ║
║    🏠 Vue Flotte      → Tous les moteurs, carte risques           ║
║    🔍 Moteur Détail   → Drill-down par moteur                    ║
║    📊 Anomalies       → Timeline + scores live                   ║
║    🔮 RUL & Tendances → Prédictions + Weibull + IC               ║
║    ⚡ CUSUM           → Détection ruptures                        ║
║    📈 Métriques IA    → Performance modèle en live               ║
║                                                                  ║
║  Usage :                                                         ║
║    streamlit run streamlit_app.py                                ║
║    streamlit run streamlit_app.py --server.port 8501             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import json
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import glob

# ── Configuration page ─────────────────────────────────────────
st.set_page_config(
    page_title="Maintenance Prédictive — IA/ML",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chemins ────────────────────────────────────────────────────
RESULTS_DIR   = "data/stream_results"
LATEST_JSON   = os.path.join(RESULTS_DIR, "latest.json")
CSV_ANOMALIES = "data/03_anomalies.csv"
CSV_RUL       = "data/rul_summary.csv"
CSV_SHAP      = "data/shap_importance.csv"
CSV_CUSUM     = "data/cusum_changepoints.csv"
CSV_WALKFWD   = "data/validation_walkforward.csv"

# ── Couleurs par niveau de risque ──────────────────────────────
RISK_COLORS = {
    "CRITIQUE": "#EF4444",
    "ÉLEVÉ"   : "#F97316",
    "MODÉRÉ"  : "#EAB308",
    "FAIBLE"  : "#22C55E",
    "inconnu" : "#94A3B8",
}

RISK_BG = {
    "CRITIQUE": "rgba(239,68,68,0.15)",
    "ÉLEVÉ"   : "rgba(249,115,22,0.15)",
    "MODÉRÉ"  : "rgba(234,179,8,0.15)",
    "FAIBLE"  : "rgba(34,197,94,0.15)",
}


# ══════════════════════════════════════════════════════════════════
#  CSS PERSONNALISÉ
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Fond principal */
    .stApp { background-color: #0f172a; color: #e2e8f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }

    /* Métriques */
    div[data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 { color: #38bdf8 !important; }

    /* Cards moteur */
    .motor-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid;
        margin-bottom: 8px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    /* Live indicator */
    .live-dot {
        display: inline-block;
        width: 10px; height: 10px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.6); }
        70%  { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }

    /* Divider */
    hr { border-color: #334155; }

    /* Plotly charts background */
    .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CHARGEMENT DONNÉES
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3)  # Rafraîchit toutes les 3 secondes
def load_stream_data():
    """Charge les données live depuis le JSON du consumer."""
    if not os.path.exists(LATEST_JSON):
        return None
    try:
        with open(LATEST_JSON) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=10)
def load_csv(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
        except Exception:
            return None
    return None


def get_demo_data():
    """Génère des données de démo si le consumer n'est pas lancé."""
    np.random.seed(int(time.time()) % 100)
    motors = {}
    profiles = {
        21: ("ÉLEVÉ",    0.57, 62.0, 1.4),
         5: ("ÉLEVÉ",    0.62, 58.0, 1.3),
        18: ("MODÉRÉ",   0.40, 52.0, 1.1),
         4: ("MODÉRÉ",   0.38, 50.0, 1.0),
        15: ("MODÉRÉ",   0.30, 47.0, 0.9),
    }
    for mid in range(1, 22):
        risk, di, temp, vib = profiles.get(mid, ("FAIBLE", 0.05 + np.random.uniform(0, 0.2),
                                                  35 + np.random.uniform(0, 15),
                                                  0.8 + np.random.uniform(0, 0.3)))
        motors[str(mid)] = {
            "motor_id"         : mid,
            "risk_level"       : risk,
            "degradation_index": di + np.random.uniform(-0.02, 0.02),
            "rul_days"         : ">90" if di < 0.5 else str(round((0.75 - di) * 200, 1)),
            "rul_num"          : 90 if di < 0.5 else (0.75 - di) * 200,
            "rul_low"          : 70,
            "rul_high"         : 90,
            "combined_score"   : di * 0.6 + np.random.uniform(0, 0.1),
            "is_anomaly"       : di > 0.45,
            "temperature"      : temp + np.random.uniform(-2, 2),
            "vibration"        : vib + np.random.uniform(-0.05, 0.05),
            "courant"          : 100 + np.random.uniform(-10, 10),
            "health_score"     : max(0, 100 - di * 100 + np.random.uniform(-5, 5)),
            "cusum_alarm"      : di > 0.55,
            "cusum_severity"   : "ÉLEVÉ" if di > 0.55 else "STABLE",
            "score_if"         : di * 0.5,
            "score_lof"        : di * 0.4,
            "score_rules"      : di * 0.7,
            "trend_slope"      : 0.001 * di,
            "alert_status_source": "ALERT" if di > 0.6 else "NORMAL",
        }

    return {
        "timestamp"   : datetime.utcnow().isoformat(),
        "n_processed" : np.random.randint(1000, 5000),
        "motors"      : motors,
        "summary"     : {
            "n_motors"   : 21,
            "n_critique" : 0,
            "n_eleve"    : 2,
            "n_modere"   : 3,
            "n_faible"   : 16,
            "avg_di"     : 0.18,
            "cusum_alarms": [5, 21],
        },
    }


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
def render_sidebar(data):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0;">
            <div style="font-size:36px">⚙️</div>
            <h2 style="color:#38bdf8; margin:8px 0">Maintenance</h2>
            <h3 style="color:#94a3b8; font-weight:400; margin:0">Prédictive IA/ML</h3>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Status live
        is_live = data is not None and "motors" in data
        if is_live:
            ts   = data.get("timestamp", "")[:19].replace("T", " ")
            n_pr = data.get("n_processed", 0)
            st.markdown(f"""
            <div style="background:#0d3320; border:1px solid #22c55e; border-radius:8px; padding:12px; margin-bottom:12px;">
                <span class="live-dot"></span><b style="color:#22c55e">LIVE — Stream actif</b>
                <div style="color:#94a3b8; font-size:12px; margin-top:4px">
                    {n_pr:,} messages traités<br>MAJ : {ts}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#2d1b0e; border:1px solid #f97316; border-radius:8px; padding:12px; margin-bottom:12px;">
                <b style="color:#f97316">⚠ Mode Démo</b>
                <div style="color:#94a3b8; font-size:12px; margin-top:4px">
                    Lancez kafka_consumer.py<br>pour les données live
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Navigation
        st.markdown("### Navigation")
        page = st.radio("", [
            "🏠 Vue Flotte",
            "🔍 Moteur Détail",
            "📊 Anomalies Live",
            "🔮 RUL & Tendances",
            "⚡ CUSUM",
            "📈 Métriques IA",
        ], label_visibility="collapsed")

        st.divider()

        # Filtres
        st.markdown("### Filtres")
        risk_filter = st.multiselect(
            "Niveau de risque",
            ["CRITIQUE", "ÉLEVÉ", "MODÉRÉ", "FAIBLE"],
            default=["CRITIQUE", "ÉLEVÉ", "MODÉRÉ", "FAIBLE"],
        )

        auto_refresh = st.toggle("Auto-refresh (3s)", value=True)
        if auto_refresh:
            time.sleep(3)
            st.rerun()

        st.divider()
        st.caption("Pipeline IA/ML V3 | Kafka + Streamlit")

    return page, risk_filter


# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — VUE FLOTTE
# ══════════════════════════════════════════════════════════════════
def page_flotte(data, risk_filter):
    st.markdown("## 🏠 Vue Flotte — 21 Moteurs")

    motors_data = data.get("motors", {})
    summary     = data.get("summary", {})

    # ── KPIs globaux ─────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Moteurs Surveillés", summary.get("n_motors", 21))
    with c2:
        n_crit = summary.get("n_critique", 0)
        st.metric("CRITIQUE", n_crit, delta=None,
                  delta_color="inverse")
    with c3:
        st.metric("ÉLEVÉ", summary.get("n_eleve", 0))
    with c4:
        st.metric("MODÉRÉ", summary.get("n_modere", 0))
    with c5:
        avg_di = summary.get("avg_di", 0)
        st.metric("DI Moyen Flotte", f"{avg_di:.3f}")

    st.divider()

    # ── Heatmap DI par moteur ─────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Indice de Dégradation par Moteur")

        ids  = sorted([int(k) for k in motors_data.keys()])
        dis  = [motors_data[str(mid)].get("degradation_index", 0) for mid in ids]
        risks= [motors_data[str(mid)].get("risk_level", "FAIBLE") for mid in ids]
        colors_bar = [RISK_COLORS.get(r, "#94A3B8") for r in risks]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"M{mid}" for mid in ids],
            y=dis,
            marker_color=colors_bar,
            marker_line_color="rgba(255,255,255,0.1)",
            marker_line_width=1,
            text=[f"{d:.3f}" for d in dis],
            textposition="outside",
            hovertemplate="<b>Moteur %{x}</b><br>DI = %{y:.4f}<extra></extra>",
        ))
        fig.add_hline(y=0.50, line_dash="dash", line_color="#F97316",
                      annotation_text="Seuil Prudence (0.50)",
                      annotation_position="right")
        fig.add_hline(y=0.75, line_dash="dash", line_color="#EF4444",
                      annotation_text="Seuil Critique (0.75)",
                      annotation_position="right")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", range=[0, 1]),
            height=350,
            margin=dict(t=30, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Distribution Risques")
        risk_counts = {}
        for v in motors_data.values():
            r = v.get("risk_level", "FAIBLE")
            if r in risk_filter:
                risk_counts[r] = risk_counts.get(r, 0) + 1

        if risk_counts:
            fig_pie = go.Figure(go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                marker_colors=[RISK_COLORS[r] for r in risk_counts],
                hole=0.5,
                textinfo="label+percent",
                textfont_size=13,
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # CUSUM alarms
        cusum_alarms = summary.get("cusum_alarms", [])
        if cusum_alarms:
            st.markdown(f"""
            <div style="background:#1a1020; border:1px solid #a855f7; border-radius:8px; padding:10px;">
                <b style="color:#a855f7">⚡ Alarmes CUSUM</b><br>
                <span style="color:#e2e8f0">Moteurs : {', '.join([f'M{m}' for m in cusum_alarms])}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Tableau moteurs ───────────────────────────
    st.markdown("### Tableau de Bord Moteurs")

    rows = []
    for mid in sorted([int(k) for k in motors_data.keys()]):
        v = motors_data[str(mid)]
        if v.get("risk_level", "FAIBLE") in risk_filter:
            rows.append({
                "Moteur"     : f"M{mid}",
                "Risque"     : v.get("risk_level", "FAIBLE"),
                "DI"         : round(v.get("degradation_index", 0), 4),
                "RUL"        : str(v.get("rul_days", ">90")) + "j",
                "Score"      : round(v.get("combined_score", 0), 4),
                "Temp (°C)"  : round(v.get("temperature", 0), 1),
                "Vibration"  : round(v.get("vibration", 0), 4),
                "Health %"   : round(v.get("health_score", 0), 1),
                "CUSUM"      : "⚠" if v.get("cusum_alarm") else "✓",
                "Anomalie"   : "🔴" if v.get("is_anomaly") else "🟢",
            })

    if rows:
        df_display = pd.DataFrame(rows)

        def color_risque(val):
            colors = {"CRITIQUE": "background-color:#3d0f0f; color:#ef4444",
                      "ÉLEVÉ":    "background-color:#3d1f0f; color:#f97316",
                      "MODÉRÉ":   "background-color:#2d2400; color:#eab308",
                      "FAIBLE":   "background-color:#0d2d1a; color:#22c55e"}
            return colors.get(val, "")

        styled = df_display.style.applymap(
            color_risque, subset=["Risque"]
        ).format({"DI": "{:.4f}", "Score": "{:.4f}"})

        st.dataframe(styled, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — DÉTAIL MOTEUR
# ══════════════════════════════════════════════════════════════════
def page_moteur_detail(data):
    st.markdown("## 🔍 Détail Moteur")

    motors_data = data.get("motors", {})
    motor_ids   = sorted([int(k) for k in motors_data.keys()])

    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        selected_mid = st.selectbox("Sélectionner un moteur",
                                    motor_ids,
                                    format_func=lambda x: f"Moteur {x}")

    v = motors_data.get(str(selected_mid), {})
    risk = v.get("risk_level", "FAIBLE")

    with col_info:
        rc = RISK_COLORS.get(risk, "#94A3B8")
        st.markdown(f"""
        <div class="motor-card" style="border-color:{rc}; background:{RISK_BG.get(risk,'#1e293b')}">
            <h3 style="color:{rc}; margin:0">Moteur {selected_mid} — {risk}</h3>
            <span style="color:#94a3b8">DI = {v.get('degradation_index',0):.4f} |
            RUL = {v.get('rul_days','>90')}j |
            Score = {v.get('combined_score',0):.4f}</span>
        </div>
        """, unsafe_allow_html=True)

    # KPIs moteur
    st.markdown("### Indicateurs Clés")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, "DI",         f"{v.get('degradation_index',0):.4f}", None),
        (c2, "Temp (°C)",  f"{v.get('temperature',0):.1f}",       None),
        (c3, "Vibration",  f"{v.get('vibration',0):.4f}",          None),
        (c4, "Health %",   f"{v.get('health_score',0):.1f}",       None),
        (c5, "IF Score",   f"{v.get('score_if',0):.4f}",           None),
        (c6, "LOF Score",  f"{v.get('score_lof',0):.4f}",          None),
    ]
    for col, label, val, delta in metrics:
        with col:
            st.metric(label, val, delta)

    # Gauge DI
    st.markdown("### Gauge — Indice de Dégradation")
    di_val = v.get("degradation_index", 0)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=di_val,
        delta={"reference": 0.30, "valueformat": ".3f"},
        title={"text": f"DI — Moteur {selected_mid}", "font": {"size": 18, "color": "#e2e8f0"}},
        number={"font": {"color": RISK_COLORS.get(risk, "#94A3B8"), "size": 36}},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#94a3b8"},
            "bar":  {"color": RISK_COLORS.get(risk, "#94A3B8"), "thickness": 0.25},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0.00, 0.30], "color": "rgba(34,197,94,0.2)"},
                {"range": [0.30, 0.50], "color": "rgba(234,179,8,0.2)"},
                {"range": [0.50, 0.75], "color": "rgba(249,115,22,0.2)"},
                {"range": [0.75, 1.00], "color": "rgba(239,68,68,0.2)"},
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 3},
                "thickness": 0.75,
                "value": 0.75,
            },
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=280,
        margin=dict(t=40, b=20, l=40, r=40),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Scores composants
    st.markdown("### Décomposition du Score d'Anomalie")
    scores = {
        "Isolation Forest": v.get("score_if", 0),
        "LOF"             : v.get("score_lof", 0),
        "Règles Métier"   : v.get("score_rules", 0),
        "Score Combiné"   : v.get("combined_score", 0),
    }
    fig_scores = go.Figure(go.Bar(
        x=list(scores.values()),
        y=list(scores.keys()),
        orientation="h",
        marker_color=["#38bdf8", "#818cf8", "#34d399", "#f97316"],
        text=[f"{s:.4f}" for s in scores.values()],
        textposition="outside",
    ))
    fig_scores.add_vline(x=0.25, line_dash="dash", line_color="#ef4444",
                         annotation_text="Seuil 0.25")
    fig_scores.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font_color="#e2e8f0",
        xaxis=dict(range=[0, 1], gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b"),
        height=250,
        margin=dict(t=20, b=20, l=20, r=60),
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    # RUL Barre avec IC
    rul_num = v.get("rul_num", 90)
    rul_low = v.get("rul_low", 70)
    rul_high= v.get("rul_high", 90)

    st.markdown("### Prédiction RUL — Intervalle de Confiance 80%")
    fig_rul = go.Figure()
    fig_rul.add_trace(go.Bar(
        x=["RUL Ensemble", "RUL Min (IC)", "RUL Max (IC)"],
        y=[rul_num, rul_low, rul_high],
        marker_color=[RISK_COLORS.get(risk), "#94a3b8", "#94a3b8"],
        text=[f"{rul_num:.1f}j", f"{rul_low:.1f}j", f"{rul_high:.1f}j"],
        textposition="outside",
    ))
    fig_rul.add_hline(y=30, line_dash="dash", line_color="#EF4444",
                      annotation_text="Urgent < 30j")
    fig_rul.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font_color="#e2e8f0",
        yaxis=dict(range=[0, 100], title="Jours", gridcolor="#1e293b"),
        height=280,
        margin=dict(t=20, b=20, l=20, r=40),
    )
    st.plotly_chart(fig_rul, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — ANOMALIES LIVE
# ══════════════════════════════════════════════════════════════════
def page_anomalies(data):
    st.markdown("## 📊 Anomalies Live")

    motors_data = data.get("motors", {})

    # Scatter plot — tous moteurs, DI vs Score
    st.markdown("### DI vs Score d'Anomalie — Tous les Moteurs")

    ids     = sorted([int(k) for k in motors_data.keys()])
    di_list = [motors_data[str(m)].get("degradation_index", 0) for m in ids]
    sc_list = [motors_data[str(m)].get("combined_score", 0) for m in ids]
    rk_list = [motors_data[str(m)].get("risk_level", "FAIBLE") for m in ids]
    al_list = [motors_data[str(m)].get("is_anomaly", False) for m in ids]

    fig_sc = go.Figure()
    for risk_level in ["FAIBLE", "MODÉRÉ", "ÉLEVÉ", "CRITIQUE"]:
        mask = [r == risk_level for r in rk_list]
        x_r  = [di_list[i] for i, m in enumerate(mask) if m]
        y_r  = [sc_list[i] for i, m in enumerate(mask) if m]
        lbl  = [f"M{ids[i]}" for i, m in enumerate(mask) if m]
        if x_r:
            fig_sc.add_trace(go.Scatter(
                x=x_r, y=y_r, mode="markers+text",
                marker=dict(size=16, color=RISK_COLORS[risk_level],
                            symbol="circle", line=dict(width=1, color="white")),
                text=lbl, textposition="top center",
                name=risk_level,
            ))

    fig_sc.add_hline(y=0.25, line_dash="dash", line_color="#f97316",
                     annotation_text="Seuil anomalie (0.25)")
    fig_sc.add_vline(x=0.50, line_dash="dash", line_color="#f97316",
                     annotation_text="Seuil DI prudence")
    fig_sc.add_vline(x=0.75, line_dash="dash", line_color="#ef4444",
                     annotation_text="Seuil DI critique")
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font_color="#e2e8f0",
        xaxis=dict(title="Indice de Dégradation (DI)", gridcolor="#1e293b", range=[0, 1]),
        yaxis=dict(title="Score Anomalie Combiné", gridcolor="#1e293b", range=[0, 1]),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor="#334155"),
        height=450,
        margin=dict(t=30, b=30, l=40, r=40),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Heatmap Temperature + Vibration
    st.markdown("### Heatmap — Température & Vibration par Moteur")
    c1, c2 = st.columns(2)

    with c1:
        temps = [motors_data[str(m)].get("temperature", 0) for m in ids]
        fig_t = go.Figure(go.Bar(
            x=[f"M{m}" for m in ids], y=temps,
            marker_color=[RISK_COLORS.get(rk_list[i]) for i in range(len(ids))],
            text=[f"{t:.1f}°" for t in temps], textposition="outside",
        ))
        fig_t.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                        annotation_text="Seuil alerte 70°C")
        fig_t.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font_color="#e2e8f0",
            title=dict(text="Température (°C)", font=dict(color="#38bdf8")),
            yaxis=dict(gridcolor="#1e293b"),
            height=320, margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with c2:
        vibs = [motors_data[str(m)].get("vibration", 0) for m in ids]
        fig_v = go.Figure(go.Bar(
            x=[f"M{m}" for m in ids], y=vibs,
            marker_color=[RISK_COLORS.get(rk_list[i]) for i in range(len(ids))],
            text=[f"{v:.3f}" for v in vibs], textposition="outside",
        ))
        fig_v.add_hline(y=1.5, line_dash="dash", line_color="#ef4444",
                        annotation_text="Seuil alerte")
        fig_v.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font_color="#e2e8f0",
            title=dict(text="Vibration (mm/s)", font=dict(color="#38bdf8")),
            yaxis=dict(gridcolor="#1e293b"),
            height=320, margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_v, use_container_width=True)

    # Tableau anomalies actives
    st.markdown("### 🔴 Anomalies Actuellement Actives")
    anomaly_motors = [
        {"Moteur": f"M{m}",
         "Score": motors_data[str(m)].get("combined_score", 0),
         "DI": motors_data[str(m)].get("degradation_index", 0),
         "Temp": motors_data[str(m)].get("temperature", 0),
         "Vibration": motors_data[str(m)].get("vibration", 0),
         "Risque": motors_data[str(m)].get("risk_level", "FAIBLE"),
         }
        for m in ids
        if motors_data[str(m)].get("is_anomaly", False)
    ]
    if anomaly_motors:
        df_anom = pd.DataFrame(anomaly_motors).sort_values("Score", ascending=False)
        st.dataframe(df_anom, use_container_width=True)
    else:
        st.success("✅ Aucune anomalie active détectée")


# ══════════════════════════════════════════════════════════════════
#  PAGE 4 — RUL & TENDANCES
# ══════════════════════════════════════════════════════════════════
def page_rul(data):
    st.markdown("## 🔮 RUL & Tendances")

    motors_data = data.get("motors", {})
    ids = sorted([int(k) for k in motors_data.keys()])

    # Timeline RUL avec IC
    st.markdown("### RUL Estimé + Intervalle de Confiance 80%")

    ruls  = [motors_data[str(m)].get("rul_num", 90)  for m in ids]
    lows  = [motors_data[str(m)].get("rul_low", 70)  for m in ids]
    highs = [motors_data[str(m)].get("rul_high", 90) for m in ids]
    risks = [motors_data[str(m)].get("risk_level", "FAIBLE") for m in ids]

    # Trier par RUL croissant
    sorted_idx = sorted(range(len(ids)), key=lambda i: ruls[i])
    ids_s  = [ids[i] for i in sorted_idx]
    ruls_s = [ruls[i] for i in sorted_idx]
    lows_s = [lows[i] for i in sorted_idx]
    highs_s= [highs[i] for i in sorted_idx]
    risks_s= [risks[i] for i in sorted_idx]

    fig_rul = go.Figure()
    # IC zone
    fig_rul.add_trace(go.Scatter(
        y=[f"M{m}" for m in ids_s],
        x=highs_s + lows_s[::-1],
        mode="lines", fill="toself",
        fillcolor="rgba(56,189,248,0.08)",
        line=dict(width=0), name="IC 80%",
    ))
    # Points RUL
    fig_rul.add_trace(go.Scatter(
        y=[f"M{m}" for m in ids_s],
        x=ruls_s,
        mode="markers+text",
        marker=dict(size=12, color=[RISK_COLORS.get(r) for r in risks_s],
                    symbol="diamond"),
        text=[f"{r:.0f}j" for r in ruls_s],
        textposition="middle right",
        name="RUL Ensemble",
    ))
    # Whiskers IC
    for i, m in enumerate(ids_s):
        fig_rul.add_shape(
            type="line",
            y0=f"M{m}", y1=f"M{m}",
            x0=lows_s[i], x1=highs_s[i],
            line=dict(color="rgba(56,189,248,0.4)", width=2),
        )

    fig_rul.add_vline(x=30, line_dash="dash", line_color="#EF4444",
                      annotation_text="Urgent")
    fig_rul.add_vline(x=60, line_dash="dash", line_color="#F97316",
                      annotation_text="Planifier")
    fig_rul.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font_color="#e2e8f0",
        xaxis=dict(title="Jours restants", gridcolor="#1e293b", range=[0, 100]),
        yaxis=dict(gridcolor="#1e293b"),
        height=600, margin=dict(t=30, b=30, l=40, r=80),
        legend=dict(bgcolor="rgba(30,41,59,0.8)"),
    )
    st.plotly_chart(fig_rul, use_container_width=True)

    # Top 5 prioritaires
    st.markdown("### ⚠ Top 5 Moteurs Prioritaires")
    rows = []
    for mid in ids:
        v = motors_data[str(mid)]
        rows.append({
            "Moteur"   : f"M{mid}",
            "DI"       : v.get("degradation_index", 0),
            "RUL"      : str(v.get("rul_days", ">90")) + "j",
            "Risque"   : v.get("risk_level", "FAIBLE"),
            "Tendance" : "↗" if v.get("trend_slope", 0) > 0 else "→",
            "Action"   : {
                "CRITIQUE": "🔴 Arrêt immédiat",
                "ÉLEVÉ":    "🟠 Intervention < 7j",
                "MODÉRÉ":   "🟡 Planifier < 30j",
                "FAIBLE":   "🟢 Surveillance",
            }.get(v.get("risk_level", "FAIBLE"), "—"),
        })

    df_top = pd.DataFrame(rows).sort_values("DI", ascending=False).head(5)
    st.dataframe(df_top, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 5 — CUSUM
# ══════════════════════════════════════════════════════════════════
def page_cusum(data):
    st.markdown("## ⚡ CUSUM — Détection de Rupture de Tendance")

    st.markdown("""
    <div style="background:#1e293b; border-left:4px solid #a855f7; padding:16px; border-radius:8px; margin-bottom:20px;">
        <b style="color:#a855f7">🧮 Algorithme CUSUM</b><br>
        <code style="color:#e2e8f0">S⁺[t] = max(0, S⁺[t-1] + (DI[t] - μ₀) - k)</code><br>
        <span style="color:#94a3b8">Alarme si S⁺[t] > h = 4σ — Détecte une dérive dès 1-5 points</span>
    </div>
    """, unsafe_allow_html=True)

    motors_data = data.get("motors", {})
    ids = sorted([int(k) for k in motors_data.keys()])

    # Statuts CUSUM
    cusum_statuses = []
    for mid in ids:
        v = motors_data[str(mid)]
        cusum_statuses.append({
            "Moteur"   : f"M{mid}",
            "Alarme"   : "⚠ OUI" if v.get("cusum_alarm") else "✓ Non",
            "S⁺ actuel": round(v.get("cusum_s_pos", 0), 4),
            "Sévérité" : v.get("cusum_severity", "STABLE"),
            "DI"       : round(v.get("degradation_index", 0), 4),
            "Risque"   : v.get("risk_level", "FAIBLE"),
        })

    df_cusum = pd.DataFrame(cusum_statuses)

    # Tri : alarmes en premier
    df_cusum["_sort"] = df_cusum["Alarme"].apply(lambda x: 0 if "OUI" in x else 1)
    df_cusum = df_cusum.sort_values(["_sort", "S⁺ actuel"], ascending=[True, False])
    df_cusum = df_cusum.drop("_sort", axis=1)

    # Mise en forme
    def color_cusum(val):
        if "OUI" in str(val):
            return "background-color:#3d0f0f; color:#ef4444; font-weight:bold"
        return ""
    def color_sev(val):
        m = {"ÉLEVÉ": "color:#ef4444", "MODÉRÉ": "color:#eab308", "STABLE": "color:#22c55e"}
        return m.get(val, "")

    styled = df_cusum.style.applymap(color_cusum, subset=["Alarme"]) \
                           .applymap(color_sev, subset=["Sévérité"])
    st.dataframe(styled, use_container_width=True, height=450)

    # Graphique S+ actuel
    st.markdown("### Valeur CUSUM S⁺ Actuelle par Moteur")
    s_vals   = [motors_data[str(m)].get("cusum_s_pos", 0) for m in ids]
    cusum_al = [motors_data[str(m)].get("cusum_alarm", False) for m in ids]
    bar_cols = ["#EF4444" if a else "#38bdf8" for a in cusum_al]

    fig_cusum = go.Figure(go.Bar(
        x=[f"M{m}" for m in ids],
        y=s_vals,
        marker_color=bar_cols,
        text=[f"{s:.3f}" for s in s_vals],
        textposition="outside",
    ))
    fig_cusum.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font_color="#e2e8f0",
        yaxis=dict(title="S⁺ CUSUM", gridcolor="#1e293b"),
        height=320, margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_cusum, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 6 — MÉTRIQUES IA
# ══════════════════════════════════════════════════════════════════
def page_metriques(data):
    st.markdown("## 📈 Métriques IA — Performance du Pipeline")

    n_proc = data.get("n_processed", 0)
    ts     = data.get("timestamp", "")[:19].replace("T", " ")

    # Stats live
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Messages Traités", f"{n_proc:,}")
    with c2:
        st.metric("Dernier Snapshot", ts[-8:] if ts else "—")
    with c3:
        motors_data = data.get("motors", {})
        n_anom = sum(1 for v in motors_data.values() if v.get("is_anomaly"))
        st.metric("Anomalies Actives", n_anom)
    with c4:
        avg_score = np.mean([v.get("combined_score", 0)
                             for v in motors_data.values()]) if motors_data else 0
        st.metric("Score Moyen Flotte", f"{avg_score:.4f}")

    st.divider()

    # Walk-forward validation
    df_wf = load_csv(CSV_WALKFWD)
    if df_wf is not None:
        st.markdown("### Walk-Forward Cross-Validation (AXE 4)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("AUC Moyenne", f"{df_wf['auc'].mean():.4f}")
        with c2:
            st.metric("AUC Std", f"±{df_wf['auc'].std():.4f}")
        with c3:
            st.metric("Folds", len(df_wf))

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            x=df_wf["fold"], y=df_wf["auc"],
            mode="lines+markers+text",
            marker=dict(size=12, color="#38bdf8"),
            text=[f"{a:.4f}" for a in df_wf["auc"]],
            textposition="top center",
            name="AUC par fold",
            line=dict(width=2),
        ))
        fig_wf.add_trace(go.Scatter(
            x=df_wf["fold"], y=df_wf["f1"],
            mode="lines+markers",
            marker=dict(size=10, color="#34d399"),
            name="F1 par fold",
            line=dict(width=2, dash="dash"),
        ))
        fig_wf.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8",
                         annotation_text="Baseline aléatoire")
        fig_wf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font_color="#e2e8f0",
            xaxis=dict(title="Fold", gridcolor="#1e293b"),
            yaxis=dict(title="Valeur", gridcolor="#1e293b", range=[0, 1]),
            legend=dict(bgcolor="rgba(30,41,59,0.8)"),
            height=300, margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # SHAP Feature Importance
    df_shap = load_csv(CSV_SHAP)
    if df_shap is not None:
        st.markdown("### SHAP — Feature Importance Globale (AXE 6)")
        top_shap = df_shap.head(12)
        v3_feats = {"vib_rms","vib_skewness","peak2peak",
                    "spectral_entropy","shape_factor","impulse_factor"}
        colors_shap = [
            "#a855f7" if r in v3_feats else "#38bdf8"
            for r in top_shap["feature"]
        ]
        fig_shap = go.Figure(go.Bar(
            y=top_shap["feature"][::-1],
            x=top_shap["importance_pct"][::-1] if "importance_pct" in top_shap else
              top_shap["importance"][::-1],
            orientation="h",
            marker_color=colors_shap[::-1],
            text=[f"{v:.1f}%" for v in (top_shap["importance_pct"][::-1]
                  if "importance_pct" in top_shap else top_shap["importance"][::-1])],
            textposition="outside",
        ))
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e293b", title="Importance (%)"),
            annotations=[dict(
                x=0.98, y=0.02, xref="paper", yref="paper",
                text="🟣 = Nouvelles features AXE 3",
                showarrow=False, font=dict(color="#a855f7", size=11),
                xanchor="right",
            )],
            height=380, margin=dict(t=20, b=20, l=20, r=80),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # Tableau récap 7 axes
    st.markdown("### Récapitulatif — 7 Axes d'Amélioration IA")
    axes_data = [
        {"Axe": "AXE 1", "Titre": "AUC corrigée vs Alert_Status",   "Statut": "✅ Actif", "Valeur": "AUC=0.597 (vs 1.0 erroné)"},
        {"Axe": "AXE 2", "Titre": "RUL Ensemble Poly+Exp+Weibull",  "Statut": "✅ Actif", "Valeur": "IC 80% calculé"},
        {"Axe": "AXE 3", "Titre": "+6 features vibratoires",         "Statut": "✅ Actif", "Valeur": "vib_rms, skew, p2p, SE, SF, IF"},
        {"Axe": "AXE 4", "Titre": "Walk-Forward CV temporel",        "Statut": "✅ Actif", "Valeur": "AUC=0.52 ±0.01 (3 folds)"},
        {"Axe": "AXE 5", "Titre": "LOF novelty=True intégré",        "Statut": "✅ Actif", "Valeur": "IF 25% + LOF 20% + Règles 55%"},
        {"Axe": "AXE 6", "Titre": "SHAP explications moteur",        "Statut": "✅ Actif", "Valeur": "Z-score vs flotte normale"},
        {"Axe": "AXE 7", "Titre": "CUSUM rupture de tendance",       "Statut": "✅ Actif", "Valeur": "S+[t] > 4σ → alarme"},
    ]
    st.dataframe(pd.DataFrame(axes_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
    <div style="background:linear-gradient(90deg,#1e40af,#7c3aed); padding:20px 30px; border-radius:12px; margin-bottom:20px;">
        <h1 style="color:white; margin:0; font-size:28px">⚙️ Maintenance Prédictive — IA/ML V3</h1>
        <p style="color:rgba(255,255,255,0.8); margin:6px 0 0">
            Pipeline Kafka → Features V3 → IF+LOF → RUL Weibull → CUSUM → Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Charger données
    stream_data = load_stream_data()
    if stream_data is None:
        st.info("ℹ️ Consumer Kafka non détecté — Affichage en mode démo")
        stream_data = get_demo_data()

    # Sidebar + navigation
    page, risk_filter = render_sidebar(stream_data)

    # Routing
    if "Vue Flotte" in page:
        page_flotte(stream_data, risk_filter)
    elif "Moteur Détail" in page:
        page_moteur_detail(stream_data)
    elif "Anomalies Live" in page:
        page_anomalies(stream_data)
    elif "RUL & Tendances" in page:
        page_rul(stream_data)
    elif "CUSUM" in page:
        page_cusum(stream_data)
    elif "Métriques IA" in page:
        page_metriques(stream_data)


if __name__ == "__main__":
    main()
