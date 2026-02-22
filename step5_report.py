"""
========================================================
 ÉTAPE 5 — RAPPORT FINAL
 Entrée : data/04_rul_results.csv + data/rul_summary.csv
 Sortie : data/rapport_final.txt + figures/fig6_report.png

 Génère :
  - Tableau récapitulatif par moteur
  - Alertes prioritaires (tri par niveau de risque)
  - Synthèse des anomalies détectées
  - Recommandations de maintenance
========================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────
INPUT_CSV       = 'data/04_rul_results.csv'
RUL_SUMMARY_CSV = 'data/rul_summary.csv'
OUTPUT_TXT      = 'data/rapport_final.txt'
FIGURES_DIR     = 'figures'
# ──────────────────────────────────────────────────────


RISK_PRIORITY = {'CRITIQUE': 0, 'ÉLEVÉ': 1, 'MODÉRÉ': 2, 'FAIBLE': 3, 'inconnu': 4}
RISK_COLOR    = {
    'CRITIQUE': '#d32f2f', 'ÉLEVÉ': '#f57c00',
    'MODÉRÉ': '#fbc02d', 'FAIBLE': '#388e3c', 'inconnu': '#9e9e9e'
}

MAINTENANCE_RECO = {
    'CRITIQUE': 'Arrêt immédiat recommandé. Inspection complète + remplacement préventif.',
    'ÉLEVÉ'   : 'Planifier une intervention dans les 7 jours. Surveillance renforcée.',
    'MODÉRÉ'  : 'Planifier une inspection dans le mois. Vérifier les seuils d\'alerte.',
    'FAIBLE'  : 'État satisfaisant. Suivi de routine. Prochaine inspection standard.',
    'inconnu' : 'Données insuffisantes. Vérifier le capteur.',
}


def generate_text_report(df: pd.DataFrame, df_rul: pd.DataFrame) -> str:
    """Génère le rapport textuel complet."""
    now    = datetime.now().strftime('%d/%m/%Y %H:%M')
    sep    = "=" * 65
    lines  = []

    lines += [
        sep,
        "  RAPPORT DE MAINTENANCE PRÉDICTIVE — IA/ML",
        f"  Généré le : {now}",
        sep,
        "",
        "1. RÉSUMÉ EXÉCUTIF",
        "-" * 40,
    ]

    # Stats globales
    n_motors     = df['motor_id'].nunique()
    n_total      = len(df)
    n_anomalies  = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
    pct_anomalies = n_anomalies / n_total * 100 if n_total > 0 else 0
    period_start = df['timestamp'].min()
    period_end   = df['timestamp'].max()

    lines += [
        f"  Moteurs surveillés        : {n_motors}",
        f"  Période analysée          : {period_start} → {period_end}",
        f"  Mesures totales           : {n_total:,}",
        f"  Anomalies détectées       : {n_anomalies:,} ({pct_anomalies:.1f}%)",
        "",
    ]

    # Distribution risque
    if 'risk_level' in df_rul.columns:
        risk_dist = df_rul['risk_level'].value_counts()
        lines.append("  Distribution des risques :")
        for level in ['CRITIQUE', 'ÉLEVÉ', 'MODÉRÉ', 'FAIBLE', 'inconnu']:
            if level in risk_dist:
                n = risk_dist[level]
                bar = '█' * n
                lines.append(f"    {level:10s} : {n:2d} moteur(s)  {bar}")
    lines.append("")

    # Alertes prioritaires
    lines += [
        "2. ALERTES PRIORITAIRES",
        "-" * 40,
    ]
    df_sorted = df_rul.copy()
    df_sorted['priority'] = df_sorted['risk_level'].map(RISK_PRIORITY).fillna(4)
    df_sorted = df_sorted.sort_values(['priority', 'current_di'], ascending=[True, False])

    for _, r in df_sorted.iterrows():
        rul_txt = f"{r['rul_days']} jours" if str(r['rul_days']) != '>90' else '>90 jours'
        lines += [
            f"  Moteur {int(r['motor_id']):2d}  |  "
            f"Risque : {r['risk_level']:10s}  |  "
            f"DI : {r['current_di']:.3f}  |  "
            f"RUL estimé : {rul_txt}",
            f"    → {MAINTENANCE_RECO.get(r['risk_level'], '')}",
            "",
        ]

    # Statistiques par moteur
    lines += [
        "3. STATISTIQUES DESCRIPTIVES PAR MOTEUR",
        "-" * 40,
    ]
    motor_stats = df.groupby('motor_id').agg(
        temp_moy   = ('temperature', 'mean'),
        vib_moy    = ('vibration', 'mean'),
        vib_max    = ('vibration', 'max'),
        n_alertes  = ('Alert_Status', lambda x: (x == 'ALERT').sum()),
        n_anomalies_detectees = ('is_anomaly', 'sum') if 'is_anomaly' in df.columns else ('motor_id', 'count'),
        health_moy = ('health_score', 'mean') if 'health_score' in df.columns else ('motor_id', 'count'),
    ).round(2)

    header = (f"  {'Moteur':>8} | {'Temp.moy':>9} | {'Vib.moy':>8} | "
              f"{'Vib.max':>8} | {'Alertes':>8} | {'Health':>7}")
    lines.append(header)
    lines.append("  " + "-" * 60)

    for mid, row in motor_stats.iterrows():
        lines.append(
            f"  {mid:>8} | {row['temp_moy']:>8.2f}° | "
            f"{row['vib_moy']:>7.4f} | {row['vib_max']:>7.4f} | "
            f"{int(row['n_alertes']):>8} | {row['health_moy']:>6.1f}%"
        )
    lines.append("")

    # Modèles utilisés
    lines += [
        "4. MÉTHODOLOGIE IA/ML",
        "-" * 40,
        "  Approche        : Non supervisée (pas d'étiquettes à l'entraînement)",
        "  Modèles         : Ensemble Isolation Forest (60%) + LOF (40%)",
        "  Features        : 23 indicateurs temporels, fréquentiels et statistiques",
        "    • Domaine temporel  : énergie, RMS, vib_energy, crest_factor",
        "    • Stats rolling     : moyenne, écart-type, kurtosis (fenêtre=20)",
        "    • Domaine fréq.     : FFT amplitude max, fréquence dominante",
        "    • Envelope (Hilbert): détection défauts roulements",
        "    • Health score      : indice composite [0–100]",
        "  RUL             : Indice de dégradation + régression polynomiale",
        "  Validation      : Alert_Status réel utilisé uniquement en évaluation",
        "",
        "5. FICHIERS PRODUITS",
        "-" * 40,
        "  data/01_raw_motor.csv        — Données brutes extraites du SQL",
        "  data/02_features_motor.csv   — Données enrichies (23 features)",
        "  data/03_anomalies.csv        — Résultats détection d'anomalies",
        "  data/04_rul_results.csv      — Résultats complets avec RUL",
        "  data/rul_summary.csv         — Résumé RUL par moteur",
        "  figures/fig1_anomaly_overview.png   — Vue globale anomalies",
        "  figures/fig2_per_motor.png          — Détail par moteur",
        "  figures/fig3_validation.png         — Matrice confusion + PR curve",
        "  figures/fig4_rul_all_motors.png     — RUL tous moteurs",
        "  figures/fig5_risk_dashboard.png     — Dashboard risque",
        "  figures/fig6_report_summary.png     — Résumé graphique final",
        "",
        sep,
    ]

    return "\n".join(lines)


def plot_report_summary(df: pd.DataFrame, df_rul: pd.DataFrame, output_dir: str):
    """Graphique récapitulatif final (tableau de bord visuel)."""
    os.makedirs(output_dir, exist_ok=True)

    df_sorted = df_rul.copy()
    df_sorted['priority'] = df_sorted['risk_level'].map(RISK_PRIORITY).fillna(4)
    df_sorted = df_sorted.sort_values(['priority', 'current_di'], ascending=[True, False])

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#1a1a2e')

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── Panel 1 : Anomaly rate par moteur ─────────
    ax1 = fig.add_subplot(gs[0, 0])
    if 'is_anomaly' in df.columns:
        anom_rate = df.groupby('motor_id')['is_anomaly'].mean() * 100
        anom_rate = anom_rate.sort_values(ascending=False)
        colors = ['#d32f2f' if v > 20 else '#f57c00' if v > 10
                  else '#fbc02d' if v > 5 else '#388e3c'
                  for v in anom_rate.values]
        ax1.barh([str(m) for m in anom_rate.index], anom_rate.values, color=colors)
        ax1.set_title('Taux d\'anomalie (%)', color='white', fontweight='bold')
        ax1.set_xlabel('%', color='white')
        ax1.set_ylabel('Moteur ID', color='white')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#16213e')
        ax1.spines[:].set_color('#444')

    # ── Panel 2 : DI actuel par moteur ────────────
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = [RISK_COLOR.get(r, '#9e9e9e') for r in df_sorted['risk_level']]
    ax2.barh(df_sorted['motor_id'].astype(str), df_sorted['current_di'],
             color=colors2, edgecolor='#333')
    ax2.axvline(0.50, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.axvline(0.75, color='red',    linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_title('Indice de dégradation', color='white', fontweight='bold')
    ax2.set_xlabel('DI [0–1]', color='white')
    ax2.set_xlim(0, 1)
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#16213e')
    ax2.spines[:].set_color('#444')

    # ── Panel 3 : Pie risque ──────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    risk_counts = df_sorted['risk_level'].value_counts()
    risk_order  = [r for r in ['CRITIQUE', 'ÉLEVÉ', 'MODÉRÉ', 'FAIBLE', 'inconnu']
                   if r in risk_counts.index]
    pie_colors  = [RISK_COLOR[r] for r in risk_order]
    pie_vals    = [risk_counts[r] for r in risk_order]
    wedges, texts, autotexts = ax3.pie(
        pie_vals, labels=risk_order, colors=pie_colors,
        autopct='%1.0f%%', startangle=90,
        textprops={'color': 'white'})
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
        at.set_color('white')
    ax3.set_title('Distribution des risques', color='white', fontweight='bold')
    ax3.set_facecolor('#16213e')

    # ── Panel 4 : Health score timeline ───────────
    ax4 = fig.add_subplot(gs[1, :2])
    if 'health_score' in df.columns:
        for mid in sorted(df['motor_id'].unique())[:8]:
            dm = df[df['motor_id'] == mid].sort_values('timestamp')
            ax4.plot(dm['timestamp'], dm['health_score'],
                     linewidth=1, alpha=0.7, label=f'M{mid}')
        ax4.axhline(50, color='orange', linestyle='--', linewidth=1.5,
                    alpha=0.8, label='Seuil 50%')
        ax4.axhline(30, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.8, label='Critique 30%')
        ax4.set_title('Health Score dans le temps (8 premiers moteurs)',
                      color='white', fontweight='bold')
        ax4.set_ylabel('Health Score [0–100]', color='white')
        ax4.set_ylim(0, 100)
        ax4.legend(fontsize=7, ncol=4, loc='lower left',
                   facecolor='#16213e', labelcolor='white')
        ax4.tick_params(colors='white')
        ax4.set_facecolor('#16213e')
        ax4.spines[:].set_color('#444')

    # ── Panel 5 : Tableau RUL ─────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#16213e')
    ax5.axis('off')

    top5 = df_sorted.head(8)
    table_data = []
    for _, r in top5.iterrows():
        rul = str(r['rul_days'])
        table_data.append([
            f"M{int(r['motor_id'])}",
            f"{r['current_di']:.3f}",
            rul + 'j',
            r['risk_level'],
        ])

    tbl = ax5.table(
        cellText   = table_data,
        colLabels  = ['Moteur', 'DI', 'RUL', 'Risque'],
        loc        = 'center',
        cellLoc    = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#555')
        if row == 0:
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            risk = table_data[row - 1][3]
            cell.set_facecolor('#1a1a3e')
            if col == 3:
                cell.set_facecolor(RISK_COLOR.get(risk, '#9e9e9e') + '55')
            cell.set_text_props(color='white')

    ax5.set_title('Top Risques', color='white', fontweight='bold', pad=10)

    # Titre global
    fig.suptitle(
        'TABLEAU DE BORD — MAINTENANCE PRÉDICTIVE (IA/ML)',
        fontsize=15, fontweight='bold', color='white', y=0.98
    )

    path = os.path.join(output_dir, 'fig6_report_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  → {path}")


def main():
    print("=" * 55)
    print(" ÉTAPE 5 — RAPPORT FINAL")
    print("=" * 55)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] Fichier introuvable : {INPUT_CSV}")
        print("→ Lancez d'abord : python step4_rul_prediction.py")
        return

    print(f"\n→ Chargement des données ...")
    df     = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    df_rul = pd.read_csv(RUL_SUMMARY_CSV)
    print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")

    # ── Rapport texte ─────────────────────────────
    print("\n→ Génération du rapport texte ...")
    report = generate_text_report(df, df_rul)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f"\n✓ Rapport sauvegardé : {OUTPUT_TXT}")

    # ── Graphique récapitulatif ───────────────────
    print("\n→ Génération du graphique récapitulatif ...")
    plot_report_summary(df, df_rul, FIGURES_DIR)

    print("\n✓ Rapport final complet !")
    print("=" * 55)


if __name__ == '__main__':
    main()
