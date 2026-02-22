"""
========================================================
 ÉTAPE 4 — PRÉDICTION RUL (Remaining Useful Life)
 Entrée : data/03_anomalies.csv
 Sortie : data/04_rul_results.csv

 Méthode :
  1. Indice de dégradation (DI) : score glissant multi-features
  2. Régression polynomiale sur DI pour estimer la tendance
  3. RUL = nb de jours avant que DI dépasse le seuil critique
  4. Niveau de risque : critique / élevé / modéré / faible
========================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────
INPUT_CSV    = 'data/03_anomalies.csv'
OUTPUT_CSV   = 'data/04_rul_results.csv'
FIGURES_DIR  = 'figures'

# Seuils de l'indice de dégradation (0–1)
DI_WARNING  = 0.50   # prudence
DI_CRITICAL = 0.75   # critique → maintenance requise
PREDICT_DAYS = 90    # horizon de prédiction (jours)
# ──────────────────────────────────────────────────────


def compute_degradation_index(group: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'Indice de Dégradation (DI) pour un moteur.
    DI ∈ [0, 1] où 0 = parfait état, 1 = panne imminente.

    Combinaison pondérée de :
      - Énergie vibratoire normalisée   (35%)
      - Kurtosis normalisé              (20%)
      - Température normalisée          (25%)
      - Score d'anomalie combiné        (20%)
    """
    g = group.sort_values('timestamp').copy()

    def norm_series(s, window=30):
        """Normalise une série avec référence glissante (début = sain)."""
        s = s.fillna(s.median())
        ref = s.iloc[:max(5, window)].mean()   # référence : état initial
        span = s.rolling(window=window, min_periods=3).mean()
        return ((span - ref) / (ref + 1e-9)).clip(0, None)

    di_vib   = norm_series(g['vib_energy_mean']  if 'vib_energy_mean'  in g else g['vib_energy'],   30)
    di_kurt  = norm_series(g['vib_kurt'].abs(),  20)
    di_temp  = norm_series(g['temp_mean']        if 'temp_mean'        in g else g['temperature'],  30)
    di_score = g['combined_score'].fillna(0)

    # Combinaison pondérée
    raw_di = (0.35 * di_vib + 0.20 * di_kurt +
              0.25 * di_temp + 0.20 * di_score)

    # Normalisation finale 0–1 par moteur
    lo, hi = raw_di.min(), raw_di.max()
    if hi > lo:
        g['degradation_index'] = ((raw_di - lo) / (hi - lo)).clip(0, 1)
    else:
        g['degradation_index'] = 0.0

    return g


def estimate_rul(group: pd.DataFrame,
                 predict_days: int = PREDICT_DAYS,
                 threshold: float  = DI_CRITICAL) -> dict:
    """
    Estime le RUL (jours) en ajustant une régression polynomiale
    sur l'indice de dégradation et en extrapolant vers le seuil.
    """
    g = group.sort_values('timestamp').copy()
    motor_id = g['motor_id'].iloc[0]

    # Temps en jours depuis la première mesure
    g['days'] = (g['timestamp'] - g['timestamp'].min()).dt.total_seconds() / 86400

    x = g['days'].values
    y = g['degradation_index'].values

    if len(x) < 5:
        return {'motor_id': motor_id, 'rul_days': None,
                'current_di': float(y[-1]) if len(y) else 0,
                'trend_slope': 0, 'risk_level': 'inconnu'}

    # Régression polynomiale degré 2 (quadratique)
    try:
        poly_coef = np.polyfit(x, y, deg=2)
        poly_fn   = np.poly1d(poly_coef)
    except Exception:
        poly_coef = np.polyfit(x, y, deg=1)
        poly_fn   = np.poly1d(poly_coef)

    current_day   = x[-1]
    current_di    = float(y[-1])
    trend_slope   = float(poly_fn(current_day + 1) - poly_fn(current_day))

    # Projection : trouver quand DI dépasse le seuil
    future_days   = np.linspace(current_day, current_day + predict_days, 500)
    future_di     = poly_fn(future_days).clip(0, 1.5)

    rul_days      = None
    idx_exceed    = np.where(future_di >= threshold)[0]
    if len(idx_exceed) > 0:
        rul_days = float(future_days[idx_exceed[0]] - current_day)
        rul_days = max(0.0, rul_days)

    # Niveau de risque
    if current_di >= DI_CRITICAL or (rul_days is not None and rul_days < 7):
        risk_level = 'CRITIQUE'
    elif current_di >= DI_WARNING or (rul_days is not None and rul_days < 21):
        risk_level = 'ÉLEVÉ'
    elif current_di >= 0.30:
        risk_level = 'MODÉRÉ'
    else:
        risk_level = 'FAIBLE'

    return {
        'motor_id'    : int(motor_id),
        'rul_days'    : round(rul_days, 1)    if rul_days is not None else '>90',
        'current_di'  : round(current_di, 4),
        'trend_slope' : round(trend_slope, 6),
        'risk_level'  : risk_level,
        'poly_coef'   : poly_coef.tolist(),
        'last_timestamp': str(g['timestamp'].max()),
    }


def plot_rul_all_motors(df: pd.DataFrame,
                        rul_results: list,
                        output_dir: str):
    """Graphique RUL et DI pour tous les moteurs (grille)."""
    os.makedirs(output_dir, exist_ok=True)
    motors    = sorted(df['motor_id'].unique())
    n_motors  = len(motors)
    ncols     = 3
    nrows     = (n_motors + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    rul_map = {r['motor_id']: r for r in rul_results}

    for idx, mid in enumerate(motors):
        ax   = axes[idx]
        dm   = df[df['motor_id'] == mid].sort_values('timestamp').copy()
        dm['days'] = (dm['timestamp'] - dm['timestamp'].min()).dt.total_seconds() / 86400

        # Historique DI
        ax.plot(dm['days'], dm['degradation_index'],
                color='steelblue', linewidth=1.5, label='DI historique')
        ax.fill_between(dm['days'], dm['degradation_index'],
                        alpha=0.2, color='steelblue')

        # Projection
        r = rul_map.get(mid)
        if r and 'poly_coef' in r and r['poly_coef']:
            x_last   = dm['days'].max()
            x_future = np.linspace(x_last, x_last + PREDICT_DAYS, 200)
            poly_fn  = np.poly1d(r['poly_coef'])
            y_future = poly_fn(x_future).clip(0, 1.2)
            ax.plot(x_future, y_future, color='orange',
                    linestyle='--', linewidth=1.5, label='Tendance prédite')

        # Seuils
        ax.axhline(DI_WARNING,  color='orange', linestyle=':', linewidth=1.2,
                   label=f'Prudence ({DI_WARNING})')
        ax.axhline(DI_CRITICAL, color='red',    linestyle='--', linewidth=1.5,
                   label=f'Critique ({DI_CRITICAL})')

        # Titre avec RUL et risque
        rul_txt = f"RUL={r['rul_days']}j" if r else ''
        risk    = r['risk_level'] if r else ''
        color_title = {'CRITIQUE': 'red', 'ÉLEVÉ': 'darkorange',
                       'MODÉRÉ': 'goldenrod', 'FAIBLE': 'green',
                       'inconnu': 'gray'}.get(risk, 'black')
        ax.set_title(f'Moteur {mid}  |  {rul_txt}  [{risk}]',
                     fontweight='bold', color=color_title)
        ax.set_xlabel('Jours depuis début')
        ax.set_ylabel('Indice de dégradation')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # Masquer axes vides
    for idx in range(n_motors, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Prédiction RUL — Tous les moteurs\n'
                 '(Indice de dégradation + tendance)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_rul_all_motors.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_risk_dashboard(rul_results: list, output_dir: str):
    """Dashboard résumé : niveau de risque et DI par moteur."""
    df_rul = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'poly_coef'}
        for r in rul_results
    ])

    color_map = {
        'CRITIQUE': '#d32f2f',
        'ÉLEVÉ':    '#f57c00',
        'MODÉRÉ':   '#fbc02d',
        'FAIBLE':   '#388e3c',
        'inconnu':  '#9e9e9e',
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Graphique 1 : DI par moteur ───────────────
    ax = axes[0]
    colors = [color_map.get(r, '#9e9e9e') for r in df_rul['risk_level']]
    bars   = ax.barh(df_rul['motor_id'].astype(str),
                     df_rul['current_di'], color=colors, edgecolor='white')
    ax.axvline(DI_WARNING,  color='orange', linestyle='--', linewidth=1.5,
               label=f'Seuil prudence ({DI_WARNING})')
    ax.axvline(DI_CRITICAL, color='red',    linestyle='--', linewidth=2,
               label=f'Seuil critique ({DI_CRITICAL})')
    ax.set_title('Indice de dégradation actuel par moteur', fontweight='bold')
    ax.set_xlabel('DI (0 = sain, 1 = critique)')
    ax.set_ylabel('Moteur ID')
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # ── Graphique 2 : Distribution risques ────────
    ax = axes[1]
    risk_counts = df_rul['risk_level'].value_counts()
    risk_order  = ['CRITIQUE', 'ÉLEVÉ', 'MODÉRÉ', 'FAIBLE', 'inconnu']
    risk_counts = risk_counts.reindex(
        [r for r in risk_order if r in risk_counts.index])
    wedge_colors = [color_map[r] for r in risk_counts.index]
    wedges, texts, autotexts = ax.pie(
        risk_counts.values, labels=risk_counts.index,
        colors=wedge_colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')
    ax.set_title('Distribution des niveaux de risque', fontweight='bold')

    plt.suptitle('Dashboard Risque — Maintenance Prédictive',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_risk_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def main():
    print("=" * 55)
    print(" ÉTAPE 4 — PRÉDICTION RUL")
    print("=" * 55)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] Fichier introuvable : {INPUT_CSV}")
        print("→ Lancez d'abord : python step3_anomaly_detection.py")
        return

    print(f"\n→ Chargement de {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")

    # ── 1. Calcul de l'indice de dégradation ─────
    print("\n→ Calcul de l'indice de dégradation (DI) ...")
    parts = []
    for mid, group in df.groupby('motor_id'):
        enriched = compute_degradation_index(group.copy())
        enriched['motor_id'] = mid
        parts.append(enriched)
    df = pd.concat(parts, ignore_index=True)

    # ── 2. Estimation RUL par moteur ──────────────
    print("\n→ Estimation RUL par moteur ...")
    rul_results = []
    for mid, group in df.groupby('motor_id'):
        r = estimate_rul(group)
        rul_results.append(r)

    # ── 3. Résumé RUL ─────────────────────────────
    df_rul_display = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'poly_coef'}
        for r in rul_results
    ])
    print("\n✓ Résultats RUL par moteur :")
    print(df_rul_display.to_string(index=False))

    # Statistiques risque
    print("\n  Distribution des risques :")
    print(df_rul_display['risk_level'].value_counts().to_string())

    critiques = df_rul_display[df_rul_display['risk_level'] == 'CRITIQUE']
    if not critiques.empty:
        print(f"\n  ⚠️  {len(critiques)} moteur(s) en état CRITIQUE !")
        print(f"  → Moteurs : {critiques['motor_id'].tolist()}")

    # ── 4. Merge résultats dans df principal ──────
    df_rul_merge = pd.DataFrame([
        {'motor_id': r['motor_id'], 'rul_days': r['rul_days'],
         'risk_level': r['risk_level'], 'current_di_summary': r['current_di']}
        for r in rul_results
    ])
    df = df.merge(df_rul_merge, on='motor_id', how='left')

    # ── 5. Graphiques ─────────────────────────────
    print(f"\n→ Génération des graphiques ...")
    plot_rul_all_motors(df, rul_results, FIGURES_DIR)
    plot_risk_dashboard(rul_results, FIGURES_DIR)

    # ── 6. Sauvegarde résultats complets ─────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Résultats complets sauvegardés : {OUTPUT_CSV}")

    # Sauvegarde résumé RUL séparé
    rul_summary_path = 'data/rul_summary.csv'
    df_rul_display.to_csv(rul_summary_path, index=False)
    print(f"✓ Résumé RUL sauvegardé       : {rul_summary_path}")
    print("=" * 55)


if __name__ == '__main__':
    main()
