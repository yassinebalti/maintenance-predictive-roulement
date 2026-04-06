"""
========================================================
 ÉTAPE 4 — PRÉDICTION RUL (Remaining Useful Life) V3
 Entrée : data/03_anomalies.csv
 Sortie : data/04_rul_results.csv

 V2 (conservé) :
   • Indice de dégradation (DI) multi-features
   • Régression polynomiale sur DI

 V3 — 2 NOUVELLES AMÉLIORATIONS :
   ✅ AXE 2 — 3 modèles de dégradation + Ensemble + IC
              Polynomial (40%) + Exponentiel (30%) + Weibull (30%)
              Intervalle de confiance à 80%
   ✅ AXE 7 — CUSUM : détection rupture de tendance
              Alarme dès que la tendance de DI change brusquement
========================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV   = 'data/03_anomalies.csv'
OUTPUT_CSV  = 'data/04_rul_results.csv'
FIGURES_DIR = 'figures'

DI_WARNING   = 0.50
DI_CRITICAL  = 0.75
PREDICT_DAYS = 90

# CUSUM paramètres
CUSUM_K_FACTOR = 0.5   # slack = k × σ
CUSUM_H_FACTOR = 4.0   # seuil alarme = h × σ


# ══════════════════════════════════════════════════════
#  INDICE DE DÉGRADATION (V2 — inchangé)
# ══════════════════════════════════════════════════════
def compute_degradation_index(group: pd.DataFrame) -> pd.DataFrame:
    """
    DI ∈ [0,1] : combinaison pondérée de 4 indicateurs.
    0 = parfait, 1 = panne imminente.
    """
    g = group.sort_values('timestamp').copy()

    def norm_series(s, window=30):
        s   = s.fillna(s.median())
        ref = s.iloc[:max(5, window)].mean()
        span = s.rolling(window=window, min_periods=3).mean()
        return ((span - ref) / (ref + 1e-9)).clip(0, None)

    col_vib  = 'vib_energy_mean' if 'vib_energy_mean' in g else 'vib_energy'
    col_temp = 'temp_mean'        if 'temp_mean'        in g else 'temperature'

    di_vib   = norm_series(g[col_vib], 30)
    di_kurt  = norm_series(g['vib_kurt'].abs(), 20)
    di_temp  = norm_series(g[col_temp], 30)
    di_score = g['combined_score'].fillna(0)

    raw_di = (0.35 * di_vib + 0.20 * di_kurt +
              0.25 * di_temp + 0.20 * di_score)

    lo, hi = raw_di.min(), raw_di.max()
    g['degradation_index'] = ((raw_di - lo) / (hi - lo + 1e-9)).clip(0, 1) if hi > lo else 0.0
    return g


# ══════════════════════════════════════════════════════
#  AXE 2 — 3 modèles + Ensemble + IC
# ══════════════════════════════════════════════════════
def estimate_rul_v3(group: pd.DataFrame) -> dict:
    """
    Estime le RUL avec 3 modèles et calcule un ensemble + IC 80%.

    Modèle 1 — Polynomiale  (40%) : DI(t) = at² + bt + c
    Modèle 2 — Exponentielle (30%) : DI(t) = a·e^(bt)
    Modèle 3 — Weibull        (30%) : F(t) = 1 - e^(-(t/η)^β)

    β Weibull :
      β = 1   → dégradation constante (usure normale)
      β > 1   → dégradation accélérée (défaut en croissance)
      β < 1   → dégradation décélérante (rodage)
    """
    g        = group.sort_values('timestamp').copy()
    motor_id = int(g['motor_id'].iloc[0])
    g['days'] = (g['timestamp'] - g['timestamp'].min()).dt.total_seconds() / 86400

    x = g['days'].values
    y = np.clip(g['degradation_index'].values, 0, 1)

    # Base de retour si données insuffisantes
    base = {
        'motor_id'        : motor_id,
        'rul_days'        : '>90',
        'rul_ensemble'    : 90,
        'rul_low'         : 70,
        'rul_high'        : 90,
        'rul_poly'        : 90,
        'rul_exp'         : 90,
        'rul_weibull'     : 90,
        'weibull_beta'    : 1.0,
        'current_di'      : float(y[-1]) if len(y) else 0.0,
        'trend_slope'     : 0.0,
        'risk_level'      : 'FAIBLE',
        'confidence'      : 0.5,
        'n_points'        : len(y),
        'poly_coef'       : [0, 0, float(y[-1]) if len(y) else 0],
        'last_timestamp'  : str(g['timestamp'].max()),
    }
    if len(x) < 5:
        return base

    if x.max() < 1:
        x = np.arange(len(x), dtype=float)

    current_day = x[-1]
    current_di  = float(y[-1])

    # ── Modèle 1 : Polynomiale ────────────────────
    try:
        poly_coef = np.polyfit(x, y, deg=min(2, len(x) - 1))
        poly_fn   = np.poly1d(poly_coef)
        slope_now = poly_fn(current_day + 1) - poly_fn(current_day)
        if slope_now <= 1e-6:
            slope_now = max(1e-6, (y[-1] - y[0]) / (current_day + 1))
        days_poly = max(0.0, (DI_CRITICAL - current_di) / slope_now)
    except Exception:
        poly_coef = [0, 0, current_di]
        poly_fn   = np.poly1d(poly_coef)
        days_poly = 90.0

    # ── Modèle 2 : Exponentielle ──────────────────
    try:
        valid = y > 0.01
        if valid.sum() >= 3:
            log_y = np.log(y[valid] + 1e-6)
            p_exp = np.polyfit(x[valid], log_y, deg=1)
            b_exp = p_exp[0]
            a_exp = np.exp(p_exp[1])
            if b_exp > 1e-6:
                t_crit  = np.log(DI_CRITICAL / (a_exp + 1e-9)) / b_exp
                days_exp = max(0.0, t_crit - current_day)
            else:
                days_exp = 90.0
        else:
            days_exp = 90.0
    except Exception:
        days_exp = 90.0

    # ── Modèle 3 : Weibull ────────────────────────
    try:
        if len(y) >= 5:
            dy  = np.diff(y)
            ddy = np.diff(dy)
            # β estimé depuis la courbure de la dégradation
            beta = 1.0 + max(0.0, np.mean(ddy) * 10.0)
            beta = float(np.clip(beta, 0.5, 3.0))
            # Paramètre d'échelle η de Weibull
            eta  = current_day / ((-np.log(1 - current_di + 1e-8)) ** (1.0 / beta) + 1e-8)
            # Temps au seuil critique
            t_crit_w   = eta * ((-np.log(1 - DI_CRITICAL)) ** (1.0 / beta))
            days_weib  = max(0.0, t_crit_w - current_day)
        else:
            beta, days_weib = 1.0, days_poly
    except Exception:
        beta, days_weib = 1.0, days_poly

    # ── Ensemble pondéré ──────────────────────────
    days_poly  = min(days_poly,  PREDICT_DAYS)
    days_exp   = min(days_exp,   PREDICT_DAYS)
    days_weib  = min(days_weib,  PREDICT_DAYS)

    rul_mean = 0.40 * days_poly + 0.30 * days_exp + 0.30 * days_weib

    # IC basé sur l'écart entre les 3 modèles
    spread   = np.std([days_poly, days_exp, days_weib])
    rul_low  = max(0.0, rul_mean - spread)
    rul_high = min(PREDICT_DAYS, rul_mean + spread)
    conf     = max(0.3, 1.0 - spread / (rul_mean + 1.0) * 0.5)

    if rul_mean >= PREDICT_DAYS:
        rul_str  = '>90'
        rul_mean = float(PREDICT_DAYS)
        rul_low  = 80.0
        rul_high = float(PREDICT_DAYS)
    else:
        rul_str  = str(round(rul_mean, 1))

    # Niveau de risque
    if current_di >= DI_CRITICAL or rul_mean < 7:
        risk = 'CRITIQUE'
    elif current_di >= DI_WARNING or rul_mean < 21:
        risk = 'ÉLEVÉ'
    elif current_di >= 0.30:
        risk = 'MODÉRÉ'
    else:
        risk = 'FAIBLE'

    return {
        'motor_id'       : motor_id,
        'rul_days'       : rul_str,
        'rul_ensemble'   : round(rul_mean, 1),
        'rul_low'        : round(rul_low, 1),
        'rul_high'       : round(rul_high, 1),
        'rul_poly'       : round(days_poly, 1),
        'rul_exp'        : round(days_exp, 1),
        'rul_weibull'    : round(days_weib, 1),
        'weibull_beta'   : round(beta, 3),
        'current_di'     : round(current_di, 4),
        'trend_slope'    : round(float(poly_fn(current_day+1) - poly_fn(current_day)), 6),
        'risk_level'     : risk,
        'confidence'     : round(conf, 3),
        'n_points'       : len(y),
        'poly_coef'      : poly_coef.tolist() if hasattr(poly_coef, 'tolist') else list(poly_coef),
        'last_timestamp' : str(g['timestamp'].max()),
    }


# ══════════════════════════════════════════════════════
#  AXE 7 — CUSUM détection rupture de tendance
# ══════════════════════════════════════════════════════
def detect_cusum(group: pd.DataFrame) -> dict:
    """
    CUSUM (CUmulative SUM) : détecte un changement de tendance dans le DI.

    S+[t] = max(0, S+[t-1] + (DI[t] - μ₀) - k)
    S-[t] = max(0, S-[t-1] - (DI[t] - μ₀) - k)
    Alarme si S+[t] > h  OU  S-[t] > h

    μ₀ = référence initiale (10 premiers points)
    k  = CUSUM_K_FACTOR × σ  (slack)
    h  = CUSUM_H_FACTOR × σ  (seuil alarme)

    Avantage vs régression : détecte la rupture en 1-5 points,
    utile pour monitoring IoT temps réel.
    """
    g        = group.sort_values('timestamp').copy()
    motor_id = int(g['motor_id'].iloc[0])
    di       = g['degradation_index'].fillna(0).values

    base = {
        'motor_id'          : motor_id,
        'cusum_alarm'       : False,
        'change_point_idx'  : None,
        'change_point_date' : None,
        'n_alarms'          : 0,
        'di_before'         : None,
        'di_after'          : None,
        'cusum_max'         : 0.0,
        'severity'          : 'STABLE',
    }

    if len(di) < 10:
        return base

    mu0   = di[:10].mean()
    sigma = max(di[:10].std(), 0.01)
    k     = CUSUM_K_FACTOR * sigma
    h     = CUSUM_H_FACTOR * sigma

    S_pos = np.zeros(len(di))
    S_neg = np.zeros(len(di))
    alarms = []

    for t in range(1, len(di)):
        S_pos[t] = max(0.0, S_pos[t-1] + (di[t] - mu0) - k)
        S_neg[t] = max(0.0, S_neg[t-1] - (di[t] - mu0) - k)
        if S_pos[t] > h or S_neg[t] > h:
            alarms.append(t)

    if not alarms:
        return base

    first = alarms[0]
    alarm_date = str(g['timestamp'].iloc[first])[:10]

    n = len(di)
    if   first < n * 0.3: severity = 'ÉLEVÉ'
    elif first < n * 0.6: severity = 'MODÉRÉ'
    else:                  severity = 'FAIBLE_ALARM'

    return {
        'motor_id'          : motor_id,
        'cusum_alarm'       : True,
        'change_point_idx'  : int(first),
        'change_point_date' : alarm_date,
        'n_alarms'          : len(alarms),
        'di_before'         : round(float(di[:first].mean()), 4) if first > 0 else 0.0,
        'di_after'          : round(float(di[first:].mean()), 4),
        'cusum_max'         : round(float(S_pos.max()), 4),
        'severity'          : severity,
    }


# ══════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════
def plot_rul_all_motors(df, rul_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    motors   = sorted(df['motor_id'].unique())
    ncols    = 3
    nrows    = (len(motors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()
    rul_map  = {r['motor_id']: r for r in rul_results}
    RCOLORS  = {'CRITIQUE': 'red', 'ÉLEVÉ': 'darkorange', 'MODÉRÉ': 'goldenrod',
                'FAIBLE': 'green', 'inconnu': 'gray'}

    for idx, mid in enumerate(motors):
        ax = axes[idx]
        dm = df[df['motor_id'] == mid].sort_values('timestamp').copy()
        dm['days'] = (dm['timestamp'] - dm['timestamp'].min()).dt.total_seconds() / 86400

        ax.plot(dm['days'], dm['degradation_index'],
                color='steelblue', linewidth=1.5, label='DI historique')
        ax.fill_between(dm['days'], dm['degradation_index'], alpha=0.15, color='steelblue')

        r = rul_map.get(mid)
        if r and 'poly_coef' in r and len(r['poly_coef']) > 0:
            x_last   = dm['days'].max()
            x_future = np.linspace(x_last, x_last + PREDICT_DAYS, 200)
            poly_fn  = np.poly1d(r['poly_coef'])
            y_future = poly_fn(x_future).clip(0, 1.2)
            ax.plot(x_future, y_future, color='orange',
                    linestyle='--', linewidth=1.5, label='Tendance (poly)')

            # IC — zone ombragée
            if r.get('rul_low') is not None and r.get('rul_high') is not None:
                ax.axvspan(x_last + r['rul_low'], x_last + r['rul_high'],
                           alpha=0.08, color='orange', label='IC 80%')

        ax.axhline(DI_WARNING,  color='orange', linestyle=':', linewidth=1.2)
        ax.axhline(DI_CRITICAL, color='red',    linestyle='--', linewidth=1.5)

        risk      = r['risk_level'] if r else 'inconnu'
        rul_txt   = f"RUL={r['rul_days']}j" if r else ''
        beta_txt  = f" β={r['weibull_beta']:.2f}" if r and 'weibull_beta' in r else ''
        ax.set_title(f'M{mid}  {rul_txt}  [{risk}]{beta_txt}',
                     fontweight='bold', color=RCOLORS.get(risk, 'black'))
        ax.set_xlabel('Jours'); ax.set_ylabel('DI')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    for idx in range(len(motors), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('RUL V3 — Ensemble Poly+Exp+Weibull + IC 80%',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_rul_all_motors.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_cusum(df, cusum_results, output_dir):
    """Visualise le CUSUM pour les moteurs avec alarme."""
    alarmed = [r for r in cusum_results if r['cusum_alarm']][:6]
    if not alarmed:
        return

    ncols = min(3, len(alarmed))
    nrows = (len(alarmed) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    SCOLORS = {'ÉLEVÉ': 'red', 'MODÉRÉ': 'orange', 'FAIBLE_ALARM': 'gold'}

    for i, r in enumerate(alarmed):
        ax = axes[i]
        dm = df[df['motor_id'] == r['motor_id']].sort_values('timestamp')
        di = dm['degradation_index'].fillna(0).values
        t  = np.arange(len(di))

        mu0   = di[:10].mean() if len(di) >= 10 else di.mean()
        sigma = max(di[:10].std() if len(di) >= 10 else di.std(), 0.01)
        k, h  = CUSUM_K_FACTOR * sigma, CUSUM_H_FACTOR * sigma

        S_pos = np.zeros(len(di))
        for pt in range(1, len(di)):
            S_pos[pt] = max(0.0, S_pos[pt-1] + (di[pt] - mu0) - k)

        ax.plot(t, di,    color='steelblue', lw=1.5, label='DI')
        ax.plot(t, S_pos, color='#ff7f0e',   lw=1.5, label='CUSUM S+', linestyle='--')
        ax.axhline(h, color='red', linestyle=':', lw=1.5, label=f'Seuil h={h:.3f}')

        cp = r['change_point_idx']
        if cp:
            color_cp = SCOLORS.get(r['severity'], 'orange')
            ax.axvline(cp, color=color_cp, linewidth=2, linestyle='-.',
                       label=f"Rupture j{cp} [{r['severity']}]")

        ax.set_title(f"M{r['motor_id']} — CUSUM ({r['severity']})",
                     fontweight='bold',
                     color=SCOLORS.get(r['severity'], 'black'))
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_xlabel('Points'); ax.set_ylabel('Valeur')

    for idx in range(len(alarmed), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('AXE 7 — CUSUM : Détection de Rupture de Tendance',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_cusum_changepoints.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_risk_dashboard(rul_results, output_dir):
    df_rul = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ('poly_coef',)}
        for r in rul_results
    ])
    RCOLOR = {'CRITIQUE': '#d32f2f', 'ÉLEVÉ': '#f57c00',
               'MODÉRÉ': '#fbc02d', 'FAIBLE': '#388e3c', 'inconnu': '#9e9e9e'}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # DI par moteur avec IC
    ax = axes[0]
    df_s = df_rul.sort_values('current_di', ascending=True)
    colors = [RCOLOR.get(r, '#9e9e9e') for r in df_s['risk_level']]
    ax.barh(df_s['motor_id'].astype(str), df_s['current_di'], color=colors)
    ax.axvline(DI_WARNING,  color='orange', linestyle='--', lw=1.5, label=f'Prudence {DI_WARNING}')
    ax.axvline(DI_CRITICAL, color='red',    linestyle='--', lw=2,   label=f'Critique {DI_CRITICAL}')
    ax.set_title('DI actuel par moteur', fontweight='bold')
    ax.set_xlabel('DI (0=sain, 1=critique)'); ax.set_xlim(0, 1)
    ax.legend(); ax.grid(axis='x', alpha=0.3)

    # RUL Ensemble + IC
    ax = axes[1]
    df_s2 = df_rul.sort_values('rul_ensemble', ascending=True)
    colors2 = [RCOLOR.get(r, '#9e9e9e') for r in df_s2['risk_level']]
    y_pos   = range(len(df_s2))
    ax.barh(list(y_pos), df_s2['rul_ensemble'], color=colors2, alpha=0.7)
    # IC whiskers
    for i, (_, row) in enumerate(df_s2.iterrows()):
        lo = row.get('rul_low',  row['rul_ensemble'])
        hi = row.get('rul_high', row['rul_ensemble'])
        ax.plot([lo, hi], [i, i], color='black', linewidth=2, solid_capstyle='round')
    ax.axvline(30, color='red',    linestyle='--', lw=1.5, label='Urgent 30j')
    ax.axvline(60, color='orange', linestyle='--', lw=1,   label='Planifier 60j')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_s2['motor_id'].astype(str))
    ax.set_title('RUL Ensemble + IC 80%\n(Poly+Exp+Weibull)', fontweight='bold')
    ax.set_xlabel('Jours'); ax.legend(); ax.grid(axis='x', alpha=0.3)

    # Pie
    ax = axes[2]
    rc   = df_rul['risk_level'].value_counts()
    ords = [r for r in ['CRITIQUE','ÉLEVÉ','MODÉRÉ','FAIBLE','inconnu'] if r in rc.index]
    wedges, texts, auts = ax.pie(
        [rc[r] for r in ords], labels=ords,
        colors=[RCOLOR[r] for r in ords],
        autopct='%1.0f%%', startangle=90
    )
    for at in auts:
        at.set_fontsize(12); at.set_fontweight('bold')
    ax.set_title('Distribution Risques V3', fontweight='bold')

    plt.suptitle('Dashboard RUL V3 — Ensemble 3 Modèles', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_risk_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    print("=" * 62)
    print(" ÉTAPE 4 — PRÉDICTION RUL V3")
    print(" AXE 2 : Ensemble Poly+Exp+Weibull + IC 80%")
    print(" AXE 7 : CUSUM détection rupture de tendance")
    print("=" * 62)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] Fichier introuvable : {INPUT_CSV}")
        print("→ Lancez d'abord : python step3_anomaly_detection.py")
        return

    print(f"\n→ Chargement de {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")

    # 1. Indice de dégradation
    print("\n→ [1/4] Calcul DI ...")
    parts = []
    for mid, group in df.groupby('motor_id'):
        enriched = compute_degradation_index(group.copy())
        enriched['motor_id'] = mid
        parts.append(enriched)
    df = pd.concat(parts, ignore_index=True)

    # 2. RUL V3 — Ensemble 3 modèles
    print("\n→ [2/4] AXE 2 — RUL Ensemble (Poly+Exp+Weibull) ...")
    rul_results = []
    print(f"  {'M':>4} | {'DI':>6} | {'Poly':>6} | {'Exp':>6} | {'Weib':>6} | "
          f"{'Ensb':>6} | {'IC':>12} | {'β':>5} | Risque")
    print("  " + "─" * 72)

    for mid, group in df.groupby('motor_id'):
        r = estimate_rul_v3(group)
        rul_results.append(r)
        ic_str = f"[{r['rul_low']:.0f}–{r['rul_high']:.0f}]"
        print(f"  {mid:>4} | {r['current_di']:>6.3f} | {r['rul_poly']:>6.1f} | "
              f"{r['rul_exp']:>6.1f} | {r['rul_weibull']:>6.1f} | "
              f"{r['rul_ensemble']:>6.1f} | {ic_str:>12} | "
              f"{r['weibull_beta']:>5.2f} | {r['risk_level']}")

    # 3. CUSUM
    print("\n→ [3/4] AXE 7 — CUSUM détection rupture de tendance ...")
    cusum_results = []
    for mid, group in df.groupby('motor_id'):
        cr = detect_cusum(group)
        cusum_results.append(cr)

    df_cusum   = pd.DataFrame(cusum_results)
    n_alarmed  = df_cusum['cusum_alarm'].sum()
    df_cusum.to_csv('data/cusum_changepoints.csv', index=False)
    print(f"  {n_alarmed}/{len(df_cusum)} moteurs avec rupture détectée")
    print(f"\n  {'M':>4} | {'Alarme':>7} | {'Date':>12} | {'Sév.':>12} | {'DI avant':>9} | {'DI après':>9}")
    print("  " + "─" * 60)
    for _, r in df_cusum.iterrows():
        alarm_str = "⚠ OUI" if r['cusum_alarm'] else "✓ Non"
        print(f"  {int(r['motor_id']):>4} | {alarm_str:>7} | "
              f"{str(r['change_point_date']):>12} | {str(r['severity']):>12} | "
              f"{str(r['di_before']):>9} | {str(r['di_after']):>9}")
    print(f"  Sauvegardé : data/cusum_changepoints.csv")

    # Rapport résumé
    df_display = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ('poly_coef',)}
        for r in rul_results
    ])
    print(f"\n  Distribution risques V3 :")
    print(df_display['risk_level'].value_counts().to_string())
    critiques = df_display[df_display['risk_level'] == 'CRITIQUE']
    if not critiques.empty:
        print(f"\n  ⚠️  {len(critiques)} moteur(s) CRITIQUE : {critiques['motor_id'].tolist()}")

    # 4. Figures
    print("\n→ [4/4] Génération des figures ...")
    # Merge RUL dans df
    df_rul_merge = pd.DataFrame([{
        'motor_id'       : r['motor_id'],
        'rul_days'       : r['rul_days'],
        'rul_ensemble'   : r['rul_ensemble'],
        'rul_low'        : r['rul_low'],
        'rul_high'       : r['rul_high'],
        'risk_level'     : r['risk_level'],
        'weibull_beta'   : r['weibull_beta'],
        'current_di_summary': r['current_di'],
    } for r in rul_results])
    df = df.merge(df_rul_merge, on='motor_id', how='left')

    plot_rul_all_motors(df, rul_results, FIGURES_DIR)
    plot_risk_dashboard(rul_results, FIGURES_DIR)
    plot_cusum(df, cusum_results, FIGURES_DIR)

    # Sauvegarde
    df_save = df.copy()
    for col in df_save.select_dtypes(include='object').columns:
        df_save[col] = df_save[col].astype(str)
    df_save.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    df_display.to_csv('data/rul_summary.csv', index=False, encoding='utf-8-sig')

    print(f"\n✓ {OUTPUT_CSV}")
    print(f"✓ data/rul_summary.csv")
    print(f"✓ data/cusum_changepoints.csv")
    print("=" * 62)


if __name__ == '__main__':
    main()
