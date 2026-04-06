"""
========================================================
 ÉTAPE 3 — DÉTECTION D'ANOMALIES (MODÈLE HYBRIDE V3)
 Entrée  : data/02_features_motor.csv
 Sortie  : data/03_anomalies.csv

 V2 (conservé) :
   ✅ Feature Importance (XAI)
   ✅ Fleet Analysis (inter-moteurs)
   ✅ Confirmation temporelle
   ✅ Score de confiance

 V3 — 4 NOUVELLES AMÉLIORATIONS IA :
   ✅ AXE 1 — AUC corrigée (vs Alert_Status usine, pas auto-ref)
   ✅ AXE 4 — Walk-forward cross-validation temporelle
   ✅ AXE 5 — LOF vraiment intégré (novelty=True sur NORMAL)
   ✅ AXE 6 — SHAP par moteur (z-score vs flotte normale)

 AJOUT step6 — DIAGNOSTIC DÉFAUTS DE ROULEMENTS :
   ✅ Identification type de défaut (BPFO/BPFI/BSF/FTF)
   ✅ Localisation par moteur
   ✅ Score de sévérité + recommandations
   ✅ Rapport JSON + figures de diagnostic
========================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV   = 'data/02_features_motor.csv'
OUTPUT_CSV  = 'data/03_anomalies.csv'
SEUILS_CSV  = 'data/seuils_moteurs.csv'
FIGURES_DIR = 'figures'

N_ESTIMATORS = 500
RANDOM_STATE = 42

# Poids ensemble V3 : IF 25% + LOF 20% + Règles 55%
W_IF    = 0.25
W_LOF   = 0.20
W_RULES = 0.55

THRESHOLD      = 0.25
CONFIRM_WINDOW = 3
CONFIRM_MIN    = 2
FLEET_SIGMA    = 2.5

PARAMS = ['temperature', 'courant', 'vibration', 'acceleration']
POIDS  = {
    'temperature' : 0.35,
    'courant'     : 0.30,
    'vibration'   : 0.25,
    'acceleration': 0.10,
}

FEATURE_IF = [
    'temperature_exceed', 'courant_exceed', 'vibration_exceed', 'acceleration_exceed',
    'temperature_ratio',  'courant_ratio',  'vibration_ratio',  'acceleration_ratio',
    'n_exceed', 'severity_score',
    'vib_energy_mean', 'vib_kurt', 'crest_factor',
    'temp_mean', 'temp_trend', 'courant_mean', 'envelope_mean', 'health_score',
    'temp_ratio_flotte', 'vib_ratio_flotte', 'courant_ratio_flotte',
    'temp_zscore_flotte', 'vib_zscore_flotte',
    # AXE 3 — nouvelles features V3
    'vib_rms', 'vib_skewness', 'peak2peak',
    'spectral_entropy', 'shape_factor', 'impulse_factor',
]


# ══════════════════════════════════════════════════════
#  V2 — fonctions existantes (inchangées)
# ══════════════════════════════════════════════════════

def calibrer_seuils(df):
    seuils = {}
    for mid in sorted(df['motor_id'].unique()):
        dm = df[df['motor_id'] == mid]
        seuils[mid] = {}
        for col in PARAMS:
            av = dm[dm['alert_parameter'] == col][col]
            if len(av) > 0:
                seuils[mid][col] = float(av.min())
            else:
                nv = dm[dm['Alert_Status'] == 'NORMAL'][col]
                seuils[mid][col] = float(nv.quantile(0.95)) if len(nv) > 0 else float(dm[col].quantile(0.95))
    return seuils


def ajouter_features_depassement(df, seuils):
    df       = df.copy()
    severity = pd.Series(0.0, index=df.index)
    for col in PARAMS:
        df[f'{col}_exceed'] = 0.0
        df[f'{col}_ratio']  = 0.0
        df[f'{col}_marge']  = 0.0
        for mid in df['motor_id'].unique():
            mask  = df['motor_id'] == mid
            seuil = seuils.get(mid, {}).get(col, df[col].quantile(0.95))
            df.loc[mask, f'{col}_exceed'] = (df.loc[mask, col] > seuil).astype(float)
            df.loc[mask, f'{col}_ratio']  = df.loc[mask, col] / (seuil + 1e-9)
            df.loc[mask, f'{col}_marge']  = df.loc[mask, col] - seuil
            severity += df[f'{col}_exceed'] * POIDS[col]
    df['n_exceed']       = df[[f'{c}_exceed' for c in PARAMS]].sum(axis=1)
    df['severity_score'] = severity.clip(0, 1)
    return df


def ajouter_features_flotte(df):
    df = df.copy()
    for col, r_col, z_col in [
        ('temperature', 'temp_ratio_flotte',    'temp_zscore_flotte'),
        ('vibration',   'vib_ratio_flotte',     'vib_zscore_flotte'),
        ('courant',     'courant_ratio_flotte',  'courant_zscore_flotte'),
    ]:
        med = df.groupby('timestamp')[col].transform('median')
        std = df.groupby('timestamp')[col].transform('std').fillna(1)
        df[r_col] = df[col] / (med + 1e-9)
        df[z_col] = (df[col] - med) / (std + 1e-9)
    df['fleet_anomaly_temp'] = (df['temp_zscore_flotte'] > FLEET_SIGMA).astype(int)
    df['fleet_anomaly_vib']  = (df['vib_zscore_flotte']  > FLEET_SIGMA).astype(int)
    df['fleet_anomaly'] = (
        (df['fleet_anomaly_temp'] == 1) | (df['fleet_anomaly_vib'] == 1)
    ).astype(int)
    n = df['fleet_anomaly'].sum()
    print(f"  Anomalies flotte : {n:,} ({n/len(df)*100:.1f}%)")
    by_motor = df.groupby('motor_id')['fleet_anomaly'].mean() * 100
    for mid, pct in by_motor[by_motor > 10].sort_values(ascending=False).items():
        ratio = df[df['motor_id'] == mid]['temp_ratio_flotte'].mean()
        print(f"  M{mid:2d} → {pct:.1f}% (temp_ratio={ratio:.2f}×)")
    return df


def appliquer_confirmation_temporelle(df):
    df = df.copy()
    df['is_anomaly_raw'] = df['is_anomaly'].copy()
    parts = []
    for mid in sorted(df['motor_id'].unique()):
        dm      = df[df['motor_id'] == mid].sort_values('timestamp').copy()
        rolling = dm['is_anomaly_raw'].rolling(CONFIRM_WINDOW, min_periods=1).sum()
        dm['is_anomaly'] = (rolling >= CONFIRM_MIN).astype(bool)
        parts.append(dm)
    df    = pd.concat(parts).sort_index()
    n_raw = df['is_anomaly_raw'].sum()
    n_con = df['is_anomaly'].sum()
    red   = (n_raw - n_con) / (n_raw + 1e-9) * 100
    print(f"  Brutes : {n_raw:,} → Confirmées : {n_con:,} (-{red:.1f}% fausses alarmes)")
    return df


def calculer_feature_importance(df, feats_used):
    normal  = ~df['is_anomaly']
    anomaly = df['is_anomaly']
    rows    = []
    for f in feats_used:
        if f not in df.columns:
            continue
        mn = df.loc[normal,  f].mean()
        ma = df.loc[anomaly, f].mean()
        sd = df[f].std() + 1e-9
        rows.append({
            'feature'     : f,
            'importance'  : round(abs(ma - mn) / sd, 4),
            'val_normal'  : round(mn, 4),
            'val_anomalie': round(ma, 4),
            'diff_pct'    : round((ma - mn) / (abs(mn) + 1e-9) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values('importance', ascending=False)


def calculer_score_confiance(df):
    df = df.copy()
    concordance = 1.0 - np.abs(df['score_if'] - df['score_rules'])
    df['confidence_score'] = (50 + concordance * 50).round(1)
    df['confidence_level'] = pd.cut(
        df['confidence_score'],
        bins=[0, 70, 85, 100],
        labels=['FAIBLE', 'MOYEN', 'ÉLEVÉ']
    )
    return df


# ══════════════════════════════════════════════════════
#  AXE 5 — LOF entraîné sur NORMAL uniquement
# ══════════════════════════════════════════════════════

def entrainer_lof(df, feats_used, X_sc):
    """
    LOF avec novelty=True :
    entraîné sur données NORMALES uniquement.
    Retourne scores normalisés [0,1] pour tout le dataset.
    """
    normal_mask = (df['Alert_Status'] == 'NORMAL').values
    X_normal    = X_sc[normal_mask]

    if len(X_normal) < 50:
        print("  ⚠ LOF : pas assez de données NORMALES, scores mis à 0")
        return np.zeros(len(df))

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.10,
        novelty=True,
        n_jobs=-1
    )
    lof.fit(X_normal)

    raw_scores = -lof.score_samples(X_sc)
    lo, hi     = raw_scores.min(), raw_scores.max()
    if hi > lo:
        normalized = (raw_scores - lo) / (hi - lo)
    else:
        normalized = np.zeros_like(raw_scores)

    auc_lof = roc_auc_score(
        (df['Alert_Status'] == 'ALERT').astype(int), normalized
    )
    print(f"  LOF (novelty) AUC vs Alert_Status : {auc_lof:.4f}")
    print(f"  corr(LOF, IF) : {np.corrcoef(normalized, df['score_if'].values)[0,1]:.4f}")
    return normalized


def entrainer_if(df, feats_used):
    avail = [c for c in feats_used if c in df.columns]
    X     = df[avail].fillna(0).values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pct_alert   = (df['Alert_Status'] == 'ALERT').mean()
    normal_mask = (df['Alert_Status'] == 'NORMAL').values
    contamination = min(float(pct_alert), 0.499)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=contamination,
        max_samples='auto',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_sc[normal_mask])

    raw_if  = -model.decision_function(X_sc)
    sc_norm = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min())

    return model, scaler, sc_norm, avail, X_sc, pct_alert


# ══════════════════════════════════════════════════════
#  AXE 4 — Walk-Forward Cross-Validation
# ══════════════════════════════════════════════════════

def walk_forward_validation(df, feats_used):
    print("Walk-Forward Cross-Validation (3 folds) :")
    df_s       = df.sort_values('timestamp').copy()
    df_s['date'] = df_s['timestamp'].dt.date
    dates      = sorted(df_s['date'].unique())

    feats_ok = [c for c in feats_used if c in df_s.columns]
    folds = [
        (dates[:14],  dates[14:21]),
        (dates[:21],  dates[21:31]),
        (dates[:31],  dates[31:]),
    ]

    results = []
    print(f"  {'Fold':>5} | {'Train':>5} | {'Test':>4} | {'AUC':>7} | {'F1':>7} | {'Prec':>7} | {'Rec':>7}")
    print(f"  {'─'*55}")

    for i, (tr_dates, te_dates) in enumerate(folds):
        train = df_s[df_s['date'].isin(tr_dates)]
        test  = df_s[df_s['date'].isin(te_dates)]
        if len(test) < 100 or len(te_dates) == 0:
            continue

        scaler_f = StandardScaler()
        X_tr     = scaler_f.fit_transform(train[feats_ok].fillna(0))
        X_te     = scaler_f.transform(test[feats_ok].fillna(0))

        X_norm   = X_tr[(train['Alert_Status'] == 'NORMAL').values]
        if len(X_norm) < 50:
            continue

        clf = IsolationForest(n_estimators=200, contamination=0.10,
                               random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_norm)

        raw  = -clf.decision_function(X_te)
        sc   = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        y_te = (test['Alert_Status'] == 'ALERT').astype(int).values

        auc  = roc_auc_score(y_te, sc)
        thr  = np.percentile(sc, 90)
        y_p  = (sc >= thr).astype(int)
        prec = precision_score(y_te, y_p, zero_division=0)
        rec  = recall_score(y_te, y_p, zero_division=0)
        f1   = f1_score(y_te, y_p, zero_division=0)

        results.append({
            'fold': i + 1, 'train_days': len(tr_dates),
            'test_days': len(te_dates), 'n_test': len(test),
            'auc': round(auc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4),
        })
        print(f"  {i+1:>5} | {len(tr_dates):>5}j | {len(te_dates):>4}j | "
              f"{auc:>7.4f} | {f1:>7.4f} | {prec:>7.4f} | {rec:>7.4f}")

    if results:
        df_r = pd.DataFrame(results)
        mu   = df_r['auc'].mean()
        sig  = df_r['auc'].std()
        print(f"  {'─'*55}")
        print(f"  AUC walk-forward : {mu:.4f} ± {sig:.4f}  (N={len(df_r)} folds)")
        df_r.to_csv('data/validation_walkforward.csv', index=False)
        print(f"  Sauvegardé : data/validation_walkforward.csv")
    return results


# ══════════════════════════════════════════════════════
#  AXE 6 — SHAP par moteur
# ══════════════════════════════════════════════════════

def calculer_shap_par_moteur(df, feats_used, X_sc, model_if):
    print("SHAP — Feature importance globale :")
    feats_ok       = [c for c in feats_used if c in df.columns]
    X_all          = X_sc
    baseline_score = -model_if.decision_function(X_all).mean()
    normal_mask    = (df['Alert_Status'] == 'NORMAL').values
    X_normal       = X_all[normal_mask]

    shap_rows = []
    for i, col in enumerate(feats_ok):
        X_perm     = X_all.copy()
        np.random.shuffle(X_perm[:, i])
        score_perm = -model_if.decision_function(X_perm).mean()
        importance = abs(score_perm - baseline_score)
        shap_rows.append({'feature': col, 'importance': round(importance, 6)})

    shap_df = pd.DataFrame(shap_rows).sort_values('importance', ascending=False)
    shap_df['rank']           = range(1, len(shap_df) + 1)
    shap_df['importance_pct'] = (
        shap_df['importance'] / shap_df['importance'].sum() * 100
    ).round(2)
    shap_df.to_csv('data/shap_importance.csv', index=False)

    for _, r in shap_df.head(10).iterrows():
        bar = '█' * int(r['importance_pct'] / 1.5)
        print(f"  {r['rank']:>3}. {r['feature']:<22} {bar:<15} {r['importance_pct']:.2f}%")

    print("\n  SHAP par moteur critique (z-score vs flotte normale) :")
    mu_normal  = X_normal.mean(axis=0)
    std_normal = X_normal.std(axis=0) + 1e-8

    motor_shap_rows = []
    anomaly_mask    = df['is_anomaly'].values.astype(bool)
    priority        = sorted(
        df[anomaly_mask]['motor_id'].value_counts().head(10).index.tolist()
    )

    for mid in priority:
        mask_m = (df['motor_id'] == mid) & anomaly_mask
        if mask_m.sum() < 3:
            mask_m = (df['motor_id'] == mid)
        X_m      = X_all[mask_m]
        z_scores = (X_m.mean(axis=0) - mu_normal) / std_normal

        top3_idx = np.argsort(np.abs(z_scores))[-3:][::-1]
        expl     = ' | '.join([
            f"{feats_ok[j]} {'+' if z_scores[j] > 0 else ''}{z_scores[j]:.1f}σ"
            for j in top3_idx
        ])
        print(f"    M{mid:2d}: {expl}")

        row = {'motor_id': mid, 'explanation': expl}
        for j, col in enumerate(feats_ok):
            row[col] = round(float(z_scores[j]), 3)
        motor_shap_rows.append(row)

    pd.DataFrame(motor_shap_rows).to_csv('data/shap_per_motor.csv', index=False)
    print("  Sauvegardés : data/shap_importance.csv, data/shap_per_motor.csv")
    return shap_df


# ══════════════════════════════════════════════════════
#  FIGURES (V2 + nouvelles V3)
# ══════════════════════════════════════════════════════

def plot_all(df, imp_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    y_true = (df['Alert_Status'] == 'ALERT').astype(int)
    y_pred = df['is_anomaly'].astype(int)

    # Fig 1 — Vue globale
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    c = np.where(df['is_anomaly'], 'red', 'steelblue')
    axes[0].scatter(df['timestamp'], df['combined_score'], c=c, s=10, alpha=0.5)
    axes[0].axhline(THRESHOLD, color='orange', linestyle='--', lw=2)
    axes[0].set_title('Score hybride V3 — IF+LOF+Règles', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Score'); axes[0].grid(alpha=0.3)
    axes[1].scatter(df['timestamp'], df['vibration'], c=c, s=10, alpha=0.5)
    axes[1].set_title('Vibration', fontsize=13, fontweight='bold'); axes[1].grid(alpha=0.3)
    axes[2].scatter(df['timestamp'], df['temperature'], c=c, s=10, alpha=0.5)
    axes[2].set_title('Température', fontsize=13, fontweight='bold'); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_anomaly_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {output_dir}/fig1_anomaly_overview.png")

    # Fig 2 — Par moteur
    motors = sorted(df['motor_id'].unique())[:6]
    fig, axes = plt.subplots(len(motors), 2, figsize=(18, 4 * len(motors)))
    for i, mid in enumerate(motors):
        dm = df[df['motor_id'] == mid].sort_values('timestamp')
        ax = axes[i, 0]
        ax.fill_between(dm['timestamp'], dm['combined_score'], alpha=0.3, color='steelblue')
        ax.scatter(dm['timestamp'][dm['is_anomaly']], dm['combined_score'][dm['is_anomaly']],
                   color='red', s=25, zorder=5, label='Confirmée')
        ax.axhline(THRESHOLD, color='orange', linestyle='--', lw=1.5)
        ax.set_title(f'M{mid} — Score hybride V3', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax = axes[i, 1]
        ax.plot(dm['timestamp'], dm['health_score'], color='green', lw=1.2)
        ax.fill_between(dm['timestamp'], dm['health_score'], alpha=0.2, color='green')
        ax.axhline(50, color='orange', linestyle='--', lw=1)
        ax.axhline(30, color='red',    linestyle='--', lw=1)
        ax.set_title(f'M{mid} — Health Score V3', fontweight='bold')
        ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_per_motor.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {output_dir}/fig2_per_motor.png")

    # Fig 3 — Validation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].figure.colorbar(im, ax=axes[0])
    axes[0].set(xticks=[0,1], yticks=[0,1],
                xticklabels=['NORMAL','ANOMALIE'],
                yticklabels=['NORMAL','ANOMALIE'],
                title='Matrice confusion V3',
                ylabel='Réel', xlabel='Prédit')
    for ii in range(2):
        for jj in range(2):
            axes[0].text(jj, ii, f'{cm[ii,jj]:,}', ha='center', va='center',
                         color='white' if cm[ii,jj] > cm.max() / 2 else 'black',
                         fontsize=13, fontweight='bold')
    prec_c, rec_c, _ = precision_recall_curve(y_true, df['combined_score'])
    try:    auc_pr = float(np.trapezoid(prec_c[::-1], rec_c[::-1]))
    except: auc_pr = float(np.abs(np.sum(np.diff(rec_c[::-1]) * prec_c[::-1][:-1])))
    axes[1].plot(rec_c, prec_c, color='darkorange', lw=2, label=f'AUC-PR={auc_pr:.3f}')
    axes[1].set_title('Precision–Recall V3', fontweight='bold')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].hist(df[df['Alert_Status']=='NORMAL']['combined_score'],
                 bins=50, alpha=0.6, color='steelblue', label='NORMAL', density=True)
    axes[2].hist(df[df['Alert_Status']=='ALERT']['combined_score'],
                 bins=50, alpha=0.6, color='red', label='ALERT', density=True)
    axes[2].axvline(THRESHOLD, color='orange', linestyle='--', lw=2)
    axes[2].set_title('Distribution scores V3', fontweight='bold')
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {output_dir}/fig3_validation.png")

    # Fig 4 — Feature importance V3
    top15    = imp_df.head(15)
    v3_feats = {'vib_rms','vib_skewness','peak2peak',
                'spectral_entropy','shape_factor','impulse_factor'}
    colors_fi = ['#d62728' if r['feature'] in v3_feats else
                  '#ff7f0e' if i < 5 else '#1f77b4'
                  for i, (_, r) in enumerate(top15.iterrows())]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].barh(top15['feature'][::-1], top15['importance'][::-1], color=colors_fi[::-1])
    axes[0].set_title('Feature Importance V3 — rouge = nouvelles features AXE 3',
                      fontweight='bold', fontsize=11)
    axes[0].set_xlabel("Score d'importance"); axes[0].grid(axis='x', alpha=0.3)
    top8 = imp_df.head(8)
    x = np.arange(len(top8)); w = 0.35
    axes[1].bar(x-w/2, top8['val_normal'],   w, label='NORMAL',   color='steelblue', alpha=0.8)
    axes[1].bar(x+w/2, top8['val_anomalie'], w, label='ANOMALIE', color='crimson',   alpha=0.8)
    axes[1].set_title('NORMAL vs ANOMALIE — Top 8 features', fontweight='bold', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top8['feature'], rotation=30, ha='right', fontsize=9)
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {output_dir}/fig_feature_importance.png")

    # Fig 5 — Fleet analysis
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ft = df.groupby('motor_id')['temp_ratio_flotte'].mean().sort_values(ascending=False)
    ct = ['#d62728' if v > 1.5 else '#ff7f0e' if v > 1.2 else '#2ca02c' for v in ft.values]
    axes[0].bar(ft.index.astype(str), ft.values, color=ct)
    axes[0].axhline(1.0, color='black',  linestyle='--', lw=1.5, label='Médiane')
    axes[0].axhline(1.5, color='orange', linestyle='--', lw=1.5, label='+50%')
    axes[0].axhline(2.0, color='red',    linestyle='--', lw=1.5, label='Critique')
    axes[0].set_title('Ratio Température vs Médiane Flotte', fontweight='bold')
    axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)
    fv = df.groupby('motor_id')['vib_zscore_flotte'].mean().sort_values(ascending=False)
    cv = ['#d62728' if v > FLEET_SIGMA else '#ff7f0e' if v > 1.5 else '#2ca02c' for v in fv.values]
    axes[1].bar(fv.index.astype(str), fv.values, color=cv)
    axes[1].axhline(FLEET_SIGMA, color='red',    linestyle='--', lw=1.5, label=f'{FLEET_SIGMA}σ')
    axes[1].axhline(1.5,         color='orange', linestyle='--', lw=1.5, label='1.5σ')
    axes[1].set_title('Z-score Vibration vs Flotte', fontweight='bold')
    axes[1].legend(); axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_fleet_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {output_dir}/fig_fleet_analysis.png")

    # Fig 6 — Contribution IF vs LOF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(df['score_if'], df['score_lof'], c=y_true,
                    cmap='RdBu_r', s=6, alpha=0.4)
    axes[0].set_xlabel('Score IF'); axes[0].set_ylabel('Score LOF')
    axes[0].set_title('IF vs LOF — Complémentarité (V3)', fontweight='bold')
    axes[0].grid(alpha=0.3)
    corr = np.corrcoef(df['score_if'], df['score_lof'])[0, 1]
    axes[0].text(0.05, 0.95, f'corr = {corr:.3f}', transform=axes[0].transAxes,
                 fontsize=11, va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    labels = [f'IF ({W_IF:.0%})', f'LOF ({W_LOF:.0%})', f'Règles ({W_RULES:.0%})']
    sizes  = [W_IF, W_LOF, W_RULES]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, pctdistance=0.75)
    axes[1].set_title('Poids Ensemble V3\n(IF+LOF+Règles)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_ensemble_lof.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {output_dir}/fig_ensemble_lof.png")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print(" ÉTAPE 3 — DÉTECTION D'ANOMALIES (MODÈLE HYBRIDE V3)")
    print(" AXE 1: AUC corrigée | AXE 4: Walk-forward CV")
    print(" AXE 5: LOF réel     | AXE 6: SHAP par moteur")
    print("=" * 68)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] Fichier introuvable : {INPUT_CSV}"); return

    print(f"\n→ Chargement de {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    df = df.dropna(subset=['vibration', 'temperature'])
    print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")
    print(f"  NORMAL: {(df['Alert_Status']=='NORMAL').sum():,} | ALERT: {(df['Alert_Status']=='ALERT').sum():,}")

    print("\n→ [1/9] Calibration des seuils par moteur ...")
    seuils = calibrer_seuils(df)
    pd.DataFrame([{'motor_id': k, **v} for k, v in seuils.items()]).to_csv(SEUILS_CSV, index=False)

    print("\n→ [2/9] Features de dépassement ...")
    df = ajouter_features_depassement(df, seuils)
    for col in PARAMS:
        n = df[f'{col}_exceed'].sum()
        print(f"  {col:<15}: {n:,} ({n/len(df)*100:.1f}%)")

    print("\n→ [3/9] Fleet Analysis ...")
    df = ajouter_features_flotte(df)

    print("\n→ [4/9] Entraînement Isolation Forest ...")
    model_if, scaler, scores_if, feats_used, X_sc, pct_alert = entrainer_if(df, FEATURE_IF)
    print(f"  {len(feats_used)} features | contamination={pct_alert:.3f}")
    df['score_if'] = scores_if

    print("\n→ [5/9] AXE 5 — LOF (novelty=True sur données NORMALES) ...")
    scores_lof      = entrainer_lof(df, feats_used, X_sc)
    df['score_lof'] = scores_lof

    print("\n→ [6/9] Score hybride V3 (IF+LOF+Règles) ...")
    sc_rules  = df['severity_score'].values
    sc_fleet  = df['fleet_anomaly'].values * 0.10
    sc_hybrid = (W_IF * scores_if + W_LOF * scores_lof +
                 W_RULES * sc_rules + sc_fleet).clip(0, 1)

    df['score_rules']       = sc_rules
    df['score_fleet']       = sc_fleet
    df['combined_score']    = sc_hybrid
    df['anomaly_threshold'] = THRESHOLD
    df['is_anomaly']        = sc_hybrid >= THRESHOLD

    n_anom = df['is_anomaly'].sum()
    print(f"  Score = {W_IF:.0%}×IF + {W_LOF:.0%}×LOF + {W_RULES:.0%}×Règles + Fleet")
    print(f"  Anomalies brutes : {n_anom:,} / {len(df):,} ({n_anom/len(df)*100:.1f}%)")

    print(f"\n→ [7/9] Confirmation temporelle (fenêtre={CONFIRM_WINDOW}, min={CONFIRM_MIN}) ...")
    df = appliquer_confirmation_temporelle(df)
    df = calculer_score_confiance(df)

    print("\n→ [8/9] AXE 1+4 — Validation (AUC vs Alert_Status) + Walk-forward ...")
    y_true = (df['Alert_Status'] == 'ALERT').astype(int)
    y_pred = df['is_anomaly'].astype(int)
    y_raw  = df['is_anomaly_raw'].astype(int)

    auc_correct = roc_auc_score(y_true, df['combined_score'])
    f1          = f1_score(y_true, y_pred, zero_division=0)
    f1_raw      = f1_score(y_true, y_raw,  zero_division=0)
    prec        = precision_score(y_true, y_pred, zero_division=0)
    rec         = recall_score(y_true, y_pred, zero_division=0)
    acc         = (y_true == y_pred).mean()
    auc_if      = roc_auc_score(y_true, df['score_if'])
    auc_lof     = roc_auc_score(y_true, df['score_lof'])

    print(f"\n  ┌{'─'*64}┐")
    print(f"  │{'RÉSULTATS MODÈLE HYBRIDE V3':^64}│")
    print(f"  ├{'─'*64}┤")
    print(f"  │  AXE 1 — AUC corrigée (vs Alert_Status) : {auc_correct:.4f}{'':>16}│")
    print(f"  │           AUC IF seul                    : {auc_if:.4f}{'':>16}│")
    print(f"  │           AUC LOF seul (AXE 5)           : {auc_lof:.4f}{'':>16}│")
    print(f"  │  Précision : {prec*100:.1f}%  Recall : {rec*100:.1f}%  F1 : {f1:.4f}{'':>13}│")
    print(f"  │  Précision globale : {acc*100:.1f}%{'':>42}│")
    print(f"  ├{'─'*64}┤")
    print(f"  │  Ensemble V3 : {W_IF:.0%}×IF + {W_LOF:.0%}×LOF + {W_RULES:.0%}×Règles{'':>29}│")
    print(f"  └{'─'*64}┘\n")
    print(classification_report(y_true, y_pred, target_names=['NORMAL','ALERT'], digits=4))

    walk_forward_validation(df, feats_used)

    print("\n  Feature Importance V3 (Top 12) :")
    imp_df = calculer_feature_importance(df, feats_used)
    imp_df.to_csv('data/feature_importance.csv', index=False)
    v3_marker = {'vib_rms','vib_skewness','peak2peak',
                 'spectral_entropy','shape_factor','impulse_factor'}
    for _, row in imp_df.head(12).iterrows():
        bar    = '█' * int(row['importance'] * 20)
        marker = ' ◀ V3' if row['feature'] in v3_marker else ''
        print(f"  {row['feature']:<22} {bar:<20} {row['importance']:.4f}{marker}")

    print("\n→ [9/9] AXE 6 — SHAP par moteur ...")
    calculer_shap_par_moteur(df, feats_used, X_sc, model_if)

    print("\n  Résumé par moteur (V3) :")
    print(f"  {'M':>4} | {'Anom':>6} | {'Taux':>6} | {'Fleet':>6} | {'IF':>6} | {'LOF':>6} | Statut")
    print("  " + "─" * 60)
    for mid in sorted(df['motor_id'].unique()):
        dm      = df[df['motor_id'] == mid]
        n_an    = dm['is_anomaly'].sum()
        taux    = n_an / len(dm) * 100
        fl      = dm['fleet_anomaly'].mean() * 100
        avg_if  = dm['score_if'].mean()
        avg_lof = dm['score_lof'].mean()
        st = ('⚠ CRITIQUE' if taux > 30 else '⚠ ÉLEVÉ' if taux > 15
              else '~ MODÉRÉ' if taux > 5 else '✓ NORMAL')
        print(f"  {mid:>4} | {n_an:>6,} | {taux:>5.1f}% | {fl:>5.1f}% | "
              f"{avg_if:>6.3f} | {avg_lof:>6.3f} | {st}")

    print(f"\n→ Génération des figures ...")
    plot_all(df, imp_df, FIGURES_DIR)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Résultats : {OUTPUT_CSV}")
    print(f"✓ Feature importance : data/feature_importance.csv")
    print(f"✓ Walk-forward : data/validation_walkforward.csv")
    print(f"✓ SHAP global : data/shap_importance.csv")
    print(f"✓ SHAP moteurs : data/shap_per_motor.csv")

    # ══════════════════════════════════════════════
    #  AJOUT — ÉTAPE 6 : Diagnostic défauts roulements
    # ══════════════════════════════════════════════
    print("\n" + "=" * 68)
    print(" ÉTAPE 6 — DIAGNOSTIC DÉFAUTS DE ROULEMENTS")
    print("=" * 68)

    try:
        from step6_bearing_fault_diagnosis import run_bearing_diagnosis

        # Récupérer uniquement les moteurs détectés en anomalie
        moteurs_anomalie = (
            df[df['is_anomaly'] == True]['motor_id']
            .unique()
            .tolist()
        )
        print(f"\n  {len(moteurs_anomalie)} moteurs en anomalie transmis à step6 :")
        print(f"  {sorted(moteurs_anomalie)}")

        diagnoses = run_bearing_diagnosis(
            anomalous_only=True,
            anomalous_motor_ids=moteurs_anomalie,
        )

        # Sauvegarder le résumé diagnostic dans data/
        if diagnoses:
            rows_diag = []
            for d in diagnoses:
                rows_diag.append({
                    'motor_id'       : d.motor_id,
                    'fault_type'     : d.fault_type,
                    'fault_label_fr' : d.fault_label_fr,
                    'severity'       : round(d.severity, 4),
                    'severity_label' : d.severity_label,
                    'confidence'     : round(d.confidence, 4),
                    'location'       : d.location,
                    'ratio_bpfo'     : round(d.ratio_bpfo, 2),
                    'ratio_bpfi'     : round(d.ratio_bpfi, 2),
                    'ratio_bsf'      : round(d.ratio_bsf, 2),
                    'ratio_ftf'      : round(d.ratio_ftf, 2),
                    'days_to_action' : d.days_to_action,
                    'recommendation' : d.recommendation,
                })
            df_diag = pd.DataFrame(rows_diag)
            df_diag.to_csv('data/bearing_diagnosis.csv', index=False)
            print(f"\n✓ Diagnostic roulements : data/bearing_diagnosis.csv")
            print(f"✓ Rapport JSON          : bearing_diagnosis_report.json")
            print(f"✓ Figures diagnostic    : figures/fig_bearing_*.png")

    except ImportError:
        print("\n  ⚠ step6_bearing_fault_diagnosis.py introuvable.")
        print("    Placez le fichier step6 dans le même dossier que step3.")
    except Exception as e:
        print(f"\n  ⚠ Erreur step6 : {e}")
        print("    Le pipeline step3 est terminé avec succès malgré cette erreur.")

    print("=" * 68)


if __name__ == '__main__':
    main()
