"""
========================================================
 ÉTAPE 3 — DÉTECTION D'ANOMALIES (NON SUPERVISÉE)
 Entrée : data/02_features_motor.csv
 Sortie : data/03_anomalies.csv

 Modèle : Ensemble Isolation Forest + LOF
   - Entraînement sur les données "normales" (Alert_Status=NORMAL)
   - Détection sur toutes les données
   - Score combiné = moyenne des deux modèles
   - Validation contre Alert_Status réel (sans l'utiliser à l'entraînement)
========================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────
INPUT_CSV      = 'data/02_features_motor.csv'
OUTPUT_CSV     = 'data/03_anomalies.csv'
FIGURES_DIR    = 'figures'
CONTAMINATION  = 0.10   # ~10% d'anomalies estimées (calibré sur 33% réel pour prudence)
N_ESTIMATORS   = 300    # Isolation Forest : nb d'arbres
LOF_NEIGHBORS  = 20     # LOF : nb de voisins
RANDOM_STATE   = 42
# ──────────────────────────────────────────────────────

FEATURE_COLS = [
    'temperature', 'courant', 'vibration', 'acceleration', 'vitesse', 'cosphi',
    'vib_energy', 'vib_energy_mean', 'vib_mean', 'vib_std', 'vib_kurt',
    'vib_max', 'crest_factor', 'temp_mean', 'temp_std', 'temp_trend',
    'courant_mean', 'courant_std', 'envelope', 'envelope_mean',
    'fft_max_amp', 'fft_dominant_freq', 'health_score'
]


def train_isolation_forest(X_train_scaled: np.ndarray) -> IsolationForest:
    model = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,
        max_samples   = 'auto',
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    model.fit(X_train_scaled)
    return model


def train_lof(X_scaled: np.ndarray) -> LocalOutlierFactor:
    """LOF est transductif → fit_predict sur tout le dataset."""
    model = LocalOutlierFactor(
        n_neighbors   = LOF_NEIGHBORS,
        contamination = CONTAMINATION,
        n_jobs        = -1,
        novelty       = False,
    )
    model.fit_predict(X_scaled)   # utilisé pour le score uniquement
    return model


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalise les scores en [0, 1] où 1 = le plus anomal."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    norm = (scores - lo) / (hi - lo)
    return norm


def plot_anomaly_overview(df: pd.DataFrame, output_dir: str):
    """Vue d'ensemble des anomalies par moteur."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Fig 1 : Score combiné dans le temps ──────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Panel 1 : Score combiné global
    ax = axes[0]
    colors = np.where(df['is_anomaly'], 'red', 'steelblue')
    ax.scatter(df['timestamp'], df['combined_score'], c=colors, s=15, alpha=0.6)
    ax.axhline(y=df['anomaly_threshold'].iloc[0], color='orange',
               linestyle='--', linewidth=2, label='Seuil détection')
    ax.set_title('Score d\'anomalie combiné (IF + LOF) — tous moteurs', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score anomalie [0–1]\n(1 = très anomal)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2 : Vibration colorée par anomalie
    ax = axes[1]
    ax.scatter(df['timestamp'], df['vibration'], c=colors, s=15, alpha=0.6)
    ax.set_title('Vibration — rouge = anomalie détectée', fontsize=13, fontweight='bold')
    ax.set_ylabel('Vibration')
    ax.grid(alpha=0.3)

    # Panel 3 : Température
    ax = axes[2]
    ax.scatter(df['timestamp'], df['temperature'], c=colors, s=15, alpha=0.6)
    ax.set_title('Température — rouge = anomalie détectée', fontsize=13, fontweight='bold')
    ax.set_ylabel('Température (°C)')
    ax.set_xlabel('Timestamp')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_anomaly_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_per_motor(df: pd.DataFrame, output_dir: str):
    """Graphique anomalies et health_score pour les 6 premiers moteurs."""
    motors = sorted(df['motor_id'].unique())[:6]
    fig, axes = plt.subplots(len(motors), 2, figsize=(18, 4 * len(motors)))

    for i, mid in enumerate(motors):
        dm = df[df['motor_id'] == mid].sort_values('timestamp')
        colors = np.where(dm['is_anomaly'], 'red', 'steelblue')

        # Score anomalie
        ax = axes[i, 0]
        ax.fill_between(dm['timestamp'], dm['combined_score'],
                        alpha=0.35, color='steelblue')
        ax.scatter(dm['timestamp'][dm['is_anomaly']],
                   dm['combined_score'][dm['is_anomaly']],
                   color='red', s=30, zorder=5, label='Anomalie')
        ax.axhline(dm['anomaly_threshold'].iloc[0], color='orange',
                   linestyle='--', linewidth=1.5)
        ax.set_title(f'Moteur {mid} — Score anomalie', fontweight='bold')
        ax.set_ylabel('Score [0–1]')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Health score
        ax = axes[i, 1]
        ax.plot(dm['timestamp'], dm['health_score'], color='green',
                linewidth=1.2, alpha=0.8)
        ax.fill_between(dm['timestamp'], dm['health_score'],
                        alpha=0.2, color='green')
        ax.axhline(50, color='orange', linestyle='--', linewidth=1,
                   label='Seuil prudence (50)')
        ax.axhline(30, color='red', linestyle='--', linewidth=1,
                   label='Seuil critique (30)')
        ax.set_title(f'Moteur {mid} — Health Score', fontweight='bold')
        ax.set_ylabel('Health (0–100)')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_per_motor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_validation(df: pd.DataFrame, output_dir: str):
    """Matrice de confusion détection vs Alert_Status réel."""
    y_true = (df['Alert_Status'] == 'ALERT').astype(int)
    y_pred = df['is_anomaly'].astype(int)

    cm = confusion_matrix(y_true, y_pred)
    labels = ['NORMAL', 'ANOMALIE']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Matrice de confusion
    ax = axes[0]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=labels, yticklabels=labels,
           title='Matrice de confusion\n(détection vs Alert_Status réel)',
           ylabel='Réel', xlabel='Prédit')
    for ii in range(2):
        for jj in range(2):
            ax.text(jj, ii, f'{cm[ii, jj]:,}',
                    ha='center', va='center',
                    color='white' if cm[ii, jj] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')

    # Courbe Precision-Recall
    ax = axes[1]
    if len(np.unique(y_true)) > 1:
        precision, recall, thresholds = precision_recall_curve(
            y_true, df['combined_score'])
        ax.plot(recall, precision, marker='.', color='darkorange',
                linewidth=2, label=f'AUC = {roc_auc_score(y_true, df["combined_score"]):.3f}')
        ax.set_title('Courbe Precision–Recall', fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def main():
    print("=" * 55)
    print(" ÉTAPE 3 — DÉTECTION D'ANOMALIES (NON SUPERVISÉE)")
    print("=" * 55)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] Fichier introuvable : {INPUT_CSV}")
        print("→ Lancez d'abord : python step2_features.py")
        return

    print(f"\n→ Chargement de {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")

    # ── Sélection des features disponibles ────────
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\n  Features utilisées ({len(available_features)}) :")
    print("  " + ", ".join(available_features))

    X = df[available_features].fillna(0).values

    # ── Normalisation ─────────────────────────────
    print("\n→ Normalisation StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Entraînement sur données NORMALES (non supervisé) ─
    # On entraîne sur les lignes étiquetées NORMAL pour apprendre
    # le profil normal, puis on prédit sur TOUT le dataset.
    normal_mask = df['Alert_Status'] == 'NORMAL'
    print(f"\n→ Entraînement Isolation Forest sur {normal_mask.sum():,} lignes normales ...")
    X_train = X_scaled[normal_mask]

    # Isolation Forest
    model_if = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,
        max_samples   = min(256, len(X_train)),
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    model_if.fit(X_train)
    scores_if_raw = model_if.decision_function(X_scaled)
    # Inverser : IF donne des scores négatifs pour anomalies
    scores_if = normalize_scores(-scores_if_raw)

    # LOF (transductif, sur tout le dataset)
    print(f"\n→ Calcul LOF sur {len(df):,} lignes ...")
    lof = LocalOutlierFactor(
        n_neighbors   = LOF_NEIGHBORS,
        contamination = CONTAMINATION,
        n_jobs        = -1,
        novelty       = False,
    )
    lof.fit_predict(X_scaled)
    scores_lof_raw = lof.negative_outlier_factor_
    scores_lof = normalize_scores(-scores_lof_raw)

    # ── Score combiné (ensemble) ─────────────────
    scores_combined = 0.6 * scores_if + 0.4 * scores_lof

    # Seuil automatique (percentile 1-contamination)
    threshold = np.percentile(scores_combined, (1 - CONTAMINATION) * 100)
    is_anomaly = scores_combined >= threshold

    df['score_if']        = scores_if
    df['score_lof']       = scores_lof
    df['combined_score']  = scores_combined
    df['anomaly_threshold'] = threshold
    df['is_anomaly']      = is_anomaly

    # ── Résultats ─────────────────────────────────
    n_anomalies = is_anomaly.sum()
    print(f"\n✓ Anomalies détectées : {n_anomalies:,} / {len(df):,} "
          f"({n_anomalies / len(df) * 100:.1f}%)")
    print(f"  Seuil de détection : {threshold:.4f}")

    # ── Validation contre Alert_Status réel ───────
    print("\n→ Validation contre Alert_Status réel :")
    y_true = (df['Alert_Status'] == 'ALERT').astype(int)
    y_pred = is_anomaly.astype(int)

    print(classification_report(y_true, y_pred,
                                target_names=['NORMAL', 'ANOMALIE'],
                                digits=3))
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, scores_combined)
        print(f"  AUC ROC : {auc:.4f}")

    # ── Top anomalies par moteur ───────────────────
    print("\n→ Top 3 anomalies par moteur :")
    top = (df[df['is_anomaly']]
           .groupby('motor_id')
           .apply(lambda g: g.nlargest(3, 'combined_score'))
           .reset_index(drop=True))
    cols_show = ['motor_id', 'timestamp', 'combined_score',
                 'temperature', 'vibration', 'health_score', 'Alert_Status']
    cols_show = [c for c in cols_show if c in top.columns]
    print(top[cols_show].to_string(index=False))

    # ── Graphiques ────────────────────────────────
    print(f"\n→ Génération des graphiques dans {FIGURES_DIR}/ ...")
    plot_anomaly_overview(df, FIGURES_DIR)
    plot_per_motor(df, FIGURES_DIR)
    plot_validation(df, FIGURES_DIR)

    # ── Sauvegarde ────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Résultats sauvegardés : {OUTPUT_CSV}")
    print("=" * 55)


if __name__ == '__main__':
    main()
