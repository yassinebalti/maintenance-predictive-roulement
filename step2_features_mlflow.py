"""
======================================================
 ÉTAPE 2 — FEATURE ENGINEERING + MLFLOW TRACKING
 
 Modifications par rapport à step2_features.py original :
   ✅ Tracking des paramètres de fenêtrage
   ✅ Log des métriques (nb features, health score moyen)
   ✅ Log du DataFrame features comme artifact
   ✅ Tags de run (version, auteur)
======================================================
"""

import mlflow
import pandas as pd
import numpy as np
import os
import sys

# ── Ajouter le dossier parent dans le path ─────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlflow_config import (
    init_mlflow, log_params_dict, log_metrics_dict,
    log_dataframe_as_artifact, EXPERIMENT_FEATURES
)

# ── Importer le step2 original ─────────────────────
# On réutilise toutes les fonctions existantes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from step2_features import (
    clean_data, add_rolling_features,
    compute_fft_features, compute_health_score,
    spectral_entropy
)

INPUT_CSV  = 'data/01_raw_motor.csv'
OUTPUT_CSV = 'data/02_features_motor.csv'

# ── Paramètres (identiques au step2 original) ──────
WINDOW_SIZE     = 20
OVERLAP         = 0.5
FFT_POINTS      = 256
HEALTH_WEIGHTS  = {
    'temperature' : 0.35,
    'courant'     : 0.30,
    'vibration'   : 0.25,
    'acceleration': 0.10,
}


def main():
    print("=" * 62)
    print(" ÉTAPE 2 — FEATURES + MLFLOW TRACKING")
    print("=" * 62)

    # ── Initialiser MLflow ─────────────────────────
    init_mlflow(EXPERIMENT_FEATURES)

    with mlflow.start_run(run_name="features_engineering_v3") as run:

        print(f"\n[MLflow] Run ID : {run.info.run_id}")

        # ── 1. Log des paramètres ──────────────────
        mlflow.set_tags({
            "projet"    : "Maintenance Prédictive Roulements",
            "step"      : "02_features",
            "version"   : "V3",
            "axe"       : "AXE_3",
        })

        log_params_dict({
            "window_size"          : WINDOW_SIZE,
            "overlap"              : OVERLAP,
            "fft_points"           : FFT_POINTS,
            "weight_temperature"   : HEALTH_WEIGHTS['temperature'],
            "weight_courant"       : HEALTH_WEIGHTS['courant'],
            "weight_vibration"     : HEALTH_WEIGHTS['vibration'],
            "weight_acceleration"  : HEALTH_WEIGHTS['acceleration'],
            "input_file"           : INPUT_CSV,
        })

        # ── 2. Charger et traiter les données ──────
        print(f"\n→ Chargement {INPUT_CSV} ...")
        df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
        n_raw = len(df)
        n_motors = df['motor_id'].nunique()
        print(f"  {n_raw:,} lignes | {n_motors} moteurs")

        mlflow.log_metric("n_lignes_brutes", n_raw)
        mlflow.log_metric("n_moteurs", n_motors)

        # ── 3. Nettoyage ───────────────────────────
        df = clean_data(df)
        n_clean = len(df)
        n_supprime = n_raw - n_clean
        mlflow.log_metric("n_lignes_nettoyees", n_clean)
        mlflow.log_metric("n_lignes_supprimees", n_supprime)
        print(f"  Après nettoyage : {n_clean:,} ({n_supprime} supprimées)")

        # ── 4. Rolling features ────────────────────
        print("\n→ Rolling features ...")
        parts = []
        for mid, group in df.groupby('motor_id'):
            enriched = add_rolling_features(group.copy(), window=WINDOW_SIZE)
            enriched['motor_id'] = mid
            parts.append(enriched)
        df = pd.concat(parts, ignore_index=True)

        # ── 5. Health score ────────────────────────
        print("→ Health score ...")
        df = compute_health_score(df)

        # ── 6. Métriques features ──────────────────
        feature_cols = [c for c in df.columns
                        if c not in ['timestamp', 'motor_id', 'Alert_Status']]
        n_features = len(feature_cols)

        # AXE 3 — Features V3 spécifiques
        v3_features = ['vib_rms', 'vib_skewness', 'peak2peak',
                       'spectral_entropy', 'shape_factor', 'impulse_factor']
        v3_present = [f for f in v3_features if f in df.columns]

        # Métriques globales
        health_mean = df['health_score'].mean() if 'health_score' in df.columns else 0
        health_std  = df['health_score'].std()  if 'health_score' in df.columns else 0

        log_metrics_dict({
            "n_features_total"     : n_features,
            "n_features_v3"        : len(v3_present),
            "health_score_moyen"   : round(health_mean, 4),
            "health_score_std"     : round(health_std, 4),
            "taux_missing_apres"   : round(df.isnull().mean().mean(), 4),
        })

        # Métriques par moteur
        for mid, group in df.groupby('motor_id'):
            h = group['health_score'].mean() if 'health_score' in group else 0
            mlflow.log_metric(f"health_score_moteur_{int(mid)}", round(h, 4))

        print(f"\n  → {n_features} features totales ({len(v3_present)} nouvelles V3)")
        print(f"  → Health score moyen flotte : {health_mean:.4f}")

        # ── 7. Sauvegarder + log artifact ─────────
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n→ Sauvegardé : {OUTPUT_CSV}")

        log_dataframe_as_artifact(df.head(500), "features_sample.csv", "data")
        mlflow.log_artifact(OUTPUT_CSV, artifact_path="data")

        # ── 8. Résumé ──────────────────────────────
        print(f"\n[MLflow] ✅ Run terminé")
        print(f"[MLflow]    Experiment : {EXPERIMENT_FEATURES}")
        print(f"[MLflow]    Run ID     : {run.info.run_id}")
        print(f"[MLflow]    UI         : http://localhost:5000")
        print("=" * 62)


if __name__ == '__main__':
    main()
