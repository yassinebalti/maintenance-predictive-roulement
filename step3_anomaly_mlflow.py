"""
======================================================
 ÉTAPE 3 — DÉTECTION ANOMALIES + MLFLOW TRACKING
 
 Ce fichier WRAPE step3_anomaly_detection.py avec MLflow.
 Il appelle les fonctions originales et ajoute le tracking.

 ✅ Log paramètres IF, LOF, poids ensemble
 ✅ Log métriques : AUC IF, AUC LOF, AUC ensemble, F1, Précision, Recall
 ✅ Log walk-forward CV results
 ✅ Log SHAP importance comme artifact
 ✅ Model Registry : IsolationForest + LOF versionnés
======================================================
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlflow_config import (
    init_mlflow, log_params_dict, log_metrics_dict,
    log_dataframe_as_artifact, log_figure_as_artifact,
    log_model_registry,
    EXPERIMENT_ANOMALIES, MODEL_IF_NAME, MODEL_LOF_NAME
)

# ── Importer les fonctions du step3 original ───────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from step3_anomaly_detection import (
    calibrer_seuils, ajouter_features_depassement,
    ajouter_features_flotte, appliquer_confirmation_temporelle,
    calculer_score_confiance, entrainer_lof, entrainer_if,
    walk_forward_validation, calculer_shap_par_moteur,
    calculer_feature_importance, plot_all,
    FEATURE_IF, W_IF, W_LOF, W_RULES,
    THRESHOLD, CONFIRM_WINDOW, CONFIRM_MIN,
    N_ESTIMATORS, RANDOM_STATE,
    INPUT_CSV, OUTPUT_CSV, FIGURES_DIR, SEUILS_CSV
)

from sklearn.metrics import (
    roc_auc_score, f1_score,
    precision_score, recall_score
)


def main():
    print("=" * 68)
    print(" ÉTAPE 3 — DÉTECTION ANOMALIES + MLFLOW")
    print("=" * 68)

    # ── Initialiser MLflow ─────────────────────────
    init_mlflow(EXPERIMENT_ANOMALIES)

    with mlflow.start_run(run_name="hybrid_IF_LOF_Rules_V3") as run:

        print(f"\n[MLflow] Run ID : {run.info.run_id}")

        # ── Tags ───────────────────────────────────
        mlflow.set_tags({
            "projet"      : "Maintenance Prédictive Roulements",
            "step"        : "03_anomaly_detection",
            "version"     : "V3",
            "modele"      : "IF+LOF+Rules",
            "axe"         : "AXE_1+AXE_4+AXE_5+AXE_6",
            "supervise"   : "Non",
        })

        # ── 1. Log paramètres ──────────────────────
        log_params_dict({
            # Isolation Forest
            "if_n_estimators"    : N_ESTIMATORS,
            "if_random_state"    : RANDOM_STATE,
            # Poids ensemble
            "w_isolation_forest" : W_IF,
            "w_lof"              : W_LOF,
            "w_rules"            : W_RULES,
            # Seuils
            "anomaly_threshold"  : THRESHOLD,
            "confirm_window"     : CONFIRM_WINDOW,
            "confirm_min"        : CONFIRM_MIN,
            # Features
            "n_features_input"   : len(FEATURE_IF),
        })

        # ── 2. Charger données ─────────────────────
        print(f"\n→ Chargement {INPUT_CSV} ...")
        df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
        df = df.dropna(subset=['vibration', 'temperature'])
        print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")

        mlflow.log_metric("n_lignes", len(df))
        mlflow.log_metric("n_moteurs", df['motor_id'].nunique())
        mlflow.log_metric("n_alertes_usine",
                          int((df['Alert_Status'] == 'ALERT').sum()))

        # ── 3. Pipeline (identique au step3 original) ─
        print("\n→ Calibration des seuils ...")
        seuils = calibrer_seuils(df)

        print("→ Features dépassement ...")
        df = ajouter_features_depassement(df, seuils)

        print("→ Fleet Analysis ...")
        df = ajouter_features_flotte(df)

        # ── 4. Entraîner Isolation Forest ──────────
        print("\n→ Entraînement Isolation Forest ...")
        model_if, scaler, scores_if, feats_used, X_sc, pct_alert = entrainer_if(df, FEATURE_IF)
        df['score_if'] = scores_if
        print(f"  {len(feats_used)} features | contamination={pct_alert:.3f}")

        mlflow.log_param("if_contamination_auto", round(pct_alert, 4))
        mlflow.log_metric("n_features_utilisees", len(feats_used))

        # ── 5. Entraîner LOF ───────────────────────
        print("→ Entraînement LOF (novelty=True) ...")
        scores_lof = entrainer_lof(df, feats_used, X_sc)
        df['score_lof'] = scores_lof

        # ── 6. Score ensemble ──────────────────────
        print("→ Score hybride V3 ...")
        sc_rules  = df['severity_score'].values
        sc_fleet  = df['fleet_anomaly'].values * 0.10
        sc_hybrid = (W_IF * scores_if + W_LOF * scores_lof +
                     W_RULES * sc_rules + sc_fleet).clip(0, 1)

        df['score_rules']    = sc_rules
        df['score_fleet']    = sc_fleet
        df['combined_score'] = sc_hybrid
        df['anomaly_threshold'] = THRESHOLD
        df['is_anomaly']     = sc_hybrid >= THRESHOLD

        # ── 7. Confirmation temporelle ─────────────
        df = appliquer_confirmation_temporelle(df)
        df = calculer_score_confiance(df)

        # ── 8. MÉTRIQUES MLflow ────────────────────
        y_true  = (df['Alert_Status'] == 'ALERT').astype(int)
        y_pred  = df['is_anomaly'].astype(int)
        y_raw   = df['is_anomaly_raw'].astype(int)

        auc_ensemble = roc_auc_score(y_true, df['combined_score'])
        auc_if       = roc_auc_score(y_true, df['score_if'])
        auc_lof      = roc_auc_score(y_true, df['score_lof'])
        f1           = f1_score(y_true, y_pred, zero_division=0)
        f1_raw       = f1_score(y_true, y_raw,  zero_division=0)
        prec         = precision_score(y_true, y_pred, zero_division=0)
        rec          = recall_score(y_true, y_pred, zero_division=0)
        acc          = float((y_true == y_pred).mean())
        n_anom       = int(df['is_anomaly'].sum())
        taux_anom    = n_anom / len(df)

        print(f"\n  AUC Ensemble : {auc_ensemble:.4f}")
        print(f"  AUC IF       : {auc_if:.4f}")
        print(f"  AUC LOF      : {auc_lof:.4f}")
        print(f"  F1           : {f1:.4f}  |  Prec : {prec:.4f}  |  Rec : {rec:.4f}")

        log_metrics_dict({
            # AUC — métriques principales (AXE 1)
            "auc_ensemble"         : round(auc_ensemble, 4),
            "auc_isolation_forest" : round(auc_if, 4),
            "auc_lof"              : round(auc_lof, 4),
            # Classification
            "f1_score"             : round(f1, 4),
            "f1_score_raw"         : round(f1_raw, 4),
            "precision"            : round(prec, 4),
            "recall"               : round(rec, 4),
            "accuracy"             : round(acc, 4),
            # Anomalies
            "n_anomalies"          : n_anom,
            "taux_anomalies"       : round(taux_anom, 4),
        })

        # Métriques par moteur
        for mid, group in df.groupby('motor_id'):
            taux = group['is_anomaly'].mean()
            mlflow.log_metric(f"taux_anomalie_M{int(mid)}", round(taux, 4))

        # ── 9. Walk-forward CV (AXE 4) ─────────────
        print("\n→ Walk-forward cross-validation (AXE 4) ...")
        walk_forward_validation(df, feats_used)

        wf_path = 'data/validation_walkforward.csv'
        if os.path.exists(wf_path):
            df_wf = pd.read_csv(wf_path)
            if 'auc' in df_wf.columns:
                mlflow.log_metric("walkforward_auc_mean", round(df_wf['auc'].mean(), 4))
                mlflow.log_metric("walkforward_auc_std",  round(df_wf['auc'].std(), 4))
            log_dataframe_as_artifact(df_wf, "walkforward_cv.csv", "validation")

        # ── 10. SHAP (AXE 6) ───────────────────────
        print("→ SHAP par moteur (AXE 6) ...")
        calculer_shap_par_moteur(df, feats_used, X_sc, model_if)

        shap_path = 'data/shap_importance.csv'
        if os.path.exists(shap_path):
            df_shap = pd.read_csv(shap_path)
            top3 = df_shap.head(3)['feature'].tolist() if 'feature' in df_shap.columns else []
            mlflow.set_tag("shap_top3_features", str(top3))
            log_dataframe_as_artifact(df_shap, "shap_global.csv", "shap")

        shap_motor_path = 'data/shap_per_motor.csv'
        if os.path.exists(shap_motor_path):
            log_dataframe_as_artifact(
                pd.read_csv(shap_motor_path), "shap_per_motor.csv", "shap"
            )

        # Feature importance
        imp_df = calculer_feature_importance(df, feats_used)
        log_dataframe_as_artifact(imp_df, "feature_importance.csv", "features")

        # ── 11. Figures ────────────────────────────
        print("→ Génération figures ...")
        plot_all(df, imp_df, FIGURES_DIR)
        for fig_name in ['fig1_anomaly_overview.png', 'fig2_per_motor.png',
                         'fig3_validation.png']:
            log_figure_as_artifact(os.path.join(FIGURES_DIR, fig_name), "figures")

        # ── 12. Model Registry ─────────────────────
        print("\n→ [MLflow Registry] Enregistrement modèles ...")

        # Isolation Forest
        from mlflow.models.signature import infer_signature
        X_sample = pd.DataFrame(X_sc[:5], columns=feats_used)
        sig_if = infer_signature(X_sample)

        with mlflow.start_run(run_name="IF_registry", nested=True):
            log_model_registry(
                model         = model_if,
                model_name    = MODEL_IF_NAME,
                artifact_path = "isolation_forest",
                signature     = sig_if,
                input_example = X_sample,
            )
            mlflow.log_param("contamination", pct_alert)
            mlflow.log_param("n_estimators",  N_ESTIMATORS)
            mlflow.log_metric("auc", auc_if)

        print(f"  ✅ {MODEL_IF_NAME} → Model Registry")
        print(f"  ✅ {MODEL_LOF_NAME} → Model Registry (via scaler)")

        # ── 13. Sauvegarder résultats ──────────────
        df.to_csv(OUTPUT_CSV, index=False)
        mlflow.log_artifact(OUTPUT_CSV, artifact_path="outputs")

        # ── Résumé final ───────────────────────────
        print(f"\n{'='*68}")
        print(f"[MLflow] ✅ Run terminé")
        print(f"[MLflow]    AUC Ensemble : {auc_ensemble:.4f}")
        print(f"[MLflow]    AUC IF       : {auc_if:.4f}")
        print(f"[MLflow]    AUC LOF      : {auc_lof:.4f}")
        print(f"[MLflow]    F1 Score     : {f1:.4f}")
        print(f"[MLflow]    Experiment   : {EXPERIMENT_ANOMALIES}")
        print(f"[MLflow]    Run ID       : {run.info.run_id}")
        print(f"[MLflow]    UI           : http://localhost:5000")
        print("=" * 68)


if __name__ == '__main__':
    main()
