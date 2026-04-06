"""
======================================================
 PIPELINE PRINCIPAL + MLFLOW
 Lance les 3 steps avec tracking MLflow intégré.
 
 Usage :
   python main_pipeline_mlflow.py          # tous les steps
   python main_pipeline_mlflow.py --step 3 # step 3 seul
   python main_pipeline_mlflow.py --compare # compare les runs

 Commandes MLflow utiles :
   mlflow ui --port 5000                   # lancer l'UI
   mlflow models list                      # voir les modèles
======================================================
"""

import mlflow
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlflow_config import (
    init_mlflow, EXPERIMENT_ANOMALIES, EXPERIMENT_RUL,
    EXPERIMENT_FEATURES, get_best_run,
    MODEL_IF_NAME, MODEL_LOF_NAME, MODEL_RUL_NAME
)


def run_step2():
    print("\n" + "━"*62)
    print(" ▶ STEP 2 — Feature Engineering")
    print("━"*62)
    from step2_features_mlflow import main as step2_main
    step2_main()


def run_step3():
    print("\n" + "━"*62)
    print(" ▶ STEP 3 — Détection Anomalies (IF + LOF)")
    print("━"*62)
    from step3_anomaly_mlflow import main as step3_main
    step3_main()


def run_step4():
    print("\n" + "━"*62)
    print(" ▶ STEP 4 — Prédiction RUL (Weibull Ensemble)")
    print("━"*62)
    from step4_rul_mlflow import main as step4_main
    step4_main()


def compare_runs():
    """Affiche un résumé comparatif des meilleurs runs MLflow."""
    print("\n" + "=" * 68)
    print(" COMPARAISON DES RUNS MLFLOW")
    print("=" * 68)

    mlflow.set_tracking_uri("http://localhost:5000")

    # Meilleur run anomalies
    best_anom = get_best_run(EXPERIMENT_ANOMALIES, metric="auc_ensemble")
    if best_anom:
        print(f"\n🏆 Meilleur run Anomalies (AUC Ensemble) :")
        print(f"   Run ID  : {best_anom['run_id']}")
        print(f"   AUC     : {best_anom['metric']:.4f}")
        print(f"   Params  : w_IF={best_anom['params'].get('w_isolation_forest')} "
              f"w_LOF={best_anom['params'].get('w_lof')} "
              f"w_rules={best_anom['params'].get('w_rules')}")

    # Meilleur run RUL
    best_rul = get_best_run(EXPERIMENT_RUL, metric="rul_moyen_flotte", ascending=True)
    if best_rul:
        print(f"\n🏆 Meilleur run RUL (RUL moyen minimal) :")
        print(f"   Run ID  : {best_rul['run_id']}")
        print(f"   RUL moy : {best_rul['metric']:.1f} jours")

    print("\n" + "=" * 68)
    print("→ Voir tous les runs : http://localhost:5000")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Maintenance Prédictive + MLflow"
    )
    parser.add_argument('--step', type=int, choices=[2, 3, 4],
                        help='Lancer un seul step (2, 3 ou 4)')
    parser.add_argument('--compare', action='store_true',
                        help='Comparer les meilleurs runs')
    args = parser.parse_args()

    if args.compare:
        compare_runs()
        return

    t_start = time.time()

    print("=" * 68)
    print(" PIPELINE MAINTENANCE PRÉDICTIVE — MLFLOW TRACKING")
    print(f" Tracking URI : http://localhost:5000")
    print("=" * 68)

    if args.step == 2:
        run_step2()
    elif args.step == 3:
        run_step3()
    elif args.step == 4:
        run_step4()
    else:
        # Tous les steps
        run_step2()
        run_step3()
        run_step4()

    elapsed = time.time() - t_start
    print(f"\n{'='*68}")
    print(f"✅ Pipeline terminé en {elapsed:.1f}s")
    print(f"📊 MLflow UI : http://localhost:5000")
    print(f"   → Expériences : {EXPERIMENT_FEATURES}")
    print(f"                   {EXPERIMENT_ANOMALIES}")
    print(f"                   {EXPERIMENT_RUL}")
    print(f"   → Modèles Registry :")
    print(f"       • {MODEL_IF_NAME}")
    print(f"       • {MODEL_LOF_NAME}")
    print(f"       • {MODEL_RUL_NAME}")
    print("=" * 68)


if __name__ == '__main__':
    main()
