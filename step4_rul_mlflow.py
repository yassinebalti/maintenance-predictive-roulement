"""
======================================================
 ÉTAPE 4 - PREDICTION RUL + MLFLOW TRACKING
 AXE 2 : Ensemble Poly+Exp+Weibull + IC 80%
 AXE 7 : CUSUM detection rupture de tendance
======================================================
"""

import mlflow
import pandas as pd
import numpy as np
import os
import sys
import pickle
from mlflow.pyfunc import PythonModel

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mlflow_config import (
    init_mlflow, log_params_dict, log_metrics_dict,
    log_dataframe_as_artifact, log_figure_as_artifact,
    EXPERIMENT_RUL, MODEL_RUL_NAME
)

from step4_rul_prediction import (
    compute_degradation_index, estimate_rul_v3, detect_cusum,
    plot_rul_all_motors, plot_risk_dashboard, plot_cusum,
    INPUT_CSV, OUTPUT_CSV, FIGURES_DIR
)

W_POLY     = 0.40
W_EXP      = 0.30
W_WEIBULL  = 0.30
CUSUM_K    = 0.5
CUSUM_H    = 4.0
RUL_CRITIQUE = 7
RUL_ELEVE    = 14
RUL_MODERATE = 30


class RULModel(PythonModel):
    """Modèle custom pour RUL dans MLflow Registry"""
    
    def load_context(self, context):
        with open(context.artifacts["rul_params"], "rb") as f:
            self.params = pickle.load(f)
    
    def predict(self, context, model_input):
        rul_results = []
        for mid, group in model_input.groupby('motor_id'):
            r = estimate_rul_v3(
                group,
                w_poly=self.params['w_poly'],
                w_exp=self.params['w_exp'],
                w_weibull=self.params['w_weibull']
            )
            rul_results.append(r)
        return pd.DataFrame(rul_results)


def main():
    print("=" * 62)
    print(" ETAPE 4 - PREDICTION RUL + MLFLOW")
    print("=" * 62)

    init_mlflow(EXPERIMENT_RUL)

    with mlflow.start_run(run_name="rul_ensemble_weibull_v3") as run:

        print(f"\n[MLflow] Run ID : {run.info.run_id}")

        mlflow.set_tags({
            "projet"  : "Maintenance Predictive Roulements",
            "step"    : "04_rul_prediction",
            "version" : "V3",
            "axe"     : "AXE_2+AXE_7",
        })

        log_params_dict({
            "w_poly"             : W_POLY,
            "w_exp"              : W_EXP,
            "w_weibull"          : W_WEIBULL,
            "ic_niveau"          : 0.80,
            "cusum_k"            : CUSUM_K,
            "cusum_h"            : CUSUM_H,
            "rul_seuil_critique" : RUL_CRITIQUE,
            "rul_seuil_eleve"    : RUL_ELEVE,
        })

        # Charger données
        print(f"\n-> Chargement {INPUT_CSV} ...")
        df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
        n_motors = df['motor_id'].nunique()
        print(f"  {len(df):,} lignes | {n_motors} moteurs")
        mlflow.log_metric("n_lignes", len(df))
        mlflow.log_metric("n_moteurs", n_motors)

        # Calcul DI
        print("\n-> Calcul indice de degradation (DI) ...")
        parts = []
        for mid, group in df.groupby('motor_id'):
            enriched = compute_degradation_index(group.copy())
            enriched['motor_id'] = mid
            parts.append(enriched)
        df = pd.concat(parts, ignore_index=True)
        di_moyen = df.groupby('motor_id')['degradation_index'].last().mean()
        mlflow.log_metric("di_moyen_flotte", round(float(di_moyen), 4))

        # RUL Ensemble
        print("\n-> RUL Ensemble (Poly+Exp+Weibull) ...")
        rul_results = []
        for mid, group in df.groupby('motor_id'):
            r = estimate_rul_v3(group)
            rul_results.append(r)

        rul_values, ic_widths, betas, dis = [], [], [], []
        for r in rul_results:
            mid  = int(r['motor_id'])
            rul  = r['rul_ensemble']
            ic_w = r['rul_high'] - r['rul_low']
            beta = r['weibull_beta']
            di   = r['current_di']
            rul_values.append(rul)
            ic_widths.append(ic_w)
            betas.append(beta)
            dis.append(di)
            mlflow.log_metric(f"rul_jours_M{mid}",         round(rul, 1))
            mlflow.log_metric(f"rul_ic_width_M{mid}",      round(ic_w, 1))
            mlflow.log_metric(f"weibull_beta_M{mid}",      round(beta, 3))
            mlflow.log_metric(f"degradation_index_M{mid}", round(di, 4))

        log_metrics_dict({
            "rul_moyen_flotte"   : round(float(np.mean(rul_values)), 1),
            "rul_min_flotte"     : round(float(np.min(rul_values)),  1),
            "rul_max_flotte"     : round(float(np.max(rul_values)),  1),
            "ic_width_moyen"     : round(float(np.mean(ic_widths)),  1),
            "weibull_beta_moyen" : round(float(np.mean(betas)),      3),
            "di_moyen"           : round(float(np.mean(dis)),        4),
        })

        df_rul = pd.DataFrame(rul_results)
        dist_risque = df_rul['risk_level'].value_counts().to_dict()
        for niveau, count in dist_risque.items():
            mlflow.log_metric(f"n_moteurs_{niveau.lower()}", int(count))

        n_critique = int((df_rul['risk_level'] == 'CRITIQUE').sum())
        mlflow.log_metric("n_moteurs_critiques", n_critique)

        print("\n  Distribution risques :")
        for niv, cnt in dist_risque.items():
            print(f"    {niv:<12} : {cnt} moteur(s)")

        # CUSUM
        print("\n-> CUSUM detection rupture ...")
        cusum_results = []
        for mid, group in df.groupby('motor_id'):
            cr = detect_cusum(group)
            cusum_results.append(cr)
        df_cusum  = pd.DataFrame(cusum_results)
        n_alarmed = int(df_cusum['cusum_alarm'].sum())
        mlflow.log_metric("n_cusum_alarmes",   n_alarmed)
        mlflow.log_metric("taux_cusum_alarme", round(n_alarmed / len(df_cusum), 4))
        for _, row in df_cusum.iterrows():
            mlflow.log_metric(f"cusum_alarm_M{int(row['motor_id'])}", int(row['cusum_alarm']))
        print(f"  {n_alarmed}/{len(df_cusum)} moteurs avec rupture detectee")

        # Artifacts CSV
        df_cusum.to_csv('data/cusum_changepoints.csv', index=False)
        log_dataframe_as_artifact(df_cusum, "cusum_changepoints.csv", "cusum")
        log_dataframe_as_artifact(df_rul,   "rul_summary.csv",        "rul")

        # Figures
        print("-> Generation figures ...")
        df_rul_merge = df_rul[[
            'motor_id', 'rul_days', 'rul_ensemble',
            'rul_low', 'rul_high', 'risk_level', 'weibull_beta', 'current_di'
        ]].rename(columns={'current_di': 'current_di_summary'})
        df_merged = df.merge(df_rul_merge, on='motor_id', how='left')
        plot_rul_all_motors(df_merged, rul_results, FIGURES_DIR)
        plot_risk_dashboard(rul_results, FIGURES_DIR)
        plot_cusum(df_merged, cusum_results, FIGURES_DIR)
        for fig_name in ['fig4_rul_all_motors.png', 'fig5_risk_dashboard.png',
                         'fig7_cusum_changepoints.png']:
            log_figure_as_artifact(os.path.join(FIGURES_DIR, fig_name), "figures")

        # ── Model Registry ─────────────────────────────────────────
        print("\n-> [MLflow Registry] Enregistrement modele RUL ...")

        # Sauvegarde params pickle
        best_rul = max(rul_results, key=lambda r: r['weibull_beta'])
        pkl_path = os.path.join("data", "rul_model.pkl")

        with open(pkl_path, "wb") as f_pkl:
            pickle.dump({
                "type"        : "RUL_Ensemble_Weibull",
                "w_poly"      : W_POLY,
                "w_exp"       : W_EXP,
                "w_weibull"   : W_WEIBULL,
                "cusum_k"     : CUSUM_K,
                "cusum_h"     : CUSUM_H,
                "best_motor"  : int(best_rul['motor_id']),
                "best_beta"   : best_rul['weibull_beta'],
            }, f_pkl)

        # Nested run pour registry
        with mlflow.start_run(run_name="RUL_registry", nested=True) as nested:
            mlflow.log_params({
                "modele"    : "Weibull+Poly+Exp",
                "w_poly"    : W_POLY,
                "w_exp"     : W_EXP,
                "w_weibull" : W_WEIBULL,
            })
            mlflow.log_metrics({
                "rul_moyen"   : round(float(np.mean(rul_values)), 1),
                "n_critiques" : n_critique,
            })

            # Log le modèle custom avec pyfunc
            artifacts = {"rul_params": pkl_path}
            mlflow.pyfunc.log_model(
                artifact_path="rul_model",
                python_model=RULModel(),
                artifacts=artifacts
            )

            nested_id = nested.info.run_id

        # Register avec URI du nested (où le modèle est logged)
        model_uri = f"runs:/{nested_id}/rul_model"
        mlflow.register_model(model_uri, MODEL_RUL_NAME)

        # Sauvegarder resultats
        df_save = df_merged.copy()
        for col in df_save.select_dtypes(include='object').columns:
            df_save[col] = df_save[col].astype(str)
        df_save.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

        print("\n" + "="*62)
        print("[MLflow] OK Run termine")
        print(f"[MLflow]    RUL moyen flotte : {np.mean(rul_values):.1f} jours")
        print(f"[MLflow]    Moteurs CRITIQUE : {n_critique}")
        print(f"[MLflow]    CUSUM alarmes    : {n_alarmed}/{len(df_cusum)}")
        print(f"[MLflow]    Run ID           : {run.info.run_id}")
        print(f"[MLflow]    UI               : http://localhost:5000")
        print("=" * 62)


if __name__ == '__main__':
    main()