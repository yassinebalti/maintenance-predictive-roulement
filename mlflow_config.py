"""
======================================================
 MLFLOW CONFIG — Maintenance Prédictive
 Configuration partagée pour tous les steps
 
 Usage : from mlflow_config import init_mlflow, log_model_registry
======================================================
"""

import mlflow
import mlflow.sklearn
import os

# ── URIs ───────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI", "./mlruns")

# ── Noms des expériences (une par step) ────────────
EXPERIMENT_FEATURES   = "01_features_engineering"
EXPERIMENT_ANOMALIES  = "02_anomaly_detection"
EXPERIMENT_RUL        = "03_rul_prediction"

# ── Noms des modèles dans le Registry ──────────────
MODEL_IF_NAME   = "IsolationForest_Maintenance"
MODEL_LOF_NAME  = "LOF_Maintenance"
MODEL_RUL_NAME  = "RUL_Weibull_Maintenance"


def init_mlflow(experiment_name: str) -> str:
    """
    Initialise MLflow et retourne l'experiment_id.
    À appeler au début de chaque step.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"[MLflow] Expérience : '{experiment_name}'")
    print(f"[MLflow] Tracking URI : {MLFLOW_TRACKING_URI}")
    return experiment.experiment_id


def log_model_registry(model, model_name: str, artifact_path: str,
                        signature=None, input_example=None,
                        tags: dict = None):
    """
    Log un modèle sklearn dans le Model Registry.
    Enregistre automatiquement comme version 'Staging'.
    """
    model_info = mlflow.sklearn.log_model(
        sk_model       = model,
        artifact_path  = artifact_path,
        registered_model_name = model_name,
        signature      = signature,
        input_example  = input_example,
    )
    print(f"[MLflow Registry] Modèle enregistré : {model_name}")
    print(f"[MLflow Registry] Run ID  : {mlflow.active_run().info.run_id}")
    return model_info


def log_params_dict(params: dict):
    """Log un dictionnaire de paramètres (filtre les valeurs non-loggables)."""
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            mlflow.log_param(k, str(v)[:250])


def log_metrics_dict(metrics: dict, step: int = None):
    """Log un dictionnaire de métriques."""
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k, float(v), step=step)
        except Exception:
            pass


def log_dataframe_as_artifact(df, filename: str, subdir: str = "data"):
    """Sauvegarde un DataFrame CSV comme artifact MLflow."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, filename)
        df.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path=subdir)


def log_figure_as_artifact(fig_path: str, subdir: str = "figures"):
    """Log un fichier image comme artifact MLflow."""
    if os.path.exists(fig_path):
        mlflow.log_artifact(fig_path, artifact_path=subdir)


def get_best_run(experiment_name: str, metric: str = "auc_ensemble",
                 ascending: bool = False) -> dict:
    """
    Retourne le meilleur run d'une expérience selon une métrique.
    Utile pour comparer les runs entre eux.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    if not runs:
        return {}
    best = runs[0]
    return {
        "run_id"  : best.info.run_id,
        "metric"  : best.data.metrics.get(metric),
        "params"  : best.data.params,
        "tags"    : best.data.tags,
    }
