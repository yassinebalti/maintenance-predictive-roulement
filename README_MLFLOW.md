# MLflow — Guide d'intégration
## Maintenance Prédictive des Roulements

---

## 1. Installation

```bash
pip install mlflow
```

---

## 2. Structure des fichiers ajoutés

```
pfemast/
├── mlflow_config.py            ← Configuration partagée (URI, noms expériences)
├── step2_features_mlflow.py    ← Step 2 + MLflow tracking
├── step3_anomaly_mlflow.py     ← Step 3 + MLflow tracking + Registry
├── step4_rul_mlflow.py         ← Step 4 + MLflow tracking + Registry
├── main_pipeline_mlflow.py     ← Orchestrateur complet
│
├── step2_features.py           ← Tes fichiers originaux (inchangés)
├── step3_anomaly_detection.py  ← (inchangés)
├── step4_rul_prediction.py     ← (inchangés)
└── mlruns/                     ← Créé automatiquement par MLflow
```

---

## 3. Lancement

### Étape 1 — Démarrer le serveur MLflow UI

```bash
# Terminal 1
mlflow ui --port 5000
```

Ouvrir dans le navigateur : **http://localhost:5000**

---

### Étape 2 — Lancer le pipeline

```bash
# Tous les steps d'un coup
python main_pipeline_mlflow.py

# OU un step seul
python main_pipeline_mlflow.py --step 2   # Features
python main_pipeline_mlflow.py --step 3   # Anomalies
python main_pipeline_mlflow.py --step 4   # RUL

# Comparer les meilleurs runs
python main_pipeline_mlflow.py --compare
```

---

## 4. Ce qui est tracké par step

### Step 2 — Feature Engineering
| Type       | Contenu |
|-----------|---------|
| Paramètres | window_size, overlap, fft_points, poids health score |
| Métriques  | n_features_total, n_features_v3, health_score_moyen par moteur |
| Artifacts  | features_sample.csv, fichier CSV complet |
| Tags       | version=V3, axe=AXE_3 |

### Step 3 — Détection Anomalies
| Type       | Contenu |
|-----------|---------|
| Paramètres | n_estimators, contamination, w_IF, w_LOF, w_rules, threshold |
| Métriques  | **auc_ensemble**, auc_IF, auc_LOF, F1, précision, recall, taux anomalie par moteur |
| Artifacts  | walkforward_cv.csv, shap_global.csv, shap_per_motor.csv, feature_importance.csv, figures |
| Registry   | `IsolationForest_Maintenance` (v1, v2...) |
| Tags       | supervise=Non, axe=AXE_1+AXE_4+AXE_5+AXE_6 |

### Step 4 — Prédiction RUL
| Type       | Contenu |
|-----------|---------|
| Paramètres | w_poly, w_exp, w_weibull, cusum_k, cusum_h, seuils alerte |
| Métriques  | rul_jours par moteur, ic_width, weibull_beta, DI, n_critiques, n_cusum_alarmes |
| Artifacts  | rul_summary.csv, cusum_changepoints.csv, fig4, fig5, fig7 |
| Registry   | `RUL_Weibull_Maintenance` (v1, v2...) |
| Tags       | axe=AXE_2+AXE_7 |

---

## 5. MLflow UI — Ce que tu verras

```
http://localhost:5000
│
├── Experiments/
│   ├── 01_features_engineering      ← Runs step 2
│   ├── 02_anomaly_detection         ← Runs step 3 (AUC, F1...)
│   └── 03_rul_prediction            ← Runs step 4 (RUL, CUSUM...)
│
└── Models/
    ├── IsolationForest_Maintenance   ← Versions IF
    ├── LOF_Maintenance               ← Versions LOF
    └── RUL_Weibull_Maintenance       ← Versions RUL
```

---

## 6. Charger un modèle depuis le Registry

```python
import mlflow.sklearn

# Charger la dernière version du modèle IF
model_if = mlflow.sklearn.load_model(
    "models:/IsolationForest_Maintenance/latest"
)

# Ou une version spécifique
model_if_v2 = mlflow.sklearn.load_model(
    "models:/IsolationForest_Maintenance/2"
)

# Prédire
scores = model_if.score_samples(X_new)
```

---

## 7. Commandes utiles

```bash
# Voir toutes les expériences
mlflow experiments list

# Voir les modèles du registry
mlflow models list

# Rechercher le meilleur run par AUC
mlflow runs search \
  --experiment-name "02_anomaly_detection" \
  --filter "metrics.auc_ensemble > 0.6" \
  --order-by "metrics.auc_ensemble DESC"
```
