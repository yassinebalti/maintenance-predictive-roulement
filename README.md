# 🚀 Pipeline IA/ML — Maintenance Prédictive V3
## Architecture Temps Réel : Kafka → ML → Streamlit → Grafana

---

## 📋 Vue d'ensemble des 3 Phases

```
Capteurs IoT
     │
     ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  PHASE 1        │     │  PHASE 2        │     │  PHASE 3        │
│  Kafka Stream   │────▶│  Streamlit UI   │     │  Prometheus     │
│                 │     │                 │     │  + Grafana      │
│  Producer       │     │  6 pages :      │     │  + Docker       │
│  Consumer ML    │────▶│  • Vue Flotte   │────▶│                 │
│  Features V3    │     │  • Moteur Détail│     │  Métriques ML   │
│  Anomalies      │     │  • Anomalies    │     │  Alertes auto   │
│  RUL Weibull    │     │  • RUL & Prédic │     │  Dashboard live │
│  CUSUM          │     │  • CUSUM        │     │  99.9% uptime   │
└─────────────────┘     │  • Métriques IA │     └─────────────────┘
                        └─────────────────┘
```

---

## 📁 Structure du Projet

```
streaming_pipeline/
├── kafka/
│   ├── kafka_producer.py      ← Simulation capteurs IoT (21 moteurs)
│   └── kafka_consumer.py      ← Pipeline ML temps réel
│
├── streamlit_app/
│   └── streamlit_app.py       ← Dashboard 6 pages (mode démo ou live)
│
├── docker/
│   ├── docker-compose.yml     ← Stack complète (8 services)
│   ├── Dockerfile.pipeline    ← Image Producer + Consumer
│   ├── Dockerfile.streamlit   ← Image Dashboard
│   ├── prometheus_exporter.py ← Exposition métriques /metrics
│   ├── prometheus.yml         ← Config scrape Prometheus
│   ├── alert_rules.yml        ← Règles d'alerte automatiques
│   ├── requirements.txt
│   ├── requirements_streamlit.txt
│   └── grafana/
│       ├── dashboards/
│       │   └── maintenance.json   ← Dashboard Grafana pré-configuré
│       └── provisioning/          ← Auto-configuration Grafana
│
└── README.md
```

---

## ⚡ PHASE 1 — Stream Processing Kafka

### Pourquoi Kafka ?
- **Robustesse** : Données persistées sur disque, replay possible
- **Scalabilité** : 3 partitions → traitement parallèle
- **Découplage** : Producer et Consumer indépendants
- **Tolérance pannes** : Offset management → reprise exactement là où on s'est arrêté

### Architecture du Consumer (Flink-like)

```
message Kafka
    │
    ▼ MotorWindowBuffer (deque taille 20)
    │
    ▼ extract_features_from_window()
    │   → 21 features V3 (vib_rms, skewness, peak2peak,
    │     spectral_entropy, shape_factor, impulse_factor...)
    │
    ▼ StreamMLModel.predict()
    │   → IF + LOF (novelty=True) + Règles métier
    │   → Ré-entraînement tous les 200 messages
    │
    ▼ StreamRULEstimator.update()
    │   → DI calculé en temps réel
    │   → RUL par régression sur historique glissant
    │
    ▼ StreamCUSUM.update()
    │   → S+[t] = max(0, S+[t-1] + (DI[t] - μ₀) - k)
    │   → Alarme si S+[t] > 4σ
    │
    ▼ StreamState.save_snapshot()
        → data/stream_results/latest.json (toutes les 5s)
        → Publication alerte vers motor.alerts si anomalie
```

### Installation sans Docker

```bash
# 1. Démarrer Kafka (via Docker juste pour Kafka)
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  bitnami/kafka:latest

# 2. Installer dépendances Python
pip install kafka-python scikit-learn scipy numpy pandas

# 3. Producer (simule les 21 moteurs)
python kafka/kafka_producer.py --interval 15

# 4. Consumer ML (pipeline complet)
python kafka/kafka_consumer.py

# Modes alternatifs :
python kafka/kafka_producer.py --motor 21          # Moteur 21 seul
python kafka/kafka_producer.py --replay data.csv   # Replay historique
python kafka/kafka_producer.py --interval 5        # 5 sec entre mesures
```

### Topics Kafka

| Topic | Partitions | Rôle |
|-------|-----------|------|
| `motor.measurements` | 3 | Mesures brutes capteurs (principal) |
| `motor.alerts` | 1 | Alertes anomalies détectées |
| `motor.heartbeat` | 1 | Battements vie pour monitoring |

---

## 📊 PHASE 2 — Dashboard Streamlit

### Pourquoi Streamlit ?
- **Interactif** : Filtres, sélection moteur, drill-down
- **Temps réel** : Auto-refresh toutes les 3 secondes (st.rerun)
- **Sans backend** : Lit directement le JSON du consumer
- **Mode démo** : Fonctionne même sans Kafka

### Pages du Dashboard

| Page | Contenu |
|------|---------|
| 🏠 Vue Flotte | Carte DI tous moteurs, pie risques, tableau complet |
| 🔍 Moteur Détail | Gauge DI, scores IF/LOF/Règles, RUL avec IC |
| 📊 Anomalies Live | Scatter DI vs Score, heatmap T° & vibration |
| 🔮 RUL & Tendances | Timeline RUL avec IC 80%, top 5 prioritaires |
| ⚡ CUSUM | Tableau statuts, graphique S+ par moteur |
| 📈 Métriques IA | Walk-forward CV, SHAP features, récap 7 axes |

### Lancement

```bash
pip install streamlit plotly pandas numpy

# Mode standard (lit data/stream_results/latest.json)
streamlit run streamlit_app/streamlit_app.py

# Accès : http://localhost:8501
# Mode démo automatique si consumer non lancé
```

---

## 📈 PHASE 3 — Monitoring & Robustesse

### Stack complète

```
Prometheus Exporter (port 8000)
    │  /metrics — format texte Prometheus
    ▼
Prometheus (port 9090)
    │  scrape toutes les 10s
    │  évalue les règles d'alerte
    ▼
Grafana (port 3000)
    │  Dashboard pré-configuré
    │  Refresh 10s
    ▼
[Alertmanager optionnel] → Email/Slack/PagerDuty
```

### Métriques Prometheus exposées

```
# Globales pipeline
ml_pipeline_messages_processed        → Messages total traités
ml_pipeline_n_critique                → Nb moteurs CRITIQUE
ml_pipeline_n_eleve                   → Nb moteurs ÉLEVÉ
ml_pipeline_avg_degradation_index     → DI moyen flotte
ml_pipeline_cusum_alarms              → Nb alarmes CUSUM actives

# Par moteur (label motor_id)
ml_motor_degradation_index{motor_id}  → DI [0-1]
ml_motor_rul_days{motor_id}           → RUL en jours
ml_motor_risk_level{motor_id}         → 0=FAIBLE, 1=MODÉRÉ, 2=ÉLEVÉ, 3=CRITIQUE
ml_motor_anomaly_score{motor_id}      → Score combiné IF+LOF+Règles
ml_motor_temperature{motor_id}        → Température °C
ml_motor_vibration{motor_id}          → Vibration mm/s
ml_motor_cusum_alarm{motor_id}        → 0 ou 1
ml_motor_health_score{motor_id}       → Health score [0-100]
ml_motor_vib_rms{motor_id}            → RMS vibratoire (AXE 3)
ml_motor_spectral_entropy{motor_id}   → Entropie spectrale (AXE 3)
```

### Alertes automatiques configurées

| Alerte | Condition | Sévérité |
|--------|-----------|----------|
| MoteurRisqueCritique | risk_level >= 3 pendant 2min | CRITICAL |
| MoteurRisqueEleve | risk_level >= 2 pendant 5min | WARNING |
| DegradationCritique | DI > 0.75 pendant 3min | CRITICAL |
| RULUrgent | RUL < 7 jours | CRITICAL |
| CUSUMRupture | cusum_alarm == 1 | WARNING |
| TemperatureCritique | T > 70°C pendant 2min | WARNING |
| PipelineInactif | 0 message/5min | CRITICAL |

---

## 🐳 Déploiement Docker Complet

### Prérequis
- Docker Desktop installé
- 4 GB RAM disponible

### Démarrage en 1 commande

```bash
# Depuis le dossier streaming_pipeline/
cd docker/
docker-compose up -d

# Vérifier que tout tourne
docker-compose ps

# Logs en direct
docker-compose logs -f consumer
docker-compose logs -f producer
```

### Accès aux services

| Service | URL | Credentials |
|---------|-----|-------------|
| 📊 Streamlit Dashboard | http://localhost:8501 | — |
| 📈 Grafana | http://localhost:3000 | admin / maintenance2024 |
| 🔥 Prometheus | http://localhost:9090 | — |
| 📨 Kafka UI | http://localhost:8080 | — |
| 📡 Métriques raw | http://localhost:8000/metrics | — |

### Commandes utiles

```bash
# Statut de tous les services
docker-compose ps

# Redémarrer un service
docker-compose restart consumer

# Voir les logs
docker-compose logs -f consumer
docker-compose logs --tail=100 producer

# Arrêt complet
docker-compose down

# Arrêt + suppression volumes
docker-compose down -v

# Rebuild après modification code
docker-compose build consumer
docker-compose up -d consumer

# Entrer dans un container
docker exec -it ml-consumer bash
```

### Sans Docker (développement local)

```bash
# Terminal 1 — Kafka (via Docker juste pour le broker)
docker run -d --name kafka -p 9092:9092 \
  -e KAFKA_CFG_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e ALLOW_PLAINTEXT_LISTENER=yes \
  bitnami/kafka:latest

# Terminal 2 — Producer
python kafka/kafka_producer.py

# Terminal 3 — Consumer ML
python kafka/kafka_consumer.py

# Terminal 4 — Dashboard
streamlit run streamlit_app/streamlit_app.py

# Terminal 5 — Exporter Prometheus
python docker/prometheus_exporter.py
```

---

## 🔧 Configuration

### Variables d'environnement

```bash
KAFKA_BOOTSTRAP=localhost:9092   # Adresse broker Kafka
```

### Paramètres ajustables (kafka_consumer.py)

```python
WINDOW_SIZE    = 20     # Buffer mesures par moteur
RETRAIN_EVERY  = 200    # Ré-entraîne modèles tous les N messages
CUSUM_K        = 0.5    # Slack CUSUM (sensibilité)
CUSUM_H        = 4.0    # Seuil alarme CUSUM
DI_CRITICAL    = 0.75   # Seuil critique DI
```

### Flux de données (sans DB)

```
kafka_consumer.py
    │
    └──► data/stream_results/latest.json  (mis à jour toutes les 5s)
              │
              ├──► streamlit_app.py  (lit le JSON, refresh 3s)
              └──► prometheus_exporter.py  (lit le JSON, expose /metrics)
```

---

## 📊 Exemple de latest.json

```json
{
  "timestamp": "2025-05-26T15:32:00Z",
  "n_processed": 4821,
  "motors": {
    "21": {
      "motor_id": 21,
      "risk_level": "ÉLEVÉ",
      "degradation_index": 0.5808,
      "rul_days": "62.4",
      "rul_low": 43.7,
      "rul_high": 81.1,
      "combined_score": 0.3412,
      "is_anomaly": true,
      "temperature": 74.2,
      "vibration": 1.42,
      "cusum_alarm": true,
      "cusum_severity": "MODÉRÉ"
    }
  },
  "summary": {
    "n_motors": 21,
    "n_critique": 0,
    "n_eleve": 2,
    "n_modere": 3,
    "n_faible": 16,
    "avg_di": 0.1847,
    "cusum_alarms": [5, 21]
  }
}
```

---

## 🎓 Aspects PFE à valoriser

1. **Architecture Event-Driven** : Kafka découple la production des données de leur analyse
2. **Fenêtrage temporel** : Simulation de Flink Sliding Windows avec deque Python
3. **ML Online** : Modèles adaptés au stream (ré-entraînement périodique sans arrêt)
4. **Observabilité** : Métriques métier exposées via Prometheus (pas juste uptime)
5. **Résilience** : restart: unless-stopped, healthchecks, consumer group offsets
6. **Séparation des concerns** : Producer / Consumer / UI / Monitoring = 4 processus indépendants
