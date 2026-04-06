"""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 1 — STREAM PROCESSING : KAFKA CONSUMER + FLINK-LIKE      ║
║  Maintenance Prédictive — Pipeline ML Temps Réel                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Remplace update.py par un vrai stream processor                 ║
║                                                                  ║
║  Pipeline par message :                                          ║
║    1. Consomme depuis motor.measurements                         ║
║    2. Buffer glissant par moteur (window=20 mesures)             ║
║    3. Feature engineering V3 en temps réel                       ║
║    4. Détection anomalies (IF+LOF+Règles)                        ║
║    5. Calcul DI + RUL Ensemble                                   ║
║    6. CUSUM détection rupture                                     ║
║    7. Écrit résultats dans fichiers JSON/CSV temps réel          ║
║    8. Publie alertes vers motor.alerts                           ║
║                                                                  ║
║  Usage :                                                         ║
║    python kafka_consumer.py                                      ║
║    python kafka_consumer.py --group mon-groupe                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import time
import logging
import argparse
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict, deque
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import os
import json as json_mod
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_MEASURES  = "motor.measurements"
TOPIC_ALERTS    = "motor.alerts"
CONSUMER_GROUP  = "maintenance-ml-pipeline"
STATE_FILE      = "data/stream_state.json"
RESULTS_DIR     = "data/stream_results"
WINDOW_SIZE     = 20      # Buffer mesures par moteur
RETRAIN_EVERY   = 200     # Ré-entraîne modèles tous les N messages
CUSUM_K         = 0.5
CUSUM_H         = 4.0
DI_CRITICAL     = 0.75
DI_WARNING      = 0.50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONSUMER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  BUFFER GLISSANT PAR MOTEUR (Fenêtre temporelle)
# ══════════════════════════════════════════════════════════════════
class MotorWindowBuffer:
    """
    Buffer circulaire (deque) par moteur.
    Simule la fenêtre glissante de Flink/Spark Streaming.
    """
    def __init__(self, motor_id: int, maxlen: int = WINDOW_SIZE):
        self.motor_id   = motor_id
        self.buffer     = deque(maxlen=maxlen)
        self.n_total    = 0
        self.cusum_s    = 0.0    # CUSUM S+
        self.cusum_mu0  = None   # Référence initiale
        self.cusum_sigma= None   # Écart-type initial

    def add(self, record: dict):
        self.buffer.append(record)
        self.n_total += 1

    def to_dataframe(self) -> pd.DataFrame:
        if len(self.buffer) == 0:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer))

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) >= 5  # Minimum pour features


# ══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING TEMPS RÉEL (V3)
# ══════════════════════════════════════════════════════════════════
def compute_spectral_entropy(signal: np.ndarray) -> float:
    if len(signal) < 8:
        return 0.0
    spec  = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    total = spec.sum()
    if total < 1e-12:
        return 0.0
    p = spec / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def extract_features_from_window(df: pd.DataFrame) -> dict:
    """
    Extrait toutes les features V3 depuis une fenêtre glissante.
    Retourne un dict de scalaires (dernière valeur calculée).
    """
    if len(df) < 3:
        return {}

    vib = df["vibration"].fillna(0).values
    tmp = df["temperature"].fillna(0).values
    cur = df["courant"].fillna(0).values
    n   = len(vib)

    # ── Features V2 ──────────────────────────────
    vib_mean    = float(vib.mean())
    vib_std     = float(vib.std()) if n > 1 else 0.0
    vib_max     = float(vib.max())
    vib_energy  = float((vib ** 2).mean())
    vib_rms     = float(np.sqrt((vib ** 2).mean()))
    vib_kurt    = float(kurtosis(vib)) if n >= 4 else 0.0
    vib_skew    = float(skew(vib))     if n >= 4 else 0.0

    mean_abs = np.abs(vib).mean() + 1e-9
    crest_factor   = vib_max / (vib_rms + 1e-9)
    shape_factor   = vib_rms / mean_abs
    impulse_factor = vib_max / mean_abs
    peak2peak      = float(vib.max() - vib.min())

    temp_mean  = float(tmp.mean())
    temp_std   = float(tmp.std()) if n > 1 else 0.0
    temp_trend = float(np.polyfit(np.arange(n), tmp, 1)[0]) if n >= 4 else 0.0

    courant_mean = float(cur.mean())
    courant_std  = float(cur.std()) if n > 1 else 0.0

    # Envelope (Hilbert)
    try:
        envelope      = np.abs(hilbert(vib))
        envelope_mean = float(envelope.mean())
    except Exception:
        envelope_mean = 0.0

    # FFT
    if n >= 4:
        spectrum     = np.abs(np.fft.rfft(vib - vib.mean()))
        fft_max_amp  = float(spectrum.max())
        fft_dom_freq = float(np.argmax(spectrum))
    else:
        fft_max_amp = fft_dom_freq = 0.0

    # Entropie spectrale (AXE 3)
    spectral_ent = compute_spectral_entropy(vib)

    # Health score V3
    def inv_norm(val, lo, hi):
        return max(0.0, min(1.0, 1.0 - (val - lo) / (hi - lo + 1e-9)))

    health_score = (
        0.28 * inv_norm(vib_energy, 0, 5)    * 100 +
        0.22 * inv_norm(temp_mean,  20, 90)  * 100 +
        0.18 * inv_norm(abs(vib_kurt), 0, 10)* 100 +
        0.15 * inv_norm(crest_factor, 1, 10) * 100 +
        0.10 * inv_norm(impulse_factor, 1, 20)*100 +
        0.07 * inv_norm(vib_rms, 0, 3)       * 100
    )

    return {
        "vib_mean"         : round(vib_mean, 4),
        "vib_std"          : round(vib_std, 4),
        "vib_max"          : round(vib_max, 4),
        "vib_energy_mean"  : round(vib_energy, 4),
        "vib_rms"          : round(vib_rms, 4),
        "vib_kurt"         : round(vib_kurt, 4),
        "vib_skewness"     : round(vib_skew, 4),
        "crest_factor"     : round(crest_factor, 4),
        "shape_factor"     : round(shape_factor, 4),
        "impulse_factor"   : round(impulse_factor, 4),
        "peak2peak"        : round(peak2peak, 4),
        "spectral_entropy" : round(spectral_ent, 4),
        "temp_mean"        : round(temp_mean, 4),
        "temp_std"         : round(temp_std, 4),
        "temp_trend"       : round(temp_trend, 6),
        "courant_mean"     : round(courant_mean, 4),
        "courant_std"      : round(courant_std, 4),
        "envelope_mean"    : round(envelope_mean, 4),
        "fft_max_amp"      : round(fft_max_amp, 4),
        "fft_dominant_freq": round(fft_dom_freq, 4),
        "health_score"     : round(health_score, 2),
    }


# ══════════════════════════════════════════════════════════════════
#  MODÈLE ML STREAM — Isolation Forest + LOF Online
# ══════════════════════════════════════════════════════════════════
class StreamMLModel:
    """
    Modèle ML adaptatif pour stream processing.
    S'ré-entraîne périodiquement sur l'historique glissant.
    """
    FEATURE_COLS = [
        "vib_energy_mean", "vib_kurt", "crest_factor", "temp_mean",
        "temp_trend", "courant_mean", "envelope_mean", "health_score",
        "vib_rms", "vib_skewness", "peak2peak", "spectral_entropy",
        "shape_factor", "impulse_factor",
    ]

    def __init__(self):
        self.model_if    = None
        self.model_lof   = None
        self.scaler      = StandardScaler()
        self.is_trained  = False
        self.n_trained   = 0
        self.history     = []     # Historique features pour ré-entraînement
        self.lock        = threading.Lock()

    def update_history(self, features: dict):
        row = [features.get(c, 0.0) for c in self.FEATURE_COLS]
        self.history.append(row)
        if len(self.history) > 5000:
            self.history = self.history[-5000:]

    def train(self):
        """Entraîne les modèles sur l'historique accumulé."""
        if len(self.history) < 50:
            return

        with self.lock:
            X       = np.array(self.history)
            X_sc    = self.scaler.fit_transform(X)
            X_norm  = X_sc  # En stream : on suppose tout normal au début

            self.model_if = IsolationForest(
                n_estimators=200, contamination=0.10,
                random_state=42, n_jobs=-1
            )
            self.model_if.fit(X_norm)

            if len(X_norm) >= 50:
                self.model_lof = LocalOutlierFactor(
                    n_neighbors=min(20, len(X_norm) - 1),
                    contamination=0.10, novelty=True, n_jobs=-1
                )
                self.model_lof.fit(X_norm)

            self.is_trained = True
            self.n_trained += 1
            log.info(f"  Modèle ré-entraîné #{self.n_trained} "
                     f"sur {len(self.history)} samples")

    def predict(self, features: dict) -> dict:
        """Score d'anomalie pour un vecteur de features."""
        if not self.is_trained:
            return {"combined_score": 0.0, "score_if": 0.0,
                    "score_lof": 0.0, "is_anomaly": False}

        row  = np.array([[features.get(c, 0.0) for c in self.FEATURE_COLS]])

        with self.lock:
            X_sc = self.scaler.transform(row)

            # IF score
            raw_if  = -self.model_if.decision_function(X_sc)[0]
            sc_if   = float(np.clip(raw_if, 0, None))

            # LOF score
            sc_lof = 0.0
            if self.model_lof is not None:
                try:
                    raw_lof = -self.model_lof.score_samples(X_sc)[0]
                    sc_lof  = float(np.clip(raw_lof, 0, None))
                except Exception:
                    pass

        # Règles métier simples
        sc_rules = 0.0
        if features.get("temp_mean", 0) > 65:
            sc_rules += 0.4
        if features.get("vib_energy_mean", 0) > 2.0:
            sc_rules += 0.3
        if features.get("crest_factor", 0) > 5:
            sc_rules += 0.2
        sc_rules = min(sc_rules, 1.0)

        # Normalisation locale IF
        sc_if_norm  = min(sc_if / 2.0, 1.0)
        sc_lof_norm = min(sc_lof / 2.0, 1.0)

        combined   = (0.25 * sc_if_norm + 0.20 * sc_lof_norm + 0.55 * sc_rules)
        is_anomaly = combined >= 0.25

        return {
            "score_if"       : round(sc_if_norm, 4),
            "score_lof"      : round(sc_lof_norm, 4),
            "score_rules"    : round(sc_rules, 4),
            "combined_score" : round(combined, 4),
            "is_anomaly"     : bool(is_anomaly),
        }


# ══════════════════════════════════════════════════════════════════
#  DI + RUL STREAM
# ══════════════════════════════════════════════════════════════════
class StreamRULEstimator:
    """Calcule DI et RUL en temps réel depuis l'historique par moteur."""

    def __init__(self, motor_id: int):
        self.motor_id  = motor_id
        self.di_history = deque(maxlen=500)
        self.ref_mean   = None  # Référence initiale (10 premières mesures)

    def update(self, features: dict, anomaly_score: float) -> dict:
        """Met à jour le DI et calcule le RUL."""
        # DI simplifié depuis les features et score
        di_raw = (
            0.35 * min(features.get("vib_energy_mean", 0) / 3.0, 1.0) +
            0.25 * min(max(0, features.get("temp_mean", 35) - 35) / 55.0, 1.0) +
            0.20 * min(abs(features.get("vib_kurt", 0)) / 10.0, 1.0) +
            0.20 * min(anomaly_score, 1.0)
        )

        # Normalisation relative (par rapport à référence initiale)
        if self.ref_mean is None and len(self.di_history) >= 10:
            self.ref_mean = np.mean(list(self.di_history)[:10])

        di = float(np.clip(di_raw, 0, 1))
        self.di_history.append(di)

        # RUL par régression polynomiale sur historique DI
        rul_days = ">90"
        rul_num  = 90.0
        rul_low  = 70.0
        rul_high = 90.0
        trend    = 0.0

        if len(self.di_history) >= 10:
            di_arr = np.array(self.di_history)
            x_arr  = np.arange(len(di_arr), dtype=float)
            try:
                coef   = np.polyfit(x_arr, di_arr, min(2, len(x_arr) - 1))
                pf     = np.poly1d(coef)
                trend  = float(pf(x_arr[-1] + 1) - pf(x_arr[-1]))
                if trend > 1e-6:
                    days_to_crit = (DI_CRITICAL - di) / trend
                    rul_num  = float(np.clip(days_to_crit * (15/1440), 0, 90))
                    rul_low  = max(0.0, rul_num * 0.7)
                    rul_high = min(90.0, rul_num * 1.3)
                    if rul_num < 90:
                        rul_days = str(round(rul_num, 1))
            except Exception:
                pass

        risk = ("CRITIQUE" if di >= DI_CRITICAL or rul_num < 7 else
                "ÉLEVÉ"   if di >= DI_WARNING   or rul_num < 21 else
                "MODÉRÉ"  if di >= 0.30 else "FAIBLE")

        return {
            "degradation_index": round(di, 4),
            "rul_days"         : rul_days,
            "rul_num"          : round(rul_num, 1),
            "rul_low"          : round(rul_low, 1),
            "rul_high"         : round(rul_high, 1),
            "trend_slope"      : round(trend, 6),
            "risk_level"       : risk,
        }


# ══════════════════════════════════════════════════════════════════
#  CUSUM STREAM — Détection rupture en temps réel
# ══════════════════════════════════════════════════════════════════
class StreamCUSUM:
    """CUSUM mis à jour à chaque mesure — détection instantanée."""

    def __init__(self, motor_id: int, k=CUSUM_K, h=CUSUM_H):
        self.motor_id  = motor_id
        self.k_factor  = k
        self.h_factor  = h
        self.S_pos     = 0.0
        self.S_neg     = 0.0
        self.mu0       = None
        self.sigma     = None
        self.init_buf  = []
        self.alarm_on  = False

    def update(self, di: float) -> dict:
        # Phase initialisation (10 premières valeurs)
        if len(self.init_buf) < 10:
            self.init_buf.append(di)
            if len(self.init_buf) == 10:
                self.mu0   = np.mean(self.init_buf)
                self.sigma = max(np.std(self.init_buf), 0.01)
            return {"cusum_alarm": False, "cusum_s_pos": 0.0, "cusum_severity": "STABLE"}

        k = self.k_factor * self.sigma
        h = self.h_factor * self.sigma

        self.S_pos = max(0.0, self.S_pos + (di - self.mu0) - k)
        self.S_neg = max(0.0, self.S_neg - (di - self.mu0) - k)

        alarm     = self.S_pos > h or self.S_neg > h
        severity  = "STABLE"
        if alarm:
            severity     = "ÉLEVÉ" if self.S_pos > 2 * h else "MODÉRÉ"
            self.alarm_on = True

        return {
            "cusum_alarm"    : alarm,
            "cusum_s_pos"    : round(self.S_pos, 4),
            "cusum_severity" : severity,
        }


# ══════════════════════════════════════════════════════════════════
#  ÉTAT GLOBAL DU STREAM — Résultats par moteur
# ══════════════════════════════════════════════════════════════════
class StreamState:
    """
    Centralise l'état temps réel de tous les moteurs.
    Écrit les résultats en JSON pour Streamlit/Grafana.
    """
    def __init__(self):
        self.motors    = {}         # motor_id → derniers résultats
        self.counters  = defaultdict(int)
        self.lock      = threading.Lock()
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def update(self, motor_id: int, result: dict):
        with self.lock:
            self.motors[motor_id]   = result
            self.counters["total"] += 1
            self.counters[f"m{motor_id}"] += 1

    def save_snapshot(self):
        """Écrit un snapshot JSON lisible par Streamlit."""
        with self.lock:
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_processed": self.counters["total"],
                "motors": {
                    str(mid): data for mid, data in self.motors.items()
                },
                "summary": self._compute_summary(),
            }
        path = os.path.join(RESULTS_DIR, "latest.json")
        with open(path, "w") as f:
            json_mod.dump(snapshot, f, default=str, indent=2)

    def _compute_summary(self) -> dict:
        if not self.motors:
            return {}
        risks  = [v.get("risk_level", "FAIBLE") for v in self.motors.values()]
        alarms = [mid for mid, v in self.motors.items()
                  if v.get("cusum_alarm", False)]
        return {
            "n_motors"       : len(self.motors),
            "n_critique"     : risks.count("CRITIQUE"),
            "n_eleve"        : risks.count("ÉLEVÉ"),
            "n_modere"       : risks.count("MODÉRÉ"),
            "n_faible"       : risks.count("FAIBLE"),
            "cusum_alarms"   : alarms,
            "avg_di"         : round(np.mean([v.get("degradation_index", 0)
                                               for v in self.motors.values()]), 4),
        }


# ══════════════════════════════════════════════════════════════════
#  CONSUMER PRINCIPAL
# ══════════════════════════════════════════════════════════════════
class MaintenanceStreamConsumer:
    def __init__(self, group_id: str = CONSUMER_GROUP):
        self.group_id   = group_id
        self.buffers    = {}     # motor_id → MotorWindowBuffer
        self.rul_estims = {}     # motor_id → StreamRULEstimator
        self.cusums     = {}     # motor_id → StreamCUSUM
        self.ml_model   = StreamMLModel()
        self.state      = StreamState()
        self.n_processed= 0
        self.producer   = None

    def _get_or_create(self, motor_id: int):
        if motor_id not in self.buffers:
            self.buffers[motor_id]    = MotorWindowBuffer(motor_id)
            self.rul_estims[motor_id] = StreamRULEstimator(motor_id)
            self.cusums[motor_id]     = StreamCUSUM(motor_id)
        return (self.buffers[motor_id],
                self.rul_estims[motor_id],
                self.cusums[motor_id])

    def process_message(self, record: dict) -> dict:
        """
        ✦ CŒUR DU STREAM PROCESSOR ✦
        Traite un message Kafka en < 100ms.
        """
        motor_id = record.get("motor_id")
        if motor_id is None:
            return {}

        buf, rul_est, cusum = self._get_or_create(motor_id)

        # 1. Ajouter au buffer glissant
        buf.add(record)
        if not buf.is_ready:
            return {}

        # 2. Feature engineering
        df       = buf.to_dataframe()
        features = extract_features_from_window(df)
        if not features:
            return {}

        # 3. Mettre à jour historique ML + ré-entraîner périodiquement
        self.ml_model.update_history(features)
        if self.n_processed % RETRAIN_EVERY == 0 and self.n_processed > 0:
            threading.Thread(target=self.ml_model.train, daemon=True).start()

        # 4. Détection anomalies
        anomaly = self.ml_model.predict(features)

        # 5. DI + RUL
        rul = rul_est.update(features, anomaly["combined_score"])

        # 6. CUSUM
        cusum_result = cusum.update(rul["degradation_index"])

        # 7. Assemblage résultat complet
        result = {
            "motor_id"   : motor_id,
            "timestamp"  : record.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms" : 0,
            # Mesures brutes
            "temperature": record.get("temperature", 0),
            "vibration"  : record.get("vibration", 0),
            "courant"    : record.get("courant", 0),
            # Features
            **features,
            # Anomalie
            **anomaly,
            # RUL
            **rul,
            # CUSUM
            **cusum_result,
            # Alert_Status source
            "alert_status_source": record.get("Alert_Status", "NORMAL"),
        }

        # 8. Publier alerte si anomalie confirmée
        if anomaly["is_anomaly"] and self.producer:
            alert = {
                "motor_id"         : motor_id,
                "timestamp"        : result["timestamp"],
                "risk_level"       : rul["risk_level"],
                "combined_score"   : anomaly["combined_score"],
                "degradation_index": rul["degradation_index"],
                "rul_days"         : rul["rul_days"],
                "cusum_alarm"      : cusum_result["cusum_alarm"],
                "temperature"      : record.get("temperature", 0),
                "vibration"        : record.get("vibration", 0),
            }
            try:
                self.producer.send(TOPIC_ALERTS, key=str(motor_id), value=alert)
            except Exception as e:
                log.warning(f"Erreur envoi alerte : {e}")

        return result

    def run(self):
        """Boucle principale de consommation Kafka."""
        log.info(f"Démarrage consumer — group={self.group_id}")
        log.info(f"Topic : {TOPIC_MEASURES}")

        # Producer pour les alertes
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
            )
        except Exception as e:
            log.warning(f"Producer alertes non disponible : {e}")

        # Entraînement initial (si données existantes)
        if self.ml_model.n_trained == 0:
            log.info("Premier entraînement en attente de données...")

        consumer = KafkaConsumer(
            TOPIC_MEASURES,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        log.info("✓ Consumer connecté — En attente de messages...")
        t_last_snapshot = time.time()
        t_start = time.time()

        try:
            for message in consumer:
                t0 = time.time()
                try:
                    record = message.value
                    result = self.process_message(record)

                    if result:
                        self.n_processed += 1
                        mid = result["motor_id"]
                        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
                        self.state.update(mid, result)

                        # Log périodique
                        if self.n_processed % 50 == 0:
                            uptime = time.time() - t_start
                            log.info(
                                f"  Traités: {self.n_processed:,} | "
                                f"Moteurs actifs: {len(self.buffers)} | "
                                f"Uptime: {uptime:.0f}s | "
                                f"Latence: {result['latency_ms']}ms"
                            )
                            # Alertes actives
                            summary = self.state._compute_summary()
                            if summary.get("n_critique", 0) + summary.get("n_eleve", 0) > 0:
                                log.warning(
                                    f"  ⚠ Risques : CRITIQUE={summary['n_critique']} "
                                    f"ÉLEVÉ={summary['n_eleve']}"
                                )

                    # Snapshot JSON toutes les 5 secondes
                    if time.time() - t_last_snapshot > 5:
                        self.state.save_snapshot()
                        t_last_snapshot = time.time()

                except Exception as e:
                    log.error(f"Erreur traitement message : {e}", exc_info=True)

        except KeyboardInterrupt:
            log.info("\n⏹ Arrêt consumer...")
        finally:
            self.state.save_snapshot()
            consumer.close()
            if self.producer:
                self.producer.flush()
                self.producer.close()
            log.info("✓ Consumer arrêté proprement")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Kafka Consumer — Pipeline ML Temps Réel")
    parser.add_argument("--group", default=CONSUMER_GROUP, help="Consumer group ID")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║   KAFKA CONSUMER V3 — Pipeline ML Temps Réel         ║
║   Features V3 | IF+LOF | RUL Weibull | CUSUM         ║
╚══════════════════════════════════════════════════════╝
""")

    os.makedirs("data/stream_results", exist_ok=True)
    consumer = MaintenanceStreamConsumer(group_id=args.group)
    consumer.run()


if __name__ == "__main__":
    main()
