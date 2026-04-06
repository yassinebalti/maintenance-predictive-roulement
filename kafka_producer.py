"""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 1 — STREAM PROCESSING : KAFKA PRODUCER                   ║
║  Maintenance Prédictive — Moteurs Industriels                    ║
╠══════════════════════════════════════════════════════════════════╣
║  Rôle : Simule les capteurs IoT des 21 moteurs                   ║
║         Envoie les mesures en temps réel vers Kafka              ║
║                                                                  ║
║  Topics Kafka :                                                  ║
║    motor.measurements  → mesures brutes capteurs                 ║
║    motor.alerts        → alertes déclenchées                     ║
║    motor.heartbeat     → battements de vie (monitoring)          ║
║                                                                  ║
║  Usage :                                                         ║
║    python kafka_producer.py                      # tous moteurs  ║
║    python kafka_producer.py --motor 21           # moteur seul   ║
║    python kafka_producer.py --interval 5         # 5 sec         ║
║    python kafka_producer.py --replay data.csv    # replay CSV    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import time
import random
import argparse
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os

# ── Configuration ──────────────────────────────────────────────
KAFKA_BOOTSTRAP    = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_MEASURES     = "motor.measurements"
TOPIC_ALERTS       = "motor.alerts"
TOPIC_HEARTBEAT    = "motor.heartbeat"
INTERVAL_SEC       = 15       # Intervalle nominal entre mesures (15 min → 15 sec en sim)
N_MOTORS           = 21
CSV_FILE           = "../data/01_raw_motor.csv"

# Profils de dégradation simulés par moteur (pour simulation réaliste)
MOTOR_PROFILES = {
    21: {"temp_bias": 30, "vib_bias": 0.5,  "degrad_rate": 0.05},  # Critique
     5: {"temp_bias": 15, "vib_bias": 0.3,  "degrad_rate": 0.03},  # Élevé
    18: {"temp_bias": 10, "vib_bias": 0.2,  "degrad_rate": 0.02},  # Modéré
     4: {"temp_bias":  8, "vib_bias": 0.15, "degrad_rate": 0.015}, # Modéré
    15: {"temp_bias":  5, "vib_bias": 0.1,  "degrad_rate": 0.01},  # Modéré
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  CONNEXION KAFKA
# ══════════════════════════════════════════════════════════════════
def create_producer(retries: int = 5) -> KafkaProducer:
    """
    Crée un KafkaProducer avec retry automatique.
    Sérialisation JSON + compression gzip.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
                compression_type="gzip",
                acks="all",               # Attendre confirmation de tous les replicas
                retries=3,
                linger_ms=50,             # Batch pendant 50ms pour efficacité
                batch_size=16384,
                max_request_size=5242880, # 5MB max
            )
            log.info(f"✓ Connecté à Kafka : {KAFKA_BOOTSTRAP}")
            return producer
        except KafkaError as e:
            log.warning(f"Tentative {attempt}/{retries} échouée : {e}")
            time.sleep(2 ** attempt)
    raise ConnectionError(f"Impossible de connecter Kafka après {retries} tentatives")


# ══════════════════════════════════════════════════════════════════
#  GÉNÉRATION DE MESURES SIMULÉES
# ══════════════════════════════════════════════════════════════════
class MotorSimulator:
    """
    Simule un moteur industriel avec dégradation progressive.
    Génère des mesures réalistes avec bruit gaussien.
    """
    def __init__(self, motor_id: int):
        self.motor_id     = motor_id
        self.profile      = MOTOR_PROFILES.get(motor_id, {
            "temp_bias": 0, "vib_bias": 0, "degrad_rate": 0.001
        })
        self.age_hours    = random.uniform(0, 5000)    # Âge initial aléatoire
        self.degradation  = random.uniform(0, 0.1)     # Dégradation initiale
        self.base_temp    = random.uniform(35, 50)
        self.base_vib     = random.uniform(0.8, 1.2)
        self.base_courant = random.uniform(80, 120)

    def generate_measurement(self) -> dict:
        """Génère une mesure avec dégradation progressive."""
        # Avancer la dégradation
        self.degradation += self.profile["degrad_rate"] * random.uniform(0.5, 1.5)
        self.degradation  = min(self.degradation, 1.0)
        self.age_hours   += INTERVAL_SEC / 3600

        # Température avec biais de dégradation + bruit
        temp = (self.base_temp
                + self.profile["temp_bias"] * self.degradation
                + random.gauss(0, 1.5))

        # Vibration — augmente avec la dégradation
        vib = (self.base_vib
               + self.profile["vib_bias"] * self.degradation
               + abs(random.gauss(0, 0.05 * (1 + self.degradation))))

        # Courant — légère variation
        courant = self.base_courant + random.gauss(0, 2)

        # Accélération
        accel = vib * random.uniform(0.8, 1.2) * (1 + 0.3 * self.degradation)

        # Vitesse (légère variation autour du nominal 1500 rpm)
        vitesse = 1500 + random.gauss(0, 10) - 20 * self.degradation

        # Cosinus phi — diminue avec la dégradation
        cosphi = max(0.7, 0.92 - 0.15 * self.degradation + random.gauss(0, 0.01))

        # THD (distorsion harmonique)
        thdi = max(0, 5 + 10 * self.degradation + random.gauss(0, 0.5))
        thdu = max(0, 2 + 5  * self.degradation + random.gauss(0, 0.3))

        # Seuils d'alerte (simples)
        alert_status = "NORMAL"
        alert_param  = None
        if temp > 70:
            alert_status, alert_param = "ALERT", "temperature"
        elif vib > 1.5:
            alert_status, alert_param = "ALERT", "vibration"
        elif courant > 150:
            alert_status, alert_param = "ALERT", "courant"

        return {
            "motor_id"       : self.motor_id,
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
            "temperature"    : round(temp, 2),
            "courant"        : round(max(0, courant), 2),
            "vibration"      : round(max(0, vib), 4),
            "acceleration"   : round(max(0, accel), 4),
            "vitesse"        : round(max(500, vitesse), 1),
            "cosphi"         : round(cosphi, 3),
            "thdi"           : round(thdi, 2),
            "thdu"           : round(thdu, 2),
            "Alert_Status"   : alert_status,
            "alert_parameter": alert_param,
            "degradation_sim": round(self.degradation, 4),  # Meta pour debug
            "age_hours"      : round(self.age_hours, 1),
            "producer_ts"    : time.time(),
        }


# ══════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════
def on_send_success(record_metadata):
    pass  # Silencieux en prod — activer en debug si besoin

def on_send_error(exc):
    log.error(f"Erreur envoi Kafka : {exc}")


# ══════════════════════════════════════════════════════════════════
#  MODE REPLAY — Rejoue un CSV historique
# ══════════════════════════════════════════════════════════════════
def replay_csv(producer: KafkaProducer, csv_path: str, speed: float = 10.0):
    """
    Rejoue les données historiques d'un CSV vers Kafka.
    speed=10 → 10× plus rapide que le temps réel.
    """
    if not os.path.exists(csv_path):
        log.error(f"Fichier introuvable : {csv_path}")
        return

    log.info(f"▶ REPLAY : {csv_path}  (vitesse ×{speed})")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "motor_id"])

    prev_ts   = None
    n_sent    = 0
    n_errors  = 0

    for _, row in df.iterrows():
        msg = row.to_dict()
        msg["timestamp"]   = str(msg["timestamp"])
        msg["producer_ts"] = time.time()
        msg["replay"]      = True

        # Respecter les intervalles temporels (accélérés)
        if prev_ts is not None:
            delta = (row["timestamp"] - prev_ts).total_seconds()
            wait  = max(0, delta / speed)
            if wait > 0:
                time.sleep(wait)

        try:
            producer.send(
                TOPIC_MEASURES,
                key=str(row["motor_id"]),
                value=msg
            ).add_errback(on_send_error)
            n_sent += 1
            if n_sent % 100 == 0:
                log.info(f"  Replay : {n_sent}/{len(df)} messages envoyés")
        except Exception as e:
            n_errors += 1
            log.warning(f"Erreur envoi : {e}")

        prev_ts = row["timestamp"]

    producer.flush()
    log.info(f"✓ Replay terminé : {n_sent} messages, {n_errors} erreurs")


# ══════════════════════════════════════════════════════════════════
#  MODE SIMULATION — Moteur unique
# ══════════════════════════════════════════════════════════════════
def run_motor_thread(producer: KafkaProducer, motor_id: int, interval: float):
    """Thread dédié à un moteur — envoie en continu."""
    sim = MotorSimulator(motor_id)
    log.info(f"  Thread M{motor_id} démarré (interval={interval}s)")

    while True:
        try:
            measure = sim.generate_measurement()

            # Envoi mesure principale
            producer.send(
                TOPIC_MEASURES,
                key=str(motor_id),
                value=measure
            ).add_callback(on_send_success).add_errback(on_send_error)

            # Envoi alerte si nécessaire
            if measure["Alert_Status"] == "ALERT":
                alert_msg = {
                    "motor_id"   : motor_id,
                    "timestamp"  : measure["timestamp"],
                    "parameter"  : measure["alert_parameter"],
                    "value"      : measure.get(measure["alert_parameter"], 0),
                    "severity"   : "HIGH" if measure["degradation_sim"] > 0.6 else "MEDIUM",
                    "degradation": measure["degradation_sim"],
                }
                producer.send(TOPIC_ALERTS, key=str(motor_id), value=alert_msg)
                log.warning(f"⚠ ALERTE M{motor_id} : {alert_msg['parameter']}")

            time.sleep(interval + random.uniform(-1, 1))  # Léger jitter

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Erreur M{motor_id} : {e}")
            time.sleep(5)


# ══════════════════════════════════════════════════════════════════
#  HEARTBEAT — Thread de monitoring
# ══════════════════════════════════════════════════════════════════
def run_heartbeat(producer: KafkaProducer):
    """Envoie un heartbeat toutes les 30s pour monitoring."""
    while True:
        try:
            hb = {
                "service"   : "kafka-producer",
                "timestamp" : datetime.now(timezone.utc).isoformat(),
                "n_motors"  : N_MOTORS,
                "status"    : "running",
                "uptime_s"  : time.monotonic(),
            }
            producer.send(TOPIC_HEARTBEAT, key="producer", value=hb)
        except Exception as e:
            log.error(f"Heartbeat erreur : {e}")
        time.sleep(30)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Kafka Producer — Capteurs Moteurs")
    parser.add_argument("--motor",    type=int,   default=None,  help="Moteur spécifique")
    parser.add_argument("--interval", type=float, default=INTERVAL_SEC, help="Intervalle (sec)")
    parser.add_argument("--replay",   type=str,   default=None,  help="CSV à rejouer")
    parser.add_argument("--speed",    type=float, default=10.0,  help="Vitesse replay")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════╗
║   KAFKA PRODUCER — Maintenance Prédictive    ║
║   Topics : motor.measurements / alerts       ║
╚══════════════════════════════════════════════╝
""")

    producer = create_producer()

    # ── Mode Replay CSV ───────────────────────────
    if args.replay:
        replay_csv(producer, args.replay, args.speed)
        return

    # ── Mode Simulation ───────────────────────────
    motors = [args.motor] if args.motor else list(range(1, N_MOTORS + 1))
    log.info(f"Simulation de {len(motors)} moteur(s), interval={args.interval}s")

    # Thread heartbeat
    hb_thread = threading.Thread(target=run_heartbeat, args=(producer,), daemon=True)
    hb_thread.start()

    # Un thread par moteur
    threads = []
    for mid in motors:
        t = threading.Thread(
            target=run_motor_thread,
            args=(producer, mid, args.interval),
            daemon=True
        )
        t.start()
        threads.append(t)
        time.sleep(0.1)  # Décalage léger pour éviter burst initial

    log.info(f"✓ {len(threads)} threads producteurs actifs")
    log.info("  CTRL+C pour arrêter")

    try:
        while True:
            time.sleep(60)
            # Stats périodiques
            log.info(f"  Status : {len(motors)} moteurs actifs | "
                     f"{datetime.now().strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        log.info("\n⏹ Arrêt demandé — flush en cours...")
        producer.flush(timeout=10)
        producer.close()
        log.info("✓ Producer arrêté proprement")


if __name__ == "__main__":
    main()
