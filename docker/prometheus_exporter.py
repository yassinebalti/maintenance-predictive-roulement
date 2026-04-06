"""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 3 — MONITORING : PROMETHEUS METRICS EXPORTER             ║
║  Maintenance Prédictive — Pipeline IA/ML                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Expose les métriques ML au format Prometheus (/metrics)         ║
║  Prometheus scrape → Grafana visualise                           ║
║                                                                  ║
║  Métriques exposées :                                            ║
║    • ml_motor_degradation_index{motor_id}                        ║
║    • ml_motor_rul_days{motor_id}                                 ║
║    • ml_motor_risk_level{motor_id}                               ║
║    • ml_motor_anomaly_score{motor_id}                            ║
║    • ml_motor_temperature{motor_id}                              ║
║    • ml_motor_vibration{motor_id}                                ║
║    • ml_motor_cusum_alarm{motor_id}                              ║
║    • ml_pipeline_messages_total                                  ║
║    • ml_pipeline_anomalies_total                                 ║
║    • ml_pipeline_latency_ms                                      ║
║    • ml_pipeline_uptime_seconds                                  ║
║                                                                  ║
║  Usage :                                                         ║
║    python prometheus_exporter.py                                 ║
║    → http://localhost:8000/metrics                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import time
import os
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

RESULTS_DIR = "data/stream_results"
LATEST_JSON = os.path.join(RESULTS_DIR, "latest.json")
METRICS_PORT = 8000
SCRAPE_INTERVAL = 5   # secondes entre lectures du JSON

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PROMETHEUS] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RISK_LEVELS_MAP = {"CRITIQUE": 3, "ÉLEVÉ": 2, "MODÉRÉ": 1, "FAIBLE": 0}


# ══════════════════════════════════════════════════════════════════
#  REGISTRE DE MÉTRIQUES (sans dépendance prometheus_client)
# ══════════════════════════════════════════════════════════════════
class MetricsRegistry:
    """
    Implémentation légère d'un registre Prometheus.
    Compatible avec le format text/plain Prometheus.
    Pas besoin de la librairie prometheus_client.
    """
    def __init__(self):
        self.gauges    = {}
        self.counters  = {}
        self.summaries = {}
        self.lock      = threading.Lock()
        self.start_time = time.time()

    def set_gauge(self, name: str, value: float, labels: dict = None):
        key = self._make_key(name, labels)
        with self.lock:
            self.gauges[key] = {
                "name"   : name,
                "value"  : value,
                "labels" : labels or {},
                "ts"     : time.time(),
            }

    def inc_counter(self, name: str, amount: float = 1, labels: dict = None):
        key = self._make_key(name, labels)
        with self.lock:
            if key not in self.counters:
                self.counters[key] = {"name": name, "value": 0, "labels": labels or {}}
            self.counters[key]["value"] += amount

    def _make_key(self, name: str, labels: dict) -> str:
        if not labels:
            return name
        lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{lbl_str}}}"

    def render(self) -> str:
        """Génère le texte au format Prometheus /metrics."""
        lines = []

        # Uptime
        uptime = time.time() - self.start_time
        lines.append("# HELP ml_pipeline_uptime_seconds Temps depuis démarrage de l'exporter")
        lines.append("# TYPE ml_pipeline_uptime_seconds gauge")
        lines.append(f"ml_pipeline_uptime_seconds {uptime:.1f}")
        lines.append("")

        # Gauges
        current_name = None
        with self.lock:
            gauge_items = list(self.gauges.items())
        for key, meta in sorted(gauge_items, key=lambda x: x[1]["name"]):
            name = meta["name"]
            if name != current_name:
                lines.append(f"# HELP {name} Métrique ML — Maintenance Prédictive")
                lines.append(f"# TYPE {name} gauge")
                current_name = name
            labels = meta["labels"]
            if labels:
                lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                lines.append(f'{name}{{{lbl_str}}} {meta["value"]:.6g}')
            else:
                lines.append(f'{name} {meta["value"]:.6g}')
        if gauge_items:
            lines.append("")

        # Counters
        current_name = None
        with self.lock:
            counter_items = list(self.counters.items())
        for key, meta in sorted(counter_items, key=lambda x: x[1]["name"]):
            name = meta["name"]
            if name != current_name:
                lines.append(f"# HELP {name}_total Compteur — {name}")
                lines.append(f"# TYPE {name}_total counter")
                current_name = name
            labels = meta["labels"]
            if labels:
                lbl_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                lines.append(f'{name}_total{{{lbl_str}}} {meta["value"]:.0f}')
            else:
                lines.append(f'{name}_total {meta["value"]:.0f}')
        if counter_items:
            lines.append("")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  COLLECTEUR — Lit le JSON et met à jour les métriques
# ══════════════════════════════════════════════════════════════════
class MLMetricsCollector:
    def __init__(self, registry: MetricsRegistry):
        self.registry    = registry
        self.last_proc   = 0
        self.n_anomalies = 0

    def collect(self):
        """Lit latest.json et met à jour toutes les métriques."""
        if not os.path.exists(LATEST_JSON):
            log.warning("Fichier JSON absent — consumer non lancé ?")
            return

        try:
            with open(LATEST_JSON) as f:
                data = json.load(f)
        except Exception as e:
            log.error(f"Erreur lecture JSON : {e}")
            return

        motors  = data.get("motors", {})
        summary = data.get("summary", {})
        n_proc  = data.get("n_processed", 0)

        # ── Métriques globales pipeline ───────────
        self.registry.set_gauge("ml_pipeline_messages_processed", n_proc)
        self.registry.set_gauge("ml_pipeline_motors_active",
                                 summary.get("n_motors", len(motors)))
        self.registry.set_gauge("ml_pipeline_avg_degradation_index",
                                 summary.get("avg_di", 0))
        self.registry.set_gauge("ml_pipeline_n_critique",
                                 summary.get("n_critique", 0))
        self.registry.set_gauge("ml_pipeline_n_eleve",
                                 summary.get("n_eleve", 0))
        self.registry.set_gauge("ml_pipeline_n_modere",
                                 summary.get("n_modere", 0))
        self.registry.set_gauge("ml_pipeline_n_faible",
                                 summary.get("n_faible", 0))
        self.registry.set_gauge("ml_pipeline_cusum_alarms",
                                 len(summary.get("cusum_alarms", [])))

        # Delta messages (taux de traitement)
        delta_proc = n_proc - self.last_proc
        self.last_proc = n_proc
        self.registry.set_gauge("ml_pipeline_messages_rate_per_scrape", delta_proc)

        # ── Métriques par moteur ──────────────────
        for mid_str, v in motors.items():
            mid = str(mid_str)
            lbl = {"motor_id": mid}

            # Dégradation
            self.registry.set_gauge(
                "ml_motor_degradation_index",
                v.get("degradation_index", 0), lbl)

            # RUL (numérique)
            rul_num = v.get("rul_num", 90)
            self.registry.set_gauge("ml_motor_rul_days", rul_num, lbl)
            self.registry.set_gauge("ml_motor_rul_low",  v.get("rul_low",  70), lbl)
            self.registry.set_gauge("ml_motor_rul_high", v.get("rul_high", 90), lbl)

            # Niveau de risque (0=FAIBLE, 1=MODÉRÉ, 2=ÉLEVÉ, 3=CRITIQUE)
            risk_num = RISK_LEVELS_MAP.get(v.get("risk_level", "FAIBLE"), 0)
            self.registry.set_gauge("ml_motor_risk_level", risk_num, lbl)

            # Score anomalie
            self.registry.set_gauge(
                "ml_motor_anomaly_score",
                v.get("combined_score", 0), lbl)
            self.registry.set_gauge("ml_motor_score_if",    v.get("score_if", 0), lbl)
            self.registry.set_gauge("ml_motor_score_lof",   v.get("score_lof", 0), lbl)
            self.registry.set_gauge("ml_motor_score_rules", v.get("score_rules", 0), lbl)

            # Anomalie (0/1)
            self.registry.set_gauge(
                "ml_motor_is_anomaly",
                1 if v.get("is_anomaly") else 0, lbl)

            # Physique
            self.registry.set_gauge("ml_motor_temperature", v.get("temperature", 0), lbl)
            self.registry.set_gauge("ml_motor_vibration",   v.get("vibration", 0), lbl)
            self.registry.set_gauge("ml_motor_courant",     v.get("courant", 0), lbl)

            # Health score
            self.registry.set_gauge("ml_motor_health_score", v.get("health_score", 0), lbl)

            # CUSUM
            self.registry.set_gauge(
                "ml_motor_cusum_alarm",
                1 if v.get("cusum_alarm") else 0, lbl)
            self.registry.set_gauge(
                "ml_motor_cusum_s_pos",
                v.get("cusum_s_pos", 0), lbl)

            # Features vibratoires
            for feat in ["vib_rms","vib_skewness","peak2peak",
                         "spectral_entropy","shape_factor","impulse_factor"]:
                self.registry.set_gauge(f"ml_motor_{feat}", v.get(feat, 0), lbl)

            # Compteur anomalies
            if v.get("is_anomaly"):
                self.registry.inc_counter("ml_motor_anomaly_events", 1, lbl)

        log.debug(f"Métriques collectées : {len(motors)} moteurs, {n_proc:,} messages")


# ══════════════════════════════════════════════════════════════════
#  SERVEUR HTTP — Endpoint /metrics
# ══════════════════════════════════════════════════════════════════
class PrometheusHandler(BaseHTTPRequestHandler):
    registry: MetricsRegistry = None

    def do_GET(self):
        if self.path == "/metrics":
            content = self.registry.render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type",
                             "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            health = json.dumps({
                "status"   : "ok",
                "timestamp": datetime.utcnow().isoformat(),
                "json_exists": os.path.exists(LATEST_JSON),
            }).encode("utf-8")
            self.wfile.write(health)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found. Use /metrics or /health")

    def log_message(self, format, *args):
        pass  # Silencieux (évite le spam de logs par Prometheus)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print(f"""
╔══════════════════════════════════════════════════════╗
║   PROMETHEUS EXPORTER — Maintenance Prédictive       ║
║   Port : {METRICS_PORT}                                     ║
║   Endpoint : http://localhost:{METRICS_PORT}/metrics         ║
║   Health   : http://localhost:{METRICS_PORT}/health          ║
╚══════════════════════════════════════════════════════╝
""")

    registry  = MetricsRegistry()
    collector = MLMetricsCollector(registry)

    # Thread collecteur (scrape JSON en arrière-plan)
    def collect_loop():
        while True:
            try:
                collector.collect()
            except Exception as e:
                log.error(f"Erreur collecteur : {e}")
            time.sleep(SCRAPE_INTERVAL)

    t = threading.Thread(target=collect_loop, daemon=True)
    t.start()
    log.info(f"Collecteur démarré (interval={SCRAPE_INTERVAL}s)")

    # Collecte initiale
    collector.collect()

    # Serveur HTTP
    PrometheusHandler.registry = registry
    server = HTTPServer(("0.0.0.0", METRICS_PORT), PrometheusHandler)
    log.info(f"✓ Exposition métriques sur http://0.0.0.0:{METRICS_PORT}/metrics")
    log.info("  CTRL+C pour arrêter")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Arrêt exporter...")
        server.shutdown()


if __name__ == "__main__":
    main()
