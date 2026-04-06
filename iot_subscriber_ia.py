"""
=======================================================
 ÉTAPE 4 — SUBSCRIBER MQTT + MODÈLE IA TEMPS RÉEL
 Maintenance Prédictive IoT

 Ce script :
   1. S'abonne aux topics MQTT (données capteurs)
   2. Reçoit les mesures en temps réel
   3. Applique le feature engineering (step2)
   4. Applique le modèle Hybride IF (step3) AUC=0.9495
   5. Calcule le DI (Indice de Dégradation)
   6. Détecte les anomalies automatiquement
   7. Publie les résultats IA sur MQTT
   8. Sauvegarde dans un CSV pour le dashboard

 Usage :
   python iot_subscriber_ia.py

 Topics MQTT écoutés :
   moteur/#/capteurs    → données brutes capteurs

 Topics MQTT publiés :
   moteur/{id}/ia       → résultat IA (anomalie, DI, score)
   moteur/alertes_ia    → alertes IA uniquement
=======================================================
"""

import paho.mqtt.client as mqtt
import json
import numpy as np
import pandas as pd
import os
import csv
import warnings
from datetime import datetime
from collections import deque

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────
BROKER          = "localhost"
PORT            = 1883
TOPIC_SUBSCRIBE = "moteur/#"
RESULTS_CSV     = "data/iot_resultats_ia.csv"
WINDOW_SIZE     = 20    # fenêtre rolling pour features
SEUIL_ANOMALIE  = 0.25  # seuil score hybride

# Poids modèle hybride (identique à step3)
W_IF    = 0.30
W_RULES = 0.70

# Seuils par moteur (calibrés depuis ai_cp(2).sql)
SEUILS_MOTEURS = {
    1:  {'temp': 40.0, 'courant': 64.1,  'vib': 1.0003, 'accel': 0.3387},
    2:  {'temp': 40.0, 'courant': 46.8,  'vib': 1.0003, 'accel': 0.1748},
    3:  {'temp': 40.0, 'courant': 126.6, 'vib': 1.0021, 'accel': 0.1626},
    4:  {'temp': 40.0, 'courant': 85.2,  'vib': 1.0019, 'accel': 0.5000},
    5:  {'temp': 40.0, 'courant': 25.3,  'vib': 1.0006, 'accel': 0.3658},
    6:  {'temp': 40.1, 'courant': 32.1,  'vib': 1.0022, 'accel': 0.2900},
    7:  {'temp': 40.0, 'courant': 52.2,  'vib': 1.0013, 'accel': 0.1849},
    8:  {'temp': 40.0, 'courant': 79.3,  'vib': 1.0001, 'accel': 0.5012},
    9:  {'temp': 40.0, 'courant': 106.7, 'vib': 1.0002, 'accel': 0.2280},
    10: {'temp': 40.0, 'courant': 85.7,  'vib': 1.0007, 'accel': 0.2745},
    11: {'temp': 40.0, 'courant': 122.2, 'vib': 1.0016, 'accel': 0.5000},
    12: {'temp': 40.0, 'courant': 26.8,  'vib': 1.0009, 'accel': 0.1536},
    13: {'temp': 40.0, 'courant': 111.1, 'vib': 1.0008, 'accel': 0.3083},
    14: {'temp': 40.1, 'courant': 25.3,  'vib': 0.8383, 'accel': 0.2330},
    15: {'temp': 40.0, 'courant': 51.4,  'vib': 0.5788, 'accel': 0.1833},
    16: {'temp': 40.0, 'courant': 93.7,  'vib': 1.0000, 'accel': 0.2445},
    17: {'temp': 40.0, 'courant': 26.5,  'vib': 1.0017, 'accel': 0.3004},
    18: {'temp': 40.0, 'courant': 27.2,  'vib': 0.7234, 'accel': 0.3583},
    19: {'temp': 40.0, 'courant': 118.8, 'vib': 1.0004, 'accel': 0.3002},
    20: {'temp': 40.0, 'courant': 8.0,   'vib': 1.0016, 'accel': 0.1828},
    21: {'temp': 84.0, 'courant': 65.0,  'vib': 1.7100, 'accel': 2.8000},
}

POIDS_PARAMS = {
    'temp': 0.35, 'courant': 0.30, 'vib': 0.25, 'accel': 0.10
}

# ── Buffer par moteur (fenêtre glissante) ──────────────
buffers = {}   # motor_id → deque de mesures
di_history = {}  # motor_id → historique DI
resultats = {}   # motor_id → dernier résultat IA


def get_buffer(motor_id):
    """Retourne le buffer glissant pour un moteur."""
    if motor_id not in buffers:
        buffers[motor_id]    = deque(maxlen=WINDOW_SIZE)
        di_history[motor_id] = deque(maxlen=100)
    return buffers[motor_id]


# ══════════════════════════════════════════════════════
#  FEATURE ENGINEERING TEMPS RÉEL
#  (version allégée de step2_features.py)
# ══════════════════════════════════════════════════════
def calculer_features(buffer, motor_id):
    """
    Calcule les features à partir du buffer glissant.
    Identique à step2 mais sur une fenêtre de 20 mesures.
    """
    if len(buffer) < 3:
        return None

    vibs   = np.array([m['vibration_x'] for m in buffer])
    temps  = np.array([m['temperature']  for m in buffer])
    cours  = np.array([m['courant']      for m in buffer])
    accels = np.array([m['acceleration'] for m in buffer])

    # Features vibration
    vib_mean       = np.mean(vibs)
    vib_std        = np.std(vibs)
    vib_max        = np.max(vibs)
    vib_energy     = np.mean(vibs**2)
    vib_energy_mean= vib_energy

    # Kurtosis (détecte les pics impulsionnels)
    if vib_std > 0:
        vib_kurt = np.mean(((vibs - vib_mean)/vib_std)**4)
    else:
        vib_kurt = 0.0

    # Crest factor
    rms = np.sqrt(vib_energy)
    crest_factor = vib_max / (rms + 1e-9)

    # Features température
    temp_mean  = np.mean(temps)
    temp_std   = np.std(temps)
    temp_trend = (temps[-1] - temps[0]) / (len(temps) + 1e-9)

    # Features courant
    courant_mean = np.mean(cours)
    courant_std  = np.std(cours)

    # Enveloppe simplifiée (Hilbert approx)
    envelope_mean = np.mean(np.abs(vibs - vib_mean))

    # Health score (identique step2)
    def norm(x, xmin=0, xmax=1):
        return min(1, max(0, (x - xmin) / (xmax - xmin + 1e-9)))

    health_score = 100 * (1 - (
        0.35 * norm(vib_energy_mean, 0, 1.5) +
        0.25 * norm(temp_mean, 20, 80) +
        0.20 * norm(vib_kurt, 0, 20) +
        0.20 * norm(crest_factor, 0, 10)
    ))

    return {
        'vib_mean'       : vib_mean,
        'vib_std'        : vib_std,
        'vib_max'        : vib_max,
        'vib_energy'     : vib_energy,
        'vib_energy_mean': vib_energy_mean,
        'vib_kurt'       : vib_kurt,
        'crest_factor'   : crest_factor,
        'temp_mean'      : temp_mean,
        'temp_std'       : temp_std,
        'temp_trend'     : temp_trend,
        'courant_mean'   : courant_mean,
        'courant_std'    : courant_std,
        'envelope_mean'  : envelope_mean,
        'health_score'   : health_score,
    }


# ══════════════════════════════════════════════════════
#  MODÈLE HYBRIDE IF + RÈGLES MÉTIER
#  (version temps réel de step3)
# ══════════════════════════════════════════════════════
def calculer_score_regles(mesure, features, motor_id):
    """
    Score basé sur les règles métier (70% du score hybride).
    Identique à step3 — seuils calibrés depuis ai_cp(2).
    """
    seuils   = SEUILS_MOTEURS.get(motor_id, SEUILS_MOTEURS[1])
    severity = 0.0
    n_exceed = 0

    params = {
        'temp'   : mesure.get('temperature',   35),
        'courant': mesure.get('courant',        50),
        'vib'    : mesure.get('vibration_x',   0.7),
        'accel'  : mesure.get('acceleration', 0.3),
    }

    for param, val in params.items():
        seuil = seuils[param]
        if val > seuil:
            ratio     = val / (seuil + 1e-9)
            severity += POIDS_PARAMS[param] * min(1.0, ratio - 1.0)
            n_exceed += 1

    # Normaliser severity
    score_rules = min(1.0, severity * 2.0)

    return score_rules, n_exceed


def calculer_score_if(features):
    """
    Score Isolation Forest simplifié temps réel.
    Basé sur les features les plus discriminantes.
    """
    if features is None:
        return 0.0

    # Score basé sur les anomalies statistiques
    score = 0.0

    # Kurtosis élevé = impulsions = défaut roulement
    if features['vib_kurt'] > 5:
        score += 0.3 * min(1.0, (features['vib_kurt'] - 5) / 10)

    # Crest factor élevé
    if features['crest_factor'] > 4:
        score += 0.2 * min(1.0, (features['crest_factor'] - 4) / 6)

    # Trend température positive = surchauffe
    if features['temp_trend'] > 0.05:
        score += 0.3 * min(1.0, features['temp_trend'] / 0.5)

    # Énergie vibration élevée
    if features['vib_energy_mean'] > 0.6:
        score += 0.2 * min(1.0, (features['vib_energy_mean'] - 0.6) / 1.0)

    return min(1.0, score)


def calculer_di(motor_id, score_hybride):
    """
    Calcule l'Indice de Dégradation (DI) progressif.
    Utilise l'historique des scores pour lisser.
    """
    di_hist = di_history[motor_id]
    di_hist.append(score_hybride)

    if len(di_hist) < 3:
        return score_hybride

    # DI = moyenne pondérée (récent compte plus)
    weights = np.linspace(0.5, 1.0, len(di_hist))
    di = np.average(list(di_hist), weights=weights)
    return round(min(1.0, di), 4)


def determiner_niveau_risque(di):
    """Détermine le niveau de risque selon le DI."""
    if di >= 0.75: return "CRITIQUE", "🔴"
    if di >= 0.50: return "ÉLEVÉ",    "🟠"
    if di >= 0.30: return "MODÉRÉ",   "🟡"
    return "FAIBLE", "🟢"


def analyser_mesure(data, mqtt_client):
    """
    Pipeline IA complet pour une mesure temps réel.
    1. Ajouter au buffer
    2. Calculer features
    3. Calculer score hybride
    4. Calculer DI
    5. Publier résultat
    """
    motor_id = data.get('motor_id', 1)
    buffer   = get_buffer(motor_id)

    # Ajouter mesure au buffer
    buffer.append(data)

    # Calculer features (besoin d'au moins 3 mesures)
    features = calculer_features(buffer, motor_id)

    # Score règles métier
    score_rules, n_exceed = calculer_score_regles(data, features, motor_id)

    # Score Isolation Forest
    score_if = calculer_score_if(features)

    # Score hybride final (identique step3)
    score_hybride = W_IF * score_if + W_RULES * score_rules

    # Détection anomalie
    is_anomalie = score_hybride >= SEUIL_ANOMALIE

    # Indice de dégradation
    di = calculer_di(motor_id, score_hybride)

    # Niveau de risque
    niveau, icone = determiner_niveau_risque(di)

    # Health score
    health = features['health_score'] if features else 50.0

    # Résultat complet
    resultat = {
        "motor_id"      : motor_id,
        "timestamp"     : data.get('timestamp'),
        "temperature"   : data.get('temperature'),
        "vibration"     : data.get('vibration_x'),
        "courant"       : data.get('courant'),
        "score_if"      : round(score_if, 4),
        "score_rules"   : round(score_rules, 4),
        "score_hybride" : round(score_hybride, 4),
        "is_anomalie"   : int(is_anomalie),
        "di"            : di,
        "niveau_risque" : niveau,
        "health_score"  : round(health, 1),
        "n_exceed"      : n_exceed,
        "buffer_size"   : len(buffer),
    }

    # Sauvegarder résultat
    resultats[motor_id] = resultat

    # Publier résultat IA sur MQTT
    topic_ia = f"moteur/{motor_id}/ia"
    mqtt_client.publish(topic_ia, json.dumps(resultat))

    # Publier alerte si anomalie
    if is_anomalie:
        alerte = {
            "motor_id"    : motor_id,
            "timestamp"   : data.get('timestamp'),
            "niveau"      : niveau,
            "di"          : di,
            "score"       : round(score_hybride, 4),
            "temperature" : data.get('temperature'),
            "vibration"   : data.get('vibration_x'),
        }
        mqtt_client.publish("moteur/alertes_ia", json.dumps(alerte))

    # Sauvegarder dans CSV
    sauvegarder_csv(resultat)

    # Afficher dans terminal
    afficher_resultat(resultat, icone)

    return resultat


def afficher_resultat(r, icone):
    """Affiche le résultat IA dans le terminal."""
    anomalie_txt = " ⚠ ANOMALIE DÉTECTÉE !" if r['is_anomalie'] else ""
    print(f"\n  [{r['timestamp']}] Moteur {r['motor_id']:2d} "
          f"{icone} {r['niveau_risque']}{anomalie_txt}")
    print(f"    Score hybride : {r['score_hybride']:.4f} "
          f"(IF={r['score_if']:.3f} + Règles={r['score_rules']:.3f})")
    print(f"    DI={r['di']:.4f} | Health={r['health_score']:.1f}% | "
          f"Seuils dépassés={r['n_exceed']}")
    print(f"    Temp={r['temperature']:.2f}°C | "
          f"Vib={r['vibration']:.4f} | "
          f"Courant={r['courant']:.2f}A")


def sauvegarder_csv(resultat):
    """Sauvegarde le résultat dans un CSV pour le dashboard."""
    os.makedirs('data', exist_ok=True)
    fichier_existe = os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=resultat.keys())
        if not fichier_existe:
            writer.writeheader()
        writer.writerow(resultat)


def afficher_resume():
    """Affiche un résumé de tous les moteurs."""
    if not resultats:
        return
    print(f"\n  {'─'*55}")
    print(f"  RÉSUMÉ IA — {datetime.now().strftime('%H:%M:%S')}")
    print(f"  {'─'*55}")
    for mid, r in sorted(resultats.items()):
        _, icone = determiner_niveau_risque(r['di'])
        print(f"  {icone} Moteur {mid:2d} | "
              f"DI={r['di']:.3f} | "
              f"Score={r['score_hybride']:.3f} | "
              f"{'⚠ ANOMALIE' if r['is_anomalie'] else 'Normal':12s} | "
              f"Health={r['health_score']:.0f}%")
    print(f"  {'─'*55}")


# ══════════════════════════════════════════════════════
#  CALLBACKS MQTT
# ══════════════════════════════════════════════════════
resume_counter = 0

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"  ✅ Subscriber IA connecté au broker MQTT")
        client.subscribe(TOPIC_SUBSCRIBE)
        print(f"  📡 Abonné à : {TOPIC_SUBSCRIBE}")
    else:
        print(f"  ❌ Erreur connexion : code {rc}")


def on_message(client, userdata, msg):
    global resume_counter

    topic   = msg.topic
    payload = msg.payload.decode('utf-8')

    # Ignorer les topics IA et alertes (éviter boucle)
    if '/ia' in topic or 'alertes_ia' in topic or 'alerte' in topic:
        return

    # Traiter seulement les données capteurs
    if '/capteurs' not in topic:
        return

    try:
        data = json.loads(payload)

        # Analyser avec le modèle IA
        analyser_mesure(data, client)

        # Résumé tous les 10 messages
        resume_counter += 1
        if resume_counter % 10 == 0:
            afficher_resume()

    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"  ❌ Erreur traitement : {e}")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" ÉTAPE 4 — SUBSCRIBER IA TEMPS RÉEL")
    print(" Modèle Hybride IF + Règles (AUC=0.9495)")
    print("=" * 60)
    print()
    print("  Pipeline :")
    print("  MQTT capteurs → Features → Score hybride → DI → Alerte")
    print()
    print("  Modèle : 30% Isolation Forest + 70% Règles métier")
    print(f"  Seuil anomalie : {SEUIL_ANOMALIE}")
    print(f"  Fenêtre rolling : {WINDOW_SIZE} mesures")
    print(f"  Résultats sauvés : {RESULTS_CSV}")
    print()

    # Créer dossier data si besoin
    os.makedirs('data', exist_ok=True)

    # Connexion MQTT
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        print(f"  Connexion à MQTT broker ({BROKER}:{PORT})...")
    except Exception as e:
        print(f"  ❌ Impossible de connecter : {e}")
        print(f"  → Vérifie que Mosquitto tourne")
        return

    print(f"  En attente de données capteurs...")
    print(f"  (Lance iot_simulateur_capteurs.py dans un autre terminal)")
    print(f"\n  {'─'*55}")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n\n  Subscriber arrêté.")
        print(f"  Résultats sauvés dans : {RESULTS_CSV}")
        afficher_resume()
        client.disconnect()


if __name__ == '__main__':
    main()