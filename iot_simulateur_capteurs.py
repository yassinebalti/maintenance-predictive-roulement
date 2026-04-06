"""
=======================================================
 SIMULATEUR CAPTEURS IoT — Maintenance Prédictive
 Simule un Raspberry Pi avec capteurs réels

 Ce script simule :
   → Accéléromètre MEMS  (vibration X, Y, Z)
   → Capteur courant Hall (ACS712)
   → Tachymètre optique  (RPM)
   → Capteur température

 Publication MQTT toutes les 5 secondes
 Topics :
   moteur/{id}/capteurs  → données brutes
   moteur/{id}/alerte    → alertes détectées
   moteur/status         → état global

 Usage :
   python iot_simulateur_capteurs.py
   python iot_simulateur_capteurs.py --moteur 5 --mode panne
=======================================================
"""

import paho.mqtt.client as mqtt
import json
import time
import numpy as np
import argparse
from datetime import datetime

# ── Configuration ──────────────────────────────────────
BROKER       = "localhost"
PORT         = 1883
INTERVAL_SEC = 5       # mesure toutes les 5 secondes

# Profils normaux par moteur (identiques à tes données SQL)
PROFILS = {
    1:  {'temp': 35, 'courant': 57,  'vib': 0.70, 'accel': 0.30, 'rpm': 1495},
    2:  {'temp': 35, 'courant': 42,  'vib': 0.70, 'accel': 0.30, 'rpm': 1495},
    3:  {'temp': 35, 'courant': 112, 'vib': 0.70, 'accel': 0.30, 'rpm': 1495},
    4:  {'temp': 35, 'courant': 76,  'vib': 0.69, 'accel': 0.30, 'rpm': 1495},
    5:  {'temp': 35, 'courant': 23,  'vib': 0.69, 'accel': 0.30, 'rpm': 1495},
    18: {'temp': 35, 'courant': 25,  'vib': 0.70, 'accel': 0.30, 'rpm': 1495},
    21: {'temp': 74, 'courant': 62,  'vib': 1.33, 'accel': 0.50, 'rpm': 1495},
}

# Seuils d'alerte par moteur
SEUILS = {
    1:  {'temp': 40, 'courant': 64,  'vib': 1.00, 'accel': 0.34},
    2:  {'temp': 40, 'courant': 47,  'vib': 1.00, 'accel': 0.17},
    3:  {'temp': 40, 'courant': 127, 'vib': 1.00, 'accel': 0.16},
    4:  {'temp': 40, 'courant': 85,  'vib': 1.00, 'accel': 0.50},
    5:  {'temp': 40, 'courant': 25,  'vib': 1.00, 'accel': 0.37},
    18: {'temp': 40, 'courant': 27,  'vib': 0.72, 'accel': 0.36},
    21: {'temp': 84, 'courant': 65,  'vib': 1.71, 'accel': 2.80},
}


class SimulateurCapteur:
    """
    Simule un capteur IoT sur Raspberry Pi.
    Modes disponibles :
      normal  → comportement sain
      panne   → dégradation progressive (surchauffe)
      roulement → défaut roulement (vibration montante)
      critique  → panne imminente
    """

    def __init__(self, motor_id, mode='normal'):
        self.motor_id  = motor_id
        self.mode      = mode
        self.step      = 0
        self.profil    = PROFILS.get(motor_id, PROFILS[1])
        self.seuil     = SEUILS.get(motor_id, SEUILS[1])
        self.start_time = time.time()

        print(f"  Moteur {motor_id} initialisé — mode : {mode.upper()}")

    def lire_temperature(self):
        """Simule capteur température DS18B20."""
        base = self.profil['temp']

        if self.mode == 'panne':
            # Surchauffe progressive +0.1°C par mesure
            base += self.step * 0.15
        elif self.mode == 'critique':
            base += self.step * 0.40

        return round(base + np.random.normal(0, 0.5), 2)

    def lire_vibration(self):
        """Simule accéléromètre MEMS ADXL355 — 3 axes."""
        base = self.profil['vib']

        if self.mode == 'roulement':
            # Défaut roulement : vibration monte progressivement
            base += self.step * 0.008
            # Ajouter impulsions périodiques (signature roulement)
            if self.step % 10 == 0:
                base += np.random.uniform(0.1, 0.3)
        elif self.mode == 'critique':
            base += self.step * 0.025

        vib_x = round(base + np.random.normal(0, 0.02), 4)
        vib_y = round(base * 0.85 + np.random.normal(0, 0.02), 4)
        vib_z = round(base * 0.60 + np.random.normal(0, 0.01), 4)
        return vib_x, vib_y, vib_z

    def lire_courant(self):
        """Simule capteur courant ACS712."""
        base = self.profil['courant']

        if self.mode == 'panne':
            # Surcharge progressive
            base += self.step * 0.05
        elif self.mode == 'critique':
            base += self.step * 0.10

        return round(base + np.random.normal(0, base * 0.03), 2)

    def lire_vitesse(self):
        """Simule tachymètre optique — RPM."""
        base = self.profil['rpm']
        if self.mode in ('panne', 'critique'):
            # Légère instabilité vitesse
            base += np.random.normal(0, 8)
        return round(base + np.random.normal(0, 3), 1)

    def detecter_alerte(self, temp, courant, vib_x, accel):
        """Détecte les dépassements de seuils."""
        seuil   = self.seuil
        alertes = []

        if temp    > seuil['temp']   : alertes.append('temperature')
        if courant > seuil['courant']: alertes.append('courant')
        if vib_x   > seuil['vib']    : alertes.append('vibration')
        if accel   > seuil['accel']  : alertes.append('acceleration')

        if not alertes:
            return 'NORMAL', None

        # Paramètre principal en alerte
        return 'ALERT', alertes[0]

    def lire_tous_capteurs(self):
        """Lit tous les capteurs et retourne un dictionnaire."""
        temp             = self.lire_temperature()
        vib_x, vib_y, vib_z = self.lire_vibration()
        courant          = self.lire_courant()
        vitesse          = self.lire_vitesse()
        accel            = round(vib_z * 0.3 + np.random.normal(0, 0.01), 4)

        alert_status, alert_param = self.detecter_alerte(
            temp, courant, vib_x, accel)

        self.step += 1

        return {
            "motor_id"      : self.motor_id,
            "timestamp"     : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "temperature"   : temp,
            "courant"       : courant,
            "vibration_x"   : vib_x,
            "vibration_y"   : vib_y,
            "vibration_z"   : vib_z,
            "acceleration"  : accel,
            "vitesse_rpm"   : vitesse,
            "cosphi"        : round(0.850 + np.random.normal(0, 0.001), 3),
            "Alert_Status"  : alert_status,
            "alert_param"   : alert_param,
            "mode_simulation": self.mode,
            "step"          : self.step,
        }


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"  ✅ Connecté au broker MQTT (localhost:{PORT})")
    else:
        print(f"  ❌ Erreur connexion MQTT : code {rc}")


def afficher_mesure(data):
    """Affiche la mesure dans le terminal de façon lisible."""
    status_icon = "🔴" if data['Alert_Status'] == 'ALERT' else "🟢"
    alert_info  = f" ← ALERTE : {data['alert_param']}" \
                  if data['alert_param'] else ""

    print(f"\n  [{data['timestamp']}] Moteur {data['motor_id']:2d} "
          f"{status_icon} {data['Alert_Status']}{alert_info}")
    print(f"    Temp={data['temperature']:6.2f}°C | "
          f"Courant={data['courant']:7.2f}A | "
          f"Vib_X={data['vibration_x']:.4f} | "
          f"RPM={data['vitesse_rpm']:.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='Simulateur capteurs IoT — Maintenance Prédictive')
    parser.add_argument('--moteur', type=int, default=5,
                        help='ID du moteur à simuler (défaut: 5)')
    parser.add_argument('--mode',
                        choices=['normal', 'panne', 'roulement', 'critique'],
                        default='panne',
                        help='Mode de simulation (défaut: panne)')
    parser.add_argument('--tous', action='store_true',
                        help='Simuler tous les moteurs simultanément')
    args = parser.parse_args()

    print("=" * 60)
    print(" SIMULATEUR IoT — Raspberry Pi (mode PC)")
    print(" Maintenance Prédictive — Capteurs Industriels")
    print("=" * 60)
    print()
    print("  Topics MQTT publiés :")
    print("    moteur/{id}/capteurs  → données brutes")
    print("    moteur/{id}/alerte    → alertes uniquement")
    print("    moteur/status         → état global")
    print()

    # Connexion MQTT
    client = mqtt.Client()
    client.on_connect = on_connect
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"  ❌ Impossible de connecter à MQTT : {e}")
        print(f"  → Vérifie que Mosquitto tourne sur le port {PORT}")
        return

    # Créer les simulateurs
    if args.tous:
        simulateurs = {
            1:  SimulateurCapteur(1,  'normal'),
            5:  SimulateurCapteur(5,  'panne'),
            18: SimulateurCapteur(18, 'roulement'),
            21: SimulateurCapteur(21, 'critique'),
        }
        print(f"\n  Simulation de 4 moteurs en parallèle :")
        print(f"  Moteur  1 → NORMAL")
        print(f"  Moteur  5 → PANNE (surchauffe progressive)")
        print(f"  Moteur 18 → ROULEMENT (vibration montante)")
        print(f"  Moteur 21 → CRITIQUE (dégradation avancée)")
    else:
        simulateurs = {
            args.moteur: SimulateurCapteur(args.moteur, args.mode)
        }

    print(f"\n  Publication toutes les {INTERVAL_SEC} secondes...")
    print(f"  Ctrl+C pour arrêter\n")
    print("  " + "─" * 56)

    mesure_count = 0

    try:
        while True:
            status_global = {}

            for motor_id, sim in simulateurs.items():
                # Lire tous les capteurs
                data = sim.lire_tous_capteurs()

                # Publier données brutes
                topic_data = f"moteur/{motor_id}/capteurs"
                client.publish(topic_data, json.dumps(data))

                # Publier alerte si détectée
                if data['Alert_Status'] == 'ALERT':
                    topic_alerte = f"moteur/{motor_id}/alerte"
                    alerte_msg = {
                        "motor_id"   : motor_id,
                        "timestamp"  : data['timestamp'],
                        "parametre"  : data['alert_param'],
                        "valeur"     : data.get(data['alert_param'], 0),
                        "niveau"     : "CRITIQUE" if sim.step > 50 else "ÉLEVÉ",
                    }
                    client.publish(topic_alerte, json.dumps(alerte_msg))

                # Afficher dans terminal
                afficher_mesure(data)

                status_global[motor_id] = {
                    "status"     : data['Alert_Status'],
                    "temperature": data['temperature'],
                    "vibration"  : data['vibration_x'],
                    "step"       : data['step'],
                }

            # Publier état global
            client.publish("moteur/status", json.dumps({
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "moteurs"  : status_global,
            }))

            mesure_count += 1
            print(f"\n  ── Mesure #{mesure_count} publiée sur MQTT ──")

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print(f"\n\n  Simulation arrêtée après {mesure_count} mesures.")
        client.loop_stop()
        client.disconnect()


if __name__ == '__main__':
    main()