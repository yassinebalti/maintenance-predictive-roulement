"""
========================================================
 ÉTAPE 1 — EXTRACTION SQL → CSV
 Source : motor_measurements (32 810 lignes, 21 moteurs)
 Colonnes extraites : toutes (temperature, courant,
   vibration, acceleration, thdi, thdu, vitesse,
   cosphi, Alert_Status, Alert_File)
========================================================
"""

import re
import json
import pandas as pd
import numpy as np
import os

# ── Configuration ──────────────────────────────────────
SQL_FILE   = 'ai_cp (2).sql'          # Fichier SQL source
OUTPUT_CSV = 'data/01_raw_motor.csv'  # Sortie brute
# ──────────────────────────────────────────────────────


def extract_motors_info(sql_content: str) -> pd.DataFrame:
    """Extrait la table motors (métadonnées moteurs)."""
    pattern = r"INSERT INTO `motors`.*?VALUES\s*(.*?);"
    match = re.search(pattern, sql_content, re.DOTALL)
    if not match:
        return pd.DataFrame()

    rows = []
    for t in re.findall(r'\(([^)]+)\)', match.group(1)):
        parts = [p.strip().strip("'") for p in t.split(',')]
        if len(parts) >= 13:
            rows.append({
                'motor_id'         : int(parts[0]),
                'name'             : parts[1],
                'model'            : parts[2],
                'manufacturer'     : parts[3],
                'power_rating_kW'  : float(parts[6])  if parts[6]  != '' else None,
                'voltage_rating_V' : float(parts[7])  if parts[7]  != '' else None,
                'current_rating_A' : float(parts[8])  if parts[8]  != '' else None,
                'speed_rpm'        : int(parts[9])    if parts[9]  != '' else None,
                'cosphi_nominal'   : float(parts[10]) if parts[10] != '' else None,
                'installation_date': parts[11],
                'location'         : parts[12],
            })
    return pd.DataFrame(rows)


def parse_alert_file(alert_json_str: str) -> dict:
    """Parse le champ Alert_File JSON et retourne le paramètre en alerte."""
    result = {'alert_parameter': None, 'alert_code': None}
    try:
        cleaned = alert_json_str.replace('\\"', '"')
        data = json.loads(cleaned)
        if data:
            result['alert_parameter'] = data[0].get('parameter')
            threshold = data[0].get('threshold', {})
            result['alert_code'] = threshold.get('code_alert')
    except Exception:
        pass
    return result


def extract_motor_measurements(sql_content: str) -> pd.DataFrame:
    """
    Extrait toutes les lignes motor_measurements du SQL.
    Gère les multiples blocs INSERT (129 blocs dans ce fichier).
    Colonnes : measurement_id, motor_id, timestamp, temperature, courant,
               vibration, acceleration, thdi, thdu, vitesse, cosphi,
               Alert_Status, alert_parameter, alert_code
    """
    pattern = r"INSERT INTO `motor_measurements`.*?VALUES\s*(.*?);"
    matches = re.findall(pattern, sql_content, re.DOTALL)

    print(f"  → {len(matches)} blocs INSERT détectés")

    all_rows = []
    for block in matches:
        tuples = re.findall(r'\((\d+,\s*\d+,\s*\x27[^)]+)\)', block)
        for t in tuples:
            # Séparer proprement (gérer la virgule dans Alert_File JSON)
            # On coupe sur les 12 premières virgules seulement
            parts = t.split(',', 12)
            parts = [p.strip().strip("'") for p in parts]
            if len(parts) < 12:
                continue

            alert_info = parse_alert_file(parts[12] if len(parts) > 12 else '[]')

            try:
                all_rows.append({
                    'measurement_id'  : int(parts[0]),
                    'motor_id'        : int(parts[1]),
                    'timestamp'       : parts[2],
                    'temperature'     : float(parts[3])  if parts[3]  not in ('NULL', '') else np.nan,
                    'courant'         : float(parts[4])  if parts[4]  not in ('NULL', '') else np.nan,
                    'vibration'       : float(parts[5])  if parts[5]  not in ('NULL', '') else np.nan,
                    'acceleration'    : float(parts[6])  if parts[6]  not in ('NULL', '') else np.nan,
                    'thdi'            : float(parts[7])  if parts[7]  not in ('NULL', '') else np.nan,
                    'thdu'            : float(parts[8])  if parts[8]  not in ('NULL', '') else np.nan,
                    'vitesse'         : float(parts[9])  if parts[9]  not in ('NULL', '') else np.nan,
                    'cosphi'          : float(parts[10]) if parts[10] not in ('NULL', '') else np.nan,
                    'Alert_Status'    : parts[11],
                    'alert_parameter' : alert_info['alert_parameter'],
                    'alert_code'      : alert_info['alert_code'],
                })
            except (ValueError, IndexError):
                continue

    return pd.DataFrame(all_rows)


def main():
    os.makedirs('data', exist_ok=True)

    print("=" * 55)
    print(" ÉTAPE 1 — EXTRACTION SQL")
    print("=" * 55)

    if not os.path.exists(SQL_FILE):
        print(f"[ERREUR] Fichier SQL introuvable : {SQL_FILE}")
        return

    print(f"\n→ Chargement de {SQL_FILE} ...")
    with open(SQL_FILE, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    print(f"  Fichier lu : {len(sql_content):,} caractères")

    # ── 1. Métadonnées moteurs ─────────────────────────
    print("\n→ Extraction table motors ...")
    df_motors = extract_motors_info(sql_content)
    df_motors.to_csv('data/motors_info.csv', index=False)
    print(f"  {len(df_motors)} moteurs extraits → data/motors_info.csv")
    if not df_motors.empty:
        print(df_motors[['motor_id', 'name', 'manufacturer', 'power_rating_kW']].to_string(index=False))

    # ── 2. Mesures moteurs ────────────────────────────
    print("\n→ Extraction motor_measurements ...")
    df = extract_motor_measurements(sql_content)

    if df.empty:
        print("[ERREUR] Aucune donnée extraite.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['motor_id', 'timestamp']).reset_index(drop=True)

    # ── 3. Rapport d'extraction ───────────────────────
    print(f"\n✓ {len(df):,} lignes extraites")
    print(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Moteurs : {sorted(df['motor_id'].unique())}")
    print(f"\n  Distribution Alert_Status :")
    print(df['Alert_Status'].value_counts().to_string())
    print(f"\n  Paramètres en alerte :")
    print(df['alert_parameter'].value_counts().to_string())

    # ── 4. Statistiques descriptives ─────────────────
    print(f"\n  Statistiques clés :")
    cols_stat = ['temperature', 'courant', 'vibration', 'acceleration', 'vitesse', 'cosphi']
    print(df[cols_stat].describe().round(3).to_string())

    # ── 5. Valeurs manquantes ─────────────────────────
    print(f"\n  Valeurs manquantes :")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "  Aucune !")

    # ── 6. Sauvegarde ────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Données brutes sauvegardées : {OUTPUT_CSV}")
    print("=" * 55)


if __name__ == '__main__':
    main()
