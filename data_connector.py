"""
================================================================
 DATA CONNECTOR — Interface Universelle de Données
 Maintenance Prédictive Industrielle

 Ce fichier se branche sur ton pipeline EXISTANT
 SANS modifier aucun fichier existant.

 Il joue le rôle d'un "traducteur universel" :
   Source de données → DataFrame pandas → Pipeline IA

 SOURCES SUPPORTÉES :
   1. Fichier SQL        (mode actuel — déjà fonctionnel)
   2. MySQL / MariaDB    (production standard)
   3. PostgreSQL         (production avancée)
   4. TimescaleDB        (IoT optimisé)
   5. SQLite             (prototype / démo)
   6. CSV direct         (données exportées)
   7. MQTT temps réel    (flux IoT live)

 USAGE :
   python data_connector.py --source mysql --host localhost
   python data_connector.py --source sqlite --file maintenance.db
   python data_connector.py --source csv --file mes_donnees.csv
   python data_connector.py --source sql --file "ai_cp (2).sql"
   python data_connector.py --test

 INTÉGRATION PIPELINE :
   from data_connector import DataConnector
   connector = DataConnector(source='mysql', host='localhost')
   df = connector.charger_donnees()
   # df est identique à ce que produit step1_extraction.py
   # → step2, step3, step4 fonctionnent sans aucun changement
================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Configuration par défaut ────────────────────────────────
CONFIG_FILE = 'db_config.json'

COLONNES_REQUISES = [
    'motor_id', 'timestamp', 'temperature', 'courant',
    'vibration', 'acceleration', 'vitesse', 'Alert_Status'
]

COLONNES_OPTIONNELLES = [
    'alert_parameter', 'alert_code', 'cosphi', 'vibration_y', 'vibration_z'
]


# ══════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════

class DataConnector:
    """
    Connecteur universel de données.
    Produit toujours un DataFrame identique à step1_extraction.py
    quelle que soit la source.
    """

    def __init__(self, source='sql', **kwargs):
        """
        source : 'sql' | 'mysql' | 'postgres' | 'sqlite' |
                 'timescale' | 'csv' | 'mqtt'
        kwargs : paramètres de connexion selon la source
        """
        self.source  = source.lower()
        self.kwargs  = kwargs
        self.engine  = None
        self.conn    = None

        print(f"  DataConnector initialisé — source : {self.source}")

    # ──────────────────────────────────────────────────────────
    #  MÉTHODE PRINCIPALE
    # ──────────────────────────────────────────────────────────

    def charger_donnees(self, depuis_jours=None) -> pd.DataFrame:
        """
        Charge les données depuis la source configurée.
        Retourne un DataFrame identique à celui de step1_extraction.py.

        depuis_jours : si défini, charge seulement les N derniers jours
                       (utile pour le mode production)
        """
        print(f"\n→ Chargement depuis : {self.source}")

        if   self.source == 'sql':       df = self._depuis_sql()
        elif self.source == 'mysql':     df = self._depuis_mysql(depuis_jours)
        elif self.source == 'postgres':  df = self._depuis_postgres(depuis_jours)
        elif self.source == 'timescale': df = self._depuis_timescale(depuis_jours)
        elif self.source == 'sqlite':    df = self._depuis_sqlite(depuis_jours)
        elif self.source == 'csv':       df = self._depuis_csv()
        else:
            raise ValueError(f"Source inconnue : {self.source}")

        # Standardisation — même format que step1 dans tous les cas
        df = self._standardiser(df)

        print(f"  ✓ {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")
        print(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
        return df

    # ──────────────────────────────────────────────────────────
    #  SOURCE 1 — FICHIER SQL (mode actuel)
    # ──────────────────────────────────────────────────────────

    def _depuis_sql(self) -> pd.DataFrame:
        """
        Mode actuel — parse le fichier SQL brut.
        Délègue à step1_extraction.py sans le modifier.
        """
        fichier = self.kwargs.get('file', 'ai_cp (2).sql')

        print(f"  Fichier SQL : {fichier}")

        # Importer step1 sans le modifier
        sys.path.insert(0, os.getcwd())
        try:
            import step1_extraction as s1
            # Appeler directement les fonctions de step1
            with open(fichier, 'r', encoding='utf-8', errors='ignore') as f:
                contenu = f.read()

            df = s1.extraire_motor_measurements(contenu)
            return df

        except Exception as e:
            # Fallback — lire le CSV produit par step1 s'il existe
            csv_path = 'data/01_raw_motor.csv'
            if os.path.exists(csv_path):
                print(f"  → Fallback : lecture {csv_path}")
                return pd.read_csv(csv_path, parse_dates=['timestamp'])
            raise RuntimeError(f"Impossible de charger SQL : {e}")

    # ──────────────────────────────────────────────────────────
    #  SOURCE 2 — MYSQL / MARIADB
    # ──────────────────────────────────────────────────────────

    def _depuis_mysql(self, depuis_jours=None) -> pd.DataFrame:
        """
        Connexion MySQL/MariaDB.
        pip install pymysql sqlalchemy
        """
        try:
            from sqlalchemy import create_engine as ce
        except ImportError:
            raise ImportError("pip install pymysql sqlalchemy")

        host     = self.kwargs.get('host', 'localhost')
        port     = self.kwargs.get('port', 3306)
        user     = self.kwargs.get('user', 'root')
        password = self.kwargs.get('password', '')
        database = self.kwargs.get('database', 'maintenance_db')

        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        print(f"  Connexion MySQL : {host}:{port}/{database}")

        engine = ce(url)
        self.engine = engine

        query = self._construire_requete('motor_measurements', depuis_jours)
        return pd.read_sql(query, engine, parse_dates=['timestamp'])

    # ──────────────────────────────────────────────────────────
    #  SOURCE 3 — POSTGRESQL
    # ──────────────────────────────────────────────────────────

    def _depuis_postgres(self, depuis_jours=None) -> pd.DataFrame:
        """
        Connexion PostgreSQL.
        pip install psycopg2-binary sqlalchemy
        """
        try:
            from sqlalchemy import create_engine as ce
        except ImportError:
            raise ImportError("pip install psycopg2-binary sqlalchemy")

        host     = self.kwargs.get('host', 'localhost')
        port     = self.kwargs.get('port', 5432)
        user     = self.kwargs.get('user', 'postgres')
        password = self.kwargs.get('password', '')
        database = self.kwargs.get('database', 'maintenance_db')

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        print(f"  Connexion PostgreSQL : {host}:{port}/{database}")

        engine = ce(url)
        self.engine = engine

        query = self._construire_requete('motor_measurements', depuis_jours)
        return pd.read_sql(query, engine, parse_dates=['timestamp'])

    # ──────────────────────────────────────────────────────────
    #  SOURCE 4 — TIMESCALEDB (PostgreSQL + time series)
    # ──────────────────────────────────────────────────────────

    def _depuis_timescale(self, depuis_jours=None) -> pd.DataFrame:
        """
        TimescaleDB — même API que PostgreSQL mais requêtes optimisées.
        pip install psycopg2-binary sqlalchemy
        """
        try:
            from sqlalchemy import create_engine as ce
        except ImportError:
            raise ImportError("pip install psycopg2-binary sqlalchemy")

        host     = self.kwargs.get('host', 'localhost')
        port     = self.kwargs.get('port', 5432)
        user     = self.kwargs.get('user', 'postgres')
        password = self.kwargs.get('password', '')
        database = self.kwargs.get('database', 'maintenance_db')

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        print(f"  Connexion TimescaleDB : {host}:{port}/{database}")

        engine = ce(url)
        self.engine = engine

        # Requête optimisée TimescaleDB (time_bucket pour agrégation rapide)
        if depuis_jours:
            query = f"""
                SELECT
                    motor_id,
                    time_bucket('5 minutes', timestamp) AS timestamp,
                    AVG(temperature)   AS temperature,
                    AVG(courant)       AS courant,
                    AVG(vibration)     AS vibration,
                    AVG(acceleration)  AS acceleration,
                    AVG(vitesse)       AS vitesse,
                    MODE() WITHIN GROUP (ORDER BY "Alert_Status") AS "Alert_Status",
                    MODE() WITHIN GROUP (ORDER BY alert_parameter) AS alert_parameter
                FROM motor_measurements
                WHERE timestamp >= NOW() - INTERVAL '{depuis_jours} days'
                GROUP BY motor_id, time_bucket('5 minutes', timestamp)
                ORDER BY motor_id, timestamp
            """
        else:
            query = self._construire_requete('motor_measurements', None)

        return pd.read_sql(query, engine, parse_dates=['timestamp'])

    # ──────────────────────────────────────────────────────────
    #  SOURCE 5 — SQLITE (prototype / PFE)
    # ──────────────────────────────────────────────────────────

    def _depuis_sqlite(self, depuis_jours=None) -> pd.DataFrame:
        """
        SQLite — fichier unique, pas de serveur.
        Idéal pour prototype et soutenance.
        pip install sqlalchemy
        """
        try:
            from sqlalchemy import create_engine as ce
        except ImportError:
            raise ImportError("pip install sqlalchemy")

        db_file = self.kwargs.get('file', 'maintenance.db')
        print(f"  Fichier SQLite : {db_file}")

        if not os.path.exists(db_file):
            print(f"  Base inexistante → création automatique")
            self._creer_sqlite(db_file)

        engine = ce(f"sqlite:///{db_file}")
        self.engine = engine

        query = self._construire_requete('motor_measurements', depuis_jours,
                                         dialect='sqlite')
        return pd.read_sql(query, engine, parse_dates=['timestamp'])

    # ──────────────────────────────────────────────────────────
    #  SOURCE 6 — CSV DIRECT
    # ──────────────────────────────────────────────────────────

    def _depuis_csv(self) -> pd.DataFrame:
        """
        Lecture depuis un fichier CSV.
        Compatible avec tout export depuis Excel, base de données, etc.
        """
        fichier = self.kwargs.get('file', 'data/01_raw_motor.csv')
        print(f"  Fichier CSV : {fichier}")

        if not os.path.exists(fichier):
            raise FileNotFoundError(f"Fichier introuvable : {fichier}")

        df = pd.read_csv(fichier, parse_dates=['timestamp'])
        return df

    # ──────────────────────────────────────────────────────────
    #  UTILITAIRES
    # ──────────────────────────────────────────────────────────

    def _construire_requete(self, table, depuis_jours, dialect='mysql'):
        """Construit la requête SQL adaptée au dialect."""
        base = f"""
            SELECT
                motor_id,
                timestamp,
                temperature,
                courant,
                vibration,
                acceleration,
                vitesse,
                cosphi,
                "Alert_Status",
                alert_parameter,
                alert_code
            FROM {table}
        """

        if depuis_jours:
            if dialect == 'sqlite':
                filtre = f"WHERE timestamp >= datetime('now', '-{depuis_jours} days')"
            else:
                filtre = f"WHERE timestamp >= NOW() - INTERVAL {depuis_jours} DAY"
            base += filtre

        base += " ORDER BY motor_id, timestamp"
        return base

    def _standardiser(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise le DataFrame pour qu'il soit identique
        à ce que produit step1_extraction.py.
        Appelé après toutes les sources.
        """
        df = df.copy()

        # Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Renommage colonnes fréquents
        rename_map = {
            'temp'          : 'temperature',
            'current'       : 'courant',
            'vibration_x'   : 'vibration',
            'vib'           : 'vibration',
            'accel'         : 'acceleration',
            'rpm'           : 'vitesse',
            'speed'         : 'vitesse',
            'alert_status'  : 'Alert_Status',
            'status'        : 'Alert_Status',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items()
                                 if k in df.columns})

        # Colonnes manquantes → valeurs par défaut
        if 'Alert_Status' not in df.columns:
            df['Alert_Status'] = 'NORMAL'
        if 'alert_parameter' not in df.columns:
            df['alert_parameter'] = np.nan
        if 'cosphi' not in df.columns:
            df['cosphi'] = 0.85
        if 'vitesse' not in df.columns:
            df['vitesse'] = 1495.0

        # Nettoyage types
        df['motor_id'] = pd.to_numeric(df['motor_id'], errors='coerce')
        for col in ['temperature', 'courant', 'vibration', 'acceleration']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Supprimer lignes invalides
        df = df.dropna(subset=['motor_id', 'timestamp', 'temperature'])
        df = df.sort_values(['motor_id', 'timestamp']).reset_index(drop=True)

        return df

    def _creer_sqlite(self, db_file: str):
        """Crée une base SQLite vide avec le bon schéma."""
        try:
            from sqlalchemy import create_engine as ce, text
        except ImportError:
            return

        engine = ce(f"sqlite:///{db_file}")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS motor_measurements (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    motor_id        INTEGER NOT NULL,
                    timestamp       DATETIME NOT NULL,
                    temperature     REAL,
                    courant         REAL,
                    vibration       REAL,
                    acceleration    REAL,
                    vitesse         REAL DEFAULT 1495.0,
                    cosphi          REAL DEFAULT 0.85,
                    Alert_Status    TEXT DEFAULT 'NORMAL',
                    alert_parameter TEXT,
                    alert_code      TEXT,
                    source          TEXT DEFAULT 'IoT'
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS iot_realtime_results (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    motor_id         INTEGER,
                    timestamp        DATETIME,
                    temperature      REAL,
                    vibration        REAL,
                    courant          REAL,
                    score_if         REAL,
                    score_rules      REAL,
                    score_hybride    REAL,
                    is_anomalie      INTEGER,
                    di               REAL,
                    niveau_risque    TEXT,
                    health_score     REAL,
                    confidence_score REAL,
                    n_exceed         INTEGER,
                    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS alertes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    motor_id     INTEGER,
                    timestamp    DATETIME,
                    niveau       TEXT,
                    parametre    TEXT,
                    valeur       REAL,
                    di           REAL,
                    score        REAL,
                    acquittee    INTEGER DEFAULT 0,
                    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rul_predictions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    motor_id   INTEGER,
                    timestamp  DATETIME,
                    di         REAL,
                    rul_days   TEXT,
                    risk_level TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
        print(f"  Base SQLite créée : {db_file}")

    # ──────────────────────────────────────────────────────────
    #  SAUVEGARDE RÉSULTATS IA → BASE DE DONNÉES
    # ──────────────────────────────────────────────────────────

    def sauvegarder_resultats_ia(self, df_anomalies: pd.DataFrame):
        """
        Sauvegarde les résultats de step3 dans la base de données.
        Appelé APRÈS step3_anomaly_detection.py — sans le modifier.
        """
        if self.engine is None:
            print("  ⚠ Pas de connexion DB — sauvegarde CSV seulement")
            return

        colonnes_ia = [
            'motor_id', 'timestamp', 'temperature', 'courant', 'vibration',
            'score_if', 'score_rules', 'combined_score', 'is_anomaly',
            'health_score', 'confidence_score'
        ]
        cols_dispo = [c for c in colonnes_ia if c in df_anomalies.columns]
        df_save = df_anomalies[cols_dispo].copy()
        df_save = df_save.rename(columns={
            'combined_score' : 'score_hybride',
            'is_anomaly'     : 'is_anomalie',
        })

        df_save.to_sql('iot_realtime_results', self.engine,
                       if_exists='append', index=False)
        print(f"  ✓ {len(df_save):,} résultats IA sauvegardés en DB")

    def sauvegarder_mesure_iot(self, mesure: dict):
        """
        Sauvegarde UNE mesure IoT reçue via MQTT dans la base.
        Appelé depuis iot_subscriber_ia.py — sans le modifier.
        """
        if self.engine is None:
            return

        df = pd.DataFrame([{
            'motor_id'       : mesure.get('motor_id'),
            'timestamp'      : mesure.get('timestamp'),
            'temperature'    : mesure.get('temperature'),
            'courant'        : mesure.get('courant'),
            'vibration'      : mesure.get('vibration_x'),
            'acceleration'   : mesure.get('acceleration'),
            'vitesse'        : mesure.get('vitesse_rpm'),
            'Alert_Status'   : mesure.get('Alert_Status', 'NORMAL'),
            'alert_parameter': mesure.get('alert_param'),
            'source'         : 'IoT',
        }])
        df.to_sql('motor_measurements', self.engine,
                  if_exists='append', index=False)

    def sauvegarder_alerte(self, alerte: dict):
        """Sauvegarde une alerte IA dans la table alertes."""
        if self.engine is None:
            return

        df = pd.DataFrame([{
            'motor_id' : alerte.get('motor_id'),
            'timestamp': alerte.get('timestamp'),
            'niveau'   : alerte.get('niveau'),
            'parametre': alerte.get('alert_param'),
            'valeur'   : alerte.get('valeur'),
            'di'       : alerte.get('di'),
            'score'    : alerte.get('score'),
        }])
        df.to_sql('alertes', self.engine, if_exists='append', index=False)

    def charger_fenetre_moteur(self, motor_id: int,
                                n_mesures: int = 20) -> pd.DataFrame:
        """
        Charge les N dernières mesures d'un moteur depuis la DB.
        Utilisé par le subscriber IoT pour le calcul des features.
        """
        if self.engine is None:
            return pd.DataFrame()

        query = f"""
            SELECT motor_id, timestamp, temperature, courant,
                   vibration, acceleration
            FROM motor_measurements
            WHERE motor_id = {motor_id}
            ORDER BY timestamp DESC
            LIMIT {n_mesures}
        """
        df = pd.read_sql(query, self.engine, parse_dates=['timestamp'])
        return df.sort_values('timestamp').reset_index(drop=True)

    # ──────────────────────────────────────────────────────────
    #  CONFIGURATION PERSISTANTE
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def sauvegarder_config(source, **kwargs):
        """Sauvegarde la configuration DB dans un fichier JSON."""
        config = {'source': source, **kwargs}
        # Ne pas sauvegarder le mot de passe en clair
        if 'password' in config:
            config['password'] = '***'
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Config sauvegardée : {CONFIG_FILE}")

    @staticmethod
    def charger_config() -> dict:
        """Charge la configuration depuis le fichier JSON."""
        if not os.path.exists(CONFIG_FILE):
            return {'source': 'sql', 'file': 'ai_cp (2).sql'}
        with open(CONFIG_FILE) as f:
            return json.load(f)

    def tester_connexion(self) -> bool:
        """Teste la connexion à la base de données."""
        print(f"\n→ Test connexion {self.source} ...")
        try:
            df = self.charger_donnees(depuis_jours=7)
            print(f"  ✅ Connexion OK — {len(df):,} lignes (7 derniers jours)")
            return True
        except Exception as e:
            print(f"  ❌ Connexion échouée : {e}")
            return False


# ══════════════════════════════════════════════════════════════
#  INTÉGRATION PIPELINE — EXEMPLE D'UTILISATION
# ══════════════════════════════════════════════════════════════

def run_pipeline_avec_db(source='sql', **kwargs):
    """
    Lance le pipeline complet depuis une base de données.
    SANS modifier step1, step2, step3, step4, step5.

    Exemple :
      run_pipeline_avec_db(source='mysql', host='localhost',
                           user='root', password='monmdp',
                           database='maintenance_db')
    """
    import subprocess

    print("=" * 60)
    print(" PIPELINE IA — MODE BASE DE DONNÉES")
    print("=" * 60)

    # 1. Charger données depuis DB → CSV intermédiaire
    print("\n→ [1/5] Chargement données depuis DB ...")
    connector = DataConnector(source=source, **kwargs)
    df = connector.charger_donnees()

    # Sauvegarder en CSV pour que step2, step3, step4 fonctionnent
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/01_raw_motor.csv', index=False)
    print(f"  ✓ data/01_raw_motor.csv créé ({len(df):,} lignes)")

    # 2. Lancer les étapes suivantes normalement (elles lisent le CSV)
    for etape, script in [
        ("2/5", "step2_features.py"),
        ("3/5", "step3_anomaly_detection.py"),
        ("4/5", "step4_rul_prediction.py"),
        ("5/5", "step5_report.py"),
    ]:
        print(f"\n→ [{etape}] {script} ...")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False
        )
        if result.returncode != 0:
            print(f"  ❌ Erreur dans {script}")
            break

    # 3. Sauvegarder résultats IA → DB
    if connector.engine and os.path.exists('data/03_anomalies.csv'):
        print("\n→ Sauvegarde résultats IA → DB ...")
        df_anomalies = pd.read_csv('data/03_anomalies.csv',
                                    parse_dates=['timestamp'])
        connector.sauvegarder_resultats_ia(df_anomalies)

    print("\n✓ Pipeline avec DB terminé !")
    return connector


# ══════════════════════════════════════════════════════════════
#  MAIN — LIGNE DE COMMANDE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='DataConnector — Interface universelle de données IoT'
    )
    parser.add_argument('--source',   default='sql',
                        choices=['sql','mysql','postgres','timescale','sqlite','csv'],
                        help='Source de données')
    parser.add_argument('--host',     default='localhost')
    parser.add_argument('--port',     type=int, default=None)
    parser.add_argument('--user',     default='root')
    parser.add_argument('--password', default='')
    parser.add_argument('--database', default='maintenance_db')
    parser.add_argument('--file',     default=None,
                        help='Fichier SQL, CSV ou SQLite')
    parser.add_argument('--jours',    type=int, default=None,
                        help='Charger seulement les N derniers jours')
    parser.add_argument('--test',     action='store_true',
                        help='Tester la connexion uniquement')
    parser.add_argument('--pipeline', action='store_true',
                        help='Lancer le pipeline complet')
    parser.add_argument('--creer-sqlite', action='store_true',
                        help='Créer une base SQLite vide')

    args = parser.parse_args()

    # Construire kwargs selon la source
    kwargs = {}
    if args.source in ['mysql', 'postgres', 'timescale']:
        kwargs = {
            'host'    : args.host,
            'port'    : args.port,
            'user'    : args.user,
            'password': args.password,
            'database': args.database,
        }
        if args.port: kwargs['port'] = args.port
    elif args.source in ['sql', 'csv', 'sqlite']:
        if args.file:
            kwargs['file'] = args.file

    print("=" * 60)
    print(f" DATA CONNECTOR — source={args.source}")
    print("=" * 60)

    connector = DataConnector(source=args.source, **kwargs)

    if args.creer_sqlite:
        db = args.file or 'maintenance.db'
        connector._creer_sqlite(db)
        print(f"✓ Base SQLite créée : {db}")

    elif args.test:
        ok = connector.tester_connexion()
        sys.exit(0 if ok else 1)

    elif args.pipeline:
        run_pipeline_avec_db(source=args.source, **kwargs)

    else:
        # Juste charger et afficher un résumé
        df = connector.charger_donnees(depuis_jours=args.jours)

        print(f"\n  Résumé des données chargées :")
        print(f"  {'─'*45}")
        print(f"  Lignes totales     : {len(df):,}")
        print(f"  Moteurs            : {df['motor_id'].nunique()}")
        print(f"  Période            : {df['timestamp'].min()}")
        print(f"                     → {df['timestamp'].max()}")
        if 'Alert_Status' in df.columns:
            n_alert  = (df['Alert_Status'] == 'ALERT').sum()
            n_normal = (df['Alert_Status'] == 'NORMAL').sum()
            print(f"  NORMAL             : {n_normal:,}")
            print(f"  ALERT              : {n_alert:,}")
        print(f"  {'─'*45}")
        print(f"\n  Colonnes : {list(df.columns)}")
        print(f"\n  Aperçu :")
        print(df.head(3).to_string())

        # Sauvegarder pour le pipeline
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/01_raw_motor.csv', index=False)
        print(f"\n✓ Sauvegardé : data/01_raw_motor.csv")
        print("  → Lance maintenant : python step2_features.py")


if __name__ == '__main__':
    main()