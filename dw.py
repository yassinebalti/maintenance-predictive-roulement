"""
========================================================
 ÉTAPE 6 — ALIMENTATION DU DATA WAREHOUSE (BI)
 Architecture : Modèle en Étoile (Star Schema)
 
 Ce script :
 1. Crée les tables de dimensions et de faits avec contraintes
 2. Construit Dim_Temps, Dim_Moteur, Dim_Modele_IA
 3. Fusionne les features et les prédictions RUL
 4. Charge les données dans Fact_Sante_Moteur
========================================================
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os

# ── Configuration de la Base de Données ────────────────
# À MODIFIER : remplace 'user' et 'password' par tes vrais identifiants
DB_URI = 'postgresql://user:password@localhost:5432/ai_cp'

# Fichiers sources (générés par les étapes précédentes)
FEATURES_CSV = 'data/02_features_motor.csv'
RUL_CSV      = 'data/04_rul_results.csv'
# ──────────────────────────────────────────────────────

def create_star_schema(engine):
    """Crée l'architecture en étoile avec clés primaires et étrangères."""
    print("→ [1/4] Création du schéma en étoile dans PostgreSQL...")
    
    drop_tables = """
    DROP TABLE IF EXISTS "Fact_Sante_Moteur" CASCADE;
    DROP TABLE IF EXISTS "Dim_Moteur" CASCADE;
    DROP TABLE IF EXISTS "Dim_Temps" CASCADE;
    DROP TABLE IF EXISTS "Dim_Modele_IA" CASCADE;
    """
    
    create_dim_moteur = """
    CREATE TABLE "Dim_Moteur" (
        motor_id INTEGER PRIMARY KEY,
        nom_moteur VARCHAR(50),
        fabricant VARCHAR(50),
        localisation VARCHAR(100)
    );
    """
    
    create_dim_temps = """
    CREATE TABLE "Dim_Temps" (
        temps_id BIGINT PRIMARY KEY,
        timestamp_complet TIMESTAMP,
        annee INTEGER,
        mois INTEGER,
        jour INTEGER,
        heure INTEGER,
        minute INTEGER,
        equipe_production VARCHAR(20)
    );
    """
    
    create_dim_modele = """
    CREATE TABLE "Dim_Modele_IA" (
        modele_id SERIAL PRIMARY KEY,
        nom_algorithme VARCHAR(100),
        version VARCHAR(20),
        description TEXT
    );
    """
    
    create_fact = """
    CREATE TABLE "Fact_Sante_Moteur" (
        fait_id SERIAL PRIMARY KEY,
        fk_motor_id INTEGER REFERENCES "Dim_Moteur"(motor_id),
        fk_temps_id BIGINT REFERENCES "Dim_Temps"(temps_id),
        fk_modele_id INTEGER REFERENCES "Dim_Modele_IA"(modele_id),
        
        -- Mesures Brutes
        temperature FLOAT,
        vibration FLOAT,
        courant FLOAT,
        
        -- Features IA
        shape_factor FLOAT,
        kurtosis FLOAT,
        fft_max_amp FLOAT,
        
        -- Résultats Prédictifs
        anomaly_score FLOAT,
        is_anomaly BOOLEAN,
        rul_days FLOAT,
        risk_level VARCHAR(20)
    );
    """
    
    with engine.begin() as conn:
        conn.execute(text(drop_tables))
        conn.execute(text(create_dim_moteur))
        conn.execute(text(create_dim_temps))
        conn.execute(text(create_dim_modele))
        conn.execute(text(create_fact))

def populate_dimensions(engine, df_merged):
    """Remplit les tables de dimensions."""
    print("→ [2/4] Alimentation des Dimensions...")
    
    # 1. Dim_Moteur
    # On extrait les moteurs uniques du dataset
    moteurs = pd.DataFrame({'motor_id': df_merged['motor_id'].unique()})
    moteurs['nom_moteur'] = 'Moteur ' + moteurs['motor_id'].astype(str)
    moteurs['fabricant'] = np.where(moteurs['motor_id'] % 2 == 0, 'Siemens', 'ABB') # Exemple simulé
    moteurs['localisation'] = 'Novation City - Ligne A'
    moteurs.to_sql('Dim_Moteur', engine, if_exists='append', index=False)
    
    # 2. Dim_Temps
    # Création d'un ID numérique unique pour le temps (ex: 202504151030)
    temps = pd.DataFrame({'timestamp_complet': df_merged['timestamp'].unique()})
    temps['temps_id'] = temps['timestamp_complet'].dt.strftime('%Y%m%d%H%M').astype(np.int64)
    temps['annee'] = temps['timestamp_complet'].dt.year
    temps['mois'] = temps['timestamp_complet'].dt.month
    temps['jour'] = temps['timestamp_complet'].dt.day
    temps['heure'] = temps['timestamp_complet'].dt.hour
    temps['minute'] = temps['timestamp_complet'].dt.minute
    
    # Définition des 3x8 (Équipes de production)
    conditions = [
        (temps['heure'] >= 6) & (temps['heure'] < 14),
        (temps['heure'] >= 14) & (temps['heure'] < 22)
    ]
    choix = ['Matin', 'Après-midi']
    temps['equipe_production'] = np.select(conditions, choix, default='Nuit')
    temps.to_sql('Dim_Temps', engine, if_exists='append', index=False)
    
    # 3. Dim_Modele_IA
    modeles = pd.DataFrame([{
        'nom_algorithme': 'Isolation Forest + Ensemble Weibull',
        'version': 'V3.0 (MLflow)',
        'description': 'Modèle hybride avec validation Walk-Forward et filtre CUSUM'
    }])
    modeles.to_sql('Dim_Modele_IA', engine, if_exists='append', index=False)

def populate_facts(engine, df_merged):
    """Remplit la table des faits."""
    print("→ [3/4] Alimentation de la Table des Faits...")
    
    # Préparation du DataFrame pour la table des faits
    fact = pd.DataFrame()
    fact['fk_motor_id'] = df_merged['motor_id']
    fact['fk_temps_id'] = df_merged['timestamp'].dt.strftime('%Y%m%d%H%M').astype(np.int64)
    fact['fk_modele_id'] = 1  # L'ID du modèle que nous venons d'insérer
    
    # Mesures et features (sélection selon ce qui existe dans tes fichiers)
    colonnes_a_garder = ['temperature', 'vibration', 'courant', 'shape_factor', 'kurtosis']
    for col in colonnes_a_garder:
        if col in df_merged.columns:
            fact[col] = df_merged[col]
        else:
            fact[col] = 0.0 # Valeur par défaut si la feature n'a pas été calculée
            
    # Si fft_max_amp existe, on l'ajoute, sinon on met 0
    fact['fft_max_amp'] = df_merged['fft_max_amp'] if 'fft_max_amp' in df_merged.columns else 0.0
    
    # Variables de prédiction (Issues de step 3 et 4)
    fact['anomaly_score'] = df_merged['combined_score'] if 'combined_score' in df_merged.columns else 0.0
    fact['is_anomaly'] = df_merged['is_anomaly'] if 'is_anomaly' in df_merged.columns else False
    fact['rul_days'] = df_merged['rul_ensemble'] if 'rul_ensemble' in df_merged.columns else 90.0
    fact['risk_level'] = df_merged['risk_level'] if 'risk_level' in df_merged.columns else 'FAIBLE'
    
    # Insertion par lots (chunksize) pour ne pas saturer la RAM
    fact.to_sql('Fact_Sante_Moteur', engine, if_exists='append', index=False, chunksize=10000)
    print(f"  ✓ {len(fact):,} faits insérés avec succès.")

def main():
    print("=" * 60)
    print(" ÉTAPE 6 — GÉNÉRATION DU DATA WAREHOUSE")
    print("=" * 60)
    
    try:
        engine = create_engine(DB_URI)
        
        print("→ Chargement des fichiers de données...")
        df_feat = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'])
        df_rul = pd.read_csv(RUL_CSV, parse_dates=['timestamp'])
        
        # Fusion des features avec les résultats IA sur l'ID et le temps
        # On supprime les colonnes dupliquées après le merge
        df_merged = pd.merge(df_feat, df_rul, on=['motor_id', 'timestamp'], how='inner', suffixes=('', '_drop'))
        df_merged = df_merged.loc[:, ~df_merged.columns.str.endswith('_drop')]
        
        # Exécution du pipeline ETL
        create_star_schema(engine)
        populate_dimensions(engine, df_merged)
        populate_facts(engine, df_merged)
        
        print("\n✅ DATA WAREHOUSE GÉNÉRÉ AVEC SUCCÈS !")
        print("Tu peux maintenant connecter Streamlit ou PowerBI directement")
        print("aux tables Dim_Moteur, Dim_Temps et Fact_Sante_Moteur.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERREUR] Impossible de générer le Data Warehouse : {e}")
        print("Vérifie que ta base PostgreSQL est allumée et que tes identifiants (DB_URI) sont corrects.")

if __name__ == '__main__':
    main()