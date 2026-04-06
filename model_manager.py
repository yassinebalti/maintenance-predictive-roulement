"""
================================================================
 MODEL MANAGER — Sauvegarde et Chargement du Modèle IA
 Maintenance Prédictive Industrielle

 Ce fichier gère la persistance du modèle Isolation Forest V2.
 Il fonctionne AVEC step3 sans le modifier.

 Usage :
   # Sauvegarder après entraînement step3
   from model_manager import ModelManager
   manager = ModelManager()
   manager.sauvegarder(model, scaler, seuils, metadata)

   # Charger pour IoT temps réel
   model, scaler, seuils = manager.charger()

   # Vérifier si un modèle existe
   manager.existe()

   # Ligne de commande
   python model_manager.py --info
   python model_manager.py --test
================================================================
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Chemins ────────────────────────────────────────────────
MODEL_DIR    = 'data'
MODEL_PATH   = 'data/model_v2.pkl'
SCALER_PATH  = 'data/scaler_v2.pkl'
SEUILS_PATH  = 'data/seuils_moteurs.csv'
META_PATH    = 'data/model_metadata.json'


class ModelManager:
    """
    Gère la sauvegarde et le chargement du modèle Isolation Forest.

    Pourquoi sauvegarder le modèle ?
    ─────────────────────────────────
    Sans sauvegarde :
      → Réentraînement à chaque démarrage du subscriber IoT (~30s)
      → Résultats légèrement différents à chaque run (random state)
      → Impossible de tracer l'évolution des performances

    Avec sauvegarde :
      → Chargement en < 0.1 seconde
      → Résultats reproductibles
      → Versioning du modèle (V1, V2, V3...)
      → Déploiement en production simplifié
    """

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        self.model_path  = model_path
        self.scaler_path = scaler_path
        self.meta_path   = META_PATH
        os.makedirs(MODEL_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────
    #  SAUVEGARDE
    # ──────────────────────────────────────────────────────

    def sauvegarder(self, model, scaler=None, seuils=None, metadata=None):
        """
        Sauvegarde le modèle entraîné + scaler + métadonnées.

        Paramètres :
            model    : IsolationForest entraîné
            scaler   : StandardScaler utilisé pour normaliser
            seuils   : dict des seuils calibrés par moteur
            metadata : dict d'informations supplémentaires
                       (AUC, date, n_features, etc.)
        """
        try:
            import joblib
        except ImportError:
            print("  ⚠ joblib non installé — pip install joblib")
            return False

        # Sauvegarder le modèle
        joblib.dump(model, self.model_path)
        print(f"  ✓ Modèle sauvegardé : {self.model_path}")

        # Sauvegarder le scaler
        if scaler is not None:
            joblib.dump(scaler, self.scaler_path)
            print(f"  ✓ Scaler sauvegardé : {self.scaler_path}")

        # Sauvegarder les seuils
        if seuils is not None and isinstance(seuils, dict):
            rows = []
            for motor_id, params in seuils.items():
                row = {'motor_id': motor_id}
                row.update(params)
                rows.append(row)
            pd.DataFrame(rows).to_csv(SEUILS_PATH, index=False)
            print(f"  ✓ Seuils sauvegardés : {SEUILS_PATH}")

        # Sauvegarder les métadonnées
        meta = {
            'version'         : 'V2',
            'date_entrainement': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'algorithme'      : 'Isolation Forest + Règles métier hybride',
            'auc_roc'         : metadata.get('auc', None) if metadata else None,
            'n_estimators'    : getattr(model, 'n_estimators', None),
            'contamination'   : getattr(model, 'contamination', None),
            'n_features'      : getattr(model, 'n_features_in_', None),
            'poids_if'        : 0.30,
            'poids_rules'     : 0.70,
            'seuil_decision'  : 0.25,
            'améliorations'   : ['XAI', 'Fleet Analysis', 'Confirmation temporelle'],
        }
        if metadata:
            meta.update(metadata)

        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Métadonnées sauvegardées : {self.meta_path}")

        return True

    # ──────────────────────────────────────────────────────
    #  CHARGEMENT
    # ──────────────────────────────────────────────────────

    def charger(self):
        """
        Charge le modèle sauvegardé.

        Retourne :
            (model, scaler, seuils) ou (None, None, None) si absent

        Usage typique dans iot_subscriber_ia.py :
            manager = ModelManager()
            model, scaler, seuils = manager.charger()
            if model is None:
                print("Pas de modèle — lancez main_pipeline.py")
        """
        try:
            import joblib
        except ImportError:
            print("  ⚠ joblib non installé — pip install joblib")
            return None, None, None

        if not os.path.exists(self.model_path):
            print(f"  ⚠ Modèle introuvable : {self.model_path}")
            print("  → Lancez : python main_pipeline.py")
            return None, None, None

        # Charger le modèle
        model = joblib.load(self.model_path)
        print(f"  ✓ Modèle chargé : {self.model_path}")

        # Charger le scaler si disponible
        scaler = None
        if os.path.exists(self.scaler_path):
            scaler = joblib.load(self.scaler_path)
            print(f"  ✓ Scaler chargé : {self.scaler_path}")

        # Charger les seuils si disponibles
        seuils = None
        if os.path.exists(SEUILS_PATH):
            df_s = pd.read_csv(SEUILS_PATH)
            seuils = {}
            for _, row in df_s.iterrows():
                mid = int(row['motor_id'])
                seuils[mid] = {
                    col: row[col]
                    for col in df_s.columns
                    if col != 'motor_id'
                }
            print(f"  ✓ Seuils chargés : {len(seuils)} moteurs")

        return model, scaler, seuils

    # ──────────────────────────────────────────────────────
    #  UTILITAIRES
    # ──────────────────────────────────────────────────────

    def existe(self) -> bool:
        """Vérifie si un modèle sauvegardé existe."""
        return os.path.exists(self.model_path)

    def infos(self) -> dict:
        """Retourne les métadonnées du modèle sauvegardé."""
        if not os.path.exists(self.meta_path):
            return {}
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def afficher_infos(self):
        """Affiche les informations du modèle sauvegardé."""
        print("\n" + "=" * 55)
        print("  INFORMATIONS MODÈLE SAUVEGARDÉ")
        print("=" * 55)

        if not self.existe():
            print("  ❌ Aucun modèle sauvegardé")
            print("  → Lancez : python main_pipeline.py")
            print("=" * 55)
            return

        meta = self.infos()
        if not meta:
            print("  ⚠ Métadonnées manquantes")
        else:
            print(f"  Version          : {meta.get('version', 'N/A')}")
            print(f"  Date entraînement: {meta.get('date_entrainement', 'N/A')}")
            print(f"  Algorithme       : {meta.get('algorithme', 'N/A')}")
            print(f"  AUC ROC          : {meta.get('auc_roc', 'N/A')}")
            print(f"  N° features      : {meta.get('n_features', 'N/A')}")
            print(f"  N° estimateurs   : {meta.get('n_estimators', 'N/A')}")
            print(f"  Contamination    : {meta.get('contamination', 'N/A')}")
            print(f"  Poids IF         : {meta.get('poids_if', 'N/A')}")
            print(f"  Poids règles     : {meta.get('poids_rules', 'N/A')}")
            amél = meta.get('améliorations', [])
            print(f"  Améliorations V2 : {', '.join(amél)}")

        # Taille des fichiers
        for label, path in [
            ("Modèle (.pkl)", self.model_path),
            ("Scaler (.pkl)", self.scaler_path),
            ("Seuils (.csv)", SEUILS_PATH),
        ]:
            if os.path.exists(path):
                taille = os.path.getsize(path) / 1024
                print(f"  {label:<20}: {taille:.1f} KB ✅")
            else:
                print(f"  {label:<20}: absent ❌")

        print("=" * 55)

    def tester(self):
        """
        Teste que le modèle chargé produit des prédictions cohérentes.
        Utile pour vérifier l'intégrité du modèle sauvegardé.
        """
        print("\n→ Test du modèle sauvegardé ...")

        if not self.existe():
            print("  ❌ Aucun modèle à tester")
            return False

        model, scaler, seuils = self.charger()
        if model is None:
            return False

        # Générer des données de test
        n_features = getattr(model, 'n_features_in_', 23)
        X_normal   = np.random.randn(50, n_features) * 0.5         # normales
        X_anomalie = np.random.randn(10, n_features) * 3.0 + 5.0   # anomalies

        if scaler:
            X_normal   = scaler.transform(X_normal)
            X_anomalie = scaler.transform(X_anomalie)

        # Prédictions
        scores_normal   = model.decision_function(X_normal)
        scores_anomalie = model.decision_function(X_anomalie)

        # Vérifications
        moy_normal   = scores_normal.mean()
        moy_anomalie = scores_anomalie.mean()

        print(f"  Score moyen NORMAL   : {moy_normal:.4f}")
        print(f"  Score moyen ANOMALIE : {moy_anomalie:.4f}")

        # Les anomalies doivent avoir des scores plus bas (IF convention)
        if moy_anomalie < moy_normal:
            print("  ✅ Modèle cohérent — anomalies bien discriminées")
            return True
        else:
            print("  ⚠ Attention — scores inattendus (données test très différentes)")
            return False


# ══════════════════════════════════════════════════════════════
#  INTÉGRATION STEP3 — À AJOUTER DANS step3_anomaly_detection.py
# ══════════════════════════════════════════════════════════════
# 
# Ce code montre COMMENT intégrer la sauvegarde dans step3
# SANS modifier step3 directement.
# Tu peux l'appeler APRÈS avoir lancé le pipeline.
#
# Exemple d'utilisation après main_pipeline.py :
#
#   from model_manager import sauvegarder_depuis_pipeline
#   sauvegarder_depuis_pipeline(auc=0.9544)

def sauvegarder_depuis_pipeline(auc=None):
    """
    Sauvegarde le modèle depuis les résultats du pipeline.
    À appeler après main_pipeline.py ou step3.

    Cette fonction reconstruit et sauvegarde le modèle
    depuis les CSV produits par step3.
    """
    print("\n→ Sauvegarde du modèle depuis le pipeline ...")

    # Vérifier que les données existent
    if not os.path.exists('data/03_anomalies.csv'):
        print("  ❌ data/03_anomalies.csv absent")
        print("  → Lancez d'abord : python main_pipeline.py")
        return False

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        import joblib

        # Charger les données
        df = pd.read_csv('data/03_anomalies.csv', parse_dates=['timestamp'])
        print(f"  {len(df):,} lignes chargées")

        # Colonnes features (identiques à step3)
        feature_cols = [
            'temperature_exceed', 'courant_exceed', 'vibration_exceed',
            'acceleration_exceed', 'temperature_ratio', 'courant_ratio',
            'vibration_ratio', 'acceleration_ratio', 'n_exceed',
            'severity_score', 'vib_energy_mean', 'vib_kurt', 'crest_factor',
            'temp_mean', 'temp_trend', 'courant_mean', 'envelope_mean',
            'health_score', 'temp_ratio_flotte', 'vib_ratio_flotte',
            'courant_ratio_flotte', 'temp_zscore_flotte', 'vib_zscore_flotte',
        ]

        # Garder seulement les colonnes disponibles
        cols_dispo = [c for c in feature_cols if c in df.columns]
        if len(cols_dispo) < 10:
            print(f"  ⚠ Seulement {len(cols_dispo)} features disponibles")

        X = df[cols_dispo].fillna(0).values

        # Réentraîner le scaler et le modèle
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        contamination = (df['Alert_Status'] == 'ALERT').mean()
        contamination = float(np.clip(contamination, 0.05, 0.49))

        model = IsolationForest(
            n_estimators=500,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        # Entraîner sur les données normales uniquement
        mask_normal = df['Alert_Status'] == 'NORMAL'
        model.fit(X_sc[mask_normal.values])

        # Charger les seuils
        seuils = None
        if os.path.exists('data/seuils_moteurs.csv'):
            df_s   = pd.read_csv('data/seuils_moteurs.csv')
            seuils = {}
            for _, row in df_s.iterrows():
                mid = int(row['motor_id'])
                seuils[mid] = {
                    col: float(row[col])
                    for col in df_s.columns
                    if col != 'motor_id'
                }

        # Sauvegarder
        manager = ModelManager()
        success = manager.sauvegarder(
            model  = model,
            scaler = scaler,
            seuils = seuils,
            metadata = {
                'auc'        : auc,
                'n_features' : len(cols_dispo),
                'features'   : cols_dispo,
                'contamination': contamination,
                'n_lignes_train': int(mask_normal.sum()),
            }
        )

        if success:
            manager.afficher_infos()

        return success

    except Exception as e:
        print(f"  ❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════
#  MAIN — LIGNE DE COMMANDE
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ModelManager — Gestion du modèle IA sauvegardé'
    )
    parser.add_argument('--info',      action='store_true',
                        help='Afficher les infos du modèle')
    parser.add_argument('--test',      action='store_true',
                        help='Tester le modèle sauvegardé')
    parser.add_argument('--sauvegarder', action='store_true',
                        help='Sauvegarder depuis le pipeline existant')
    parser.add_argument('--auc',       type=float, default=None,
                        help='AUC ROC à enregistrer dans les métadonnées')
    args = parser.parse_args()

    manager = ModelManager()

    if args.info:
        manager.afficher_infos()

    elif args.test:
        manager.afficher_infos()
        manager.tester()

    elif args.sauvegarder:
        sauvegarder_depuis_pipeline(auc=args.auc)

    else:
        # Par défaut — afficher les infos
        manager.afficher_infos()
        if manager.existe():
            print("\n  Commandes disponibles :")
            print("  python model_manager.py --info")
            print("  python model_manager.py --test")
            print("  python model_manager.py --sauvegarder --auc 0.9544")
        else:
            print("\n  → Pour sauvegarder le modèle après le pipeline :")
            print("  python model_manager.py --sauvegarder --auc 0.9544")


if __name__ == '__main__':
    main()
