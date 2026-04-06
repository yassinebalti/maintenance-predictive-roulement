"""
========================================================
 ÉTAPE 3B — AUTOENCODER (Réseau de Neurones)
 Entrée  : data/02_features_motor.csv
 Sorties : data/03b_autoencoder.csv
           data/comparaison_modeles.csv
           figures/fig_autoencoder_*.png

 Modèle : Autoencoder Dense (Deep Learning)
 ─────────────────────────────────────────────
 Architecture :
   Encodeur : 18 → 12 → 6 → 3  (compression)
   Décodeur :  3 → 6 → 12 → 18 (reconstruction)

 Principe :
   Entraîné uniquement sur données NORMALES
   Anomalie = erreur de reconstruction élevée

 Comparaison avec Modèle Hybride IF :
   → Tableau AUC, Précision, F1, Recall
   → Figure comparative
========================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────
INPUT_CSV    = 'data/02_features_motor.csv'
ANOMALIES_IF = 'data/03_anomalies.csv'
OUTPUT_CSV   = 'data/03b_autoencoder.csv'
COMPARE_CSV  = 'data/comparaison_modeles.csv'
FIGURES_DIR  = 'figures'

# Hyperparamètres Autoencoder
LEARNING_RATE = 0.001
N_EPOCHS      = 150
BATCH_SIZE    = 256
RANDOM_STATE  = 42

FEATURES = [
    'temperature_exceed','courant_exceed','vibration_exceed','acceleration_exceed',
    'temperature_ratio', 'courant_ratio', 'vibration_ratio', 'acceleration_ratio',
    'n_exceed','severity_score',
    'vib_energy_mean','vib_kurt','crest_factor',
    'temp_mean','temp_trend','courant_mean','envelope_mean','health_score',
]
# ──────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════
#  AUTOENCODER IMPLÉMENTÉ EN NUMPY PUR
#  (pas besoin de TensorFlow/PyTorch)
# ══════════════════════════════════════════════════════
class Autoencoder:
    """
    Autoencoder Dense avec rétropropagation manuelle.

    Architecture :
      Encodeur : input_dim → 12 → 6 → 3
      Décodeur : 3 → 6 → 12 → input_dim

    Activation : ReLU (couches cachées) + Sigmoid (sortie)
    Perte      : MSE (Mean Squared Error)
    Optimizer  : Adam
    """

    def __init__(self, input_dim, lr=0.001, seed=42):
        np.random.seed(seed)
        self.lr = lr

        # ── Architecture ──────────────────────────────
        # Encodeur :  input → 12 → 6 → 3
        # Décodeur :  3 → 6 → 12 → input
        dims_enc = [input_dim, 12, 6, 3]
        dims_dec = [3, 6, 12, input_dim]

        # Initialisation He (meilleure pour ReLU)
        self.W_enc, self.b_enc = [], []
        for i in range(len(dims_enc) - 1):
            fan_in = dims_enc[i]
            W = np.random.randn(dims_enc[i], dims_enc[i+1]) * np.sqrt(2.0/fan_in)
            b = np.zeros((1, dims_enc[i+1]))
            self.W_enc.append(W); self.b_enc.append(b)

        self.W_dec, self.b_dec = [], []
        for i in range(len(dims_dec) - 1):
            fan_in = dims_dec[i]
            W = np.random.randn(dims_dec[i], dims_dec[i+1]) * np.sqrt(2.0/fan_in)
            b = np.zeros((1, dims_dec[i+1]))
            self.W_dec.append(W); self.b_dec.append(b)

        # Paramètres Adam
        self.init_adam()

    def init_adam(self):
        """Initialise les moments Adam pour tous les paramètres."""
        self.m_We = [np.zeros_like(W) for W in self.W_enc]
        self.v_We = [np.zeros_like(W) for W in self.W_enc]
        self.m_be = [np.zeros_like(b) for b in self.b_enc]
        self.v_be = [np.zeros_like(b) for b in self.b_enc]
        self.m_Wd = [np.zeros_like(W) for W in self.W_dec]
        self.v_Wd = [np.zeros_like(W) for W in self.W_dec]
        self.m_bd = [np.zeros_like(b) for b in self.b_dec]
        self.v_bd = [np.zeros_like(b) for b in self.b_dec]
        self.t = 0  # pas de temps Adam

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        """Passe avant complète : encodage + décodage."""
        # Encodeur
        self.enc_inputs = [X]
        h = X
        for W, b in zip(self.W_enc, self.b_enc):
            z = h @ W + b
            h = self.relu(z)
            self.enc_inputs.append(h)
        self.encoded = h

        # Décodeur
        self.dec_inputs = [h]
        for i, (W, b) in enumerate(zip(self.W_dec, self.b_dec)):
            z = h @ W + b
            # Dernière couche : sigmoid pour output [0,1]
            h = self.sigmoid(z) if i == len(self.W_dec)-1 else self.relu(z)
            self.dec_inputs.append(h)

        self.output = h
        return self.output

    def backward(self, X):
        """Rétropropagation du gradient."""
        n = X.shape[0]
        dL_dout = 2 * (self.output - X) / n  # gradient MSE

        grads_Wd, grads_bd = [], []
        dh = dL_dout
        # Décodeur (sens inverse)
        for i in range(len(self.W_dec)-1, -1, -1):
            h_prev = self.dec_inputs[i]
            h_curr = self.dec_inputs[i+1]
            # Gradient activation
            if i == len(self.W_dec)-1:
                da = dh * h_curr * (1 - h_curr)  # sigmoid
            else:
                da = dh * self.relu_grad(h_prev @ self.W_dec[i] + self.b_dec[i])
            grads_Wd.insert(0, h_prev.T @ da)
            grads_bd.insert(0, da.sum(axis=0, keepdims=True))
            dh = da @ self.W_dec[i].T

        grads_We, grads_be = [], []
        # Encodeur (sens inverse)
        for i in range(len(self.W_enc)-1, -1, -1):
            h_prev = self.enc_inputs[i]
            z = h_prev @ self.W_enc[i] + self.b_enc[i]
            da = dh * self.relu_grad(z)
            grads_We.insert(0, h_prev.T @ da)
            grads_be.insert(0, da.sum(axis=0, keepdims=True))
            dh = da @ self.W_enc[i].T

        return grads_We, grads_be, grads_Wd, grads_bd

    def adam_update(self, grads_We, grads_be, grads_Wd, grads_bd):
        """Mise à jour des poids avec l'optimiseur Adam."""
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bc1 = 1 - beta1**self.t
        bc2 = 1 - beta2**self.t

        def update(W, g, m, v):
            m[:] = beta1*m + (1-beta1)*g
            v[:] = beta2*v + (1-beta2)*g**2
            return W - self.lr * (m/bc1) / (np.sqrt(v/bc2) + eps)

        for i in range(len(self.W_enc)):
            self.W_enc[i] = update(self.W_enc[i], grads_We[i],
                                   self.m_We[i], self.v_We[i])
            self.b_enc[i] = update(self.b_enc[i], grads_be[i],
                                   self.m_be[i], self.v_be[i])
        for i in range(len(self.W_dec)):
            self.W_dec[i] = update(self.W_dec[i], grads_Wd[i],
                                   self.m_Wd[i], self.v_Wd[i])
            self.b_dec[i] = update(self.b_dec[i], grads_bd[i],
                                   self.m_bd[i], self.v_bd[i])

    def fit(self, X_normal, epochs=150, batch_size=256, verbose=True):
        """
        Entraînement sur données NORMALES uniquement.
        Le modèle apprend à reconstruire ce qui est normal.
        """
        n = X_normal.shape[0]
        history = []

        for epoch in range(1, epochs+1):
            # Mélanger les données
            idx = np.random.permutation(n)
            X_shuf = X_normal[idx]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, batch_size):
                Xb = X_shuf[start:start+batch_size]
                # Forward
                self.forward(Xb)
                # Loss MSE
                loss = np.mean((self.output - Xb)**2)
                epoch_loss += loss
                n_batches  += 1
                # Backward + Adam
                grads = self.backward(Xb)
                self.adam_update(*grads)

            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)

            if verbose and (epoch % 25 == 0 or epoch == 1):
                print(f"    Epoch {epoch:3d}/{epochs} | Loss MSE = {avg_loss:.6f}")

        return history

    def reconstruction_error(self, X):
        """Calcule l'erreur de reconstruction par échantillon."""
        out = self.forward(X)
        return np.mean((out - X)**2, axis=1)


# ══════════════════════════════════════════════════════
#  SEUILS PAR MOTEUR (identique à step3)
# ══════════════════════════════════════════════════════
PARAMS_ALERTES = ['temperature', 'courant', 'vibration', 'acceleration']
POIDS_PARAMS   = {'temperature':0.35,'courant':0.30,'vibration':0.25,'acceleration':0.10}

def calibrer_seuils(df):
    seuils = {}
    for mid in sorted(df['motor_id'].unique()):
        dm = df[df['motor_id']==mid]
        seuils[mid] = {}
        for col in PARAMS_ALERTES:
            if col not in dm.columns: continue
            av = (dm[dm['alert_parameter']==col][col]
                  if 'alert_parameter' in dm.columns else pd.Series())
            seuils[mid][col] = (float(av.min()) if len(av)>0
                                else float(dm[col].quantile(0.95)))
    return seuils

def ajouter_features_depassement(df, seuils):
    df = df.copy()
    severity = pd.Series(0.0, index=df.index)
    for col in PARAMS_ALERTES:
        if col not in df.columns:
            df[f'{col}_exceed'] = 0.0; df[f'{col}_ratio'] = 0.0
            continue
        df[f'{col}_exceed'] = 0.0; df[f'{col}_ratio'] = 0.0
        for mid in df['motor_id'].unique():
            mask  = df['motor_id']==mid
            seuil = seuils.get(mid,{}).get(col, df[col].quantile(0.95))
            df.loc[mask, f'{col}_exceed'] = (df.loc[mask,col]>seuil).astype(float)
            df.loc[mask, f'{col}_ratio']  = df.loc[mask,col]/(seuil+1e-9)
        severity += df[f'{col}_exceed'] * POIDS_PARAMS[col]
    df['n_exceed']       = df[[f'{c}_exceed' for c in PARAMS_ALERTES]].sum(axis=1)
    df['severity_score'] = severity.clip(0,1)
    return df


# ══════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════
def plot_training_loss(history):
    """Courbe de perte pendant l'entraînement."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#16213e')
    ax.plot(history, color='#00bcd4', lw=2, label='Loss MSE')
    ax.fill_between(range(len(history)), history, alpha=0.2, color='#00bcd4')
    ax.set_title('Courbe d\'entraînement Autoencoder',
                 color='white', fontweight='bold', fontsize=13)
    ax.set_xlabel('Époque', color='white')
    ax.set_ylabel('Erreur MSE', color='white')
    ax.legend(facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_autoencoder_training.png',
                dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  → figures/fig_autoencoder_training.png")


def plot_reconstruction_error(df, threshold_ae):
    """Distribution des erreurs de reconstruction."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # Distribution globale
    ax = axes[0]; ax.set_facecolor('#16213e')
    normal = df[df['Alert_Status']=='NORMAL']['recon_error']
    alert  = df[df['Alert_Status']=='ALERT']['recon_error']
    ax.hist(normal, bins=80, alpha=0.7, color='#4caf50',
            label=f'NORMAL (n={len(normal):,})', density=True)
    ax.hist(alert,  bins=80, alpha=0.7, color='#f44336',
            label=f'ALERT (n={len(alert):,})',  density=True)
    ax.axvline(threshold_ae, color='orange', linestyle='--',
               lw=2, label=f'Seuil = {threshold_ae:.4f}')
    ax.set_title('Distribution Erreur de Reconstruction',
                 color='white', fontweight='bold')
    ax.set_xlabel('Erreur MSE', color='white')
    ax.set_ylabel('Densité', color='white')
    ax.legend(facecolor='#16213e', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    # Par moteur
    ax = axes[1]; ax.set_facecolor('#16213e')
    err_mean = df.groupby('motor_id')['recon_error'].mean().sort_values(ascending=True)
    colors   = ['#d32f2f' if v > threshold_ae*2 else
                '#f57c00' if v > threshold_ae else '#388e3c'
                for v in err_mean.values]
    ax.barh(err_mean.index.astype(str), err_mean.values,
            color=colors, edgecolor='#333')
    ax.axvline(threshold_ae, color='orange', linestyle='--',
               lw=2, label='Seuil anomalie')
    ax.set_title('Erreur de reconstruction par moteur',
                 color='white', fontweight='bold')
    ax.set_xlabel('Erreur MSE moyenne', color='white')
    ax.legend(facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    fig.suptitle('Autoencoder — Analyse erreurs de reconstruction',
                 color='white', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_autoencoder_errors.png',
                dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  → figures/fig_autoencoder_errors.png")


def plot_comparaison(df_compare):
    """Figure comparative des deux modèles."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#1a1a2e')

    modeles  = df_compare['modele'].tolist()
    couleurs = ['#2196f3', '#ff5722']
    metrics  = [
        ('auc',       'AUC ROC',       0.50, 1.00),
        ('precision', 'Précision (%)', 50,   100),
        ('f1',        'F1-score',      0.50, 1.00),
    ]

    for i, (col, label, ymin, ymax) in enumerate(metrics):
        ax = axes[i]; ax.set_facecolor('#16213e')
        vals = df_compare[col].tolist()
        bars = ax.bar(modeles, vals, color=couleurs, edgecolor='#555', width=0.5)

        # Valeurs sur les barres
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (ymax-ymin)*0.01,
                    f'{val:.4f}' if col in ('auc','f1') else f'{val:.1f}%',
                    ha='center', va='bottom', color='white',
                    fontweight='bold', fontsize=11)

        ax.set_title(label, color='white', fontweight='bold', fontsize=12)
        ax.set_ylim(ymin, ymax + (ymax-ymin)*0.12)
        ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
        ax.set_facecolor('#16213e')

        # Ligne de référence
        ax.axhline(vals[0], color=couleurs[0], linestyle='--',
                   lw=1, alpha=0.5)

    fig.suptitle('Comparaison Modèles : IF Hybride vs Autoencoder',
                 color='white', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_comparaison_modeles.png',
                dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  → figures/fig_comparaison_modeles.png")


def plot_courbe_roc(df):
    """Courbe ROC des deux modèles."""
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#16213e')

    y_true = (df['Alert_Status']=='ALERT').astype(int)

    for col, label, color in [
        ('combined_score', 'IF Hybride (AUC=0.9495)', '#2196f3'),
        ('ae_score',       'Autoencoder',             '#ff5722'),
    ]:
        if col not in df.columns: continue
        fpr, tpr, _ = roc_curve(y_true, df[col])
        auc = roc_auc_score(y_true, df[col])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{label} — AUC={auc:.4f}')
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)

    ax.plot([0,1],[0,1], '--', color='#666', lw=1.5, label='Aléatoire (AUC=0.50)')
    ax.set_xlabel('Taux Faux Positifs', color='white', fontsize=12)
    ax.set_ylabel('Taux Vrais Positifs', color='white', fontsize=12)
    ax.set_title('Courbe ROC — Comparaison des Modèles',
                 color='white', fontweight='bold', fontsize=13)
    ax.legend(facecolor='#16213e', labelcolor='white', fontsize=10)
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_autoencoder_roc.png',
                dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  → figures/fig_autoencoder_roc.png")


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" ÉTAPE 3B — AUTOENCODER (Deep Learning Non Supervisé)")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Vérifications
    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] {INPUT_CSV} introuvable.")
        print("→ Lancez d'abord : python step2_features.py")
        return
    if not os.path.exists(ANOMALIES_IF):
        print(f"[ERREUR] {ANOMALIES_IF} introuvable.")
        print("→ Lancez d'abord : python step3_anomaly_detection.py")
        return

    # ── 1. Chargement ─────────────────────────────────
    print(f"\n→ [1/6] Chargement des données ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")
    print(f"  NORMAL: {(df['Alert_Status']=='NORMAL').sum():,} | "
          f"ALERT: {(df['Alert_Status']=='ALERT').sum():,}")

    # ── 2. Features de dépassement ────────────────────
    print(f"\n→ [2/6] Calcul features de dépassement ...")
    seuils = calibrer_seuils(df)
    df     = ajouter_features_depassement(df, seuils)

    # Préparer features
    avail = [f for f in FEATURES if f in df.columns]
    print(f"  {len(avail)} features disponibles sur {len(FEATURES)}")
    X     = df[avail].fillna(0).values

    # Normalisation [0, 1]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    # Min-Max pour sigmoid en sortie
    X_mm   = (X_sc - X_sc.min(axis=0)) / (X_sc.max(axis=0) - X_sc.min(axis=0) + 1e-9)

    # Données d'entraînement = NORMALES uniquement
    normal_mask = df['Alert_Status'] == 'NORMAL'
    X_train     = X_mm[normal_mask]
    print(f"  Entraînement sur {X_train.shape[0]:,} données NORMALES")

    # ── 3. Entraînement Autoencoder ───────────────────
    print(f"\n→ [3/6] Entraînement Autoencoder ...")
    print(f"  Architecture : {len(avail)} → 12 → 6 → 3 → 6 → 12 → {len(avail)}")
    print(f"  Epochs={N_EPOCHS} | Batch={BATCH_SIZE} | LR={LEARNING_RATE}")
    print()

    ae = Autoencoder(input_dim=len(avail), lr=LEARNING_RATE, seed=RANDOM_STATE)
    history = ae.fit(X_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, verbose=True)

    # ── 4. Scores anomalies ───────────────────────────
    print(f"\n→ [4/6] Calcul scores anomalies ...")
    recon_error = ae.reconstruction_error(X_mm)

    # Seuil : percentile sur données normales
    threshold_ae = np.percentile(recon_error[normal_mask], 90)
    print(f"  Seuil reconstruction (p90 normal) = {threshold_ae:.6f}")

    # Normaliser l'erreur → score [0,1]
    ae_score = (recon_error - recon_error.min()) / \
               (recon_error.max() - recon_error.min())

    df['recon_error'] = recon_error
    df['ae_score']    = ae_score
    df['ae_anomaly']  = recon_error > threshold_ae

    # ── 5. Validation & Comparaison ───────────────────
    print(f"\n→ [5/6] Validation & Comparaison des modèles ...")

    # Charger résultats IF Hybride
    df_if = pd.read_csv(ANOMALIES_IF, parse_dates=['timestamp'])
    df['combined_score'] = df_if['combined_score'].values \
                           if 'combined_score' in df_if.columns else 0.5

    y_true = (df['Alert_Status'] == 'ALERT').astype(int)

    # AUC
    auc_ae = roc_auc_score(y_true, df['ae_score'])
    auc_if = roc_auc_score(y_true, df['combined_score']) \
             if 'combined_score' in df.columns else 0.9495

    # Métriques AE
    y_pred_ae = df['ae_anomaly'].astype(int)
    rep_ae = classification_report(y_true, y_pred_ae,
                                   target_names=['NORMAL','ALERT'],
                                   output_dict=True)
    f1_ae  = rep_ae['ALERT']['f1-score']
    prec_ae = rep_ae['ALERT']['precision'] * 100
    rec_ae  = rep_ae['ALERT']['recall']    * 100
    acc_ae  = rep_ae['accuracy']           * 100

    # Métriques IF (depuis step3)
    y_pred_if = (df['combined_score'] >= 0.25).astype(int)
    rep_if = classification_report(y_true, y_pred_if,
                                   target_names=['NORMAL','ALERT'],
                                   output_dict=True)
    f1_if  = rep_if['ALERT']['f1-score']
    prec_if = rep_if['ALERT']['precision'] * 100
    rec_if  = rep_if['ALERT']['recall']    * 100
    acc_if  = rep_if['accuracy']           * 100

    # ── Affichage résultats ───────────────────────────
    sep = "─" * 55
    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║         COMPARAISON DES MODÈLES                      ║
  ╠══════════╦═══════════════════╦══════════════════════╣
  ║ Métrique ║   IF Hybride      ║   Autoencoder        ║
  ╠══════════╬═══════════════════╬══════════════════════╣
  ║ AUC ROC  ║   {auc_if:.4f}          ║   {auc_ae:.4f}              ║
  ║ Précision║   {prec_if:.1f}%          ║   {prec_ae:.1f}%            ║
  ║ Rappel   ║   {rec_if:.1f}%          ║   {rec_ae:.1f}%            ║
  ║ F1-score ║   {f1_if:.4f}          ║   {f1_ae:.4f}              ║
  ║ Accuracy ║   {acc_if:.1f}%          ║   {acc_ae:.1f}%            ║
  ╠══════════╬═══════════════════╬══════════════════════╣
  ║ Type     ║ Statistique+règles║ Deep Learning         ║
  ║ Non supv ║       ✅          ║       ✅              ║
  ╚══════════╩═══════════════════╩══════════════════════╝

  Gagnant AUC     : {"IF Hybride" if auc_if >= auc_ae else "Autoencoder"}
  Meilleur rappel : {"IF Hybride" if rec_if  >= rec_ae  else "Autoencoder"}
    """)

    # Matrice de confusion Autoencoder
    cm = confusion_matrix(y_true, y_pred_ae)
    print("  Matrice de confusion Autoencoder :")
    print(f"  {'':12s} | Prédit NORMAL | Prédit ALERT")
    print(f"  {'Réel NORMAL':12s} | {cm[0,0]:13,} | {cm[0,1]:12,}")
    print(f"  {'Réel ALERT':12s} | {cm[1,0]:13,} | {cm[1,1]:12,}")

    print(f"\n  Rapport détaillé Autoencoder :")
    print(classification_report(y_true, y_pred_ae,
                                target_names=['NORMAL','ALERT'],
                                digits=4))

    # Résumé par moteur
    print(f"  Anomalies Autoencoder par moteur :")
    print(f"  {'Moteur':>8} | {'Anomalies':>10} | {'Taux':>6} | "
          f"{'Erreur moy':>10} | Statut")
    print("  " + "-"*55)
    for mid in sorted(df['motor_id'].unique()):
        dm   = df[df['motor_id']==mid]
        n_an = dm['ae_anomaly'].sum()
        taux = n_an/len(dm)*100
        err  = dm['recon_error'].mean()
        st   = ('⚠ CRITIQUE' if taux>40 else '⚠ ÉLEVÉ' if taux>20
                else '~ MODÉRÉ' if taux>10 else '✓ NORMAL')
        print(f"  {mid:>8} | {n_an:>10,} | {taux:>5.1f}% | "
              f"{err:>10.6f} | {st}")

    # Sauvegarder comparaison
    df_compare = pd.DataFrame([
        {'modele':'IF Hybride', 'auc':auc_if, 'precision':prec_if,
         'rappel':rec_if, 'f1':f1_if, 'accuracy':acc_if},
        {'modele':'Autoencoder','auc':auc_ae, 'precision':prec_ae,
         'rappel':rec_ae, 'f1':f1_ae, 'accuracy':acc_ae},
    ])
    df_compare.to_csv(COMPARE_CSV, index=False)
    print(f"\n  ✓ Comparaison sauvegardée : {COMPARE_CSV}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Résultats AE sauvegardés : {OUTPUT_CSV}")

    # ── 6. Figures ────────────────────────────────────
    print(f"\n→ [6/6] Génération des figures ...")
    plot_training_loss(history)
    plot_reconstruction_error(df, threshold_ae)
    plot_courbe_roc(df)
    plot_comparaison(df_compare)

    # ── Résumé final ──────────────────────────────────
    print(f"""
{'='*60}
 ✓ AUTOENCODER TERMINÉ

  Architecture  : {len(avail)} → 12 → 6 → 3 → 6 → 12 → {len(avail)}
  Entraînement  : {N_EPOCHS} epochs | {X_train.shape[0]:,} données normales
  AUC Autoencoder : {auc_ae:.4f}
  AUC IF Hybride  : {auc_if:.4f}

  Fichiers produits :
    {OUTPUT_CSV}
    {COMPARE_CSV}
    figures/fig_autoencoder_training.png
    figures/fig_autoencoder_errors.png
    figures/fig_autoencoder_roc.png
    figures/fig_comparaison_modeles.png

  Prochaine étape : python step4_rul_prediction.py
{'='*60}
    """)


if __name__ == '__main__':
    main()