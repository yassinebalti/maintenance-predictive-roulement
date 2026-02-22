"""
========================================================
 ÉTAPE 2 — NETTOYAGE + FEATURE ENGINEERING
 Entrée  : data/01_raw_motor.csv
 Sortie  : data/02_features_motor.csv

 Features créées :
   • Domaine temporel   : vib_energy, vib_rms_ratio
   • Rolling statistics : vib_mean/std/kurt (fenêtre 20)
   • Envelope (Hilbert) : détection impulsions roulements
   • FFT                : fft_dominant_freq, fft_max_amp
   • Santé moteur       : health_score (0–100)
   • Dérive thermique   : temp_trend (rolling slope)
========================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import hilbert
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────
INPUT_CSV  = 'data/01_raw_motor.csv'
OUTPUT_CSV = 'data/02_features_motor.csv'
WINDOW     = 20   # Fenêtre rolling (nombre de mesures)
# ──────────────────────────────────────────────────────


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage des données brutes.
    - Supprime les outliers physiquement impossibles
    - Impute les valeurs manquantes par interpolation
    - Supprime les doublons horodatés par moteur
    """
    n_init = len(df)

    # Supprimer doublons
    df = df.drop_duplicates(subset=['motor_id', 'timestamp'])

    # Seuils physiques réalistes pour moteurs industriels
    limits = {
        'temperature' : (0,   120),   # °C
        'courant'     : (0,   500),   # A
        'vibration'   : (0,   50),    # mm/s ou g
        'acceleration': (0,   50),    # g
        'vitesse'     : (0,   4000),  # RPM
        'cosphi'      : (0,   1.2),   # facteur de puissance
    }

    for col, (lo, hi) in limits.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            df.loc[mask, col] = np.nan

    # Interpolation linéaire par moteur
    numeric_cols = ['temperature', 'courant', 'vibration', 'acceleration',
                    'vitesse', 'cosphi', 'thdi', 'thdu']
    df = df.sort_values(['motor_id', 'timestamp'])
    df[numeric_cols] = (df.groupby('motor_id')[numeric_cols]
                          .transform(lambda g: g.interpolate(method='linear', limit=5)))

    # Supprime les lignes où vibration ou temperature manque encore
    df = df.dropna(subset=['vibration', 'temperature'])

    n_final = len(df)
    print(f"  Nettoyage : {n_init:,} → {n_final:,} lignes "
          f"({n_init - n_final:,} supprimées)")
    return df.reset_index(drop=True)


def compute_fft_features(signal: np.ndarray, n: int = 5) -> tuple:
    """
    Calcule les features FFT sur un signal :
    - fft_max_amp    : amplitude max du spectre
    - fft_dominant_f : fréquence (index) du pic dominant
    """
    if len(signal) < 4:
        return 0.0, 0.0
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    fft_max  = float(spectrum.max())
    fft_dom  = float(np.argmax(spectrum))
    return fft_max, fft_dom


def add_rolling_features(group: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Features rolling par fenêtre glissante (par moteur).
    Toutes les stats sont calculées sur la vibration.
    """
    g = group.sort_values('timestamp').copy()
    vib = g['vibration']
    tmp = g['temperature']
    cur = g['courant']

    roll_vib = vib.rolling(window=window, min_periods=3)
    roll_tmp = tmp.rolling(window=window, min_periods=3)
    roll_cur = cur.rolling(window=window, min_periods=3)

    # ── Vibration stats ────────────────────────────
    g['vib_mean']   = roll_vib.mean()
    g['vib_std']    = roll_vib.std().fillna(0)
    g['vib_max']    = roll_vib.max()
    g['vib_energy'] = vib ** 2                              # énergie instantanée
    g['vib_energy_mean'] = g['vib_energy'].rolling(window=window, min_periods=3).mean()

    # Kurtosis rolling (sensible aux chocs roulements)
    g['vib_kurt'] = roll_vib.apply(
        lambda x: kurtosis(x, nan_policy='omit') if len(x) >= 4 else 0.0,
        raw=True
    ).fillna(0)

    # Crest factor = pic / RMS → détecte les impulsions
    rms_roll = np.sqrt((vib**2).rolling(window=window, min_periods=3).mean())
    g['crest_factor'] = (g['vib_max'] / (rms_roll + 1e-9)).fillna(1.0)

    # ── Température stats ─────────────────────────
    g['temp_mean'] = roll_tmp.mean()
    g['temp_std']  = roll_tmp.std().fillna(0)
    # Pente thermique (dérive = signe de surchauffe)
    g['temp_trend'] = roll_tmp.apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 4 else 0.0,
        raw=True
    ).fillna(0)

    # ── Courant stats ─────────────────────────────
    g['courant_mean'] = roll_cur.mean()
    g['courant_std']  = roll_cur.std().fillna(0)

    # ── Envelope (Hilbert) ────────────────────────
    # Détecte les impulsions hautes fréquences (défauts roulements)
    try:
        analytic  = hilbert(vib.fillna(0).values)
        envelope  = np.abs(analytic)
        g['envelope']      = envelope
        g['envelope_mean'] = pd.Series(envelope).rolling(window=window, min_periods=3).mean().values
    except Exception:
        g['envelope']      = 0.0
        g['envelope_mean'] = 0.0

    # ── FFT par fenêtre ───────────────────────────
    fft_maxs, fft_doms = [], []
    vib_vals = vib.fillna(0).values
    for i in range(len(vib_vals)):
        start = max(0, i - window + 1)
        seg   = vib_vals[start:i + 1]
        fm, fd = compute_fft_features(seg)
        fft_maxs.append(fm)
        fft_doms.append(fd)
    g['fft_max_amp']      = fft_maxs
    g['fft_dominant_freq'] = fft_doms

    return g


def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score de santé global (0 = critique, 100 = parfait).
    Basé sur la normalisation inverse des indicateurs clés.
    """
    g = df.copy()

    def normalize_inv(series, p_low=5, p_high=95):
        lo = series.quantile(p_low / 100)
        hi = series.quantile(p_high / 100)
        if hi == lo:
            return pd.Series(50.0, index=series.index)
        norm = (series - lo) / (hi - lo)
        return (1 - norm.clip(0, 1)) * 100

    s_vib   = normalize_inv(g['vib_energy_mean'])
    s_temp  = normalize_inv(g['temp_mean'])
    s_kurt  = normalize_inv(g['vib_kurt'].abs())
    s_crest = normalize_inv(g['crest_factor'])

    g['health_score'] = (
        0.35 * s_vib   +
        0.25 * s_temp  +
        0.20 * s_kurt  +
        0.20 * s_crest
    ).clip(0, 100)
    return g


def main():
    print("=" * 55)
    print(" ÉTAPE 2 — NETTOYAGE + FEATURE ENGINEERING")
    print("=" * 55)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] Fichier introuvable : {INPUT_CSV}")
        print("→ Lancez d'abord : python step1_extraction.py")
        return

    print(f"\n→ Chargement de {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    print(f"  {len(df):,} lignes chargées | {df['motor_id'].nunique()} moteurs")

    # ── 1. Nettoyage ──────────────────────────────
    print("\n→ Nettoyage des données ...")
    df = clean_data(df)

    # ── 2. Features rolling (par moteur) ─────────
    print(f"\n→ Calcul des features rolling (fenêtre={WINDOW}) ...")
    # Process each motor separately to avoid pandas groupby key-drop issue
    parts = []
    for mid, group in df.groupby('motor_id'):
        enriched = add_rolling_features(group.copy(), WINDOW)
        enriched['motor_id'] = mid   # garantit que motor_id est présent
        parts.append(enriched)
    df_feat = pd.concat(parts, ignore_index=True)
    print(f"  Features ajoutées : {df_feat.shape[1]} colonnes")

    # ── 3. Score de santé ─────────────────────────
    print("\n→ Calcul du score de santé (health_score) ...")
    parts2 = []
    for mid, group in df_feat.groupby('motor_id'):
        g2 = compute_health_score(group.copy())
        parts2.append(g2)
    df_feat = pd.concat(parts2, ignore_index=True)

    # ── 4. Remplissage NaN résiduels ─────────────
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(0)

    # ── 5. Rapport ────────────────────────────────
    print(f"\n✓ Dataset enrichi : {df_feat.shape[0]:,} lignes × {df_feat.shape[1]} colonnes")
    print("\n  Nouvelles features créées :")
    new_cols = [c for c in df_feat.columns
                if c not in ['measurement_id', 'motor_id', 'timestamp',
                              'Alert_Status', 'alert_parameter', 'alert_code',
                              'temperature', 'courant', 'vibration', 'acceleration',
                              'thdi', 'thdu', 'vitesse', 'cosphi']]
    for col in new_cols:
        print(f"    • {col}")

    print(f"\n  Health score (sample) :")
    sample = df_feat.groupby('motor_id')['health_score'].mean().round(1)
    print(sample.to_string())

    # ── 6. Sauvegarde ────────────────────────────
    df_feat.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Dataset sauvegardé : {OUTPUT_CSV}")
    print("=" * 55)


if __name__ == '__main__':
    main()
