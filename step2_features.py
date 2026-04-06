"""
========================================================
 ÉTAPE 2 — NETTOYAGE + FEATURE ENGINEERING (V3)
 Entrée  : data/01_raw_motor.csv
 Sortie  : data/02_features_motor.csv

 NETTOYAGE (V2 conservé intact) :
   • Suppression doublons horodatés
   • Détection valeurs physiquement impossibles
   • Détection gaps temporels (trous > 1 heure)
   • Interpolation linéaire intelligente (limit=5)

 FEATURES V2 (conservées) :
   • Rolling : vib_mean/std/kurt, crest_factor
   • Envelope (Hilbert), FFT, health_score, temp_trend

 AXE 3 — 6 NOUVELLES FEATURES VIBRATOIRES :
   • vib_rms          → Root Mean Square        (roulements usés)
   • vib_skewness     → Asymétrie distribution  (chocs/impacts)
   • peak2peak        → Amplitude max-min        (usure mécanique)
   • spectral_entropy → Complexité spectre FFT  (engrenages)
   • shape_factor     → RMS / Mean abs           (cavitation)
   • impulse_factor   → Peak / Mean abs          (chocs précoces)
========================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
import os
import warnings
warnings.filterwarnings('ignore')

INPUT_CSV  = 'data/01_raw_motor.csv'
OUTPUT_CSV = 'data/02_features_motor.csv'
WINDOW     = 20

LIMITES_PHYSIQUES = {
    'temperature' : (5,   120),
    'courant'     : (0,   500),
    'vibration'   : (0,    50),
    'acceleration': (0,    50),
    'vitesse'     : (500, 4000),
    'cosphi'      : (0.5,  1.0),
}
INTERVALLE_NOMINAL_MIN = 15
SEUIL_GAP_MIN          = 60


# ══════════════════════════════════════════════════════
#  NETTOYAGE (V2 — inchangé)
# ══════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n_init  = len(df)
    rapport = []

    n_avant = len(df)
    df = df.drop_duplicates(subset=['motor_id', 'timestamp'])
    rapport.append(f"  {'⚠' if n_avant-len(df) else '✓'} Doublons : {n_avant-len(df):,}")

    df = df.sort_values(['motor_id', 'timestamp']).reset_index(drop=True)

    for col, (lo, hi) in LIMITES_PHYSIQUES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
        rapport.append(f"  {'⚠' if n else '✓'} {col:<14} [{lo},{hi}] : {n:,} corrigées")

    gaps = []
    for mid in df['motor_id'].unique():
        ts = df.loc[df['motor_id']==mid, 'timestamp'].sort_values()
        deltas = ts.diff().dropna().dt.total_seconds() / 60
        for idx, g in deltas[deltas > SEUIL_GAP_MIN].items():
            gaps.append({'motor_id': mid, 'gap_min': round(g, 1)})
    rapport.append(f"  {'⚠' if gaps else '✓'} Gaps temporels : {len(gaps)}")

    numeric_cols = ['temperature','courant','vibration','acceleration',
                    'vitesse','cosphi','thdi','thdu']
    cols_ok = [c for c in numeric_cols if c in df.columns]
    n_nan   = df[cols_ok].isna().sum().sum()
    df[cols_ok] = df.groupby('motor_id')[cols_ok].transform(
        lambda g: g.interpolate(method='linear', limit=5))
    rapport.append(f"  ✓ Interpolées : {n_nan - df[cols_ok].isna().sum().sum():,}")

    df = df.dropna(subset=['vibration', 'temperature'])

    print(f"\n  {'─'*50}")
    print(f"  NETTOYAGE V3")
    print(f"  {'─'*50}")
    for l in rapport:
        print(l)
    print(f"  {'─'*50}")
    print(f"  {n_init:,} → {len(df):,} lignes")
    print(f"  {'─'*50}")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════
#  AXE 3 — Entropie spectrale
# ══════════════════════════════════════════════════════
def spectral_entropy(signal: np.ndarray) -> float:
    """
    Entropie spectrale = -Σ p(f)·log(p(f))
    Valeur haute → spectre étalé → défaut engrenage
    Valeur basse → fréquences dominantes nettes → normal
    """
    if len(signal) < 8:
        return 0.0
    spec  = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    total = spec.sum()
    if total < 1e-12:
        return 0.0
    p = spec / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_fft_features(signal: np.ndarray) -> tuple:
    if len(signal) < 4:
        return 0.0, 0.0
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    return float(spectrum.max()), float(np.argmax(spectrum))


# ══════════════════════════════════════════════════════
#  FEATURES ROLLING — V3 (V2 + 6 nouvelles)
# ══════════════════════════════════════════════════════
def add_rolling_features(group: pd.DataFrame, window: int) -> pd.DataFrame:
    g   = group.sort_values('timestamp').copy()
    vib = g['vibration']
    tmp = g['temperature']
    cur = g['courant']

    roll_vib = vib.rolling(window=window, min_periods=3)
    roll_tmp = tmp.rolling(window=window, min_periods=3)
    roll_cur = cur.rolling(window=window, min_periods=3)

    # ── V2 features ───────────────────────────────
    g['vib_mean']        = roll_vib.mean()
    g['vib_std']         = roll_vib.std().fillna(0)
    g['vib_max']         = roll_vib.max()
    g['vib_energy']      = vib ** 2
    g['vib_energy_mean'] = g['vib_energy'].rolling(window=window, min_periods=3).mean()
    g['vib_kurt']        = roll_vib.apply(
        lambda x: kurtosis(x, nan_policy='omit') if len(x) >= 4 else 0.0,
        raw=True
    ).fillna(0)
    rms_roll         = np.sqrt((vib**2).rolling(window=window, min_periods=3).mean())
    g['crest_factor'] = (g['vib_max'] / (rms_roll + 1e-9)).fillna(1.0)

    g['temp_mean']  = roll_tmp.mean()
    g['temp_std']   = roll_tmp.std().fillna(0)
    g['temp_trend'] = roll_tmp.apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 4 else 0.0,
        raw=True
    ).fillna(0)
    g['courant_mean'] = roll_cur.mean()
    g['courant_std']  = roll_cur.std().fillna(0)

    try:
        analytic         = hilbert(vib.fillna(0).values)
        envelope         = np.abs(analytic)
        g['envelope']      = envelope
        g['envelope_mean'] = pd.Series(envelope).rolling(window=window, min_periods=3).mean().values
    except Exception:
        g['envelope']      = 0.0
        g['envelope_mean'] = 0.0

    vib_v      = vib.fillna(0).values
    fft_maxs, fft_doms = [], []
    for i in range(len(vib_v)):
        start = max(0, i - window + 1)
        fm, fd = compute_fft_features(vib_v[start:i + 1])
        fft_maxs.append(fm)
        fft_doms.append(fd)
    g['fft_max_amp']       = fft_maxs
    g['fft_dominant_freq'] = fft_doms

    # ── AXE 3 — 6 Nouvelles features V3 ──────────

    # 1. vib_rms — énergie continue (roulements usés)
    g['vib_rms'] = rms_roll.fillna(0)

    # 2. vib_skewness — asymétrie (chocs impulsifs)
    g['vib_skewness'] = roll_vib.apply(
        lambda x: float(skew(x)) if len(x) >= 4 else 0.0,
        raw=True
    ).fillna(0)

    # 3. peak2peak — amplitude max-min (usure mécanique)
    g['peak2peak'] = roll_vib.apply(
        lambda x: float(x.max() - x.min()) if len(x) >= 3 else 0.0,
        raw=True
    ).fillna(0)

    # 4. spectral_entropy — complexité fréquentielle (engrenages)
    sp_ent = []
    for i in range(len(vib_v)):
        start = max(0, i - window + 1)
        sp_ent.append(spectral_entropy(vib_v[start:i + 1]))
    g['spectral_entropy'] = sp_ent

    # 5. shape_factor = RMS / |mean| (cavitation pompes)
    mean_abs_roll       = vib.abs().rolling(window=window, min_periods=3).mean()
    g['shape_factor']   = (g['vib_rms'] / (mean_abs_roll + 1e-9)).fillna(1.0)

    # 6. impulse_factor = Peak / |mean| (chocs précoces roulements)
    g['impulse_factor'] = (g['vib_max'] / (mean_abs_roll + 1e-9)).fillna(1.0)

    return g


# ══════════════════════════════════════════════════════
#  HEALTH SCORE V3 — intègre nouvelles features
# ══════════════════════════════════════════════════════
def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()

    def norm_inv(s, p_lo=5, p_hi=95):
        lo, hi = s.quantile(p_lo/100), s.quantile(p_hi/100)
        if hi == lo:
            return pd.Series(50.0, index=s.index)
        return (1 - ((s - lo) / (hi - lo)).clip(0, 1)) * 100

    # V2 composantes
    s_vib   = norm_inv(g['vib_energy_mean'])
    s_temp  = norm_inv(g['temp_mean'])
    s_kurt  = norm_inv(g['vib_kurt'].abs())
    s_crest = norm_inv(g['crest_factor'])
    # V3 — nouvelles composantes
    s_imp   = norm_inv(g['impulse_factor'])   # chocs précoces
    s_rms   = norm_inv(g['vib_rms'])          # énergie continue

    g['health_score'] = (
        0.28 * s_vib   +
        0.22 * s_temp  +
        0.18 * s_kurt  +
        0.15 * s_crest +
        0.10 * s_imp   +   # AXE 3
        0.07 * s_rms       # AXE 3
    ).clip(0, 100)
    return g


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" ÉTAPE 2 — FEATURE ENGINEERING V3")
    print(" AXE 3 : +6 features vibratoires")
    print("         vib_rms | skewness | peak2peak")
    print("         spectral_entropy | shape_factor | impulse_factor")
    print("=" * 60)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERREUR] Fichier introuvable : {INPUT_CSV}")
        print("→ Lancez d'abord : python step1_extraction.py")
        return

    print(f"\n→ Chargement de {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['timestamp'])
    print(f"  {len(df):,} lignes | {df['motor_id'].nunique()} moteurs")

    print("\n→ Nettoyage ...")
    df = clean_data(df)

    print(f"\n→ Features rolling V3 (fenêtre={WINDOW}) ...")
    parts = []
    for mid, group in df.groupby('motor_id'):
        enriched            = add_rolling_features(group.copy(), WINDOW)
        enriched['motor_id'] = mid
        parts.append(enriched)
        print(f"  M{mid:2d} ✓", end='  ')
    print()
    df_feat = pd.concat(parts, ignore_index=True)

    print("\n→ Health score V3 ...")
    parts2 = []
    for mid, group in df_feat.groupby('motor_id'):
        parts2.append(compute_health_score(group.copy()))
    df_feat = pd.concat(parts2, ignore_index=True)

    # Remplissage NaN résiduels
    num_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[num_cols] = df_feat[num_cols].fillna(0)

    # Rapport
    v3_feats = ['vib_rms','vib_skewness','peak2peak',
                'spectral_entropy','shape_factor','impulse_factor']
    total_feats = df_feat.shape[1]

    print(f"\n{'='*60}")
    print(f"  RÉSULTAT FEATURE ENGINEERING V3")
    print(f"{'='*60}")
    print(f"  Lignes   : {df_feat.shape[0]:,}")
    print(f"  Colonnes : {total_feats}  (+6 vs V2)")
    print(f"\n  AXE 3 — Nouvelles features :")
    print(f"  {'Feature':<22} {'Min':>8} {'Mean':>8} {'Max':>8}  Cible")
    print(f"  {'─'*62}")
    cibles = {
        'vib_rms'          : 'Roulements usés',
        'vib_skewness'     : 'Chocs / impacts',
        'peak2peak'        : 'Usure mécanique',
        'spectral_entropy' : 'Défauts engrenages',
        'shape_factor'     : 'Cavitation pompes',
        'impulse_factor'   : 'Chocs impulsifs précoces',
    }
    for c in v3_feats:
        if c in df_feat.columns:
            mn = df_feat[c].min()
            av = df_feat[c].mean()
            mx = df_feat[c].max()
            print(f"  {c:<22} {mn:>8.3f} {av:>8.3f} {mx:>8.3f}  → {cibles[c]}")

    print(f"\n  Health score moyen (V3 vs V2) :")
    hs = df_feat.groupby('motor_id')['health_score'].mean().round(1)
    for mid, v in hs.items():
        bar = '█' * int(v / 5)
        print(f"    M{mid:2d}: {v:5.1f}  {bar}")

    df_feat.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Sauvegardé : {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == '__main__':
    main()
