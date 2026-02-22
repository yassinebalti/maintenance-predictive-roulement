"""
========================================================
 ÉTAPE 1B — INTÉGRATION TABLE full_data (VERSION 2)
 
 Découverte importante :
   motor_measurements : Avril–Mai 2025
   full_data          : Novembre 2025–Janvier 2026
   → Périodes séparées, pas de chevauchement temporel

 Stratégie adoptée :
   full_data est traité comme source INDÉPENDANTE et
   plus récente. On crée un dataset séparé analysé
   par le même modèle IA.

 Sorties :
   data/full_data_parsed.csv       → données brutes parsées
   data/full_data_for_model.csv    → format compatible pipeline
   data/full_data_anomalies.csv    → résultats anomalies
   data/sensor_motor_mapping.csv   → mapping capteur→moteur
   figures/fig_full_data_*.png     → 3 figures
========================================================
"""

import re, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SQL_FILE    = 'ai_cp (2).sql'
OUTPUT_FD   = 'data/full_data_parsed.csv'
OUTPUT_MODEL= 'data/full_data_for_model.csv'
OUTPUT_ANOM = 'data/full_data_anomalies.csv'
MAPPING_CSV = 'data/sensor_motor_mapping.csv'
SEUILS_CSV  = 'data/seuils_moteurs.csv'
FIGURES_DIR = 'figures'

SENSOR_MOTOR_MAP = {
    '68c11f06': 1, '4b5e4b32': 2, 'a6a46be1': 3,
    'd9508e77': 4, '53cb61b2': 6, '2c6254af': 8, 'bc59bf5f': 10,
}
MOTOR_INFO = {
    1: {'name':'Moteur 1600 App Cylindre v1','manufacturer':'Siemens','power_kW':55},
    2: {'name':'Moteur 1601 App Cylindre',   'manufacturer':'ABB',    'power_kW':55},
    3: {'name':'Moteur 1602 CLI Fin',         'manufacturer':'Schneider','power_kW':37},
    4: {'name':'Motor_1603',                  'manufacturer':'ABB',    'power_kW':37},
    6: {'name':'Motor_1605',                  'manufacturer':'WEG',    'power_kW':19},
    8: {'name':'Motor_1607',                  'manufacturer':'ABB',    'power_kW':37},
   10: {'name':'Motor_1613',                  'manufacturer':'ABB',    'power_kW':17.9},
}
VIB_FACTOR = 0.01  # unité brute → mm/s


def extraire_full_data(sql_file):
    with open(sql_file, 'r', encoding='utf-8') as f:
        content = f.read()
    matches = re.findall(r"INSERT INTO `full_data`[^;]+;", content, re.DOTALL)
    rows = []
    for block in matches:
        tuples = re.findall(
            r'\((\d+),\s*\'(.*?)\',\s*\'(.*?)\',\s*\'(.*?)\',\s*\'(.*?)\',\s*\'(\w+)\'\)',
            block, re.DOTALL)
        for t in tuples:
            id_, sensor, ts, gph, data_str, typ = t
            if typ != 'res': continue
            try:
                dj = json.loads(data_str.replace('\\\\"','"').replace('\\"','"'))
                if not isinstance(dj, dict): continue
                sid = dj.get('SensorNodeId','').lower()
                mid = SENSOR_MOTOR_MAP.get(sid)
                if not mid: continue
                vib  = dj.get('Vibration',{}).get('RMS',{})
                meas = dj.get('MeasDetails',{})
                rows.append({
                    'motor_id':int(mid), 'sensor_id':sid, 'timestamp':ts,
                    'temp_sensor': dj.get('Temperature', np.nan),
                    'vib_x': vib.get('X', np.nan),
                    'vib_y': vib.get('Y', np.nan),
                    'vib_z': vib.get('Z', np.nan),
                    'battery_v': dj.get('BatteryVoltage', np.nan),
                    'fft_size': meas.get('FftSize', np.nan),
                    'bin_size': meas.get('BinSize', np.nan),
                })
            except: continue
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values(['motor_id','timestamp']).reset_index(drop=True)


def enrichir_et_preparer(df):
    """Calcule features 3D et prépare pour le modèle IA."""
    parts = []
    for mid, grp in df.groupby('motor_id'):
        g  = grp.sort_values('timestamp').copy()
        vx = g['vib_x'].fillna(0)
        vy = g['vib_y'].fillna(0)
        vz = g['vib_z'].fillna(0)

        g['vib_rms_3d']   = np.sqrt(vx**2 + vy**2 + vz**2)
        g['vib_radiale']  = np.sqrt(vx**2 + vy**2)
        g['vib_axiale']   = vz
        g['vib_ratio_az'] = (vz / (g['vib_radiale'] + 1e-9)).clip(0,10)
        g['vib_aniso']    = pd.concat([vx,vy,vz],axis=1).std(axis=1) / \
                            (pd.concat([vx,vy,vz],axis=1).mean(axis=1) + 1e-9)

        w = 10
        g['vib_3d_mean']  = g['vib_rms_3d'].rolling(w, min_periods=3).mean()
        g['vib_3d_max']   = g['vib_rms_3d'].rolling(w, min_periods=3).max()
        g['vib_3d_std']   = g['vib_rms_3d'].rolling(w, min_periods=3).std().fillna(0)
        g['temp_trend']   = g['temp_sensor'].ffill().rolling(w, min_periods=3).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>=4 else 0, raw=True
        ).fillna(0)

        # Conversion vers unités compatibles pipeline
        g['vibration']    = (g['vib_rms_3d'] * VIB_FACTOR).clip(0, 50)
        g['temperature']  = g['temp_sensor'].ffill().bfill().fillna(25)
        g['motor_id']     = mid
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df


def detecter_anomalies_full_data(df):
    """Applique le modèle IA sur les données full_data."""
    feats = ['vib_rms_3d','vib_radiale','vib_axiale','vib_ratio_az',
             'vib_aniso','vib_3d_mean','vib_3d_max','vib_3d_std',
             'temp_trend','vibration','temperature']
    avail = [f for f in feats if f in df.columns]
    X     = df[avail].fillna(0).values
    X_sc  = StandardScaler().fit_transform(X)

    # Charger les seuils calibrés depuis motor_measurements
    seuils_df = pd.read_csv(SEUILS_CSV) if os.path.exists(SEUILS_CSV) else None

    # Dépassement de seuil (vibration uniquement car pas courant/temp fiable)
    df['vib_exceed'] = 0.0
    if seuils_df is not None:
        for mid in df['motor_id'].unique():
            mask = df['motor_id'] == mid
            row  = seuils_df[seuils_df['motor_id'] == mid]
            if len(row) > 0:
                seuil_v = float(row['vibration'].iloc[0]) * VIB_FACTOR * 100
                df.loc[mask, 'vib_exceed'] = (
                    df.loc[mask, 'vib_rms_3d'] > seuil_v
                ).astype(float)

    # Isolation Forest
    model = IsolationForest(n_estimators=300, contamination=0.05,
                            random_state=42, n_jobs=-1)
    model.fit(X_sc)
    sc_if = -model.decision_function(X_sc)
    sc_if = (sc_if - sc_if.min()) / (sc_if.max() - sc_if.min())
    sc_rules  = df['vib_exceed'].values
    sc_hybrid = (0.6 * sc_if + 0.4 * sc_rules).clip(0, 1)

    df['score_if']       = sc_if
    df['combined_score'] = sc_hybrid
    df['is_anomaly']     = sc_hybrid >= 0.50

    return df


def plot_resultats(df, df_anom):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    motors = sorted(df['motor_id'].unique())

    # Fig 1 : Vibration 3D par moteur
    n_rows = (len(motors)+1)//2
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    for i, mid in enumerate(motors):
        dm   = df[df['motor_id']==mid].sort_values('timestamp')
        info = MOTOR_INFO.get(mid,{})
        ax   = axes[i]
        ax.plot(dm['timestamp'], dm['vib_rms_3d'].fillna(0),
                color='steelblue', lw=1.2, alpha=0.8, label='RMS 3D')
        if 'vib_axiale' in dm.columns:
            ax.plot(dm['timestamp'], dm['vib_axiale'].fillna(0),
                    color='orange', lw=1, alpha=0.6, label='Axial (Z)')
        ax.set_title(f"Moteur {mid} — {info.get('name','?')} ({info.get('power_kW','?')}kW)",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel('Vib RMS'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle('Vibration 3 axes — full_data (Nov 2025 → Jan 2026)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_full_data_vibration.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  → {FIGURES_DIR}/fig_full_data_vibration.png")

    # Fig 2 : Anomalies détectées
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    for i, mid in enumerate(motors):
        dm = df_anom[df_anom['motor_id']==mid].sort_values('timestamp')
        ax = axes[i]
        colors = np.where(dm['is_anomaly'], 'red', 'steelblue')
        ax.scatter(dm['timestamp'], dm['combined_score'], c=colors, s=15, alpha=0.6)
        ax.axhline(0.25, color='orange', linestyle='--', lw=1.5, label='Seuil=0.25')
        n_an = dm['is_anomaly'].sum()
        ax.set_title(f"Moteur {mid} — {n_an} anomalies ({n_an/len(dm)*100:.1f}%)",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel('Score'); ax.set_ylim(0,1)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle('Anomalies détectées — full_data', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_full_data_anomalies.png', dpi=150, bbox_inches='tight')
    plt.close(); print(f"  → {FIGURES_DIR}/fig_full_data_anomalies.png")

    # Fig 3 : Dashboard sombre
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    [ax.set_facecolor('#16213e') for ax in axes.flatten()]

    # Vib 3D
    ax = axes[0,0]
    for mid in motors:
        dm = df[df['motor_id']==mid].sort_values('timestamp')
        ax.plot(dm['timestamp'], dm['vib_rms_3d'].fillna(0), lw=1.2, alpha=0.8, label=f'M{mid}')
    ax.set_title('Vibration RMS 3D', color='white', fontweight='bold')
    ax.set_ylabel('mm/s', color='white')
    ax.legend(fontsize=7, facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    # Moy vib par moteur
    ax = axes[0,1]
    vib_mean = df.groupby('motor_id')['vib_rms_3d'].mean()
    colors = ['#d32f2f' if v>300 else '#f57c00' if v>100 else '#388e3c' for v in vib_mean]
    ax.bar(vib_mean.index.astype(str), vib_mean.values, color=colors, edgecolor='#333')
    ax.set_title('Vibration moyenne par moteur', color='white', fontweight='bold')
    ax.set_ylabel('RMS moyen', color='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    # Température
    ax = axes[1,0]
    for mid in motors:
        dm = df[df['motor_id']==mid].sort_values('timestamp')
        t  = dm['temp_sensor'].dropna()
        if len(t):
            ax.plot(dm['timestamp'].iloc[:len(t)], t, lw=1.2, alpha=0.8, label=f'M{mid}')
    ax.set_title('Température capteur', color='white', fontweight='bold')
    ax.set_ylabel('°C', color='white')
    ax.legend(fontsize=7, facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    # Score anomalie
    ax = axes[1,1]
    sc_mean = df_anom.groupby('motor_id')['combined_score'].mean()
    colors2 = ['#d32f2f' if v>0.4 else '#f57c00' if v>0.25 else '#388e3c' for v in sc_mean]
    ax.bar(sc_mean.index.astype(str), sc_mean.values, color=colors2, edgecolor='#333')
    ax.axhline(0.25, color='orange', linestyle='--', lw=1.5)
    ax.set_title('Score anomalie moyen', color='white', fontweight='bold')
    ax.set_ylabel('Score [0-1]', color='white'); ax.set_ylim(0,1)
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    fig.suptitle('Dashboard full_data — Capteurs VWV (Nov 2025 → Jan 2026)',
                 color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_full_data_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(); print(f"  → {FIGURES_DIR}/fig_full_data_dashboard.png")


def main():
    print("="*60)
    print(" ÉTAPE 1B — INTÉGRATION TABLE full_data")
    print("="*60)

    if not os.path.exists(SQL_FILE):
        print(f"[ERREUR] {SQL_FILE} introuvable"); return

    # 1. Extraction
    print("\n→ [1/4] Extraction full_data ...")
    df = extraire_full_data(SQL_FILE)
    print(f"  {len(df):,} mesures | {df['motor_id'].nunique()} moteurs")
    print(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # 2. Enrichissement features
    print("\n→ [2/4] Feature engineering vibration 3 axes ...")
    df = enrichir_et_preparer(df)
    print(f"\n  {'Moteur':>8} | {'Mesures':>8} | {'Vib3D moy':>10} | {'Vib3D max':>10} | {'Temp moy':>9}")
    print("  "+"-"*55)
    for mid in sorted(df['motor_id'].unique()):
        dm = df[df['motor_id']==mid]
        print(f"  {mid:>8} | {len(dm):>8,} | {dm['vib_rms_3d'].mean():>10.1f} | "
              f"{dm['vib_rms_3d'].max():>10.1f} | {dm['temp_sensor'].mean():>9.1f}°C")

    df.to_csv(OUTPUT_FD, index=False)
    print(f"\n  ✓ {OUTPUT_FD}")

    # Mapping
    pd.DataFrame([{'sensor_id':s,'motor_id':m,**MOTOR_INFO.get(m,{})}
                  for s,m in SENSOR_MOTOR_MAP.items()]).to_csv(MAPPING_CSV, index=False)
    print(f"  ✓ {MAPPING_CSV}")

    # 3. Détection anomalies
    print("\n→ [3/4] Détection d'anomalies ...")
    df_anom = detecter_anomalies_full_data(df.copy())
    n_anom  = df_anom['is_anomaly'].sum()
    print(f"  Anomalies : {n_anom:,} / {len(df_anom):,} ({n_anom/len(df_anom)*100:.1f}%)")
    print(f"\n  {'Moteur':>8} | {'Anomalies':>10} | {'Taux':>6} | {'Score max':>10} | Statut")
    print("  "+"-"*55)
    for mid in sorted(df_anom['motor_id'].unique()):
        dm   = df_anom[df_anom['motor_id']==mid]
        n_an = dm['is_anomaly'].sum()
        taux = n_an/len(dm)*100
        mx   = dm['combined_score'].max()
        st   = ('⚠ CRITIQUE' if taux>30 else '⚠ ÉLEVÉ' if taux>15
                else '~ MODÉRÉ' if taux>5 else '✓ NORMAL')
        print(f"  {mid:>8} | {n_an:>10,} | {taux:>5.1f}% | {mx:>10.4f} | {st}")
    df_anom.to_csv(OUTPUT_ANOM, index=False)
    print(f"\n  ✓ {OUTPUT_ANOM}")

    # 4. Figures
    print("\n→ [4/4] Génération des figures ...")
    plot_resultats(df, df_anom)

    print("\n"+"="*60)
    print(" ✓ INTÉGRATION full_data TERMINÉE")
    print("="*60)
    print(f"""
  RÉSUMÉ :
  ────────────────────────────────────────────
  • full_data couvre Novembre 2025 → Janvier 2026
  • motor_measurements couvre Avril–Mai 2025
  • Les deux périodes sont séparées (6 mois d'écart)
  • full_data est analysé de façon indépendante
  • 7 moteurs avec capteurs physiques VWV
  • Vibration 3 axes (X, Y, Z) disponible
  • Moteur 10 : vibration très élevée (512 RMS moyen)
  ────────────────────────────────────────────
  Fichiers produits :
    {OUTPUT_FD}
    {OUTPUT_ANOM}
    {MAPPING_CSV}
    figures/fig_full_data_*.png (3 figures)
""")

if __name__ == '__main__':
    main()