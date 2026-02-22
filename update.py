"""
╔══════════════════════════════════════════════════════════════╗
║         MISE À JOUR INCRÉMENTALE INTELLIGENTE                ║
║         Maintenance Prédictive — update.py                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Comment ça marche :                                         ║
║  1. Tu reçois un nouveau fichier SQL de ton collègue        ║
║  2. Tu lances : python update.py nom_fichier.sql            ║
║  3. Le script détecte UNIQUEMENT les nouvelles lignes        ║
║  4. Il met à jour tous les CSV + modèles + figures           ║
║  5. En quelques secondes, tout est à jour !                  ║
║                                                              ║
║  Usage :                                                     ║
║    python update.py ai_cp_juin.sql                          ║
║    python update.py ai_cp_juin.sql --force                  ║
║    python update.py --status                                 ║
║    python update.py --history                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, re, sys, json, time, shutil, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import kurtosis
from scipy.signal import hilbert
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
DATA_DIR          = 'data'
FIGURES_DIR       = 'figures'
HISTORY_DIR       = 'data/historique'
STATE_FILE        = 'data/update_state.json'
RAW_CSV           = 'data/01_raw_motor.csv'
FEATURES_CSV      = 'data/02_features_motor.csv'
ANOMALIES_CSV     = 'data/03_anomalies.csv'
RUL_CSV           = 'data/04_rul_results.csv'
RUL_SUMMARY_CSV   = 'data/rul_summary.csv'

WINDOW            = 20
CONTAMINATION     = 0.10
N_ESTIMATORS      = 300
LOF_NEIGHBORS     = 20
RANDOM_STATE      = 42
RETRAIN_THRESHOLD = 500    # réentraîner si > 500 nouvelles lignes
DI_WARNING        = 0.50
DI_CRITICAL       = 0.75
PREDICT_DAYS      = 90

FEATURE_COLS = [
    'temperature','courant','vibration','acceleration','vitesse','cosphi',
    'vib_energy','vib_energy_mean','vib_mean','vib_std','vib_kurt',
    'vib_max','crest_factor','temp_mean','temp_std','temp_trend',
    'courant_mean','courant_std','envelope','envelope_mean',
    'fft_max_amp','fft_dominant_freq','health_score'
]

RISK_PRIORITY = {'CRITIQUE': 0, 'ÉLEVÉ': 1, 'MODÉRÉ': 2, 'FAIBLE': 3}
RISK_COLOR    = {
    'CRITIQUE': '#d32f2f', 'ÉLEVÉ': '#f57c00',
    'MODÉRÉ':   '#fbc02d', 'FAIBLE': '#388e3c', 'inconnu': '#9e9e9e'
}

# ── Couleurs console ─────────────────────────────────────────
class C:
    GREEN  = '\033[92m'; YELLOW = '\033[93m'; RED  = '\033[91m'
    CYAN   = '\033[96m'; BOLD   = '\033[1m';  RESET = '\033[0m'

def log(msg, level='info'):
    now = datetime.now().strftime('%H:%M:%S')
    icons = {'info':'→','ok':'✓','warn':'⚠','error':'✗','title':'◆'}
    colors = {'info':C.CYAN,'ok':C.GREEN,'warn':C.YELLOW,'error':C.RED,'title':C.BOLD}
    ic = icons.get(level,'→'); co = colors.get(level,'')
    print(f"{co}[{now}] {ic}{C.RESET} {msg}")


# ══════════════════════════════════════════════════════════════
#  GESTION DE L'ÉTAT
# ══════════════════════════════════════════════════════════════
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE,'r',encoding='utf-8') as f:
            return json.load(f)
    return {
        'last_update': None, 'last_sql_file': None,
        'last_measurement_id': 0,
        'last_timestamp': '2000-01-01 00:00:00',
        'total_rows': 0, 'update_count': 0, 'history': []
    }

def save_state(state: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STATE_FILE,'w',encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
#  EXTRACTION SQL
# ══════════════════════════════════════════════════════════════
def parse_alert_file(s: str) -> dict:
    result = {'alert_parameter': None, 'alert_code': None}
    try:
        data = json.loads(s.replace('\\"','"'))
        if data:
            result['alert_parameter'] = data[0].get('parameter')
            result['alert_code'] = data[0].get('threshold',{}).get('code_alert')
    except Exception:
        pass
    return result

def extract_from_sql(sql_file: str) -> pd.DataFrame:
    log(f"Lecture de {sql_file} ...")
    with open(sql_file,'r',encoding='utf-8') as f:
        content = f.read()
    matches = re.findall(
        r"INSERT INTO `motor_measurements`.*?VALUES\s*(.*?);",
        content, re.DOTALL
    )
    log(f"{len(matches)} blocs INSERT trouvés")

    rows = []
    for block in matches:
        for t in re.findall(r'\((\d+,\s*\d+,\s*\x27[^)]+)\)', block):
            parts = [p.strip().strip("'") for p in t.split(',', 12)]
            if len(parts) < 12: continue
            ai = parse_alert_file(parts[12] if len(parts) > 12 else '[]')
            try:
                rows.append({
                    'measurement_id': int(parts[0]),
                    'motor_id'      : int(parts[1]),
                    'timestamp'     : parts[2],
                    'temperature'   : float(parts[3])  if parts[3]  not in ('NULL','') else np.nan,
                    'courant'       : float(parts[4])  if parts[4]  not in ('NULL','') else np.nan,
                    'vibration'     : float(parts[5])  if parts[5]  not in ('NULL','') else np.nan,
                    'acceleration'  : float(parts[6])  if parts[6]  not in ('NULL','') else np.nan,
                    'thdi'          : float(parts[7])  if parts[7]  not in ('NULL','') else np.nan,
                    'thdu'          : float(parts[8])  if parts[8]  not in ('NULL','') else np.nan,
                    'vitesse'       : float(parts[9])  if parts[9]  not in ('NULL','') else np.nan,
                    'cosphi'        : float(parts[10]) if parts[10] not in ('NULL','') else np.nan,
                    'Alert_Status'  : parts[11],
                    'alert_parameter': ai['alert_parameter'],
                    'alert_code'    : ai['alert_code'],
                })
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values(['motor_id','timestamp']).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
#  DÉTECTION NOUVELLES LIGNES
# ══════════════════════════════════════════════════════════════
def detect_new_rows(df_sql: pd.DataFrame, state: dict, force: bool) -> pd.DataFrame:
    if force:
        log("Mode FORCE : retraitement complet", 'warn')
        return df_sql
    last_id = state.get('last_measurement_id', 0)
    last_ts = pd.to_datetime(state.get('last_timestamp', '2000-01-01'))
    mask = (df_sql['measurement_id'] > last_id) | (df_sql['timestamp'] > last_ts)
    return df_sql[mask].copy()


# ══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def compute_features_group(g: pd.DataFrame, window: int) -> pd.DataFrame:
    g   = g.sort_values('timestamp').copy()
    vib = g['vibration'].fillna(0)
    tmp = g['temperature'].fillna(g['temperature'].median() if not g['temperature'].isna().all() else 35)
    cur = g['courant'].fillna(g['courant'].median() if not g['courant'].isna().all() else 50)

    rv = vib.rolling(window=window, min_periods=3)
    rt = tmp.rolling(window=window, min_periods=3)
    rc = cur.rolling(window=window, min_periods=3)

    g['vib_mean']        = rv.mean()
    g['vib_std']         = rv.std().fillna(0)
    g['vib_max']         = rv.max()
    g['vib_energy']      = vib ** 2
    g['vib_energy_mean'] = g['vib_energy'].rolling(window=window, min_periods=3).mean()
    g['vib_kurt']        = rv.apply(
        lambda x: kurtosis(x, nan_policy='omit') if len(x) >= 4 else 0.0, raw=True
    ).fillna(0)

    rms_roll = np.sqrt((vib**2).rolling(window=window, min_periods=3).mean())
    g['crest_factor'] = (g['vib_max'] / (rms_roll + 1e-9)).fillna(1.0)

    g['temp_mean']  = rt.mean()
    g['temp_std']   = rt.std().fillna(0)
    g['temp_trend'] = rt.apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 4 else 0.0, raw=True
    ).fillna(0)

    g['courant_mean'] = rc.mean()
    g['courant_std']  = rc.std().fillna(0)

    try:
        env = np.abs(hilbert(vib.values))
        g['envelope']      = env
        g['envelope_mean'] = pd.Series(env).rolling(window=window, min_periods=3).mean().values
    except Exception:
        g['envelope'] = g['envelope_mean'] = 0.0

    fft_maxs, fft_doms = [], []
    vib_vals = vib.values
    for i in range(len(vib_vals)):
        seg = vib_vals[max(0, i-window+1):i+1]
        if len(seg) >= 4:
            spec = np.abs(np.fft.rfft(seg - seg.mean()))
            fft_maxs.append(float(spec.max()))
            fft_doms.append(float(np.argmax(spec)))
        else:
            fft_maxs.append(0.0); fft_doms.append(0.0)
    g['fft_max_amp']       = fft_maxs
    g['fft_dominant_freq'] = fft_doms

    def norm_inv(s):
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        if hi == lo: return pd.Series(50.0, index=s.index)
        return (1 - ((s-lo)/(hi-lo)).clip(0,1)) * 100

    g['health_score'] = (
        0.35 * norm_inv(g['vib_energy_mean'].fillna(0)) +
        0.25 * norm_inv(g['temp_mean'].fillna(g['temperature'])) +
        0.20 * norm_inv(g['vib_kurt'].abs()) +
        0.20 * norm_inv(g['crest_factor'])
    ).clip(0, 100)

    return g.fillna(0)


def add_features_incremental(df_new: pd.DataFrame,
                              df_existing: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for mid in df_new['motor_id'].unique():
        # Contexte historique pour les rolling windows
        hist_cols = [c for c in df_existing.columns if c in df_new.columns]
        ctx  = df_existing[df_existing['motor_id'] == mid][hist_cols].tail(WINDOW * 2)
        combined = pd.concat([ctx, df_new[df_new['motor_id'] == mid]], ignore_index=True)
        combined = compute_features_group(combined, WINDOW)
        combined['motor_id'] = mid
        # Garder uniquement les nouvelles lignes
        new_ids = df_new[df_new['motor_id'] == mid]['measurement_id'].values
        parts.append(combined[combined['measurement_id'].isin(new_ids)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
#  ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════
def update_anomaly_scores(df_full: pd.DataFrame, n_new: int) -> pd.DataFrame:
    avail   = [c for c in FEATURE_COLS if c in df_full.columns]
    X       = df_full[avail].fillna(0).values
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    if n_new >= RETRAIN_THRESHOLD:
        log(f"Réentraînement complet ({n_new} nouvelles lignes)", 'warn')
    else:
        log(f"Mise à jour modèle ({n_new} nouvelles lignes)")

    normal  = df_full['Alert_Status'] == 'NORMAL'
    X_train = X_sc[normal]

    if_model = IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=CONTAMINATION,
        max_samples=min(256, len(X_train)), random_state=RANDOM_STATE, n_jobs=-1
    )
    if_model.fit(X_train)
    sc_if_raw = if_model.decision_function(X_sc)

    lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, contamination=CONTAMINATION,
                             n_jobs=-1, novelty=False)
    lof.fit_predict(X_sc)
    sc_lof_raw = lof.negative_outlier_factor_

    def norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    sc_if   = norm(-sc_if_raw)
    sc_lof  = norm(-sc_lof_raw)
    sc_comb = 0.6 * sc_if + 0.4 * sc_lof
    thresh  = np.percentile(sc_comb, (1 - CONTAMINATION) * 100)

    df_out = df_full.copy()
    df_out['score_if']          = sc_if
    df_out['score_lof']         = sc_lof
    df_out['combined_score']    = sc_comb
    df_out['anomaly_threshold'] = thresh
    df_out['is_anomaly']        = sc_comb >= thresh
    return df_out


# ══════════════════════════════════════════════════════════════
#  RUL
# ══════════════════════════════════════════════════════════════
def update_rul(df: pd.DataFrame) -> tuple:
    def compute_di(grp):
        g = grp.sort_values('timestamp').copy()
        def norm_ref(s, w=30):
            s   = s.fillna(s.median())
            ref = s.iloc[:max(5,w)].mean()
            return ((s.rolling(w, min_periods=3).mean() - ref) / (ref + 1e-9)).clip(0, None)
        raw = (0.35 * norm_ref(g.get('vib_energy_mean', g['vib_energy']), 30) +
               0.20 * norm_ref(g['vib_kurt'].abs(), 20) +
               0.25 * norm_ref(g.get('temp_mean', g['temperature']), 30) +
               0.20 * g['combined_score'].fillna(0))
        lo, hi = raw.min(), raw.max()
        g['degradation_index'] = ((raw-lo)/(hi-lo+1e-9)).clip(0,1) if hi > lo else 0.0
        return g

    parts = []
    for mid, grp in df.groupby('motor_id'):
        e = compute_di(grp.copy())
        e['motor_id'] = mid
        parts.append(e)
    df = pd.concat(parts, ignore_index=True)

    results = []
    for mid, grp in df.groupby('motor_id'):
        g = grp.sort_values('timestamp').copy()
        g['days'] = (g['timestamp'] - g['timestamp'].min()).dt.total_seconds() / 86400
        x, y = g['days'].values, g['degradation_index'].values
        current_di = float(y[-1]) if len(y) else 0
        rul_days = '>90'; trend = 0.0
        if len(x) >= 5:
            try:
                coef = np.polyfit(x, y, deg=min(2, len(x)-1))
                fn   = np.poly1d(coef)
                trend = float(fn(x[-1]+1) - fn(x[-1]))
                fut   = np.linspace(x[-1], x[-1]+PREDICT_DAYS, 500)
                exc   = np.where(fn(fut).clip(0,1.5) >= DI_CRITICAL)[0]
                if len(exc): rul_days = round(float(fut[exc[0]] - x[-1]), 1)
            except Exception: pass
        risk = ('CRITIQUE' if current_di >= DI_CRITICAL or
                              (isinstance(rul_days, float) and rul_days < 7)
                else 'ÉLEVÉ'  if current_di >= DI_WARNING or
                              (isinstance(rul_days, float) and rul_days < 21)
                else 'MODÉRÉ' if current_di >= 0.30
                else 'FAIBLE')
        results.append({
            'motor_id': int(mid), 'rul_days': rul_days,
            'current_di': round(current_di, 4), 'trend_slope': round(trend, 6),
            'risk_level': risk, 'last_timestamp': str(g['timestamp'].max())
        })
    return df, pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════
def update_figures(df: pd.DataFrame, df_rul: pd.DataFrame):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    now = datetime.now().strftime('%d/%m/%Y %H:%M')

    df_sorted = df_rul.copy()
    df_sorted['_p'] = df_sorted['risk_level'].map(RISK_PRIORITY).fillna(4)
    df_sorted = df_sorted.sort_values(['_p','current_di'], ascending=[True,False])

    # ── Dashboard risque ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')

    ax = axes[0]
    ax.set_facecolor('#16213e')
    colors = [RISK_COLOR.get(r,'#9e9e9e') for r in df_sorted['risk_level']]
    ax.barh(df_sorted['motor_id'].astype(str), df_sorted['current_di'],
            color=colors, edgecolor='#333')
    ax.axvline(DI_WARNING,  color='orange', linestyle='--', lw=1.5, alpha=0.8)
    ax.axvline(DI_CRITICAL, color='red',    linestyle='--', lw=2,   alpha=0.8)
    ax.set_title(f'Indice de dégradation — MAJ {now}', color='white', fontweight='bold')
    ax.set_xlabel('DI [0–1]', color='white'); ax.set_xlim(0,1)
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    ax = axes[1]
    ax.set_facecolor('#16213e')
    rc = df_sorted['risk_level'].value_counts()
    ro = [r for r in ['CRITIQUE','ÉLEVÉ','MODÉRÉ','FAIBLE'] if r in rc.index]
    _, _, ats = ax.pie([rc[r] for r in ro], labels=ro,
                       colors=[RISK_COLOR[r] for r in ro],
                       autopct='%1.0f%%', startangle=90,
                       textprops={'color':'white'})
    for at in ats: at.set_fontsize(12); at.set_fontweight('bold')
    ax.set_title('Distribution risques', color='white', fontweight='bold')
    fig.suptitle(f'Dashboard Maintenance Prédictive — {now}',
                 color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig5_risk_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    log("fig5_risk_dashboard.png mis à jour", 'ok')

    # ── Health score évolution ────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#16213e')
    if 'health_score' in df.columns:
        for mid in sorted(df['motor_id'].unique()):
            dm = df[df['motor_id']==mid].sort_values('timestamp')
            ax.plot(dm['timestamp'], dm['health_score'], lw=1.2, alpha=0.7, label=f'M{mid}')
    ax.axhline(50, color='orange', linestyle='--', lw=1.5, alpha=0.8, label='Prudence 50%')
    ax.axhline(30, color='red',    linestyle='--', lw=1.5, alpha=0.8, label='Critique 30%')
    ax.set_title(f'Health Score — Tous moteurs (MAJ {now})', color='white', fontweight='bold')
    ax.set_ylabel('Health Score [0–100]', color='white'); ax.set_ylim(0,100)
    ax.legend(fontsize=7, ncol=5, loc='lower left',
              facecolor='#16213e', labelcolor='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig_health_evolution.png',
                dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    log("fig_health_evolution.png généré", 'ok')


# ══════════════════════════════════════════════════════════════
#  RAPPORT MISE À JOUR
# ══════════════════════════════════════════════════════════════
def generate_update_report(n_new, n_total, df_rul, sql_file, duration) -> str:
    now  = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    sep  = "=" * 60
    lines = [sep,
             f"  RAPPORT MISE À JOUR — {now}",
             f"  Fichier SQL     : {sql_file}",
             f"  Nouvelles lignes: {n_new:,}",
             f"  Total historique: {n_total:,}",
             f"  Durée           : {duration:.1f}s",
             "", "  ÉTAT DES MOTEURS :", "-"*40]
    df_s = df_rul.sort_values('risk_level',
                              key=lambda x: x.map(RISK_PRIORITY).fillna(4))
    for _, r in df_s.iterrows():
        flag = {'CRITIQUE':'🔴','ÉLEVÉ':'🟠','MODÉRÉ':'🟡','FAIBLE':'🟢'}.get(r['risk_level'],'⚪')
        rul  = f"{r['rul_days']}j" if str(r['rul_days']) != '>90' else '>90j'
        lines.append(f"  {flag} Moteur {int(r['motor_id']):2d} | "
                     f"DI={r['current_di']:.3f} | RUL={rul:>6} | {r['risk_level']}")
    lines.append(sep)
    report = "\n".join(lines)
    with open('data/update_log.txt','a',encoding='utf-8') as f:
        f.write(report + "\n\n")
    return report


# ══════════════════════════════════════════════════════════════
#  COMMANDES UTILITAIRES
# ══════════════════════════════════════════════════════════════
def show_status():
    state = load_state()
    print(f"\n{C.CYAN}{C.BOLD}{'═'*55}\n  ÉTAT DU SYSTÈME\n{'═'*55}{C.RESET}")
    if not state['last_update']:
        print(f"{C.YELLOW}  Aucune mise à jour. Lancez : python main_pipeline.py{C.RESET}\n")
        return
    print(f"  Dernière MAJ     : {state['last_update']}")
    print(f"  Fichier SQL      : {state['last_sql_file']}")
    print(f"  Total mesures    : {state['total_rows']:,}")
    print(f"  Nombre de MAJ    : {state['update_count']}")
    if os.path.exists(RUL_SUMMARY_CSV):
        df_rul = pd.read_csv(RUL_SUMMARY_CSV)
        print("\n  État moteurs :")
        for _, r in df_rul.sort_values('risk_level',
            key=lambda x: x.map(RISK_PRIORITY).fillna(4)).iterrows():
            col = (C.RED if r['risk_level']=='CRITIQUE' else
                   C.YELLOW if r['risk_level'] in ('ÉLEVÉ','MODÉRÉ') else C.GREEN)
            print(f"  {col}Moteur {int(r['motor_id']):2d} | "
                  f"DI={r['current_di']:.3f} | {r['risk_level']}{C.RESET}")
    print()

def show_history():
    state = load_state()
    print(f"\n{C.CYAN}{C.BOLD}{'═'*55}\n  HISTORIQUE DES MISES À JOUR\n{'═'*55}{C.RESET}")
    history = state.get('history', [])
    if not history:
        print(f"{C.YELLOW}  Aucun historique.{C.RESET}\n"); return
    for i, h in enumerate(history, 1):
        print(f"  {i}. {h.get('date','?')} | {h.get('sql_file','?')} | "
              f"+{h.get('new_rows',0):,} lignes | Total: {h.get('total_rows',0):,}")
    print()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Mise à jour incrémentale')
    parser.add_argument('sql_file', nargs='?', default=None, help='Nouveau fichier SQL')
    parser.add_argument('--force',   action='store_true', help='Retraitement complet')
    parser.add_argument('--status',  action='store_true', help='État actuel')
    parser.add_argument('--history', action='store_true', help='Historique des MAJ')
    args = parser.parse_args()

    if args.status:  show_status();  return
    if args.history: show_history(); return

    if not args.sql_file:
        print(f"{C.RED}Usage : python update.py <fichier.sql> [--force]{C.RESET}")
        print("        python update.py --status")
        print("        python update.py --history")
        return

    if not os.path.exists(args.sql_file):
        log(f"Fichier introuvable : {args.sql_file}", 'error'); return
    if not os.path.exists(RAW_CSV):
        log("Données de base absentes. Lancez d'abord : python main_pipeline.py", 'error'); return

    t0 = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"\n{C.CYAN}{C.BOLD}")
    print("╔══════════════════════════════════════════════════════╗")
    print("║          MISE À JOUR INCRÉMENTALE INTELLIGENTE       ║")
    print(f"║  Fichier : {args.sql_file:<43}║")
    print(f"╚══════════════════════════════════════════════════════╝{C.RESET}\n")

    state = load_state()

    # ── 1. Extraction ────────────────────────────
    log("ÉTAPE 1 — Extraction SQL", 'title')
    df_sql = extract_from_sql(args.sql_file)
    log(f"{len(df_sql):,} lignes extraites", 'ok')

    # ── 2. Détection nouvelles lignes ────────────
    log("ÉTAPE 2 — Détection nouvelles lignes", 'title')
    df_new = detect_new_rows(df_sql, state, args.force)
    n_new  = len(df_new)

    if n_new == 0:
        log("Aucune nouvelle ligne — données déjà à jour !", 'warn')
        log(f"Dernière MAJ : {state.get('last_update','jamais')}")
        log("Utilisez --force pour forcer le retraitement.", 'warn')
        return

    log(f"{n_new:,} nouvelles lignes détectées !", 'ok')
    log(f"Période : {df_new['timestamp'].min()} → {df_new['timestamp'].max()}")
    log(f"Moteurs concernés : {sorted(df_new['motor_id'].unique())}")

    # ── 3. Sauvegarde historique ─────────────────
    log("ÉTAPE 3 — Sauvegarde historique", 'title')
    os.makedirs(HISTORY_DIR, exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    bdir = os.path.join(HISTORY_DIR, f"{ts}_{os.path.splitext(os.path.basename(args.sql_file))[0]}")
    os.makedirs(bdir, exist_ok=True)
    for f in [RAW_CSV, RUL_SUMMARY_CSV]:
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(bdir, os.path.basename(f)))
    log(f"Sauvegarde → {bdir}", 'ok')

    # ── 4. Fusion données brutes ─────────────────
    log("ÉTAPE 4 — Fusion avec historique", 'title')
    df_existing = pd.read_csv(RAW_CSV, parse_dates=['timestamp'])
    df_raw_full = (pd.concat([df_existing, df_new], ignore_index=True)
                   .drop_duplicates(subset=['motor_id','timestamp'])
                   .sort_values(['motor_id','timestamp'])
                   .reset_index(drop=True))
    n_total = len(df_raw_full)
    df_raw_full.to_csv(RAW_CSV, index=False)
    log(f"01_raw_motor.csv : {n_total:,} lignes totales", 'ok')

    # ── 5. Features ──────────────────────────────
    log("ÉTAPE 5 — Feature engineering", 'title')
    df_existing_feat = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'])
    df_new_feat = add_features_incremental(df_new, df_existing_feat)
    df_feat_full = (pd.concat([df_existing_feat, df_new_feat], ignore_index=True)
                    .drop_duplicates(subset=['motor_id','timestamp'])
                    .sort_values(['motor_id','timestamp'])
                    .reset_index(drop=True))
    df_feat_full.to_csv(FEATURES_CSV, index=False)
    log(f"02_features_motor.csv : {len(df_feat_full):,} lignes", 'ok')

    # ── 6. Anomalies ─────────────────────────────
    log("ÉTAPE 6 — Détection anomalies", 'title')
    df_anom = update_anomaly_scores(df_feat_full, n_new)
    n_anom  = df_anom['is_anomaly'].sum()
    log(f"Anomalies : {n_anom:,} / {len(df_anom):,} ({n_anom/len(df_anom)*100:.1f}%)", 'ok')
    df_anom.to_csv(ANOMALIES_CSV, index=False)
    log("03_anomalies.csv mis à jour", 'ok')

    # ── 7. RUL ───────────────────────────────────
    log("ÉTAPE 7 — Prédiction RUL", 'title')
    df_rul_full, df_rul_summary = update_rul(df_anom)
    df_rul_full.to_csv(RUL_CSV, index=False)
    df_rul_summary.to_csv(RUL_SUMMARY_CSV, index=False)
    log("04_rul_results.csv + rul_summary.csv mis à jour", 'ok')

    # ── 8. Figures ───────────────────────────────
    log("ÉTAPE 8 — Mise à jour figures", 'title')
    update_figures(df_rul_full, df_rul_summary)

    # ── 9. Rapport ───────────────────────────────
    duration = time.time() - t0
    log("ÉTAPE 9 — Rapport de mise à jour", 'title')
    report = generate_update_report(n_new, n_total, df_rul_summary, args.sql_file, duration)
    print(f"\n{report}")

    # ── 10. Sauvegarde état ──────────────────────
    state.update({
        'last_update'         : datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'last_sql_file'       : args.sql_file,
        'last_measurement_id' : int(df_raw_full['measurement_id'].max()),
        'last_timestamp'      : str(df_raw_full['timestamp'].max()),
        'total_rows'          : n_total,
        'update_count'        : state.get('update_count', 0) + 1,
    })
    state.setdefault('history', []).append({
        'date': state['last_update'], 'sql_file': args.sql_file,
        'new_rows': n_new, 'total_rows': n_total,
    })
    save_state(state)

    # ── Résumé final ─────────────────────────────
    print(f"\n{C.GREEN}{C.BOLD}{'═'*55}")
    print("  ✓ MISE À JOUR TERMINÉE AVEC SUCCÈS !")
    print(f"{'═'*55}{C.RESET}")
    print(f"  Durée            : {duration:.1f}s")
    print(f"  Nouvelles lignes : {n_new:,}")
    print(f"  Total historique : {n_total:,}")
    print(f"\n  Moteurs à surveiller :")
    df_warn = df_rul_summary[df_rul_summary['risk_level'].isin(['CRITIQUE','ÉLEVÉ','MODÉRÉ'])]
    df_warn = df_warn.sort_values('risk_level', key=lambda x: x.map(RISK_PRIORITY).fillna(4))
    for _, r in df_warn.iterrows():
        col = C.RED if r['risk_level']=='CRITIQUE' else C.YELLOW
        print(f"  {col}⚠ Moteur {int(r['motor_id']):2d} — "
              f"{r['risk_level']} (DI={r['current_di']:.3f}){C.RESET}")
    if df_warn.empty:
        print(f"  {C.GREEN}Tous les moteurs sont en état FAIBLE ✓{C.RESET}")
    print()

if __name__ == '__main__':
    main()
