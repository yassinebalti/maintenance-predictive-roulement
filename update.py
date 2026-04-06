"""
╔══════════════════════════════════════════════════════════════╗
║         MISE À JOUR INCRÉMENTALE V3                          ║
║         Maintenance Prédictive — update.py                   ║
╠══════════════════════════════════════════════════════════════╣
║  Modèle : Hybride IF+LOF+Règles V3 (7 axes améliorés)       ║
║  Nouvelles features : vib_rms, skewness, peak2peak,         ║
║                       spectral_entropy, shape/impulse_factor ║
║  RUL : Ensemble Poly+Exp+Weibull + IC                        ║
║  CUSUM : Détection rupture de tendance                       ║
║                                                              ║
║  Usage :                                                     ║
║    python update.py nouveau_fichier.sql                      ║
║    python update.py nouveau_fichier.sql --force              ║
║    python update.py --status                                 ║
║    python update.py --history                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, re, sys, json, time, shutil, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import kurtosis, skew
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
DATA_DIR        = 'data'
FIGURES_DIR     = 'figures'
HISTORY_DIR     = 'data/historique'
STATE_FILE      = 'data/update_state.json'
RAW_CSV         = 'data/01_raw_motor.csv'
FEATURES_CSV    = 'data/02_features_motor.csv'
ANOMALIES_CSV   = 'data/03_anomalies.csv'
RUL_CSV         = 'data/04_rul_results.csv'
RUL_SUMMARY_CSV = 'data/rul_summary.csv'
SEUILS_CSV      = 'data/seuils_moteurs.csv'

WINDOW            = 20
N_ESTIMATORS      = 500
RANDOM_STATE      = 42
RETRAIN_THRESHOLD = 500
DI_WARNING        = 0.50
DI_CRITICAL       = 0.75
PREDICT_DAYS      = 90

# V3 — Poids Ensemble
W_IF      = 0.25
W_LOF     = 0.20
W_RULES   = 0.55
THRESHOLD = 0.25

# CUSUM
CUSUM_K = 0.5
CUSUM_H = 4.0

PARAMS_ALERTES = ['temperature','courant','vibration','acceleration']
POIDS_PARAMS   = {'temperature':0.35,'courant':0.30,'vibration':0.25,'acceleration':0.10}

FEATURE_IF = [
    'temperature_exceed','courant_exceed','vibration_exceed','acceleration_exceed',
    'temperature_ratio','courant_ratio','vibration_ratio','acceleration_ratio',
    'n_exceed','severity_score',
    'vib_energy_mean','vib_kurt','crest_factor',
    'temp_mean','temp_trend','courant_mean','envelope_mean','health_score',
    # AXE 3 — nouvelles features
    'vib_rms','vib_skewness','peak2peak','spectral_entropy','shape_factor','impulse_factor',
]

RISK_PRIORITY = {'CRITIQUE':0,'ÉLEVÉ':1,'MODÉRÉ':2,'FAIBLE':3}
RISK_COLOR    = {'CRITIQUE':'#d32f2f','ÉLEVÉ':'#f57c00',
                 'MODÉRÉ':'#fbc02d','FAIBLE':'#388e3c'}


class C:
    GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'
    BOLD='\033[1m';   RESET='\033[0m';   CYAN='\033[96m'


# ══════════════════════════════════════════════════════════════
#  PARSING SQL
# ══════════════════════════════════════════════════════════════
def parse_sql(sql_file: str) -> pd.DataFrame:
    with open(sql_file, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r"INSERT INTO `motor_measurements`.*?VALUES\s*(.*?);"
    matches = re.findall(pattern, content, re.DOTALL)
    rows    = []
    for block in matches:
        for t in re.findall(r'\((\d+,\s*\d+,\s*\x27[^)]+)\)', block):
            parts = [p.strip().strip("'") for p in t.split(',', 12)]
            if len(parts) < 12:
                continue
            try:
                rows.append({
                    'measurement_id': int(parts[0]),
                    'motor_id'      : int(parts[1]),
                    'timestamp'     : pd.to_datetime(parts[2]),
                    'temperature'   : float(parts[3])  if parts[3]  not in ('NULL','') else np.nan,
                    'courant'       : float(parts[4])  if parts[4]  not in ('NULL','') else np.nan,
                    'vibration'     : float(parts[5])  if parts[5]  not in ('NULL','') else np.nan,
                    'acceleration'  : float(parts[6])  if parts[6]  not in ('NULL','') else np.nan,
                    'thdi'          : float(parts[7])  if parts[7]  not in ('NULL','') else np.nan,
                    'thdu'          : float(parts[8])  if parts[8]  not in ('NULL','') else np.nan,
                    'vitesse'       : float(parts[9])  if parts[9]  not in ('NULL','') else np.nan,
                    'cosphi'        : float(parts[10]) if parts[10] not in ('NULL','') else np.nan,
                    'Alert_Status'  : parts[11],
                    'alert_parameter': None,
                    'alert_code'    : None,
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  FEATURES V3 (identiques à step2)
# ══════════════════════════════════════════════════════════════
def spectral_entropy(signal):
    if len(signal) < 8: return 0.0
    spec  = np.abs(np.fft.rfft(signal - signal.mean()))**2
    total = spec.sum()
    if total < 1e-12: return 0.0
    p = spec / total; p = p[p>0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def compute_features(group, window=WINDOW):
    g   = group.sort_values('timestamp').copy()
    vib = g['vibration']
    tmp = g['temperature']
    cur = g['courant']

    rv = vib.rolling(window=window, min_periods=3)
    g['vib_mean']        = rv.mean()
    g['vib_std']         = rv.std().fillna(0)
    g['vib_max']         = rv.max()
    g['vib_energy']      = vib**2
    g['vib_energy_mean'] = g['vib_energy'].rolling(window=window, min_periods=3).mean()
    g['vib_kurt']        = rv.apply(lambda x: kurtosis(x, nan_policy='omit') if len(x)>=4 else 0.0, raw=True).fillna(0)
    rms_r                = np.sqrt((vib**2).rolling(window=window, min_periods=3).mean())
    g['crest_factor']    = (g['vib_max'] / (rms_r+1e-9)).fillna(1.0)

    rt = tmp.rolling(window=window, min_periods=3)
    g['temp_mean']  = rt.mean()
    g['temp_std']   = rt.std().fillna(0)
    g['temp_trend'] = rt.apply(lambda x: np.polyfit(np.arange(len(x)),x,1)[0] if len(x)>=4 else 0.0, raw=True).fillna(0)

    rc2 = cur.rolling(window=window, min_periods=3)
    g['courant_mean'] = rc2.mean()
    g['courant_std']  = rc2.std().fillna(0)

    try:
        envelope         = np.abs(hilbert(vib.fillna(0).values))
        g['envelope']      = envelope
        g['envelope_mean'] = pd.Series(envelope).rolling(window=window, min_periods=3).mean().values
    except Exception:
        g['envelope'] = g['envelope_mean'] = 0.0

    vib_v      = vib.fillna(0).values
    fft_maxs, fft_doms = [], []
    for i in range(len(vib_v)):
        s = max(0, i-window+1)
        seg = vib_v[s:i+1]
        if len(seg) >= 4:
            spec = np.abs(np.fft.rfft(seg-seg.mean()))
            fft_maxs.append(float(spec.max())); fft_doms.append(float(np.argmax(spec)))
        else:
            fft_maxs.append(0.0); fft_doms.append(0.0)
    g['fft_max_amp']       = fft_maxs
    g['fft_dominant_freq'] = fft_doms

    # AXE 3 — nouvelles features
    g['vib_rms']       = rms_r.fillna(0)
    g['vib_skewness']  = rv.apply(lambda x: float(skew(x)) if len(x)>=4 else 0.0, raw=True).fillna(0)
    g['peak2peak']     = rv.apply(lambda x: float(x.max()-x.min()) if len(x)>=3 else 0.0, raw=True).fillna(0)
    sp_ent = []
    for i in range(len(vib_v)):
        sp_ent.append(spectral_entropy(vib_v[max(0,i-window+1):i+1]))
    g['spectral_entropy'] = sp_ent
    ma_roll             = vib.abs().rolling(window=window, min_periods=3).mean()
    g['shape_factor']   = (g['vib_rms']  / (ma_roll+1e-9)).fillna(1.0)
    g['impulse_factor'] = (g['vib_max']  / (ma_roll+1e-9)).fillna(1.0)

    # Health score V3
    def ni(s): lo,hi=s.quantile(.05),s.quantile(.95); return (1-((s-lo)/(hi-lo+1e-9)).clip(0,1))*100 if hi>lo else pd.Series(50.,index=s.index)
    g['health_score'] = (0.28*ni(g['vib_energy_mean']) + 0.22*ni(g['temp_mean']) +
                         0.18*ni(g['vib_kurt'].abs())  + 0.15*ni(g['crest_factor']) +
                         0.10*ni(g['impulse_factor'])   + 0.07*ni(g['vib_rms'])).clip(0,100)
    return g


# ══════════════════════════════════════════════════════════════
#  SEUILS + FEATURES DÉPASSEMENT
# ══════════════════════════════════════════════════════════════
def calibrer_seuils(df):
    seuils = {}
    for mid in sorted(df['motor_id'].unique()):
        dm = df[df['motor_id']==mid]
        seuils[mid] = {}
        for col in PARAMS_ALERTES:
            av = dm[dm.get('alert_parameter','x')==col][col] if 'alert_parameter' in dm.columns else pd.Series()
            nv = dm[dm['Alert_Status']=='NORMAL'][col]
            seuils[mid][col] = float(av.min()) if len(av)>0 else (float(nv.quantile(0.95)) if len(nv)>0 else float(dm[col].quantile(0.95)))
    return seuils


def features_depassement(df, seuils):
    df = df.copy()
    sev = pd.Series(0., index=df.index)
    for col in PARAMS_ALERTES:
        df[f'{col}_exceed'] = 0.; df[f'{col}_ratio'] = 0.
        for mid in df['motor_id'].unique():
            mask  = df['motor_id']==mid
            s_val = seuils.get(mid,{}).get(col, df[col].quantile(0.95))
            df.loc[mask, f'{col}_exceed'] = (df.loc[mask,col]>s_val).astype(float)
            df.loc[mask, f'{col}_ratio']  = df.loc[mask,col]/(s_val+1e-9)
        sev += df[f'{col}_exceed']*POIDS_PARAMS[col]
    df['n_exceed']       = df[[f'{c}_exceed' for c in PARAMS_ALERTES]].sum(axis=1)
    df['severity_score'] = sev.clip(0,1)
    return df


def features_flotte(df):
    df = df.copy()
    for col, rc, zc in [('temperature','temp_ratio_flotte','temp_zscore_flotte'),
                         ('vibration','vib_ratio_flotte','vib_zscore_flotte'),
                         ('courant','courant_ratio_flotte','courant_zscore_flotte')]:
        med = df.groupby('timestamp')[col].transform('median')
        std = df.groupby('timestamp')[col].transform('std').fillna(1)
        df[rc] = df[col]/(med+1e-9)
        df[zc] = (df[col]-med)/(std+1e-9)
    df['fleet_anomaly'] = ((df['temp_zscore_flotte']>2.5)|(df['vib_zscore_flotte']>2.5)).astype(int)
    return df


# ══════════════════════════════════════════════════════════════
#  DÉTECTION ANOMALIES V3 (IF + LOF + Règles)
# ══════════════════════════════════════════════════════════════
def detect_anomalies(df, seuils, retrain=True):
    df = features_depassement(df, seuils)
    df = features_flotte(df)

    feats_ok = [c for c in FEATURE_IF if c in df.columns]
    X        = df[feats_ok].fillna(0).values
    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)

    normal_mask = (df['Alert_Status']=='NORMAL').values
    X_normal    = X_sc[normal_mask] if normal_mask.sum()>50 else X_sc

    # IF
    clf_if  = IsolationForest(n_estimators=N_ESTIMATORS, contamination=0.10,
                               random_state=RANDOM_STATE, n_jobs=-1)
    clf_if.fit(X_normal)
    raw_if  = -clf_if.decision_function(X_sc)
    sc_if   = (raw_if-raw_if.min())/(raw_if.max()-raw_if.min()+1e-8)

    # LOF V3 (novelty=True)
    if len(X_normal) >= 50:
        lof     = LocalOutlierFactor(n_neighbors=20, contamination=0.10, novelty=True, n_jobs=-1)
        lof.fit(X_normal)
        raw_lof = -lof.score_samples(X_sc)
        sc_lof  = (raw_lof-raw_lof.min())/(raw_lof.max()-raw_lof.min()+1e-8)
    else:
        sc_lof = np.zeros(len(df))

    sc_rules    = df['severity_score'].values
    sc_fleet    = df['fleet_anomaly'].values * 0.10
    combined    = (W_IF*sc_if + W_LOF*sc_lof + W_RULES*sc_rules + sc_fleet).clip(0,1)

    df['score_if']       = sc_if
    df['score_lof']      = sc_lof
    df['score_rules']    = sc_rules
    df['combined_score'] = combined
    df['is_anomaly']     = combined >= THRESHOLD
    return df


# ══════════════════════════════════════════════════════════════
#  RUL V3 (Ensemble Poly+Exp+Weibull + IC)
# ══════════════════════════════════════════════════════════════
def compute_di(group):
    g = group.sort_values('timestamp').copy()
    def ns(s, w=30):
        s=s.fillna(s.median()); ref=s.iloc[:max(5,w)].mean()
        return ((s.rolling(w,min_periods=3).mean()-ref)/(ref+1e-9)).clip(0,None)
    col_v = 'vib_energy_mean' if 'vib_energy_mean' in g.columns else 'vib_energy'
    col_t = 'temp_mean'        if 'temp_mean'        in g.columns else 'temperature'
    raw   = 0.35*ns(g[col_v]) + 0.20*ns(g['vib_kurt'].abs()) + 0.25*ns(g[col_t]) + 0.20*g['combined_score'].fillna(0)
    lo,hi = raw.min(),raw.max()
    g['degradation_index'] = ((raw-lo)/(hi-lo+1e-9)).clip(0,1) if hi>lo else 0.0
    return g


def estimate_rul(group):
    g   = group.sort_values('timestamp').copy()
    mid = int(g['motor_id'].iloc[0])
    g['days'] = (g['timestamp']-g['timestamp'].min()).dt.total_seconds()/86400
    x, y = g['days'].values, np.clip(g['degradation_index'].values,0,1)

    if len(x)<5 or x.max()<0.1:
        return {'motor_id':mid,'rul_days':'>90','rul_ensemble':90,'rul_low':70,'rul_high':90,
                'rul_poly':90,'rul_exp':90,'rul_weibull':90,'weibull_beta':1.0,
                'current_di':float(y[-1]) if len(y) else 0,'trend_slope':0,'risk_level':'FAIBLE',
                'confidence':0.5,'poly_coef':[0,0,float(y[-1]) if len(y) else 0]}

    try:
        pc=np.polyfit(x,y,min(2,len(x)-1)); pf=np.poly1d(pc)
        sl=max(1e-6,pf(x[-1]+1)-pf(x[-1]))
        dp=max(0.,min(PREDICT_DAYS,(DI_CRITICAL-y[-1])/sl))
    except: dp=90.; pc=[0,0,y[-1]]; pf=np.poly1d(pc); sl=0.

    try:
        valid=y>0.01
        if valid.sum()>=3:
            logy=np.log(y[valid]+1e-6); pe=np.polyfit(x[valid],logy,1)
            be=pe[0]; ae=np.exp(pe[1])
            de=max(0.,min(PREDICT_DAYS,np.log(DI_CRITICAL/(ae+1e-9))/be-x[-1])) if be>1e-6 else 90.
        else: de=90.
    except: de=90.

    try:
        beta=float(np.clip(1.+max(0.,np.mean(np.diff(np.diff(y)))*10.),0.5,3.))
        eta=x[-1]/((-np.log(1-y[-1]+1e-8))**(1./beta)+1e-8)
        dw=max(0.,min(PREDICT_DAYS,eta*((-np.log(1-DI_CRITICAL))**(1./beta))-x[-1]))
    except: beta=1.; dw=dp

    rul_m = min(PREDICT_DAYS, 0.4*dp+0.3*de+0.3*dw)
    spread= np.std([dp,de,dw])
    rul_l = max(0.,rul_m-spread); rul_h=min(PREDICT_DAYS,rul_m+spread)
    conf  = max(0.3,1.-spread/(rul_m+1.)*0.5)

    risk=('CRITIQUE' if y[-1]>=DI_CRITICAL or rul_m<7 else 'ÉLEVÉ' if y[-1]>=DI_WARNING or rul_m<21
          else 'MODÉRÉ' if y[-1]>=0.30 else 'FAIBLE')

    return {'motor_id':mid,'rul_days':('>90' if rul_m>=PREDICT_DAYS else str(round(rul_m,1))),
            'rul_ensemble':round(rul_m,1),'rul_low':round(rul_l,1),'rul_high':round(rul_h,1),
            'rul_poly':round(dp,1),'rul_exp':round(de,1),'rul_weibull':round(dw,1),
            'weibull_beta':round(beta,3),'current_di':round(float(y[-1]),4),
            'trend_slope':round(float(sl),6),'risk_level':risk,'confidence':round(conf,3),
            'poly_coef':list(pc)}


# ══════════════════════════════════════════════════════════════
#  CUSUM V3
# ══════════════════════════════════════════════════════════════
def detect_cusum(group):
    g   = group.sort_values('timestamp').copy()
    mid = int(g['motor_id'].iloc[0])
    di  = g['degradation_index'].fillna(0).values
    if len(di)<10:
        return {'motor_id':mid,'cusum_alarm':False,'change_point_date':None,'severity':'STABLE'}
    mu0,sigma=di[:10].mean(),max(di[:10].std(),0.01)
    k,h=CUSUM_K*sigma,CUSUM_H*sigma
    Sp=np.zeros(len(di)); alarms=[]
    for t in range(1,len(di)):
        Sp[t]=max(0.,Sp[t-1]+(di[t]-mu0)-k)
        if Sp[t]>h: alarms.append(t)
    if not alarms:
        return {'motor_id':mid,'cusum_alarm':False,'change_point_date':None,'severity':'STABLE'}
    first=alarms[0]
    sev=('ÉLEVÉ' if first<len(di)*0.3 else 'MODÉRÉ' if first<len(di)*0.6 else 'FAIBLE_ALARM')
    return {'motor_id':mid,'cusum_alarm':True,
            'change_point_date':str(g['timestamp'].iloc[first])[:10],'severity':sev,
            'di_before':round(float(di[:first].mean()),4),'di_after':round(float(di[first:].mean()),4)}


# ══════════════════════════════════════════════════════════════
#  GESTION ÉTAT
# ══════════════════════════════════════════════════════════════
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'last_update':None,'n_total':0,'updates':[]}


def save_state(state):
    with open(STATE_FILE,'w') as f:
        json.dump(state, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
#  FIGURES UPDATE
# ══════════════════════════════════════════════════════════════
def plot_update_summary(df_new, rul_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df_rul = pd.DataFrame([{k:v for k,v in r.items() if k!='poly_coef'} for r in rul_results])
    fig, axes = plt.subplots(1,3, figsize=(18,5))

    # DI
    colors = [RISK_COLOR.get(r,'#9e9e9e') for r in df_rul['risk_level']]
    axes[0].barh(df_rul['motor_id'].astype(str), df_rul['current_di'], color=colors)
    axes[0].axvline(DI_WARNING,color='orange',linestyle='--',lw=1.5)
    axes[0].axvline(DI_CRITICAL,color='red',linestyle='--',lw=2)
    axes[0].set_title('DI par moteur (MAJ)',fontweight='bold')
    axes[0].set_xlabel('DI'); axes[0].grid(axis='x',alpha=0.3)

    # RUL + IC
    df_s2 = df_rul.sort_values('rul_ensemble')
    yp = range(len(df_s2))
    axes[1].barh(list(yp), df_s2['rul_ensemble'],
                  color=[RISK_COLOR.get(r,'#9e9e9e') for r in df_s2['risk_level']], alpha=0.7)
    for i,(_,row) in enumerate(df_s2.iterrows()):
        axes[1].plot([row.get('rul_low',row['rul_ensemble']), row.get('rul_high',row['rul_ensemble'])],
                     [i,i], color='black', lw=2)
    axes[1].set_yticks(list(yp)); axes[1].set_yticklabels(df_s2['motor_id'].astype(str))
    axes[1].axvline(30,color='red',linestyle='--',lw=1.5)
    axes[1].set_title('RUL Ensemble + IC 80%',fontweight='bold')
    axes[1].set_xlabel('Jours'); axes[1].grid(axis='x',alpha=0.3)

    # Anomalie temporelle
    if 'combined_score' in df_new.columns:
        motors = df_new.groupby('motor_id')['is_anomaly'].mean().nlargest(5).index
        for mid in motors:
            dm = df_new[df_new['motor_id']==mid].sort_values('timestamp')
            axes[2].plot(dm['timestamp'],dm['combined_score'],lw=1,alpha=0.7,label=f'M{mid}')
        axes[2].axhline(THRESHOLD,color='orange',linestyle='--',lw=1.5)
        axes[2].set_title('Scores hybrides V3\n(5 moteurs à risque)',fontweight='bold')
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.suptitle(f'Mise à jour V3 — {datetime.now().strftime("%d/%m/%Y %H:%M")}',
                 fontsize=12,fontweight='bold')
    plt.tight_layout()
    path=os.path.join(output_dir,'fig_update_latest.png')
    plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════
#  PROCESSUS DE MISE À JOUR
# ══════════════════════════════════════════════════════════════
def run_update(sql_file, force=False):
    print(f"\n{C.CYAN}{C.BOLD}╔═══════════════════════════════════════╗")
    print(f"║  MISE À JOUR V3 — {datetime.now().strftime('%H:%M:%S')}          ║")
    print(f"╚═══════════════════════════════════════╝{C.RESET}\n")

    state = load_state()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # 1. Parsing nouvelles données
    print(f"→ Parsing SQL : {sql_file} ...")
    t0       = time.time()
    df_new   = parse_sql(sql_file)
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
    print(f"  {len(df_new):,} nouvelles mesures ({time.time()-t0:.1f}s)")

    if len(df_new) == 0:
        print(f"{C.RED}[ERREUR] Aucune donnée extraite.{C.RESET}"); return

    # 2. Fusion avec historique
    if os.path.exists(RAW_CSV) and not force:
        df_hist = pd.read_csv(RAW_CSV, parse_dates=['timestamp'])
        df_all  = pd.concat([df_hist, df_new]).drop_duplicates(
            subset=['motor_id','timestamp']).sort_values(['motor_id','timestamp'])
        n_new_unique = len(df_all) - len(df_hist)
        print(f"  {n_new_unique:,} nouvelles mesures uniques ajoutées")
        if n_new_unique == 0 and not force:
            print(f"{C.YELLOW}  ⚠ Aucune nouvelle donnée. Utilisez --force pour recalculer.{C.RESET}")
            return
    else:
        df_all = df_new.sort_values(['motor_id','timestamp'])
        print(f"  Mode complet (pas d'historique ou --force)")

    # Sauvegarde archive
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutil.copy2(sql_file, os.path.join(HISTORY_DIR, f'update_{ts_str}.sql'))

    # 3. Features V3
    print(f"\n→ Calcul features V3 ...")
    parts = []
    for mid, group in df_all.groupby('motor_id'):
        parts.append(compute_features(group.copy()))
    df_feat = pd.concat(parts, ignore_index=True)
    num_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat[num_cols] = df_feat[num_cols].fillna(0)
    df_feat.to_csv(FEATURES_CSV, index=False)
    print(f"  {df_feat.shape[0]:,} lignes × {df_feat.shape[1]} features")

    # 4. Seuils
    print(f"\n→ Calibration seuils ...")
    seuils = calibrer_seuils(df_feat)
    pd.DataFrame([{'motor_id':k,**v} for k,v in seuils.items()]).to_csv(SEUILS_CSV, index=False)

    # 5. Détection anomalies V3
    print(f"\n→ Détection anomalies (IF+LOF+Règles V3) ...")
    df_anom = detect_anomalies(df_feat, seuils)
    # Confirmation temporelle
    parts2 = []
    for mid, group in df_anom.groupby('motor_id'):
        g2 = group.sort_values('timestamp').copy()
        rolling = g2['is_anomaly'].rolling(3, min_periods=1).sum()
        g2['is_anomaly'] = (rolling >= 2).astype(bool)
        parts2.append(g2)
    df_anom = pd.concat(parts2, ignore_index=True)
    n_anom  = df_anom['is_anomaly'].sum()
    print(f"  Anomalies confirmées : {n_anom:,} ({n_anom/len(df_anom)*100:.1f}%)")
    df_anom.to_csv(ANOMALIES_CSV, index=False)

    # 6. DI + RUL V3
    print(f"\n→ Calcul DI + RUL V3 (Ensemble Poly+Exp+Weibull) ...")
    parts3 = []
    for mid, group in df_anom.groupby('motor_id'):
        parts3.append(compute_di(group.copy()))
    df_di = pd.concat(parts3, ignore_index=True)

    rul_results = []
    for mid, group in df_di.groupby('motor_id'):
        rul_results.append(estimate_rul(group))

    # 7. CUSUM V3
    print(f"\n→ CUSUM détection rupture ...")
    cusum_results = []
    for mid, group in df_di.groupby('motor_id'):
        cusum_results.append(detect_cusum(group))
    df_cusum   = pd.DataFrame(cusum_results)
    n_alarmed  = df_cusum['cusum_alarm'].sum()
    print(f"  {n_alarmed}/{len(df_cusum)} moteurs avec rupture CUSUM")
    df_cusum.to_csv('data/cusum_changepoints.csv', index=False)

    # Sauvegarde RUL
    df_rul = pd.DataFrame([{k:v for k,v in r.items() if k!='poly_coef'} for r in rul_results])
    df_rul.to_csv(RUL_SUMMARY_CSV, index=False)

    # Merge final
    df_rul_m = df_rul[['motor_id','rul_days','rul_ensemble','rul_low','rul_high',
                         'risk_level','weibull_beta','current_di']].copy()
    df_di    = df_di.merge(df_rul_m, on='motor_id', how='left')
    df_di.to_csv(RUL_CSV, index=False, encoding='utf-8-sig')

    # 8. Figures + état
    print(f"\n→ Génération figures ...")
    plot_update_summary(df_anom, rul_results, FIGURES_DIR)

    state['last_update'] = str(datetime.now())
    state['n_total']     = len(df_all)
    state['updates'].append({'ts':ts_str, 'n_new':len(df_new), 'n_anom':int(n_anom)})
    save_state(state)
    df_all.to_csv(RAW_CSV, index=False)

    # 9. Résumé
    print(f"\n{'='*55}")
    print(f"{C.GREEN}{C.BOLD}✓ MISE À JOUR V3 COMPLÈTE{C.RESET}")
    print(f"  Total mesures : {len(df_all):,}")
    print(f"  Anomalies     : {n_anom:,} ({n_anom/len(df_anom)*100:.1f}%)")
    print(f"  CUSUM alarmes : {n_alarmed}/{len(df_cusum)}")
    print(f"\n  {'M':>4} | {'DI':>6} | {'RUL':>8} | {'β':>5} | {'CUSUM':>6} | Risque")
    print(f"  {'─'*48}")
    cusum_map = {r['motor_id']: r for r in cusum_results}
    for _, r in df_rul.sort_values('current_di', ascending=False).head(10).iterrows():
        mid    = int(r['motor_id'])
        cusum  = '⚠' if cusum_map.get(mid,{}).get('cusum_alarm') else '✓'
        beta   = f"{r.get('weibull_beta',1.):.2f}"
        risk   = r['risk_level']
        color  = (C.RED if risk in ('CRITIQUE','ÉLEVÉ') else
                  C.YELLOW if risk=='MODÉRÉ' else C.GREEN)
        print(f"  {color}{mid:>4} | {r['current_di']:>6.3f} | {str(r['rul_days']):>8} | "
              f"{beta:>5} | {cusum:>6} | {risk}{C.RESET}")
    print(f"{'='*55}")


def print_status():
    state = load_state()
    print(f"\n{'='*50}")
    print(f"  STATUS — Maintenance Prédictive V3")
    print(f"{'='*50}")
    print(f"  Dernière MAJ  : {state.get('last_update','Jamais')}")
    print(f"  Total mesures : {state.get('n_total',0):,}")
    print(f"  Nb mises à jour: {len(state.get('updates',[]))}")
    if os.path.exists(RUL_SUMMARY_CSV):
        df_rul = pd.read_csv(RUL_SUMMARY_CSV)
        print(f"\n  Distribution risques :")
        print(df_rul['risk_level'].value_counts().to_string())
        crit = df_rul[df_rul['risk_level'].isin(['CRITIQUE','ÉLEVÉ'])]
        if not crit.empty:
            print(f"\n  ⚠ Prioritaires :")
            for _, r in crit.sort_values('current_di', ascending=False).iterrows():
                print(f"    M{int(r['motor_id']):2d} DI={r['current_di']:.3f} "
                      f"RUL={r.get('rul_days','?')}j [{r['risk_level']}]")
    print(f"{'='*50}")


def print_history():
    state = load_state()
    updates = state.get('updates', [])
    if not updates:
        print("Aucune mise à jour enregistrée."); return
    print(f"\n  Historique ({len(updates)} mises à jour) :")
    print(f"  {'#':>4} | {'Timestamp':>20} | {'Nouvelles':>10} | {'Anomalies':>10}")
    print(f"  {'─'*50}")
    for i, u in enumerate(updates[-10:], 1):
        print(f"  {i:>4} | {u['ts']:>20} | {u['n_new']:>10,} | {u['n_anom']:>10,}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Mise à jour incrémentale V3')
    parser.add_argument('sql_file', nargs='?', help='Fichier SQL à intégrer')
    parser.add_argument('--force',   action='store_true', help='Forcer recalcul complet')
    parser.add_argument('--status',  action='store_true', help='Afficher statut')
    parser.add_argument('--history', action='store_true', help='Afficher historique')
    args = parser.parse_args()

    if args.status:
        print_status(); return
    if args.history:
        print_history(); return
    if not args.sql_file:
        parser.print_help(); sys.exit(1)
    if not os.path.exists(args.sql_file):
        print(f"{C.RED}[ERREUR] Fichier introuvable : {args.sql_file}{C.RESET}"); sys.exit(1)

    run_update(args.sql_file, force=args.force)


if __name__ == '__main__':
    main()
