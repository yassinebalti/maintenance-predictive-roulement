"""
=============================================================================
STEP 6 — DIAGNOSTIC DES DÉFAUTS DE ROULEMENTS PAR ANALYSE SPECTRALE
=============================================================================
Ce module s'intègre APRÈS step3_anomaly_detection.py dans votre pipeline.

Il prend les moteurs détectés comme ANOMALIE et effectue :
  1. Calcul des fréquences caractéristiques de roulement (BPFO, BPFI, BSF, FTF)
  2. Analyse spectrale FFT + analyse d'enveloppe
  3. Classification du type de défaut (piste ext., piste int., bille, cage)
  4. Localisation (côté Drive End vs Non-Drive End)
  5. Score de sévérité par type de défaut
  6. Rapport de diagnostic par moteur

USAGE :
    python step6_bearing_fault_diagnosis.py

DÉPENDANCES :
    pip install numpy scipy pandas matplotlib scikit-learn mlflow
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
import os
import json
import logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Paramètres d'acquisition (à adapter à vos capteurs VWV)
SAMPLE_RATE = 10_000          # Hz — taux d'échantillonnage vibration
WINDOW_SEC  = 2.0             # secondes par fenêtre d'analyse
OVERLAP     = 0.5             # 50 % chevauchement

# Seuil d'énergie spectrale pour confirmer un défaut (ratio vs bruit)
FAULT_ENERGY_THRESHOLD = 3.0  # × énergie de fond

# Paramètres roulement par défaut (roulement générique 6205)
# À remplacer par les vrais paramètres de vos roulements
DEFAULT_BEARING = {
    "n_balls":    9,           # nombre de billes
    "ball_dia":   7.938,       # mm — diamètre bille
    "pitch_dia":  38.5,        # mm — diamètre primitif
    "contact_angle": 0.0,      # degrés (0 = roulement à billes standard)
}

# Mapping moteur → paramètres spécifiques (ajoutez vos roulements réels ici)
MOTOR_BEARING_PARAMS = {
    # Moteur 1600 — App Cylindre 55kW
    1: {"n_balls": 9, "ball_dia": 7.938, "pitch_dia": 38.5, "contact_angle": 0.0,
        "rpm_nominal": 1480},
    2: {"n_balls": 9, "ball_dia": 7.938, "pitch_dia": 38.5, "contact_angle": 0.0,
        "rpm_nominal": 1480},
    # Moteur 1602 CLI Fin 37kW
    3: {"n_balls": 8, "ball_dia": 8.731, "pitch_dia": 42.0, "contact_angle": 0.0,
        "rpm_nominal": 1470},
    4: {"n_balls": 8, "ball_dia": 8.731, "pitch_dia": 42.0, "contact_angle": 0.0,
        "rpm_nominal": 1470},
    # Valeur par défaut pour les autres moteurs
    **{i: {**DEFAULT_BEARING, "rpm_nominal": 1475} for i in range(5, 22)},
}


# ---------------------------------------------------------------------------
# 1. CALCUL DES FRÉQUENCES CARACTÉRISTIQUES DE ROULEMENT
# ---------------------------------------------------------------------------

@dataclass
class BearingFrequencies:
    """Fréquences caractéristiques d'un roulement à une vitesse donnée."""
    motor_id: int
    rpm: float
    fr: float      # fréquence de rotation (Hz)
    bpfo: float    # Ball Pass Frequency Outer race — piste extérieure
    bpfi: float    # Ball Pass Frequency Inner race — piste intérieure
    bsf: float     # Ball Spin Frequency — bille
    ftf: float     # Fundamental Train Frequency — cage
    harmonics: int = 5


def compute_bearing_frequencies(motor_id: int, rpm: float) -> BearingFrequencies:
    """
    Calcule les fréquences caractéristiques à partir de la géométrie du roulement.
    
    Formules AGMA/ISO :
      BPFO = (n/2) × fr × (1 − (d/D)×cos(α))
      BPFI = (n/2) × fr × (1 + (d/D)×cos(α))
      BSF  = (D/2d) × fr × (1 − ((d/D)×cos(α))²)
      FTF  = (fr/2) × (1 − (d/D)×cos(α))
    """
    params = MOTOR_BEARING_PARAMS.get(motor_id, {**DEFAULT_BEARING, "rpm_nominal": 1475})
    
    n   = params["n_balls"]
    d   = params["ball_dia"]
    D   = params["pitch_dia"]
    a   = np.radians(params["contact_angle"])
    
    fr   = rpm / 60.0
    ratio = (d / D) * np.cos(a)
    
    bpfo = (n / 2) * fr * (1 - ratio)
    bpfi = (n / 2) * fr * (1 + ratio)
    bsf  = (D / (2 * d)) * fr * (1 - ratio**2)
    ftf  = (fr / 2) * (1 - ratio)
    
    return BearingFrequencies(
        motor_id=motor_id,
        rpm=rpm,
        fr=fr,
        bpfo=bpfo,
        bpfi=bpfi,
        bsf=bsf,
        ftf=ftf,
    )


# ---------------------------------------------------------------------------
# 2. ANALYSE SPECTRALE — FFT + ENVELOPPE
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, low: float, high: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """Filtre passe-bande Butterworth."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def compute_fft(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (freqs, amplitudes) du spectre FFT unilatéral."""
    n      = len(signal)
    freqs  = fftfreq(n, d=1.0/fs)[:n//2]
    amps   = (2.0 / n) * np.abs(fft(signal))[:n//2]
    return freqs, amps


def envelope_analysis(signal: np.ndarray, fs: float,
                       demod_band: Tuple[float, float] = (2000, 4000)
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyse d'enveloppe (démodulation AM) :
      1. Filtrage dans la bande de résonance
      2. Signal analytique (Hilbert)
      3. FFT de l'enveloppe
    """
    filtered  = bandpass_filter(signal, demod_band[0], demod_band[1], fs)
    analytic  = hilbert(filtered)
    envelope  = np.abs(analytic)
    # Centrer l'enveloppe
    envelope -= envelope.mean()
    freqs, amps = compute_fft(envelope, fs)
    return freqs, amps


def spectral_energy_around(freqs: np.ndarray, amps: np.ndarray,
                             target_freq: float, n_harmonics: int = 3,
                             bandwidth: float = 2.0) -> float:
    """
    Énergie spectrale cumulée autour des n harmoniques d'une fréquence cible.
    bandwidth : ±Hz autour de chaque harmonique.
    """
    energy = 0.0
    for k in range(1, n_harmonics + 1):
        hf    = k * target_freq
        mask  = (freqs >= hf - bandwidth) & (freqs <= hf + bandwidth)
        energy += np.sum(amps[mask] ** 2)
    return energy


def background_noise_energy(freqs: np.ndarray, amps: np.ndarray,
                              target_freqs: List[float], n_harmonics: int = 3,
                              bandwidth: float = 2.0) -> float:
    """Énergie de fond en excluant les zones de fréquences de défaut."""
    mask = np.ones(len(freqs), dtype=bool)
    for tf in target_freqs:
        for k in range(1, n_harmonics + 1):
            hf   = k * tf
            zone = (freqs >= hf - bandwidth) & (freqs <= hf + bandwidth)
            mask &= ~zone
    return np.mean(amps[mask] ** 2) if mask.sum() > 0 else 1e-10


# ---------------------------------------------------------------------------
# 3. CLASSIFICATION DU TYPE DE DÉFAUT
# ---------------------------------------------------------------------------

@dataclass
class FaultDiagnosis:
    """Résultat du diagnostic pour un moteur."""
    motor_id: int
    rpm: float
    bearing_freqs: BearingFrequencies
    
    # Ratios énergie / bruit pour chaque type
    ratio_bpfo: float = 0.0    # piste extérieure
    ratio_bpfi: float = 0.0    # piste intérieure
    ratio_bsf:  float = 0.0    # bille
    ratio_ftf:  float = 0.0    # cage
    
    # Classification finale
    fault_type: str = "NORMAL"        # OUTER_RACE | INNER_RACE | BALL | CAGE | NORMAL
    fault_label_fr: str = "Normal"
    severity: float = 0.0             # 0-1
    severity_label: str = "FAIBLE"
    confidence: float = 0.0
    
    # Localisation
    location: str = "Inconnu"         # Drive End | Non-Drive End | Indéterminé
    
    # Recommandation
    recommendation: str = ""
    days_to_action: int = 90


FAULT_LABELS = {
    "OUTER_RACE": "Défaut piste extérieure (BPFO)",
    "INNER_RACE": "Défaut piste intérieure (BPFI)",
    "BALL":       "Défaut de bille (BSF)",
    "CAGE":       "Défaut de cage (FTF)",
    "NORMAL":     "Normal — aucun défaut détecté",
}

SEVERITY_THRESHOLDS = {
    # (ratio_energy_vs_noise) → (label, days_to_action, recommendation)
    10.0: ("CRITIQUE", 7,  "Arrêt immédiat recommandé — remplacement roulement"),
    5.0:  ("ÉLEVÉ",   14,  "Planifier remplacement sous 2 semaines"),
    3.0:  ("MODÉRÉ",  30,  "Surveiller de près — planifier intervention"),
    1.5:  ("FAIBLE",  90,  "Surveillance continue — pas d'action immédiate"),
    0.0:  ("NORMAL",  180, "Aucune action requise"),
}


def classify_fault(ratios: Dict[str, float],
                   threshold: float = FAULT_ENERGY_THRESHOLD) -> Tuple[str, float, float]:
    """
    Détermine le type de défaut dominant et la sévérité.
    
    Returns: (fault_type, severity_ratio, confidence)
    """
    fault_map = {
        "OUTER_RACE": ratios.get("bpfo", 0),
        "INNER_RACE": ratios.get("bpfi", 0),
        "BALL":       ratios.get("bsf",  0),
        "CAGE":       ratios.get("ftf",  0),
    }
    
    max_fault = max(fault_map, key=fault_map.get)
    max_ratio = fault_map[max_fault]
    
    if max_ratio < threshold:
        return "NORMAL", max_ratio, 1.0 - (max_ratio / threshold)
    
    # Confiance = ratio du dominant sur la somme totale
    total = sum(fault_map.values()) + 1e-10
    confidence = fault_map[max_fault] / total
    
    # Sévérité normalisée 0-1 (log scale)
    severity = min(1.0, np.log1p(max_ratio) / np.log1p(20.0))
    
    return max_fault, severity, confidence


def get_severity_label(ratio: float) -> Tuple[str, int, str]:
    """Retourne (label, jours avant action, recommandation)."""
    for threshold, (label, days, rec) in sorted(
            SEVERITY_THRESHOLDS.items(), reverse=True):
        if ratio >= threshold:
            return label, days, rec
    return "NORMAL", 180, "Aucune action requise"


# ---------------------------------------------------------------------------
# 4. PIPELINE PRINCIPAL DE DIAGNOSTIC
# ---------------------------------------------------------------------------

def diagnose_motor(motor_id: int,
                   vibration_signal: np.ndarray,
                   rpm: Optional[float] = None,
                   fs: float = SAMPLE_RATE,
                   location: str = "Drive End") -> FaultDiagnosis:
    """
    Diagnostic complet pour un moteur donné.
    
    Args:
        motor_id      : identifiant moteur (1-21)
        vibration_signal : signal vibratoire brut (numpy array)
        rpm           : vitesse réelle (si None, utilise la valeur nominale)
        fs            : fréquence d'échantillonnage
        location      : "Drive End" ou "Non-Drive End"
    
    Returns:
        FaultDiagnosis avec type de défaut, sévérité et recommandations
    """
    params  = MOTOR_BEARING_PARAMS.get(motor_id, {**DEFAULT_BEARING, "rpm_nominal": 1475})
    rpm     = rpm or params["rpm_nominal"]
    
    bf      = compute_bearing_frequencies(motor_id, rpm)
    
    # --- FFT directe ---
    freqs_fft, amps_fft = compute_fft(vibration_signal, fs)
    
    # --- Analyse d'enveloppe (bande haute fréquence) ---
    freqs_env, amps_env = envelope_analysis(vibration_signal, fs)
    
    # --- Énergie de fond ---
    target_list = [bf.bpfo, bf.bpfi, bf.bsf, bf.ftf]
    
    bg_fft = background_noise_energy(freqs_fft, amps_fft, target_list)
    bg_env = background_noise_energy(freqs_env, amps_env, target_list)
    
    # Combine FFT + enveloppe (enveloppe plus sensible aux défauts de roulement)
    def combined_ratio(target_freq):
        e_fft = spectral_energy_around(freqs_fft, amps_fft, target_freq)
        e_env = spectral_energy_around(freqs_env, amps_env, target_freq)
        r_fft = e_fft / (bg_fft + 1e-10)
        r_env = e_env / (bg_env + 1e-10)
        return 0.35 * r_fft + 0.65 * r_env   # enveloppe plus pondérée
    
    ratios = {
        "bpfo": combined_ratio(bf.bpfo),
        "bpfi": combined_ratio(bf.bpfi),
        "bsf":  combined_ratio(bf.bsf),
        "ftf":  combined_ratio(bf.ftf),
    }
    
    fault_type, severity, confidence = classify_fault(ratios)
    sev_label, days, rec = get_severity_label(max(ratios.values()))
    
    return FaultDiagnosis(
        motor_id=motor_id,
        rpm=rpm,
        bearing_freqs=bf,
        ratio_bpfo=ratios["bpfo"],
        ratio_bpfi=ratios["bpfi"],
        ratio_bsf=ratios["bsf"],
        ratio_ftf=ratios["ftf"],
        fault_type=fault_type,
        fault_label_fr=FAULT_LABELS[fault_type],
        severity=severity,
        severity_label=sev_label,
        confidence=confidence,
        location=location,
        recommendation=rec,
        days_to_action=days,
    )


# ---------------------------------------------------------------------------
# 5. SIMULATION — Générateur de signaux synthétiques de test
#    (à remplacer par la lecture de vos vraies données Kafka/CSV)
# ---------------------------------------------------------------------------

def simulate_bearing_signal(fault_type: str = "OUTER_RACE",
                             rpm: float = 1480,
                             fs: float = SAMPLE_RATE,
                             duration: float = 2.0,
                             snr_db: float = 20.0) -> np.ndarray:
    """
    Génère un signal vibratoire synthétique avec un défaut de roulement.
    Utile pour tester le pipeline sans données réelles.
    
    fault_type : "OUTER_RACE" | "INNER_RACE" | "BALL" | "CAGE" | "NORMAL"
    """
    t     = np.arange(0, duration, 1.0/fs)
    n     = len(t)
    
    # Signal de base : harmoniques de rotation
    fr    = rpm / 60.0
    sig   = 0.5 * np.sin(2 * np.pi * fr * t)
    sig  += 0.2 * np.sin(2 * np.pi * 2 * fr * t)
    
    # Paramètres roulement 6205
    n_b, d, D = 9, 7.938, 38.5
    ratio = d / D
    
    fault_freqs = {
        "OUTER_RACE": (n_b / 2) * fr * (1 - ratio),
        "INNER_RACE": (n_b / 2) * fr * (1 + ratio),
        "BALL":       (D / (2 * d)) * fr * (1 - ratio**2),
        "CAGE":       (fr / 2) * (1 - ratio),
    }
    
    if fault_type in fault_freqs:
        ff     = fault_freqs[fault_type]
        # Impulsions périodiques modulées (modèle de défaut réaliste)
        impact = np.zeros(n)
        period = int(fs / ff)
        for i in range(0, n, period):
            if i < n:
                # Impulsion exponentielle amortie
                decay_len = min(period // 2, n - i)
                decay     = np.exp(-np.linspace(0, 5, decay_len))
                impact[i:i + decay_len] += 2.0 * decay * np.sin(
                    2 * np.pi * 3000 * np.linspace(0, decay_len/fs, decay_len)
                )
        sig += impact
    
    # Ajout de bruit blanc
    signal_power = np.mean(sig ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    sig         += np.random.normal(0, np.sqrt(noise_power), n)
    
    return sig


# ---------------------------------------------------------------------------
# 6. VISUALISATION — Rapport spectral par moteur
# ---------------------------------------------------------------------------

def plot_motor_diagnosis(diagnosis: FaultDiagnosis,
                          signal: np.ndarray,
                          fs: float = SAMPLE_RATE,
                          save_path: Optional[str] = None):
    """Génère le rapport visuel complet pour un moteur."""
    
    bf = diagnosis.bearing_freqs
    
    # Calcul des spectres
    freqs_fft, amps_fft = compute_fft(signal, fs)
    freqs_env, amps_env = envelope_analysis(signal, fs)
    t = np.linspace(0, len(signal)/fs, len(signal))
    
    # Couleurs par sévérité
    sev_colors = {
        "CRITIQUE": "#FF1744",
        "ÉLEVÉ":    "#FF6D00",
        "MODÉRÉ":   "#FFC400",
        "FAIBLE":   "#00BFA5",
        "NORMAL":   "#00C853",
    }
    sev_color = sev_colors.get(diagnosis.severity_label, "#00C853")
    
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    # --- Titre ---
    fig.suptitle(
        f"DIAGNOSTIC ROULEMENT — Moteur {diagnosis.motor_id}  |  "
        f"{diagnosis.fault_label_fr}  |  Sévérité : {diagnosis.severity_label}  |  "
        f"Confiance : {diagnosis.confidence*100:.0f}%",
        fontsize=13, fontweight="bold", color=sev_color, y=0.98
    )
    
    ax_style = dict(facecolor="#161b22", tick_params=dict(colors="white"),
                    label_color="white")
    
    def style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#adbac7")
        ax.set_title(title, color="#e6edf3", fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, color="#adbac7", fontsize=9)
        ax.set_ylabel(ylabel, color="#adbac7", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
    
    # --- [0,0] Signal temporel ---
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(t[:int(0.1*fs)], signal[:int(0.1*fs)], color="#58a6ff", lw=0.7, alpha=0.9)
    style_ax(ax0, f"Signal vibratoire — M{diagnosis.motor_id} ({diagnosis.location})",
             "Temps (s)", "Amplitude (g)")
    
    # --- [0,2] Ratios par type de défaut ---
    ax1 = fig.add_subplot(gs[0, 2])
    labels  = ["BPFO\n(Ext.)", "BPFI\n(Int.)", "BSF\n(Bille)", "FTF\n(Cage)"]
    ratios  = [diagnosis.ratio_bpfo, diagnosis.ratio_bpfi,
               diagnosis.ratio_bsf,  diagnosis.ratio_ftf]
    colors  = []
    for r in ratios:
        if   r >= 10: colors.append("#FF1744")
        elif r >= 5:  colors.append("#FF6D00")
        elif r >= 3:  colors.append("#FFC400")
        else:         colors.append("#00BFA5")
    
    bars = ax1.barh(labels, ratios, color=colors, edgecolor="#30363d")
    ax1.axvline(FAULT_ENERGY_THRESHOLD, color="#FFC400", ls="--", lw=1.5,
                label=f"Seuil = {FAULT_ENERGY_THRESHOLD}×")
    ax1.set_xlabel("Ratio Énergie / Bruit", color="#adbac7", fontsize=9)
    style_ax(ax1, "Score par type de défaut")
    ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="#adbac7")
    for bar, ratio in zip(bars, ratios):
        ax1.text(ratio + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{ratio:.1f}×", va="center", color="#e6edf3", fontsize=8)
    
    # --- [1,0-1] Spectre FFT avec marqueurs fréquences de défaut ---
    ax2 = fig.add_subplot(gs[1, :2])
    mask = freqs_fft < 500
    ax2.semilogy(freqs_fft[mask], amps_fft[mask] + 1e-10,
                  color="#79c0ff", lw=0.8, alpha=0.8)
    
    freq_info = [
        (bf.bpfo, "#FF6D00", "BPFO"),
        (bf.bpfi, "#FF1744", "BPFI"),
        (bf.bsf,  "#00BFA5", "BSF"),
        (bf.ftf,  "#BC8CFF", "FTF"),
        (bf.fr,   "#3FB950", "fr"),
    ]
    for base_f, color, label in freq_info:
        for k in range(1, 4):
            hf = k * base_f
            if hf < 500:
                lbl = label if k == 1 else None
                ax2.axvline(hf, color=color, ls="--", lw=1.0, alpha=0.7, label=lbl)
    
    style_ax(ax2, f"Spectre FFT (0-500 Hz) — RPM={diagnosis.rpm:.0f}",
             "Fréquence (Hz)", "Amplitude (log)")
    ax2.legend(ncol=5, fontsize=8, facecolor="#161b22", labelcolor="#adbac7",
               loc="upper right")
    
    # --- [1,2] Spectre d'enveloppe ---
    ax3 = fig.add_subplot(gs[1, 2])
    mask_e = freqs_env < 300
    ax3.plot(freqs_env[mask_e], amps_env[mask_e],
              color="#F78166", lw=0.8, alpha=0.9)
    for base_f, color, label in freq_info[:4]:
        for k in range(1, 4):
            hf = k * base_f
            if hf < 300:
                ax3.axvline(hf, color=color, ls="--", lw=1.0, alpha=0.7)
    style_ax(ax3, "Spectre d'enveloppe (0-300 Hz)",
             "Fréquence (Hz)", "Amplitude")
    
    # --- [2, :] Tableau de bord résumé ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    
    table_data = [
        ["Paramètre", "Valeur", "Paramètre", "Valeur"],
        ["Moteur ID",           str(diagnosis.motor_id),
         "Type de défaut",      diagnosis.fault_label_fr],
        ["RPM mesurée",         f"{diagnosis.rpm:.0f} tr/min",
         "Sévérité",            diagnosis.severity_label],
        ["Localisation",        diagnosis.location,
         "Score sévérité",      f"{diagnosis.severity:.3f}"],
        ["BPFO (piste ext.)",   f"{bf.bpfo:.2f} Hz",
         "Ratio BPFO/bruit",    f"{diagnosis.ratio_bpfo:.1f}×"],
        ["BPFI (piste int.)",   f"{bf.bpfi:.2f} Hz",
         "Ratio BPFI/bruit",    f"{diagnosis.ratio_bpfi:.1f}×"],
        ["BSF (bille)",         f"{bf.bsf:.2f} Hz",
         "Ratio BSF/bruit",     f"{diagnosis.ratio_bsf:.1f}×"],
        ["FTF (cage)",          f"{bf.ftf:.2f} Hz",
         "Ratio FTF/bruit",     f"{diagnosis.ratio_ftf:.1f}×"],
        ["Confiance diagnostic",f"{diagnosis.confidence*100:.0f}%",
         "Jours avant action",  str(diagnosis.days_to_action)],
        ["RECOMMANDATION", diagnosis.recommendation, "", ""],
    ]
    
    col_colors = [["#1c2128"]*4] + [["#0d1117", "#161b22"]*2] * (len(table_data)-2) + [["#21262d"]*4]
    
    tbl = ax4.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        cellColours=col_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    
    # Style en-tête
    for j in range(4):
        tbl[0, j].set_text_props(color="#e6edf3", fontweight="bold")
        tbl[0, j].set_facecolor("#21262d")
    
    for i in range(1, len(table_data)):
        for j in range(4):
            tbl[i, j].set_text_props(color="#adbac7")
    
    # Couleur ligne recommandation
    for j in range(4):
        tbl[len(table_data)-1, j].set_facecolor(sev_color + "33")
        tbl[len(table_data)-1, j].set_text_props(color=sev_color, fontweight="bold")
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor="#0d1117")
        logger.info(f"Figure sauvegardée : {save_path}")
    
    plt.close()


# ---------------------------------------------------------------------------
# 7. RAPPORT FLOTTE — Vue d'ensemble tous moteurs
# ---------------------------------------------------------------------------

def plot_fleet_fault_report(diagnoses: List[FaultDiagnosis],
                              save_path: Optional[str] = None):
    """Tableau de bord flotte — synthèse des diagnostics."""
    
    if not diagnoses:
        logger.warning("Aucun diagnostic à afficher.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("RAPPORT FLOTTE — Diagnostic défauts de roulements",
                 fontsize=14, fontweight="bold", color="#e6edf3", y=1.01)
    
    fault_types  = [d.fault_type for d in diagnoses]
    motor_ids    = [f"M{d.motor_id}" for d in diagnoses]
    severities   = [d.severity for d in diagnoses]
    sev_labels   = [d.severity_label for d in diagnoses]
    
    fault_colors = {
        "OUTER_RACE": "#FF6D00",
        "INNER_RACE": "#FF1744",
        "BALL":       "#00BFA5",
        "CAGE":       "#BC8CFF",
        "NORMAL":     "#3FB950",
    }
    fault_colors_list = [fault_colors.get(ft, "#adbac7") for ft in fault_types]
    
    sev_colors_list = []
    for sl in sev_labels:
        m = {"CRITIQUE": "#FF1744", "ÉLEVÉ": "#FF6D00",
             "MODÉRÉ": "#FFC400", "FAIBLE": "#00BFA5", "NORMAL": "#3FB950"}
        sev_colors_list.append(m.get(sl, "#adbac7"))
    
    # [0] Type de défaut par moteur
    ax0 = axes[0]
    ax0.set_facecolor("#161b22")
    bars = ax0.barh(motor_ids, severities, color=fault_colors_list, edgecolor="#30363d")
    ax0.set_xlabel("Score sévérité [0-1]", color="#adbac7")
    ax0.set_title("Sévérité par moteur (couleur = type de défaut)",
                   color="#e6edf3", fontweight="bold")
    ax0.tick_params(colors="#adbac7")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#30363d")
    
    # Légende types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=FAULT_LABELS[ft][:25])
                       for ft, c in fault_colors.items()]
    ax0.legend(handles=legend_elements, fontsize=7, facecolor="#161b22",
                labelcolor="#adbac7", loc="lower right")
    
    # [1] Distribution des types de défauts
    ax1 = axes[1]
    ax1.set_facecolor("#161b22")
    from collections import Counter
    ft_counts = Counter(fault_types)
    labels    = [FAULT_LABELS[ft][:20] for ft in ft_counts]
    sizes     = list(ft_counts.values())
    colors_p  = [fault_colors.get(ft, "#adbac7") for ft in ft_counts]
    
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors_p,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "#e6edf3", "fontsize": 8},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_color("#0d1117")
        at.set_fontweight("bold")
    ax1.set_title("Distribution des types de défauts",
                   color="#e6edf3", fontweight="bold")
    
    # [2] Tableau récapitulatif
    ax2 = axes[2]
    ax2.set_facecolor("#161b22")
    ax2.axis("off")
    
    rows = [[f"M{d.motor_id}",
             d.fault_label_fr[:22] + ("…" if len(d.fault_label_fr) > 22 else ""),
             d.severity_label,
             f"{d.days_to_action}j",
             d.location[:10]]
            for d in sorted(diagnoses, key=lambda x: -x.severity)]
    
    tbl = ax2.table(
        cellText=rows,
        colLabels=["Moteur", "Défaut", "Sévérité", "Action", "Localisation"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    
    sev_bg = {"CRITIQUE": "#3d0000", "ÉLEVÉ": "#3d1a00",
               "MODÉRÉ": "#3d3000", "FAIBLE": "#003d2e", "NORMAL": "#003d0f"}
    
    for i, d in enumerate(sorted(diagnoses, key=lambda x: -x.severity)):
        bg = sev_bg.get(d.severity_label, "#161b22")
        for j in range(5):
            tbl[i+1, j].set_facecolor(bg)
            tbl[i+1, j].set_text_props(color="#e6edf3")
    
    for j in range(5):
        tbl[0, j].set_facecolor("#21262d")
        tbl[0, j].set_text_props(color="#e6edf3", fontweight="bold")
    
    ax2.set_title("Classement par priorité d'intervention",
                   color="#e6edf3", fontweight="bold", pad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor="#0d1117")
        logger.info(f"Rapport flotte sauvegardé : {save_path}")
    
    plt.close()


# ---------------------------------------------------------------------------
# 8. EXPORT JSON — Intégration avec MLflow / dashboard Streamlit
# ---------------------------------------------------------------------------

def export_diagnoses_json(diagnoses: List[FaultDiagnosis],
                           path: str = "bearing_diagnosis_report.json"):
    """Exporte les diagnostics en JSON pour intégration dashboard."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_motors_diagnosed": len(diagnoses),
        "summary": {
            "critique": sum(1 for d in diagnoses if d.severity_label == "CRITIQUE"),
            "élevé":    sum(1 for d in diagnoses if d.severity_label == "ÉLEVÉ"),
            "modéré":   sum(1 for d in diagnoses if d.severity_label == "MODÉRÉ"),
            "faible":   sum(1 for d in diagnoses if d.severity_label == "FAIBLE"),
            "normal":   sum(1 for d in diagnoses if d.severity_label == "NORMAL"),
        },
        "motors": []
    }
    
    for d in sorted(diagnoses, key=lambda x: -x.severity):
        report["motors"].append({
            "motor_id":       d.motor_id,
            "rpm":            d.rpm,
            "fault_type":     d.fault_type,
            "fault_label_fr": d.fault_label_fr,
            "severity":       round(d.severity, 4),
            "severity_label": d.severity_label,
            "confidence":     round(d.confidence, 4),
            "location":       d.location,
            "ratios": {
                "bpfo": round(d.ratio_bpfo, 2),
                "bpfi": round(d.ratio_bpfi, 2),
                "bsf":  round(d.ratio_bsf, 2),
                "ftf":  round(d.ratio_ftf, 2),
            },
            "bearing_freqs": {
                "fr":   round(d.bearing_freqs.fr, 2),
                "bpfo": round(d.bearing_freqs.bpfo, 2),
                "bpfi": round(d.bearing_freqs.bpfi, 2),
                "bsf":  round(d.bearing_freqs.bsf, 2),
                "ftf":  round(d.bearing_freqs.ftf, 2),
            },
            "recommendation":  d.recommendation,
            "days_to_action":  d.days_to_action,
        })
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Rapport JSON exporté : {path}")
    return report


# ---------------------------------------------------------------------------
# 9. INTERFACE AVEC VOTRE PIPELINE EXISTANT
#    Remplacez simulate_motor_data() par la lecture de vos vraies données
# ---------------------------------------------------------------------------

def load_motor_data_from_pipeline(motor_id: int,
                                   data_dir: str = "data") -> Optional[np.ndarray]:
    """
    Charge le signal vibratoire depuis vos fichiers CSV existants.
    
    Adaptez cette fonction à votre format de données réel.
    Votre pipeline produit des fichiers dans data/ avec colonnes :
    timestamp, motor_id, vibration_x, vibration_y, vibration_z, temperature, current
    """
    # Tentative de chargement depuis le répertoire data/
    candidates = [
        f"{data_dir}/motor_{motor_id:02d}_vibration.csv",
        f"{data_dir}/full_data_motor_{motor_id}.csv",
        f"{data_dir}/M{motor_id}_raw.csv",
    ]
    
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Cherche la colonne de vibration
            vib_cols = [c for c in df.columns if "vibr" in c.lower() or "rms" in c.lower()]
            if vib_cols:
                sig = df[vib_cols[0]].dropna().values.astype(float)
                logger.info(f"M{motor_id} : {len(sig)} échantillons depuis {path}")
                return sig
    
    logger.warning(f"M{motor_id} : aucun fichier trouvé, signal simulé utilisé")
    return None


def simulate_motor_data(motor_id: int) -> Tuple[np.ndarray, str]:
    """
    Simule des données pour les tests.
    Retourne (signal, fault_type simulé).
    """
    # Distribution réaliste pour la démonstration
    fault_distribution = {
        1:  ("OUTER_RACE", 15),
        2:  ("OUTER_RACE", 18),
        3:  ("OUTER_RACE", 20),
        4:  ("INNER_RACE", 25),
        5:  ("BALL",       22),
        6:  ("OUTER_RACE", 16),
        7:  ("NORMAL",     35),
        8:  ("OUTER_RACE", 20),
        9:  ("NORMAL",     40),
        10: ("CAGE",       25),
        11: ("NORMAL",     38),
        12: ("OUTER_RACE", 15),
        13: ("NORMAL",     42),
        14: ("INNER_RACE", 18),
        15: ("BALL",       20),
        16: ("NORMAL",     35),
        17: ("NORMAL",     40),
        18: ("INNER_RACE", 22),
        19: ("NORMAL",     38),
        20: ("OUTER_RACE", 25),
        21: ("OUTER_RACE", 8),   # critique — RUL=1.7j
    }
    
    ft, snr = fault_distribution.get(motor_id, ("NORMAL", 35))
    params  = MOTOR_BEARING_PARAMS.get(motor_id, {**DEFAULT_BEARING, "rpm_nominal": 1475})
    sig     = simulate_bearing_signal(ft, rpm=params["rpm_nominal"],
                                       fs=SAMPLE_RATE, snr_db=snr)
    return sig, ft


# ---------------------------------------------------------------------------
# 10. POINT D'ENTRÉE PRINCIPAL
# ---------------------------------------------------------------------------

def run_bearing_diagnosis(
        motor_ids: Optional[List[int]] = None,
        anomalous_only: bool = False,
        anomalous_motor_ids: Optional[List[int]] = None,
) -> List[FaultDiagnosis]:
    """
    Lance le diagnostic sur tous les moteurs (ou seulement les anomalies).
    
    Args:
        motor_ids            : liste des IDs à analyser (None = tous 1-21)
        anomalous_only       : si True, utilise anomalous_motor_ids
        anomalous_motor_ids  : résultat de step3 (moteurs en anomalie)
    
    Returns:
        Liste de FaultDiagnosis
    """
    if motor_ids is None:
        motor_ids = list(range(1, 22))
    
    if anomalous_only and anomalous_motor_ids is not None:
        motor_ids = anomalous_motor_ids
        logger.info(f"Diagnostic limité aux {len(motor_ids)} moteurs en anomalie")
    
    diagnoses = []
    
    for mid in motor_ids:
        logger.info(f"=== Diagnostic Moteur {mid} ===")
        
        # Chargement données (réelles ou simulées)
        signal = load_motor_data_from_pipeline(mid)
        if signal is None:
            signal, true_fault = simulate_motor_data(mid)
            logger.info(f"  → Simulation : défaut réel = {true_fault}")
        
        # Diagnostic Drive End (principal)
        diag_de = diagnose_motor(mid, signal, location="Drive End")
        
        # Optionnel : diagnostic Non-Drive End (si capteur dispo)
        # diag_nde = diagnose_motor(mid, signal_nde, location="Non-Drive End")
        
        diagnoses.append(diag_de)
        
        logger.info(f"  → Résultat : {diag_de.fault_label_fr} | "
                    f"Sévérité : {diag_de.severity_label} | "
                    f"Confiance : {diag_de.confidence*100:.0f}%")
        
        # Graphique individuel
        plot_path = OUTPUT_DIR / f"fig_bearing_diagnosis_M{mid:02d}.png"
        plot_motor_diagnosis(diag_de, signal, save_path=str(plot_path))
    
    # Rapport flotte
    fleet_path = OUTPUT_DIR / "fig_bearing_fleet_report.png"
    plot_fleet_fault_report(diagnoses, save_path=str(fleet_path))
    
    # Export JSON
    export_diagnoses_json(diagnoses, path="bearing_diagnosis_report.json")
    
    # Résumé console
    logger.info("\n" + "="*60)
    logger.info("RÉSUMÉ DIAGNOSTIC FLOTTE")
    logger.info("="*60)
    
    for d in sorted(diagnoses, key=lambda x: -x.severity):
        icon = {"CRITIQUE": "🔴", "ÉLEVÉ": "🟠", "MODÉRÉ": "🟡",
                "FAIBLE": "🟢", "NORMAL": "✅"}.get(d.severity_label, "⚪")
        logger.info(f"{icon} M{d.motor_id:2d} | {d.fault_label_fr[:35]:35s} | "
                    f"Sévérité : {d.severity_label:8s} | "
                    f"Action dans : {d.days_to_action}j")
    
    return diagnoses


# ---------------------------------------------------------------------------
# INTÉGRATION DANS VOTRE PIPELINE — Appel depuis step3_anomaly_detection.py
# ---------------------------------------------------------------------------
#
# Dans step3_anomaly_detection.py, ajoutez à la fin :
#
#   from step6_bearing_fault_diagnosis import run_bearing_diagnosis
#
#   # moteurs_anomalie = liste des motor_id avec anomalie détectée
#   diagnoses = run_bearing_diagnosis(
#       anomalous_only=True,
#       anomalous_motor_ids=moteurs_anomalie
#   )
#
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    np.random.seed(42)
    random.seed(42)
    
    logger.info("Démarrage du module de diagnostic des défauts de roulements")
    logger.info("Pipeline : step6_bearing_fault_diagnosis.py")
    logger.info("-" * 60)
    
    # Test complet sur tous les moteurs (simulation)
    # En production : passer anomalous_motor_ids depuis step3
    diagnoses = run_bearing_diagnosis(
        motor_ids=list(range(1, 22)),
        anomalous_only=False,
    )
    
    logger.info(f"\n✅ Diagnostic terminé pour {len(diagnoses)} moteurs")
    logger.info(f"📊 Figures sauvegardées dans : {OUTPUT_DIR}/")
    logger.info(f"📄 Rapport JSON : bearing_diagnosis_report.json")
