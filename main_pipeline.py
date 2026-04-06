"""
========================================================
 PIPELINE PRINCIPAL V3 — MAINTENANCE PRÉDICTIVE
 Orchestrateur complet

 Usage :
   python main_pipeline.py                    # pipeline complet
   python main_pipeline.py --step 4           # étape spécifique
   python main_pipeline.py --from-step 3      # depuis une étape
   python main_pipeline.py --skip-fulldata    # sans step1b
   python main_pipeline.py --skip-autoencoder # sans step3b

 Étapes :
   1 → step1_extraction.py         Extraction SQL (motor_measurements)
   2 → step1b_full_data.py         Intégration full_data (VWV)
   3 → step2_features.py           Feature Engineering V3 (+6 features AXE 3)
   4 → step3_anomaly_detection.py  Hybride V3 (AXE 1+4+5+6) + Diagnostic roulements
   5 → step3b_autoencoder.py       Autoencoder + Comparaison
   6 → step4_rul_prediction.py     RUL V3 (AXE 2+7)
   7 → step5_report.py             Rapport Final V3

 7 AXES D'AMÉLIORATION IA :
   AXE 1 — AUC corrigée vs Alert_Status (step3)
   AXE 2 — RUL Ensemble Poly+Exp+Weibull + IC (step4)
   AXE 3 — 6 nouvelles features vibratoires (step2)
   AXE 4 — Walk-forward cross-validation (step3)
   AXE 5 — LOF novelty=True intégré (step3)
   AXE 6 — SHAP par moteur (step3)
   AXE 7 — CUSUM détection rupture (step4)

 AJOUT step6 — Diagnostic défauts roulements (appelé depuis step3) :
   → Lancé automatiquement à la fin de step3_anomaly_detection.py
   → Identifie le type exact de défaut (BPFO/BPFI/BSF/FTF) par moteur
   → Produit : data/bearing_diagnosis.csv + bearing_diagnosis_report.json
========================================================
"""

import os
import sys
import time
import argparse
from datetime import datetime


class C:
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BLUE   = '\033[94m'
    BOLD   = '\033[1m'
    RESET  = '\033[0m'
    CYAN   = '\033[96m'


def print_header():
    print(f"""
{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║     PIPELINE IA/ML V3 — MAINTENANCE PRÉDICTIVE              ║
║     IF+LOF+Règles | RUL Weibull | CUSUM | SHAP | Walk-fwd   ║
║     + Diagnostic défauts roulements (BPFO/BPFI/BSF/FTF)     ║
╚══════════════════════════════════════════════════════════════╝
{C.RESET}""")


def run_step(step_num, script, description):
    print(f"\n{C.BOLD}{'─'*60}")
    print(f"  ÉTAPE {step_num} — {description}")
    print(f"{'─'*60}{C.RESET}")

    if not os.path.exists(script):
        print(f"{C.RED}[ERREUR] Script introuvable : {script}{C.RESET}")
        return False

    start   = time.time()
    ret     = os.system(f'python "{script}"')
    elapsed = time.time() - start

    if ret == 0:
        print(f"\n{C.GREEN}✓ Étape {step_num} terminée en {elapsed:.1f}s{C.RESET}")
        return True
    else:
        print(f"\n{C.RED}✗ Étape {step_num} échouée (code={ret}){C.RESET}")
        return False


def check_prerequisites():
    sql_files = [f for f in os.listdir('.') if f.endswith('.sql')]
    if not sql_files:
        print(f"{C.RED}[ERREUR] Aucun fichier .sql trouvé dans le répertoire courant.{C.RESET}")
        return False
    expected = 'ai_cp (2).sql'
    if os.path.exists(expected):
        print(f"{C.GREEN}✓ Fichier SQL : {expected}{C.RESET}")
    else:
        print(f"{C.YELLOW}⚠ Renommage → {expected}{C.RESET}")
        try:
            os.rename(sql_files[0], expected)
            print(f"{C.GREEN}  ✓ Renommé{C.RESET}")
        except Exception as e:
            print(f"{C.RED}  ✗ {e}{C.RESET}")

    # Vérification step6
    if os.path.exists('step6_bearing_fault_diagnosis.py'):
        print(f"{C.GREEN}✓ step6_bearing_fault_diagnosis.py trouvé{C.RESET}")
    else:
        print(f"{C.YELLOW}⚠ step6_bearing_fault_diagnosis.py absent — "
              f"le diagnostic roulements sera ignoré{C.RESET}")

    return True


def print_fichiers_produits():
    print(f"\n  {C.BOLD}── Données V3 ──────────────────────────────────{C.RESET}")
    fichiers = [
        ('data/01_raw_motor.csv',                'Extraction SQL brute'),
        ('data/02_features_motor.csv',           'Features V3 (31 colonnes, +6 AXE 3)'),
        ('data/03_anomalies.csv',                'Anomalies IF+LOF+Règles V3'),
        ('data/seuils_moteurs.csv',              'Seuils calibrés par moteur'),
        ('data/04_rul_results.csv',              'RUL Ensemble + IC Weibull'),
        ('data/rul_summary.csv',                 'Résumé RUL 21 moteurs'),
        ('data/cusum_changepoints.csv',          'CUSUM ruptures (AXE 7)'),
        ('data/validation_walkforward.csv',      'Walk-forward CV (AXE 4)'),
        ('data/shap_importance.csv',             'Importance features (AXE 6)'),
        ('data/shap_per_motor.csv',              'SHAP par moteur (AXE 6)'),
        ('data/feature_importance.csv',          'Feature importance XAI'),
        # ── AJOUT step6 ──────────────────────────────────────
        ('data/bearing_diagnosis.csv',           'Diagnostic défauts roulements (step6)'),
        ('bearing_diagnosis_report.json',        'Rapport JSON diagnostic (step6)'),
        # ─────────────────────────────────────────────────────
        ('data/rapport_final.txt',               'Rapport texte V3'),
    ]
    for path, desc in fichiers:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  {C.GREEN}✓{C.RESET} {path:<42} ({size:>6.0f} KB)  {desc}")
        else:
            print(f"  {C.YELLOW}·{C.RESET} {path:<42} (non généré)")

    print(f"\n  {C.BOLD}── Figures V3 ──────────────────────────────────{C.RESET}")
    if os.path.exists('figures'):
        for f in sorted(os.listdir('figures')):
            size = os.path.getsize(f'figures/{f}') / 1024
            print(f"  {C.GREEN}✓{C.RESET} figures/{f:<40} ({size:>5.0f} KB)")


def print_axes_summary():
    print(f"\n  {C.BOLD}{C.CYAN}── 7 AXES D'AMÉLIORATION ──────────────────────{C.RESET}")
    axes = [
        ("AXE 1", "AUC corrigée vs Alert_Status",     "data/validation_walkforward.csv"),
        ("AXE 2", "RUL Ensemble Poly+Exp+Weibull + IC","data/rul_summary.csv"),
        ("AXE 3", "+6 features vibratoires",           "data/02_features_motor.csv"),
        ("AXE 4", "Walk-forward cross-validation",     "data/validation_walkforward.csv"),
        ("AXE 5", "LOF novelty=True dans l'ensemble",  "data/03_anomalies.csv"),
        ("AXE 6", "SHAP explications par moteur",      "data/shap_per_motor.csv"),
        ("AXE 7", "CUSUM rupture de tendance",         "data/cusum_changepoints.csv"),
    ]
    for num, desc, path in axes:
        done  = '✓' if os.path.exists(path) else '·'
        color = C.GREEN if os.path.exists(path) else C.YELLOW
        print(f"  {color}{done}{C.RESET} {C.BOLD}{num}{C.RESET} — {desc}")

    # ── Résumé step6 ────────────────────────────────────────
    print(f"\n  {C.BOLD}{C.CYAN}── DIAGNOSTIC ROULEMENTS (step6) ───────────────{C.RESET}")
    diag_path = 'data/bearing_diagnosis.csv'
    json_path = 'bearing_diagnosis_report.json'
    if os.path.exists(diag_path) and os.path.exists(json_path):
        import json, pandas as pd
        try:
            df_d = pd.read_csv(diag_path)
            with open(json_path) as f:
                report = json.load(f)
            summary = report.get('summary', {})
            print(f"  {C.GREEN}✓{C.RESET} {C.BOLD}step6{C.RESET} — Diagnostic défauts de roulements")
            print(f"       Moteurs diagnostiqués : {len(df_d)}")
            print(f"       🔴 Critique : {summary.get('critique', 0)}  "
                  f"🟠 Élevé : {summary.get('élevé', 0)}  "
                  f"🟡 Modéré : {summary.get('modéré', 0)}  "
                  f"🟢 Faible : {summary.get('faible', 0)}")
            # Top priorités
            critiques = df_d[df_d['severity_label'].isin(['CRITIQUE', 'ÉLEVÉ'])]
            if not critiques.empty:
                print(f"\n  {C.RED}{C.BOLD}⚠ Moteurs prioritaires (diagnostic roulement) :{C.RESET}")
                for _, r in critiques.sort_values('severity', ascending=False).iterrows():
                    print(f"  {C.RED}  M{int(r['motor_id']):2d} | {r['fault_label_fr'][:35]:35s} | "
                          f"Sévérité : {r['severity_label']:8s} | "
                          f"Action dans : {r['days_to_action']}j{C.RESET}")
        except Exception:
            print(f"  {C.GREEN}✓{C.RESET} {C.BOLD}step6{C.RESET} — Fichiers générés")
    else:
        print(f"  {C.YELLOW}·{C.RESET} step6 — non exécuté ou step6 absent")


def main():
    print_header()

    parser = argparse.ArgumentParser(description='Pipeline Maintenance Prédictive V3')
    parser.add_argument('--step',             type=int,            help='Étape unique (1-7)')
    parser.add_argument('--from-step',        type=int, default=1, help='Depuis étape N')
    parser.add_argument('--skip-fulldata',    action='store_true', help='Ignorer step1b')
    parser.add_argument('--skip-autoencoder', action='store_true', help='Ignorer step3b')
    args = parser.parse_args()

    # NOTE : step6 n'est PAS une étape séparée du pipeline principal.
    # Il est appelé automatiquement à la fin de step3_anomaly_detection.py.
    steps = [
        (1, 'step1_extraction.py',        'Extraction SQL → motor_measurements'),
        (2, 'step1b_full_data.py',         'Intégration full_data (capteurs VWV)'),
        (3, 'step2_features.py',           'Feature Engineering V3 (+6 features AXE 3)'),
        (4, 'step3_anomaly_detection.py',  'Hybride V3 : AXE 1+4+5+6 + Diagnostic roulements'),
        (5, 'step3b_autoencoder.py',       'Autoencoder Deep Learning + Comparaison'),
        (6, 'step4_rul_prediction.py',     'RUL V3 : AXE 2+7 (Weibull+CUSUM)'),
        (7, 'step5_report.py',             'Rapport Final V3'),
    ]

    skip_nums = []
    if args.skip_fulldata:
        skip_nums.append(2)
        print(f"{C.YELLOW}⚠ --skip-fulldata : step1b ignoré{C.RESET}")
    if args.skip_autoencoder:
        skip_nums.append(5)
        print(f"{C.YELLOW}⚠ --skip-autoencoder : step3b ignoré{C.RESET}")

    steps_filtered = [s for s in steps if s[0] not in skip_nums]
    steps_renum    = [(i + 1, s[1], s[2]) for i, s in enumerate(steps_filtered)]

    os.makedirs('data',    exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print(f"  Répertoire : {os.getcwd()}")
    print(f"  Démarré le : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    if not check_prerequisites():
        sys.exit(1)

    if args.step:
        steps_to_run = [s for s in steps_renum if s[0] == args.step]
        if not steps_to_run:
            print(f"{C.RED}[ERREUR] Étape invalide : {args.step}{C.RESET}")
            sys.exit(1)
    else:
        steps_to_run = [s for s in steps_renum if s[0] >= args.from_step]

    print(f"\n  {'─'*55}")
    print(f"  Plan ({len(steps_to_run)} étapes) :")
    for num, script, desc in steps_to_run:
        print(f"    {num}. {desc}")
    print(f"  {'─'*55}\n")

    t0     = time.time()
    failed = False

    for step_num, script, description in steps_to_run:
        if not run_step(step_num, script, description):
            print(f"\n{C.RED}{C.BOLD}Pipeline interrompu à l'étape {step_num}.{C.RESET}")
            print(f"→ Relancez : python main_pipeline.py --from-step {step_num}")
            failed = True
            break

    total = time.time() - t0
    print(f"\n{'='*60}")

    if not failed:
        print(f"{C.GREEN}{C.BOLD}")
        print("  ✓ PIPELINE V3 COMPLET")
        print(f"{C.RESET}")
        print(f"  Durée totale : {total:.1f}s")
        print_axes_summary()
        print_fichiers_produits()

        # Résumé RUL si disponible
        if os.path.exists('data/rul_summary.csv'):
            import pandas as pd
            df_rul   = pd.read_csv('data/rul_summary.csv')
            critiques = df_rul[df_rul['risk_level'].isin(['CRITIQUE', 'ÉLEVÉ'])]
            if not critiques.empty:
                print(f"\n  {C.RED}{C.BOLD}⚠ Moteurs prioritaires (RUL) :{C.RESET}")
                for _, r in critiques.sort_values('current_di', ascending=False).iterrows():
                    rul = str(r.get('rul_days', r.get('rul_ensemble', '?')))
                    print(f"  {C.RED}  M{int(r['motor_id']):2d} | DI={r['current_di']:.3f} | "
                          f"RUL={rul}j | {r['risk_level']}{C.RESET}")
    else:
        print(f"{C.RED}  ✗ Pipeline incomplet{C.RESET}")
        sys.exit(1)

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
