"""
========================================================
 PIPELINE PRINCIPAL — MAINTENANCE PRÉDICTIVE
 Orchestrateur complet : Extraction → Features →
   Anomalies → RUL → Rapport

 Usage :
   python main_pipeline.py                  # pipeline complet
   python main_pipeline.py --step 3        # étape spécifique
   python main_pipeline.py --from-step 2   # depuis une étape
========================================================
"""

import os
import sys
import time
import argparse
from datetime import datetime


# ── Couleurs console ───────────────────────────────────
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
╔══════════════════════════════════════════════════════╗
║     PIPELINE IA/ML — MAINTENANCE PRÉDICTIVE          ║
║     Moteurs industriels | Approche non supervisée    ║
╚══════════════════════════════════════════════════════╝
{C.RESET}""")


def run_step(step_num: int, script: str, description: str) -> bool:
    """Exécute une étape du pipeline et mesure le temps."""
    print(f"\n{C.BOLD}{'─'*55}")
    print(f"  ÉTAPE {step_num} — {description}")
    print(f"{'─'*55}{C.RESET}")

    if not os.path.exists(script):
        print(f"{C.RED}[ERREUR] Script introuvable : {script}{C.RESET}")
        return False

    start = time.time()
    ret   = os.system(f'python "{script}"')
    elapsed = time.time() - start

    if ret == 0:
        print(f"\n{C.GREEN}✓ Étape {step_num} terminée en {elapsed:.1f}s{C.RESET}")
        return True
    else:
        print(f"\n{C.RED}✗ Étape {step_num} échouée (code={ret}){C.RESET}")
        return False


def check_prerequisites():
    """Vérifie que le fichier SQL source est présent."""
    sql_files = [f for f in os.listdir('.') if f.endswith('.sql')]
    if not sql_files:
        print(f"{C.RED}[ERREUR] Aucun fichier .sql trouvé dans le répertoire courant.{C.RESET}")
        print("→ Copiez votre fichier SQL ici, puis relancez.")
        return False

    # Cherche le bon fichier SQL
    expected = 'ai_cp (2).sql'
    if os.path.exists(expected):
        print(f"{C.GREEN}✓ Fichier SQL trouvé : {expected}{C.RESET}")
    else:
        print(f"{C.YELLOW}⚠ Fichier SQL trouvé : {sql_files[0]}{C.RESET}")
        print(f"  Attendu : '{expected}'")
        print(f"  Renommage automatique...")
        try:
            os.rename(sql_files[0], expected)
            print(f"{C.GREEN}  ✓ Renommé{C.RESET}")
        except Exception as e:
            print(f"{C.RED}  ✗ Impossible de renommer : {e}{C.RESET}")
            print(f"  → Modifiez SQL_FILE dans step1_extraction.py")

    return True


def main():
    print_header()

    # ── Arguments CLI ─────────────────────────────
    parser = argparse.ArgumentParser(description='Pipeline Maintenance Prédictive')
    parser.add_argument('--step', type=int, default=None,
                        help='Exécuter uniquement cette étape (1-5)')
    parser.add_argument('--from-step', type=int, default=1,
                        help='Démarrer à partir de cette étape (défaut=1)')
    args = parser.parse_args()

    # Définition des étapes
    steps = [
        (1, 'step1_extraction.py',        'Extraction SQL → CSV brut'),
        (2, 'step2_features.py',           'Nettoyage + Feature Engineering'),
        (3, 'step3_anomaly_detection.py',  'Détection d\'anomalies (IF + LOF)'),
        (4, 'step4_rul_prediction.py',     'Prédiction RUL'),
        (5, 'step5_report.py',             'Rapport final'),
    ]

    # Créer les répertoires nécessaires
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print(f"  Répertoire de travail : {os.getcwd()}")
    print(f"  Démarré le           : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    # Vérification prérequis
    if not check_prerequisites():
        sys.exit(1)

    # Sélection des étapes à exécuter
    if args.step:
        steps_to_run = [s for s in steps if s[0] == args.step]
        if not steps_to_run:
            print(f"{C.RED}[ERREUR] Étape invalide : {args.step} (valeurs 1-5){C.RESET}")
            sys.exit(1)
    else:
        steps_to_run = [s for s in steps if s[0] >= args.from_step]

    print(f"\n  Étapes à exécuter : {[s[0] for s in steps_to_run]}\n")

    # ── Exécution du pipeline ─────────────────────
    pipeline_start = time.time()
    failed         = False

    for step_num, script, description in steps_to_run:
        success = run_step(step_num, script, description)
        if not success:
            print(f"\n{C.RED}{C.BOLD}Pipeline interrompu à l'étape {step_num}.{C.RESET}")
            print("→ Corrigez l'erreur et relancez avec --from-step " + str(step_num))
            failed = True
            break

    # ── Résumé final ─────────────────────────────
    total_time = time.time() - pipeline_start
    print(f"\n{'='*55}")

    if not failed:
        print(f"{C.GREEN}{C.BOLD}")
        print("  ✓ PIPELINE COMPLET — Tous les résultats sont prêts !")
        print(f"{C.RESET}")
        print(f"  Durée totale : {total_time:.1f}s")
        print(f"\n  Fichiers produits :")
        output_files = [
            ('data/01_raw_motor.csv',      'Données brutes'),
            ('data/02_features_motor.csv', 'Features engineering'),
            ('data/03_anomalies.csv',      'Résultats anomalies'),
            ('data/04_rul_results.csv',    'Résultats RUL'),
            ('data/rul_summary.csv',       'Résumé RUL'),
            ('data/rapport_final.txt',     'Rapport texte'),
        ]
        for path, desc in output_files:
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024
                print(f"  {C.GREEN}✓{C.RESET} {path:<35} {desc} ({size:.0f} KB)")
            else:
                print(f"  {C.YELLOW}?{C.RESET} {path:<35} (non généré)")

        print(f"\n  Figures sauvegardées dans : figures/")
        for f in sorted(os.listdir('figures')) if os.path.exists('figures') else []:
            print(f"    • {f}")
    else:
        print(f"{C.RED}  ✗ Pipeline incomplet — vérifiez les erreurs ci-dessus{C.RESET}")
        sys.exit(1)

    print(f"\n{'='*55}\n")


if __name__ == '__main__':
    main()
