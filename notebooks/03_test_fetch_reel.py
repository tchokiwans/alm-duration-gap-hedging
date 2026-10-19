"""
alm-duration-gap-hedging — notebooks/03_test_fetch_reel.py

Test isole de fetch_bdf_curve() : celle qui a besoin d'un vrai acces
reseau vers banque-france.fr, que je n'ai pas pu tester moi-meme.

Lance ce script (`python 03_test_fetch_reel.py`, peu importe le dossier
courant) et copie-moi tout ce qu'il affiche — que ce soit un succes ou
une erreur, c'est exactement l'info dont j'ai besoin pour ajuster
curve.py si necessaire.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from curve import fetch_bdf_curve

print("Appel de fetch_bdf_curve() en cours...\n")

try:
    tenors, yields = fetch_bdf_curve()
    print(">>> SUCCES <<<")
    print("Tenors (annees) :", tenors)
    print("Rendements (%)  :", yields * 100)
except Exception as e:
    print(">>> ECHEC <<<")
    print(f"Type d'erreur : {type(e).__name__}")
    print(f"Message       : {e}")
    print("\nTraceback complet (a copier-coller integralement) :")
    import traceback
    traceback.print_exc()