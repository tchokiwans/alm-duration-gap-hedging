"""
alm-duration-gap-hedging — notebooks/02_curve_bootstrap.py

Validation du bootstrap : compare le rendement PAR interpole (l'entree)
au taux zero-coupon obtenu (la sortie) pour chaque annee. Sert a verifier
que le bootstrap fait bien quelque chose de coherent avant de l'utiliser
pour pricer le portefeuille et le passif.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
# chemin calcule depuis l'emplacement du fichier -> marche peu importe
# le dossier courant d'ou le script est lance (terminal, VS Code, etc.)

import numpy as np
from curve import bootstrap_zero_curve, MAX_TENOR

# Rendements PAR de secours (utilises tant que fetch_bdf_curve() n'est pas
# verifie en conditions reelles) — ancrage reel : OAT 10 ans ~4.10%, debut
# septembre 2026.
PAR_TENORS = np.array([1, 2, 5, 10, 30])
PAR_YIELDS = np.array([0.0320, 0.0350, 0.0385, 0.0410, 0.0435])

zero_tenors, zero_rates = bootstrap_zero_curve(PAR_TENORS, PAR_YIELDS, MAX_TENOR)

print(f"{'T':>3}{'Par (interpole)':>17}{'Zero (bootstrap)':>18}{'Ecart':>10}")
for T in zero_tenors.astype(int):
    par_i = np.interp(T, PAR_TENORS, PAR_YIELDS)
    z_i = zero_rates[T - 1]
    print(f"{T:>3}{par_i*100:>16.3f}%{z_i*100:>17.3f}%{(z_i-par_i)*10000:>9.2f} bp")

print("\nSur courbe croissante : zero > par a chaque maturite (attendu — "
      "le rendement par 'moyenne' des flux actualises a des taux plus bas "
      "que le taux terminal, ce qui le tire sous le vrai taux zero).")