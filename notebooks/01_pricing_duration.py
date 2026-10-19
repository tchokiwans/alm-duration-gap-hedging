"""
alm-duration-gap-hedging — notebooks/01_pricing_duration.py

Exploration : pricing du portefeuille, duration Macaulay/modifiee,
convexite, DV01 (analytique + numerique), et agregation au niveau
portefeuille (DV01 total, duration ponderee par la valeur).

A convertir en notebook Jupyter si tu preferes ce format — la logique
reutilisable vit dans src/, ce fichier ne fait qu'appeler et afficher.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
# chemin calcule depuis l'emplacement du fichier -> marche peu importe
# le dossier courant d'ou le script est lance (terminal, VS Code, etc.)

from datetime import date
from bonds import BONDS, NOMINAL, price_bond
from curve import zero_rate

VAL_DATE = date(2026, 9, 21)

# --- Tableau du portefeuille, obligation par obligation ---------------------

print(f"{'Obligation':<16}{'Mat(ans)':>9}{'Prix':>13}{'Macaulay':>10}{'Modifiee':>10}{'Convexite':>11}{'DV01':>10}")
results = []
total_price, total_dv01, total_mac_w, total_mod_w = 0.0, 0.0, 0.0, 0.0
for name, isin, coupon, maturity in BONDS:
    r = price_bond(coupon, maturity, VAL_DATE)
    r["name"], r["isin"] = name, isin
    mat_years = (maturity - VAL_DATE).days / 365.25
    results.append(r)
    total_price += r["price"]
    total_dv01 += r["dv01"]
    total_mac_w += r["macaulay"] * r["price"]
    total_mod_w += r["modified"] * r["price"]
    print(f"{name:<16}{mat_years:>9.2f}{r['price']:>13,.0f}{r['macaulay']:>10.2f}"
          f"{r['modified']:>10.2f}{r['convexity']:>11.2f}{r['dv01']:>10,.0f}")

# --- Agregation portefeuille --------------------------------------------

print(f"\n{'PORTEFEUILLE':<16}{'':>9}{total_price:>13,.0f}"
      f"{total_mac_w/total_price:>10.2f}{total_mod_w/total_price:>10.2f}{'':>11}{total_dv01:>10,.0f}")
print("(Macaulay et duration modifiee : moyennes ponderees par la valeur de marche."
      " DV01 : simple somme — c'est ce qui rend le DV01 pratique pour agreger un book.)")

# --- Validation ordre 1 vs ordre 2 (OAT 2.50% 2030) --------------------------

print("\nValidation (OAT 2.50% 2030), bump uniforme de la courbe :")
bond = results[1]
for bump_bp in [10, 50, 100, 200, 300]:
    bump = bump_bp / 10000
    bumped = sum(cf / (1 + zero_rate(t) + bump) ** t for _, t, cf in bond["cfs"])
    real_pct = (bumped / bond["price"] - 1) * 100
    lin_pct = -bond["modified"] * bump * 100
    quad_pct = (-bond["modified"] * bump + 0.5 * bond["convexity"] * bump**2) * 100
    print(f"  +{bump_bp:>3}bp : reel = {real_pct:+.3f}%   "
          f"duration seule = {lin_pct:+.3f}% (ecart {real_pct - lin_pct:+.4f})   "
          f"duration+convexite = {quad_pct:+.3f}% (ecart {real_pct - quad_pct:+.4f})")

# --- Validation DV01 : analytique vs numerique (repricing complet) ----------

print("\nValidation DV01, toutes obligations (analytique = D_mod x Prix x 1bp) :")
for r in results:
    p_up = sum(cf / (1 + zero_rate(t) + 0.00005) ** t for _, t, cf in r["cfs"])
    p_down = sum(cf / (1 + zero_rate(t) - 0.00005) ** t for _, t, cf in r["cfs"])
    dv01_numerique = p_down - p_up
    print(f"  {r['name']:<16} analytique = {r['dv01']:>8,.0f}   "
          f"numerique = {dv01_numerique:>8,.0f}   ecart = {r['dv01']-dv01_numerique:+.1f}")