"""
alm-duration-gap-hedging — notebooks/05_swap_hedge.py

Etape 5 : premiere couverture — UN swap receveur 10 ans, dimensionne sur le DV01.

  1. Anatomie du swap : jambe fixe, jambe flottante, valeur nulle a l'origine.
  2. Pourquoi son DV01 est celui d'une obligation 10 ans.
  3. Dimensionnement : quel notionnel annule le DV01 de bilan ?
  4. Test de la couverture sur des chocs paralleles de -300 a +300bp.
  5. Ce qui reste : la COURBURE. D'ou elle vient, et l'indice pour l'Etape 6.
  6. Graphique : figures/05_swap_10y_courbure.png

Vocabulaire
  PV (present value)     = valeur actuelle : ce que vaut AUJOURD'HUI un flux futur
  DF (discount factor)   = facteur d'actualisation : DF(t) = (1 + taux zero(t)) ** (-t),
                           la valeur aujourd'hui de 1 EUR recu dans t ans
  notionnel              = montant de reference sur lequel les interets sont calcules.
                           Dans un swap, il n'est JAMAIS echange.
  receveur / payeur      = on recoit / on paie le taux FIXE (et l'inverse pour le flottant)
"""

import sys
import os
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
FIG_DIR = os.path.join(HERE, "..", "figures")

import numpy as np

from curve import curve_source
from hedging import (discount_factor, floating_leg_value, par_swap_rate,
                     swap_fixed_leg_cfs)
from scenarios import (VAL_DATE, balance_sheet, dv01_hedge_notional, make_swaps,
                       parallel, parallel_dv01, shocked_zero_curve, stress_table,
                       surplus_change, swap_profile)
from sensitivity import price_with_curve

SHOW = False  # True : ouvre aussi la fenetre du graphique
TENOR = 10
UNIT = 1_000_000


def check(label: str, ok: bool) -> None:
    print(f"  [{'OK' if ok else 'ECHEC'}] {label}")


print("Chargement de la courbe de taux (jusqu'a 15 s si le site de la Banque de France est lent)...", flush=True)
print("Courbe utilisee :", curve_source())

# --- 1. Anatomie d'un swap receveur 10 ans --------------------------------------

zt, zr = shocked_zero_curve()  # courbe zero-coupon de depart
K = par_swap_rate(TENOR, zt, zr)
fixed_cfs = swap_fixed_leg_cfs(K, UNIT, TENOR, VAL_DATE)
df_T = discount_factor(TENOR, zt, zr)
annuity = sum(discount_factor(t, zt, zr) for t in range(1, TENOR + 1))
pv_fixed = price_with_curve(fixed_cfs, zt, zr)
pv_float = floating_leg_value(UNIT, TENOR, zt, zr)

print(f"\n=== 1. Anatomie d'un swap receveur {TENOR} ans, notionnel {UNIT:,} EUR ===")
print(f"DF({TENOR})  = {df_T:.4f}   -> 1 EUR recu dans {TENOR} ans vaut {df_T:.4f} EUR aujourd'hui")
print(f"Annuite = somme des DF(1..{TENOR}) = {annuity:.4f}   (valeur aujourd'hui de 1 EUR recu chaque annee)")
print(f"Taux fixe d'equilibre K = (1 - DF({TENOR})) / annuite = {K:.4%}")
print(f"PV jambe fixe      = K x N x annuite   = {pv_fixed:>12,.2f} EUR   (on RECOIT ces coupons)")
print(f"PV jambe flottante = N x (1 - DF({TENOR})) = {pv_float:>12,.2f} EUR   (on PAIE le flottant)")
print(f"Valeur du swap = PV fixe - PV flottante    = {round(pv_fixed - pv_float, 2) + 0.0:>12,.2f} EUR")
check("valeur du swap nulle a la signature (c'est la definition du taux 'au pair')",
      abs(pv_fixed - pv_float) < 1e-6)

# --- 2. Un swap = une obligation au pair - son nominal ------------------------------

bond_cfs = fixed_cfs + [(date(VAL_DATE.year + TENOR, VAL_DATE.month, VAL_DATE.day), float(TENOR), UNIT)]
pv_bond = price_with_curve(bond_cfs, zt, zr)


def bond_value(bp: float) -> float:
    z_t, z_r = shocked_zero_curve(parallel(bp))
    return price_with_curve(bond_cfs, z_t, z_r)


dv01_bond = bond_value(0) - bond_value(1)
curv_bond = (bond_value(100) + bond_value(-100)) / 2 - bond_value(0)
dv01_swap, curv_swap = swap_profile(TENOR, UNIT)

print("\n=== 2. Un swap receveur = obligation au pair - son nominal ===")
print(f"Obligation {TENOR} ans, coupon K = {K:.3%}, remboursee au pair : prix = {pv_bond:,.2f} EUR")
print("(= le nominal : c'est exactement ce que veut dire 'taux par'.)")
print("Jambe fixe + principal = obligation ; jambe flottante = N - N x DF(T) ;")
print("donc swap = [obligation] - N. Soustraire une CONSTANTE (N) ne change aucune sensibilite.")
print(f"  DV01 du swap         : {dv01_swap:>8.2f} EUR par {UNIT:,} EUR de notionnel")
print(f"  DV01 de l'obligation : {dv01_bond:>8.2f} EUR")
check("le swap a exactement le meme DV01 que l'obligation", np.isclose(dv01_swap, dv01_bond, rtol=1e-9))
approx = annuity * UNIT * 0.0001
print("Formule : un swap au pair vaut (K - taux par(T) actuel) x annuite, donc pour +1bp :")
print(f"  DV01 ~ annuite x notionnel x 1bp = {annuity:.4f} x {UNIT:,} x 0.0001 = {approx:,.2f} EUR")
check("DV01 mesure = annuite x notionnel x 1bp (a 0,1 % pres)", abs(dv01_swap / approx - 1) < 1e-3)

# --- 3. Dimensionnement ----------------------------------------------------------------

dv = parallel_dv01()
gap = dv["passif"] - dv["actif"]
notional = round(dv01_hedge_notional(TENOR), -2)
swaps_b = make_swaps([TENOR], [notional])
total_assets = balance_sheet()["actif"]

print("\n=== 3. Dimensionnement : quel notionnel annule le DV01 de bilan ? ===")
print(f"Variation du surplus pour +1bp, sans swap : {gap:>+10,.0f} EUR   <- le 'trou' a combler")
print(f"DV01 d'un swap receveur {TENOR} ans, par {UNIT:,} EUR de notionnel : {dv01_swap:,.2f} EUR")
print(f"Notionnel = {gap:,.0f} / {dv01_swap:,.2f} x {UNIT:,} = {notional:,.0f} EUR "
      f"({notional / total_assets:.0%} de l'actif)")
after = surplus_change(parallel(1), swaps_b)
print(f"Variation du surplus pour +1bp, AVEC le swap : {round(after, 2) + 0.0:>+10,.2f} EUR")
check("swap sans valeur a la signature", abs(balance_sheet(swaps=swaps_b)["swaps"]) < 1e-3)
check("DV01 de bilan neutralise (|variation pour +1bp| < 1 EUR)", abs(after) < 1.0)

# --- 4. Test sur des chocs paralleles ------------------------------------------------

grid = [-300, -200, -100, -50, 50, 100, 200, 300]
table = stress_table({"Sans swap": [], f"Swap {TENOR} ans": swaps_b},
                     {f"{s:+d}bp": parallel(s) for s in grid})
print("\n=== 4. Variation du surplus (EUR) apres un choc PARALLELE ===")
print(table.to_string(float_format=lambda v: f"{v:>14,.0f}"))
print(f"\nPire perte : sans swap {table['Sans swap'].min():,.0f} EUR ; "
      f"avec swap {table[f'Swap {TENOR} ans'].min():,.0f} EUR.")
print("Le swap aplatit presque tout : pour un petit choc, la perte devient negligeable.")
print("Mais regarde le signe : avec le swap, on perd des deux cotes (hausse ET baisse).")

# --- 5. Ce qui reste : la courbure ----------------------------------------------------

print("\n=== 5. La perte residuelle : une courbure (cloche) ===")
col = f"Swap {TENOR} ans"
print("On coupe la variation en deux :")
print("  partie PAIRE   = [dS(+s) + dS(-s)] / 2  : ce qu'on perd (ou gagne) dans les DEUX sens")
print("  partie IMPAIRE = [dS(+s) - dS(-s)] / 2  : l'asymetrie (pente residuelle, effets d'ordre 3)")
print(f"\n{'Choc +/-':>9}{'Partie paire':>15}{'Partie impaire':>16}{'Paire / paire(100)':>20}{'(s/100)^2':>11}")
even_100 = (table.loc["+100bp", col] + table.loc["-100bp", col]) / 2
for s in (100, 200, 300):
    up, down = table.loc[f"{s:+d}bp", col], table.loc[f"{-s:+d}bp", col]
    even, odd = (up + down) / 2, (up - down) / 2
    print(f"{s:>7}bp{even:>15,.0f}{odd:>16,.0f}{even / even_100:>20.2f}{(s / 100) ** 2:>11.2f}")
print("\nSi la perte est bien 'quadratique', doubler le choc la multiplie par 4 (colonne de droite).")
print("Pourquoi une cloche ? Apres le swap, actif couvert et passif ont la MEME pente en taux")
print("(tangentes confondues), mais le passif est plus 'courbe' : il s'ecarte plus de sa tangente")
print("que l'actif. Donc, dans les DEUX sens, l'ecart (passif - actif couvert) augmente : le surplus baisse.")

print("\nCourbure par 1 000 000 EUR de swap receveur (choc parallele +/-100bp) :")
print(f"{'Instrument':<30}{'DV01':>10}{'Courbure':>12}{'Courbure/DV01':>16}")
for T in (5, 10, 20):
    d, c = swap_profile(T, UNIT)
    print(f"{f'Swap receveur {T} ans':<30}{d:>10,.1f}{c:>12,.0f}{c / d:>16.1f}")
print(f"{f'Obligation {TENOR} ans, coupon K':<30}{dv01_bond:>10,.1f}{curv_bond:>12,.0f}{curv_bond / dv01_bond:>16.1f}")
check("courbure du swap = courbure de l'obligation : un swap n'est PAS lineaire",
      np.isclose(curv_swap, curv_bond, rtol=1e-9))
print("\nIndice pour l'Etape 6 : le rapport Courbure/DV01 CROIT avec la maturite. Un swap long")
print("apporte beaucoup de courbure pour peu de DV01 ; un swap court l'inverse. En les combinant,")
print("on peut regler le DV01 ET la courbure separement.")

# --- 6. Graphique ------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except ImportError:
    plt = None
    print("\n(matplotlib absent : `pip install matplotlib` pour generer le graphique)")

if plt is not None:
    shocks = list(range(-300, 301, 25))
    fine = stress_table({"Sans swap": [], col: swaps_b}, {f"{s:+d}": parallel(s) for s in shocks})
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax1.plot(shocks, fine["Sans swap"] / 1e6, color="#d62728", lw=2, label="Sans couverture")
    ax1.plot(shocks, fine[col] / 1e6, color="#1f77b4", lw=2, label=f"Avec swap {TENOR} ans")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Choc parallèle sur la courbe (bp)")
    ax1.set_ylabel("Variation du surplus (M€)")
    ax1.set_title("Le swap annule la pente...")
    ax1.legend(frameon=False)

    ax2.plot(shocks, fine[col] / 1e3, color="#1f77b4", lw=2, label=f"Avec swap {TENOR} ans")
    ax2.plot(shocks, even_100 * (np.array(shocks) / 100) ** 2 / 1e3, color="gray", ls="--",
             label="Parabole de courbure seule")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Choc parallèle sur la courbe (bp)")
    ax2.set_ylabel("Variation du surplus (k€)")
    ax2.set_title("... mais laisse une cloche (courbure)")
    ax2.legend(frameon=False)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "05_swap_10y_courbure.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nGraphique enregistre : {os.path.abspath(path)}")
    if SHOW:
        plt.show()
    plt.close(fig)