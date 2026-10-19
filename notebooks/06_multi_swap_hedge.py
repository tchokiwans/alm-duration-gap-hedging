"""
alm-duration-gap-hedging — notebooks/06_multi_swap_hedge.py

Etape 6 : le swap 10 ans (notebook 05) laisse deux residus : une COURBURE
(perte dans les deux sens) et un risque de FORME de courbe (le passif est
long, l'actif est au milieu). On les traite avec plusieurs swaps de
maturites differentes.

  Strategie A : aucune couverture
  Strategie B : 1 swap 10 ans, calibre sur le DV01 (notebook 05)
  Strategie C : 2 swaps (5 ans + 20 ans), calibres sur le DV01 ET la courbure
  Strategie E : 5 swaps (1, 2, 5, 10, 20 ans), calibres pivot par pivot (KR01)
  (La strategie D, avec options, vient a l'Etape 7.)

Vocabulaire
  KR01        = perte (EUR) si le taux d'UN pivot monte de 1bp, les autres fixes
  receveur    = on recoit le taux fixe (notionnel > 0) ; payeur : notionnel < 0
  conditionnement (cond) = sensibilite d'un systeme lineaire aux petites erreurs :
                un cond de 10 est sain ; 1e12 signifie que la solution est du bruit.
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
FIG_DIR = os.path.join(HERE, "..", "figures")

import numpy as np

from curve import RISK_PIVOTS, curve_source, get_par_curve
from hedging import discount_factor, solve_swap_hedge, swap_kr01_matrix
from scenarios import (VAL_DATE, balance_sheet, curvature, dv01_hedge_notional,
                       make_swaps, parallel, shocked_zero_curve,
                       solve_dv01_curvature_hedge, stress_table, surplus_change,
                       surplus_exposure)

SHOW = False  # True : ouvre aussi les fenetres des graphiques


def check(label: str, ok: bool) -> None:
    print(f"  [{'OK' if ok else 'ECHEC'}] {label}")


print("Chargement de la courbe de taux (jusqu'a 15 s si le site de la Banque de France est lent)...", flush=True)
print("Courbe utilisee :", curve_source())
par_t, par_y = get_par_curve()
actif = balance_sheet()["actif"]
expo = surplus_exposure()  # variation du surplus pour +1bp a chaque pivot, sans couverture

# --- 1. L'empreinte de chaque swap ------------------------------------------------------

MATS = [1, 2, 5, 10, 20]
M = swap_kr01_matrix(MATS, par_t, par_y, VAL_DATE)  # M[pivot, swap] pour 1 000 000 EUR

print("\n=== 1. L'empreinte de chaque swap : KR01 par 1 000 000 EUR de notionnel receveur ===")
print(f"{'Pivot':>9}" + "".join(f"{f'swap {T}a':>10}" for T in MATS) + f"{'A combler':>13}")
M_show = np.round(M, 1) + 0.0  # + 0.0 : evite l'affichage "-0.0"
for i, p in enumerate(RISK_PIVOTS):
    print(f"{p:>5} ans" + "".join(f"{M_show[i, j]:>10,.1f}" for j in range(len(MATS))) + f"{expo[i]:>+13,.0f}")
print("\nChaque colonne dit OU un swap est sensible sur la courbe. 'A combler' = variation du surplus")
print("pour +1bp a chaque pivot. On cherche les notionnels n (en millions d'EUR) tels que :")
print("      M x n = A combler        (5 equations, une par pivot ; 5 inconnues, une par swap)")
print("c'est-a-dire : la PERTE des swaps pour +1bp compense, pivot par pivot, le GAIN du surplus.")

print("\nRemarque : la matrice est presque diagonale. Pourquoi ? Un swap AU PAIR de maturite T vaut,")
print("apres un choc :   V = (K - taux par(T) apres choc) x annuite(T)")
print("Il 'rend' K alors que le marche rend maintenant par(T) : il ne depend QUE du taux par a la")
print("maturite T. Le swap 5 ans ne voit que le pivot 5 ans, etc. Le 20 ans, situe entre les pivots")
print("10 et 30 ans, se repartit a parts egales entre les deux. Annuite(T) = somme des DF(1..T).")
zt0, zr0 = shocked_zero_curve()
annuity = {T: sum(discount_factor(t, zt0, zr0) for t in range(1, T + 1)) for T in MATS}
pivots = list(RISK_PIVOTS)
diag_ok = all(np.isclose(M[pivots.index(T), j], annuity[T] * 100, rtol=2e-3)
              for j, T in enumerate(MATS) if T in pivots)
split_ok = np.isclose(M[pivots.index(10), MATS.index(20)] + M[pivots.index(30), MATS.index(20)],
                      annuity[20] * 100, rtol=2e-3)
check("KR01 du swap T au pivot T = annuite(T) x 1 000 000 EUR x 1bp (a 0,2 % pres), T = 1, 2, 5, 10",
      diag_ok)
check("KR01 du swap 20 ans : somme sur les pivots 10 et 30 ans = annuite(20) x 1 000 000 EUR x 1bp",
      split_ok)

# --- 2. Quelles maturites choisir ? --------------------------------------------------------

print("\n=== 2. Quelles maturites de swaps choisir ? ===")
print(f"{'Maturites (ans)':<18}{'cond(M)':>10}{'Brut (M EUR)':>14}{'Pire KR01 residuel':>20}  Verdict")
for mats in ([1, 2, 5, 10, 20], [2, 5, 10, 20], [1, 3, 5, 7, 10], [2, 5, 10, 15, 20]):
    n, Mc = solve_swap_hedge(expo, mats, par_t, par_y, VAL_DATE)
    resid = float(np.abs(expo - Mc @ (n / 1e6)).max())
    cond = np.linalg.cond(Mc)
    gross = np.abs(n).sum() / 1e6
    if not np.isfinite(cond) or resid > 1_000:
        verdict = "un pivot est hors de portee : risque residuel"
    elif cond > 1e6:
        verdict = "mal conditionne : notionnels absurdes"
    else:
        verdict = "exploitable"
    gross_txt = f"{gross:,.1f}" if gross < 1e6 else f"{gross:.1e}"
    print(f"{str(mats):<18}{cond:>10.3g}{gross_txt:>14}{resid:>20,.1f}  {verdict}")
print("\n  [1,2,5,10,20]   : un swap par pivot (le 20 ans couvre la fin du passif, 18 ans) -> retenu.")
print(f"  [2,5,10,20]     : on abandonne le pivot 1 an ({abs(expo[0]):,.0f} EUR/bp seulement) : un swap de moins, residu tolere.")
print("  [1,3,5,7,10]    : aucun swap n'atteint le pivot 30 ans, qui porte l'exposition du passif long.")
print("  [2,5,10,15,20]  : les swaps 10, 15 et 20 ans n'agissent que sur DEUX pivots (10 et 30 ans) :")
print("                    trois swaps pour deux degres de liberte, l'un est redondant. Le systeme est")
print("                    singulier ; sa solution 'exacte' est un empilement absurde de positions geantes")
print("                    qui se compensent. Regle : pas plus de swaps que de pivots a couvrir par zone.")

# --- 3. Les quatre strategies ----------------------------------------------------------------

n_b = dv01_hedge_notional(10)
n_c = solve_dv01_curvature_hedge([5, 20])
n_e, _ = solve_swap_hedge(expo, MATS, par_t, par_y, VAL_DATE)

strategies = {
    "A. Aucune": [],
    "B. Swap 10 ans": make_swaps([10], [n_b]),
    "C. 5 ans + 20 ans": make_swaps([5, 20], n_c),
    "E. 5 swaps (KR01)": make_swaps(MATS, n_e),
}

print("\n=== 3. Composition des strategies (+ = receveur, - = payeur) ===")
for name, swaps in strategies.items():
    if not swaps:
        print(f"{name:<20}aucune position")
        continue
    legs = ", ".join(f"{N / 1e6:+.2f} M a {T} ans" for T, K, N in swaps)
    gross = sum(abs(N) for _, _, N in swaps)
    print(f"{name:<20}{legs}")
    print(f"{'':<20}notionnel brut = {gross / 1e6:.1f} M EUR, soit {gross / actif:.0%} de l'actif")
print("\nB annule le DV01. C annule le DV01 ET la courbure (systeme 2x2 : un swap long apporte la")
print("courbure, un swap court corrige le DV01). E annule le KR01 a chaque pivot (systeme 5x5).")

# --- 4. Test sur la batterie de chocs ------------------------------------------------------------

table = stress_table(strategies)
shown = table.copy()
shown.loc["PIRE CAS"] = table.min()
print("\n=== 4. Variation du surplus (EUR) sous chaque choc ===")
print(shown.to_string(float_format=lambda v: f"{v:,.0f}"))
print(f"\nSurplus de depart : {balance_sheet()['surplus']:,.0f} EUR.")

# --- 5. Risque residuel -----------------------------------------------------------------------------

print("\n=== 5. Risque qui reste ===")
print("a) Variation du surplus pour +1bp a chaque pivot (EUR) -- 0 = pivot couvert")
resid = {name: surplus_exposure(swaps) for name, swaps in strategies.items()}
print(f"{'Pivot':>9}" + "".join(f"{name:>20}" for name in strategies))
for i, p in enumerate(RISK_PIVOTS):
    print(f"{p:>5} ans" + "".join(f"{np.round(resid[name][i]) + 0.0:>+20,.0f}" for name in strategies))

print("\nb) Mesures globales (choc parallele)")
print(f"{'':<20}{'Surplus pour +1bp':>20}{'Courbure +/-100bp':>20}")
for name, swaps in strategies.items():
    print(f"{name:<20}{np.round(surplus_change(parallel(1), swaps), 1) + 0.0:>+20,.1f}"
          f"{np.round(curvature(swaps, 100)) + 0.0:>+20,.0f}")

worst = table.min()
print("\nPire perte sur la batterie :")
for name in strategies:
    print(f"  {name:<20}{worst[name]:>14,.0f} EUR")

# --- 6. Ce que ces resultats montrent... et ne montrent pas -------------------------------------------

print("\n=== 6. Ce que le modele montre, et ce qu'il ne permet PAS de conclure ===")
print(f"Le modele montre : avec 5 swaps, le risque de taux est annule pivot par pivot ; la pire")
print(f"perte de la batterie tombe a {abs(worst['E. 5 swaps (KR01)']):,.0f} EUR (contre {abs(worst['A. Aucune']):,.0f} sans couverture).")
print("Il ne permet PAS de conclure que le risque est nul :")
print("  - Test 'dans l'echantillon' : les chocs sont construits sur les memes 5 pivots que la")
print("    couverture. Un choc que la courbe a 5 pivots ne sait pas decrire (ex. +50bp sur le 7 ans")
print("    seul) n'est pas teste.")
print("  - Une seule courbe pour actualiser l'actif ET les swaps : aucun risque de base (l'ecart")
print("    OAT / swap peut bouger en vrai).")
print("  - Passif deterministe : pas de rachats dependant des taux, pas d'option cachee.")
print(f"  - Aucun cout : marge (collateral), contrepartie, ecarts de prix, renouvellement. La strategie E")
print(f"    exige un notionnel brut de {sum(abs(N) for _, _, N in strategies['E. 5 swaps (KR01)']) / actif:.0%} de l'actif, "
      "avec de grosses positions qui se compensent.")

# --- 7. Graphiques -------------------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("\n(matplotlib absent : `pip install matplotlib` pour generer les graphiques)")

if plt is not None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    colors = {"A. Aucune": "#d62728", "B. Swap 10 ans": "#ff7f0e",
              "C. 5 ans + 20 ans": "#1f77b4", "E. 5 swaps (KR01)": "#2ca02c"}
    os.makedirs(FIG_DIR, exist_ok=True)

    # Figure 1 : variation du surplus selon le choc parallele + pire perte (echelle log)
    shocks = list(range(-300, 301, 25))
    fine = stress_table(strategies, {f"{s:+d}": parallel(s) for s in shocks})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for name in strategies:
        ax1.plot(shocks, fine[name] / 1e6, lw=2, color=colors[name], label=name)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Choc parallèle sur la courbe (bp)")
    ax1.set_ylabel("Variation du surplus (M€)")
    ax1.set_title("Chaque couverture aplatit un peu plus la courbe")
    ax1.legend(frameon=False, fontsize=8)

    names = list(strategies)
    losses = [abs(worst[n]) for n in names]
    ax2.bar(range(len(names)), losses, color=[colors[n] for n in names])
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.split(".")[0] for n in names])
    ax2.set_ylabel("Pire perte sur les 12 chocs (€, échelle log)")
    ax2.set_title("Pire perte de la batterie")
    for i, v in enumerate(losses):
        ax2.text(i, v * 1.15, f"{v:,.0f}", ha="center", fontsize=8)
    ax2.set_ylim(top=max(losses) * 3)
    path1 = os.path.join(FIG_DIR, "06_comparaison_strategies.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"\nGraphique enregistre : {os.path.abspath(path1)}")
    if SHOW:
        plt.show()
    plt.close(fig)

    # Figure 2 : risque residuel pivot par pivot
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(RISK_PIVOTS))
    w = 0.2
    for k, name in enumerate(strategies):
        ax.bar(x + (k - 1.5) * w, resid[name] / 1e3, w, color=colors[name], label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} ans" for p in RISK_PIVOTS])
    ax.set_ylabel("Variation du surplus pour +1bp au pivot (k€)")
    ax.set_title("Le risque de taux qui reste, pivot par pivot")
    ax.legend(frameon=False, fontsize=8)
    path2 = os.path.join(FIG_DIR, "06_kr01_residuel.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Graphique enregistre : {os.path.abspath(path2)}")
    if SHOW:
        plt.show()
    plt.close(fig)