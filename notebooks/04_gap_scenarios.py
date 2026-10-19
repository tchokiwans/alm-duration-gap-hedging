"""
alm-duration-gap-hedging — notebooks/04_gap_scenarios.py

Etape 4 : OU est le risque de taux du bilan, et que se passe-t-il si la
courbe bouge, SANS aucune couverture ?

  1. Bilan de depart : valeur de l'actif, du passif, surplus.
  2. DV01 "bilan" et gap de duration : le risque resume en un chiffre.
  3. KR01 : le meme risque, pivot par pivot (1, 2, 5, 10, 30 ans).
  4. Scenarios de choc : parallele, pentification, aplatissement.
  5. Graphique : figures/04_kr01_actif_passif.png

Rappel de vocabulaire
  bp   = point de base = 0,01 % = 0,0001
  DV01 = perte (EUR) si TOUS les taux montent de 1bp
  KR01 = perte (EUR) si le taux d'UN SEUL pivot monte de 1bp
  surplus = actif - passif : ce qui reste pour les fonds propres
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
# chemin calcule depuis l'emplacement du fichier -> marche peu importe
# le dossier courant d'ou le script est lance (terminal, VS Code, etc.)
FIG_DIR = os.path.join(HERE, "..", "figures")

import numpy as np

from curve import RISK_PIVOTS, curve_source
from scenarios import (SCENARIOS, balance_sheet, kr01_actif, kr01_passif,
                       parallel_dv01, surplus_exposure)

SHOW = False  # True : ouvre aussi la fenetre du graphique (bloque le script jusqu'a sa fermeture)

# --- 1. Bilan de depart ------------------------------------------------------

print("Chargement de la courbe de taux (jusqu'a 15 s si le site de la Banque de France est lent)...", flush=True)
print("Courbe utilisee :", curve_source())
base = balance_sheet()
print("\n=== 1. Bilan de depart ===")
print(f"Actif   (5 OAT)          : {base['actif']:>14,.0f} EUR")
print(f"Passif  (provision TMG)  : {base['passif']:>14,.0f} EUR")
print(f"Surplus (actif - passif) : {base['surplus']:>14,.0f} EUR")

# --- 2. DV01 de bilan et gap de duration ---------------------------------------

dv = parallel_dv01()
dur_actif = dv["actif"] / (base["actif"] * 0.0001)
dur_passif = dv["passif"] / (base["passif"] * 0.0001)
ratio_pa = base["passif"] / base["actif"]
gap_duree = dur_actif - ratio_pa * dur_passif

print("\n=== 2. DV01 de bilan et gap de duration ===")
print(f"DV01 actif   : {dv['actif']:>9,.0f} EUR perdus si les taux montent de 1bp")
print(f"DV01 passif  : {dv['passif']:>9,.0f} EUR de passif en MOINS si les taux montent de 1bp")
print(f"Variation du surplus pour +1bp : {dv['passif'] - dv['actif']:>+9,.0f} EUR")
print(f"\nDuree effective actif  = DV01 / (valeur x 1bp) : {dur_actif:.2f} ans")
print(f"Duree effective passif                         : {dur_passif:.2f} ans")
print(f"Gap de duration = D_actif - (Passif/Actif) x D_passif = {gap_duree:+.2f} an(s)")
print("Lecture : gap < 0 -> le passif est plus long que l'actif. Les taux BAISSENT : le passif"
      "\ns'alourdit plus vite que l'actif ne gagne -> le surplus baisse. C'est le risque a couvrir.")
print("(Duree 'effective' = bump-and-reprice de la courbe ; elle differe un peu de la duree"
      "\nmodifiee du notebook 01, qui decale UN SEUL taux, le YTM. Pour la couverture, on garde"
      "\nla definition qui est aussi celle des scenarios : la courbe choquee.)")

# --- 3. KR01 : le risque pivot par pivot ------------------------------------------

ka, kp, expo = kr01_actif(), kr01_passif(), surplus_exposure()

print("\n=== 3. KR01 : ou, sur la courbe, se situe le risque ? ===")
print(f"{'Pivot':>9}{'KR01 actif':>14}{'KR01 passif':>14}{'Surplus pour +1bp':>20}")
for p, a, l, e in zip(RISK_PIVOTS, ka, kp, expo):
    print(f"{p:>5} ans{a:>14,.0f}{l:>14,.0f}{e:>+20,.0f}")
print(f"{'Somme':>9}{ka.sum():>14,.0f}{kp.sum():>14,.0f}{expo.sum():>+20,.0f}")

print("\nControles :")
for label, kr_sum, dv01 in (("actif", ka.sum(), dv["actif"]), ("passif", kp.sum(), dv["passif"])):
    rel = abs(kr_sum - dv01) / dv01
    print(f"  [{'OK' if rel < 0.005 else 'ECHEC'}] somme des KR01 {label} = DV01 {label} "
          f"({kr_sum:,.0f} vs {dv01:,.0f}, ecart {rel:.2%} : effet de 2e ordre du choc de 1bp)")

print("\nLecture de la derniere colonne (variation du surplus si CE pivot monte de 1bp) :")
print("  > 0 : le surplus gagne si le pivot monte, perd s'il baisse (trop de passif long ici)")
print("  < 0 : le surplus perd si le pivot monte (trop d'actif ici)")

# --- 4. Scenarios de choc, sans couverture ------------------------------------

print("\n=== 4. Scenarios de choc (aucune couverture) ===")
print("Definition des chocs (bp) aux pivots", [int(p) for p in RISK_PIVOTS])
for name, shock in SCENARIOS.items():
    print(f"  {name:<20}", [f"{s:+.0f}" for s in shock])

print(f"\n{'Scenario':<20}{'Actif':>14}{'Passif':>14}{'Surplus':>14}{'Var. surplus':>15}")
negatifs = []
for name, shock in SCENARIOS.items():
    b = balance_sheet(shock)
    print(f"{name:<20}{b['actif']:>14,.0f}{b['passif']:>14,.0f}{b['surplus']:>14,.0f}"
          f"{b['surplus'] - base['surplus']:>+15,.0f}")
    if b["surplus"] < 0:
        negatifs.append(name)
print(f"\nSurplus de depart : {base['surplus']:,.0f} EUR. "
      f"Scenarios ou il devient negatif : {', '.join(negatifs) if negatifs else 'aucun'}.")

# --- 5. Graphique ------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("\n(matplotlib absent : `pip install matplotlib` pour generer le graphique)")

if plt is not None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})
    x = np.arange(len(RISK_PIVOTS))
    labels = [f"{p} ans" for p in RISK_PIVOTS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    w = 0.38
    ax1.bar(x - w / 2, ka / 1e3, w, label="Actif (5 OAT)", color="#1f77b4")
    ax1.bar(x + w / 2, kp / 1e3, w, label="Passif (provision TMG)", color="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("KR01 (k€ perdus pour +1bp au pivot)")
    ax1.set_title("Où se situe la sensibilité aux taux ?")
    ax1.legend(frameon=False)

    colors = ["#2ca02c" if v > 0 else "#ff7f0e" for v in expo]
    ax2.bar(x, expo / 1e3, color=colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Variation du surplus pour +1bp (k€)")
    ax2.set_title("Le risque à couvrir, pivot par pivot")
    ax2.text(0.02, 0.97, "vert : le surplus gagne si le taux monte\norange : le surplus perd si le taux monte",
             transform=ax2.transAxes, va="top", fontsize=8)

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "04_kr01_actif_passif.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nGraphique enregistre : {os.path.abspath(path)}")
    if SHOW:
        plt.show()
    plt.close(fig)