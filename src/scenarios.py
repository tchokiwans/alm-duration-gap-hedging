"""
alm-duration-gap-hedging — src/scenarios.py

Chocs de courbe et revalorisation complete du bilan : actif (5 OAT),
passif (provision TMG synthetique), swaps de couverture, surplus.

Convention de DV01 "bilan" : perte de valeur pour un choc de +1bp sur les
5 pivots de la courbe PAR (puis re-bootstrap). C'est la meme definition
pour l'actif, le passif et les swaps, et c'est celle des scenarios — sinon
un notionnel "calibre" sur une definition laisserait un petit risque
residuel sous une autre.

Les chocs sont definis par des points d'ancrage (maturite -> bp)
interpoles lineairement aux pivots de risque : ils restent valables meme
si la source de la courbe change.

Vocabulaire
  bp (point de base)  : 0,01 % = 0,0001.
  surplus             : valeur de l'actif + valeur des swaps - valeur du passif.
  pivot               : l'une des 5 maturites (1, 2, 5, 10, 30 ans) sur
                        lesquelles on mesure et on choque la courbe.
  courbure            : partie PAIRE de la variation du surplus pour un choc
                        parallele +/- s :  [dS(+s) + dS(-s)] / 2.
                        = 0 si le surplus est une droite du niveau des taux,
                        < 0 si c'est une "cloche" (U inverse) : on perd dans
                        les deux sens.
"""

from datetime import date
import numpy as np
import pandas as pd

from bonds import BONDS, NOMINAL, cash_flows
from curve import RISK_PIVOTS, bootstrap_zero_curve, get_par_curve, key_rate_01
from hedging import par_swap_rate, swap_key_rate_01, swap_value
from liabilities import liability_cash_flows
from sensitivity import price_with_curve

VAL_DATE = date(2026, 9, 21)

# Flux futurs : ils ne dependent pas de la courbe, on les calcule une fois.
BOND_CFS = [cash_flows(coupon, maturity, VAL_DATE, NOMINAL) for _, _, coupon, maturity in BONDS]
LIAB_CFS = liability_cash_flows(VAL_DATE)


# --- Definition des chocs ----------------------------------------------------

def parallel(bp: float) -> np.ndarray:
    """Choc parallele : meme amplitude (en bp) sur tous les pivots."""
    return np.full(len(RISK_PIVOTS), float(bp))


def _anchored(anchors: dict) -> np.ndarray:
    maturities = sorted(anchors)
    return np.interp(RISK_PIVOTS, maturities, [anchors[m] for m in maturities])


# Pentification : le court baisse, le long monte. Aplatissement : l'inverse.
STEEPENING = _anchored({1: -50, 2: -50, 5: 0, 10: 50, 30: 75})
FLATTENING = _anchored({1: 50, 2: 50, 5: 0, 10: -50, 30: -75})
# Papillon : le milieu de la courbe (5 ans) bouge, les extremites non.
BUTTERFLY = _anchored({1: 0, 2: 25, 5: 50, 10: 25, 30: 0})

SCENARIOS = {
    "Parallele +100bp": parallel(100),
    "Parallele -100bp": parallel(-100),
    "Pentification": STEEPENING,
    "Aplatissement": FLATTENING,
}

# Batterie de stress utilisee pour comparer les strategies de couverture.
STRESS_BATTERY = {
    "Parallele -300bp": parallel(-300),
    "Parallele -200bp": parallel(-200),
    "Parallele -100bp": parallel(-100),
    "Parallele +100bp": parallel(100),
    "Parallele +200bp": parallel(200),
    "Parallele +300bp": parallel(300),
    "Pentification": STEEPENING,
    "Pentification x2": 2 * STEEPENING,
    "Aplatissement": FLATTENING,
    "Aplatissement x2": 2 * FLATTENING,
    "Papillon +": BUTTERFLY,
    "Papillon -": -BUTTERFLY,
}


# --- Revalorisation du bilan ----------------------------------------------------

def shocked_zero_curve(shock_bp=None):
    """Courbe zero-coupon obtenue en choquant les pivots de la courbe par."""
    par_t, par_y = get_par_curve()
    shock = np.zeros(len(par_t)) if shock_bp is None else np.asarray(shock_bp, dtype=float)
    return bootstrap_zero_curve(par_t, par_y + shock / 10000)


def make_swaps(maturities, notionals):
    """Swaps AU PAIR a la date de valorisation, sous la forme
    (maturite, taux fixe K, notionnel signe) : notionnel > 0 = receveur,
    < 0 = payeur."""
    zt, zr = shocked_zero_curve()
    return [(int(T), par_swap_rate(int(T), zt, zr), float(N))
            for T, N in zip(maturities, notionals)]


def balance_sheet(shock_bp=None, swaps=()):
    """Valeur de l'actif, du passif, des swaps et du surplus sous un choc."""
    zt, zr = shocked_zero_curve(shock_bp)
    actif = sum(price_with_curve(cfs, zt, zr) for cfs in BOND_CFS)
    passif = price_with_curve(LIAB_CFS, zt, zr)
    swaps_pv = sum(swap_value(K, N, T, VAL_DATE, zt, zr) for T, K, N in swaps)
    return {"actif": actif, "passif": passif, "swaps": swaps_pv,
            "surplus": actif + swaps_pv - passif}


def surplus_change(shock_bp, swaps=()) -> float:
    """Variation du surplus (EUR) entre la courbe de depart et la courbe
    choquee, avec les memes swaps (leur taux fixe K reste celui de depart)."""
    return balance_sheet(shock_bp, swaps)["surplus"] - balance_sheet(None, swaps)["surplus"]


def stress_table(strategies: dict, battery: dict = None) -> pd.DataFrame:
    """
    Variation du surplus (EUR) de chaque strategie sous chaque choc.
    strategies : {nom: liste de swaps (T, K, N)}   -- [] = pas de couverture
    battery    : {nom du choc: vecteur de chocs aux pivots} (defaut : STRESS_BATTERY)
    Renvoie un DataFrame : une ligne par choc, une colonne par strategie.
    """
    battery = STRESS_BATTERY if battery is None else battery
    base = {name: balance_sheet(swaps=swaps)["surplus"] for name, swaps in strategies.items()}
    data = {name: [balance_sheet(shock, swaps)["surplus"] - base[name]
                   for shock in battery.values()]
            for name, swaps in strategies.items()}
    return pd.DataFrame(data, index=list(battery))


# --- Sensibilites du bilan ----------------------------------------------------------

def parallel_dv01() -> dict:
    """DV01 'bilan' de l'actif et du passif (EUR perdus pour +1bp parallele)."""
    base, up = balance_sheet(), balance_sheet(parallel(1))
    return {"actif": base["actif"] - up["actif"], "passif": base["passif"] - up["passif"]}


def kr01_actif() -> np.ndarray:
    """KR01 du portefeuille obligataire, pivot par pivot (EUR par bp)."""
    par_t, par_y = get_par_curve()
    return sum(key_rate_01(cfs, par_t, par_y)[0] for cfs in BOND_CFS)


def kr01_passif() -> np.ndarray:
    """KR01 du passif, pivot par pivot (EUR par bp)."""
    par_t, par_y = get_par_curve()
    return key_rate_01(LIAB_CFS, par_t, par_y)[0]


def kr01_swaps(swaps) -> np.ndarray:
    """KR01 d'un ensemble de swaps, pivot par pivot (EUR par bp)."""
    par_t, par_y = get_par_curve()
    total = np.zeros(len(par_t))
    for T, K, N in swaps:
        total += swap_key_rate_01(K, N, T, VAL_DATE, par_t, par_y)
    return total


def surplus_exposure(swaps=()) -> np.ndarray:
    """
    Variation du SURPLUS (EUR) si le taux d'UN pivot monte de +1bp, les
    autres restant fixes :   KR01(passif) - KR01(actif) - KR01(swaps).
      > 0 : le surplus GAGNE si ce pivot monte (et perd s'il baisse) ;
      < 0 : le surplus PERD si ce pivot monte.
    Sans swap, c'est le "gap" a combler ; avec des swaps, c'est le risque
    qui reste. Une couverture parfaite au premier ordre = vecteur nul.
    """
    return kr01_passif() - kr01_actif() - kr01_swaps(swaps)


# --- Courbure et dimensionnement des couvertures -----------------------------------------

def curvature(swaps=(), amplitude_bp: float = 100.0) -> float:
    """Courbure du surplus (EUR) : partie PAIRE de sa variation pour un choc
    parallele +/- amplitude, [dS(+s) + dS(-s)] / 2. Negative = cloche."""
    up = surplus_change(parallel(amplitude_bp), swaps)
    down = surplus_change(parallel(-amplitude_bp), swaps)
    return (up + down) / 2


def swap_profile(maturity: int, unit: float = 1_000_000, amplitude_bp: float = 100.0):
    """
    (DV01, courbure) d'un swap receveur AU PAIR de `maturity` ans pour `unit`
    EUR de notionnel, sous chocs paralleles de la courbe par.
    DV01      = V(0) - V(+1bp)               (perte pour +1bp)
    courbure  = [V(+s) + V(-s)] / 2 - V(0)   (partie paire, s = amplitude)
    """
    _, K, N = make_swaps([maturity], [unit])[0]

    def value(bp: float) -> float:
        zt, zr = shocked_zero_curve(parallel(bp))
        return swap_value(K, N, maturity, VAL_DATE, zt, zr)

    dv01 = value(0) - value(1)
    curv = (value(amplitude_bp) + value(-amplitude_bp)) / 2 - value(0)
    return dv01, curv


def dv01_hedge_notional(maturity: int, unit: float = 1_000_000) -> float:
    """Notionnel (EUR) du swap receveur de `maturity` ans qui annule le DV01
    de bilan :  N = (DV01 passif - DV01 actif) / DV01 du swap par EUR."""
    dv = parallel_dv01()
    swap_dv01, _ = swap_profile(maturity, unit)
    return (dv["passif"] - dv["actif"]) / swap_dv01 * unit


def solve_dv01_curvature_hedge(maturities, amplitude_bp: float = 100.0,
                                unit: float = 1_000_000) -> np.ndarray:
    """
    Notionnels signes (EUR) de DEUX swaps qui annulent a la fois le DV01 de
    bilan et la courbure du surplus : systeme lineaire 2x2.
        N1 x dv01_1  + N2 x dv01_2  =  DV01 passif - DV01 actif
        N1 x courb_1 + N2 x courb_2 =  - courbure du surplus non couvert
    """
    if len(maturities) != 2:
        raise ValueError("Il faut exactement 2 maturites : 2 equations (DV01 et courbure).")
    dv = parallel_dv01()
    target = np.array([dv["passif"] - dv["actif"], -curvature((), amplitude_bp)])
    profiles = np.array([swap_profile(T, unit, amplitude_bp) for T in maturities])
    n_units = np.linalg.solve(profiles.T, target)   # colonnes = swaps, lignes = (DV01, courbure)
    return n_units * unit