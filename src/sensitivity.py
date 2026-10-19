"""
alm-duration-gap-hedging — src/sensitivity.py

Moteur de sensibilite generique : opere sur n'importe quelle liste de
cash-flows (date, t, montant), qu'ils viennent d'une obligation ou d'un
passif d'assurance — un passif n'est qu'un cash-flow qui n'a pas choisi
son camp. Extrait de bonds.py pour etre partage avec liabilities.py sans
dupliquer les formules.
"""

import numpy as np
from scipy.optimize import brentq
from curve import zero_rate


def price_and_macaulay(cfs):
    """Prix = flux actualises sur la courbe. Macaulay = moyenne ponderee des t_i."""
    price = sum(cf / (1 + zero_rate(t)) ** t for _, t, cf in cfs)
    macaulay = sum(t * cf / (1 + zero_rate(t)) ** t for _, t, cf in cfs) / price
    return price, macaulay


def find_ytm(cfs, target_price):
    """Taux plat unique qui redonne le meme prix (root-finding)."""
    f = lambda y: sum(cf / (1 + y) ** t for _, t, cf in cfs) - target_price
    return brentq(f, -0.05, 0.30)


def modified_duration(macaulay: float, ytm: float) -> float:
    """D_mod = D_Mac / (1 + y) — sensibilite : DeltaPrix/Prix ~ -D_mod x Delta_y."""
    return macaulay / (1 + ytm)


def convexity(cfs, price: float, ytm: float) -> float:
    """C = (1/Prix) x Sum[ t_i(t_i+1) x CF_i / (1+y)^(t_i+2) ]."""
    return sum(t * (t + 1) * cf / (1 + ytm) ** (t + 2) for _, t, cf in cfs) / price


def dv01(modified: float, price: float) -> float:
    """DV01 = D_mod x Prix x 1bp — variation de Prix (EUR) pour un choc de 1bp."""
    return modified * price * 0.0001


def sensitivity_profile(cfs):
    """Calcule prix, YTM, Macaulay, modifiee, convexite, DV01 pour un cash-flow."""
    price, macaulay = price_and_macaulay(cfs)
    ytm = find_ytm(cfs, price)
    mod = modified_duration(macaulay, ytm)
    return dict(price=price, ytm=ytm, macaulay=macaulay, modified=mod,
                convexity=convexity(cfs, price, ytm),
                dv01=dv01(mod, price), cfs=cfs)


def price_with_curve(cfs, tenors, rates):
    """Prix seul, avec une courbe explicite (pas la courbe 'live' par
    defaut) -- utilise pour reprice sous un choc de courbe donne (KRD,
    scenarios), sans dependre de get_curve()."""
    return sum(cf / (1 + np.interp(t, tenors, rates)) ** t for _, t, cf in cfs)