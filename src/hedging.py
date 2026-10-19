"""
alm-duration-gap-hedging — src/hedging.py

Couverture par swaps de taux receveurs (recoit fixe / paie flottant).
Un notionnel NEGATIF = swap payeur (paie fixe / recoit flottant) : toutes
les formules sont lineaires en le notionnel.

Jambe fixe : traitee comme un cash-flow explicite (coupon K x N chaque
annee, PAS de principal a l'echeance -- c'est ce qui distingue un swap
d'une obligation) -- reutilise price_with_curve() de sensitivity.py.

Jambe flottante : PV = N x (1 - DF(T)). Ce n'est pas une approximation
de calcul mais le resultat exact du "telescopage" des flux flottants
actualises (cours Turinici, Dauphine, §3.1.2, eq. 3.16 : BV(t) = B(t,T0) -
B(t,Tn), avec B(t,T0) = 1 quand le swap demarre aujourd'hui). Hypotheses :
une seule courbe pour projeter et actualiser, pas de spread de base.

Resolution de la couverture : un swap de maturite T a un profil de KR01
(sensibilite pivot par pivot) qui lui est propre. Choisir les notionnels
de plusieurs swaps revient a resoudre un systeme lineaire : M x n = gap,
ou M[p, j] = KR01 du swap j au pivot p, et gap[p] = KR01(passif) -
KR01(actif) au pivot p.
"""

from datetime import date
import numpy as np

from curve import bootstrap_zero_curve
from sensitivity import price_with_curve


def swap_fixed_leg_cfs(K: float, notional: float, tenor: int, val_date: date):
    """Cash-flows de la jambe fixe : coupon annuel K x N, PAS de principal."""
    return [(date(val_date.year + t, val_date.month, val_date.day), float(t), K * notional)
            for t in range(1, tenor + 1)]


def discount_factor(t: float, tenors, rates) -> float:
    """DF(t) = (1 + taux zero(t)) ** (-t) : valeur aujourd'hui de 1 EUR recu dans t ans."""
    return (1 + np.interp(t, tenors, rates)) ** (-t)


def floating_leg_value(notional: float, tenor: int, tenors, rates) -> float:
    """PV jambe flottante = N x (1 - DF(T))."""
    return notional * (1 - discount_factor(tenor, tenors, rates))


def par_swap_rate(tenor: int, tenors, rates) -> float:
    """Taux fixe K tel que le swap ait une valeur nulle a l'origine."""
    annuity = sum(discount_factor(t, tenors, rates) for t in range(1, tenor + 1))
    return (1 - discount_factor(tenor, tenors, rates)) / annuity


def swap_value(K: float, notional: float, tenor: int, val_date: date, tenors, rates) -> float:
    """Valeur du swap RECEVEUR (recoit fixe K, paie flottant)."""
    fixed_cfs = swap_fixed_leg_cfs(K, notional, tenor, val_date)
    pv_fixed = price_with_curve(fixed_cfs, tenors, rates)
    pv_floating = floating_leg_value(notional, tenor, tenors, rates)
    return pv_fixed - pv_floating


def swap_dv01(K: float, notional: float, tenor: int, val_date: date,
              par_tenors, par_yields, bump: float = 0.0001) -> float:
    """DV01 du swap receveur : perte de valeur pour un choc parallele +1bp
    de la courbe par, calcule par bump-and-reprice complet (jambes fixe ET
    flottante)."""
    t0, r0 = bootstrap_zero_curve(par_tenors, par_yields)
    v0 = swap_value(K, notional, tenor, val_date, t0, r0)
    t1, r1 = bootstrap_zero_curve(par_tenors, par_yields + bump)
    v1 = swap_value(K, notional, tenor, val_date, t1, r1)
    return v0 - v1


def swap_key_rate_01(K: float, notional: float, tenor: int, val_date: date,
                      par_tenors, par_yields, bump: float = 0.0001):
    """KR01 du swap, pivot par pivot -- bump-and-reprice direct de
    swap_value() (fixe + flottante ensemble), pas de decomposition des
    deux jambes separement : plus simple et sans piege de signe."""
    base_tenors, base_rates = bootstrap_zero_curve(par_tenors, par_yields)
    v0 = swap_value(K, notional, tenor, val_date, base_tenors, base_rates)
    kr01 = []
    for i in range(len(par_tenors)):
        bumped = par_yields.copy()
        bumped[i] += bump
        t, r = bootstrap_zero_curve(par_tenors, bumped)
        v1 = swap_value(K, notional, tenor, val_date, t, r)
        kr01.append(v0 - v1)
    return np.array(kr01)


def swap_kr01_matrix(maturities, par_tenors, par_yields, val_date, unit: float = 1_000_000):
    """
    Matrice M (pivots x maturites) : M[p, j] = KR01 au pivot p d'un swap
    receveur AU PAIR de maturite maturities[j], pour `unit` EUR de notionnel.
    """
    base_t, base_r = bootstrap_zero_curve(par_tenors, par_yields)
    cols = []
    for T in maturities:
        K = par_swap_rate(int(T), base_t, base_r)
        cols.append(swap_key_rate_01(K, unit, int(T), val_date, par_tenors, par_yields))
    return np.column_stack(cols)


def solve_swap_hedge(gap_kr01, maturities, par_tenors, par_yields, val_date,
                     unit: float = 1_000_000):
    """
    Notionnels signes (EUR) des swaps de maturites `maturities` qui
    annulent le gap de KR01 (passif - actif), au sens des moindres carres :
    exact si autant de swaps que de pivots (et matrice inversible), sinon
    meilleur compromis.
    Renvoie (notionnels_en_EUR, matrice_M_par_`unit`).
    """
    M = swap_kr01_matrix(maturities, par_tenors, par_yields, val_date, unit)
    n_units, *_ = np.linalg.lstsq(M, np.asarray(gap_kr01, dtype=float), rcond=None)
    return n_units * unit, M