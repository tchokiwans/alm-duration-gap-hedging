"""
alm-duration-gap-hedging — src/bonds.py

Cash-flows du portefeuille actif : 5 OAT reelles, coupon annuel,
remboursement au pair. Le pricing et les sensibilites (Macaulay,
modifiee, convexite, DV01) sont calcules par le moteur generique de
sensitivity.py.
"""

from datetime import date
from sensitivity import sensitivity_profile

BONDS = [
    # (nom, ISIN, coupon, date d'echeance)
    ("OAT 0.75% 2028", "FR0013341682", 0.0075, date(2028, 11, 25)),
    ("OAT 2.50% 2030", "FR0011883966", 0.0250, date(2030, 5, 25)),
    ("OAT 0.00% 2031", "FR0014002WK3", 0.0000, date(2031, 11, 25)),
    ("OAT 5.75% 2032", "FR0000187635", 0.0575, date(2032, 10, 25)),
    ("OAT 3.70% 2036", "FR0014018YR0", 0.0370, date(2036, 11, 25)),
]
NOMINAL = 10_000_000


def year_frac(d1: date, d2: date) -> float:
    """Fraction d'annee entre deux dates (ACT/365.25, approximation)."""
    return (d2 - d1).days / 365.25


def coupon_dates(maturity: date, val_date: date) -> list[date]:
    """Dates de coupon annuel, en remontant depuis l'echeance."""
    dates = []
    d = maturity
    while d > val_date:
        dates.append(d)
        try:
            d = date(d.year - 1, d.month, d.day)
        except ValueError:
            d = date(d.year - 1, d.month, 28)
    return sorted(dates)


def cash_flows(coupon: float, maturity: date, val_date: date, nominal: float):
    """Liste (date, t, montant) des flux futurs d'une obligation."""
    cfs = []
    for d in coupon_dates(maturity, val_date):
        cf = nominal * coupon + (nominal if d == maturity else 0.0)
        cfs.append((d, year_frac(val_date, d), cf))
    return cfs


def price_bond(coupon: float, maturity: date, val_date: date, nominal: float = NOMINAL):
    """Calcule prix, YTM, duration Macaulay, modifiee, convexite et DV01."""
    cfs = cash_flows(coupon, maturity, val_date, nominal)
    return sensitivity_profile(cfs)