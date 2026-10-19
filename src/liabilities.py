"""
alm-duration-gap-hedging — src/liabilities.py

Passif synthetique : provision de type epargne avec taux minimum garanti
(TMG). CELA EST ENTIEREMENT FICTIF — aucune donnee, calibration ou methodologie
utilisée. Le run-off (partie de la provision qui sort chaque
annee) est un parametre invente pour l'exercice, pas un taux de rachat
calibre.

Mecanique : chaque annee, la provision est creditee du TMG, puis une
fraction fixe ("taux de sortie") est payee aux assures. A l'horizon, le
solde restant est verse en totalite (maturite finale de la generation).
Le pricing et les sensibilites reutilisent le meme moteur generique que
les obligations (sensitivity.py) — à noter qu'un passif est un cash-flow comme un
autre.
"""

from datetime import date
from sensitivity import sensitivity_profile

# --- Parametres du passif synthetique (100% hypothétique) ------------------------
RESERVE_0 = 55_000_000   # provision mathematique initiale (EUR)
TMG = 0.025               # taux minimum garanti annuel
TAUX_SORTIE = 0.08        # fraction de la provision (post-credit TMG) payee chaque annee
HORIZON = 18               # derniere annee : solde entierement verse


def liability_cash_flows(val_date: date, reserve0: float = RESERVE_0,
                          tmg: float = TMG, taux_sortie: float = TAUX_SORTIE,
                          horizon: int = HORIZON):
    """Genere le calendrier (date, t, montant) des sorties de la provision."""
    cfs = []
    reserve = reserve0
    for t in range(1, horizon + 1):
        reserve_creditee = reserve * (1 + tmg)
        if t < horizon:
            cf = reserve_creditee * taux_sortie
            reserve = reserve_creditee - cf
        else:
            cf = reserve_creditee   # derniere annee : on solde tout
            reserve = 0.0
        d = date(val_date.year + t, val_date.month, val_date.day)
        cfs.append((d, float(t), cf))
    return cfs


def price_liability(val_date: date, **kwargs):
    """Calcule prix (= valeur actuelle des engagements), YTM: taux actuariel si detenu jusqu'à maturite, Macaulay,
    modifiee, convexite et DV01 du passif — meme logique que price_bond()."""
    cfs = liability_cash_flows(val_date, **kwargs)
    return sensitivity_profile(cfs)