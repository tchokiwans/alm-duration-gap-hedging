"""
alm-duration-gap-hedging — src/curve.py

Courbe zero-coupon utilisee pour actualiser les flux du portefeuille et
du passif. Construite en deux temps :

  1. fetch_bdf_curve() recupere les rendements PAR publies par la Banque
     de France ("Taux indicatifs des bons du Tresor et OAT" — 1m/3m/6m/
     9m/1an/2ans/5ans/10ans/30ans). Ce sont des rendements d'obligations
     reelles a coupon, donc des YTM point par point, PAS des taux
     zero-coupon directement utilisables.
  2. bootstrap_zero_curve() convertit ces rendements par en vrais taux
     zero-coupon (bootstrap standard, coupon annuel — convention OAT).

Pivots de risque : quelle que soit la source, la courbe par est ramenee
aux 5 points RISK_PIVOTS (1, 2, 5, 10, 30 ans). Ce sont ces 5 points que
les scenarios choquent, que les KR01 mesurent et que la couverture
neutralise.

Cache : la courbe est chargee UNE SEULE FOIS par session Python. (Version
precedente : zero_rate() relancait un fetch reseau a chaque appel, soit
des centaines de requetes HTTP par script — corrige.)

ATTENTION sur fetch_bdf_curve() : jamais teste en conditions reelles (le
site etait en maintenance quand tu l'as lance, et mon environnement n'a
pas d'acces a banque-france.fr). Si le fetch echoue, un avertissement est
affiche UNE fois et la courbe indicative de repli est utilisee ; la
fonction curve_source() dit laquelle des deux est en cours d'usage.
"""

import functools
import io
import urllib.request
import warnings

import numpy as np
import pandas as pd

MAX_TENOR = 20  # annees ; couvre le plus long flux du passif (18 ans) et un swap 20 ans

# Les 5 points sur lesquels on raisonne en risque (scenarios, KR01, couverture).
RISK_PIVOTS = np.array([1, 2, 5, 10, 30])

# Rendements par de repli (ancrage reel : OAT 10 ans ~4.10 %, debut sept. 2026).
# Utilises seulement si fetch_bdf_curve() echoue.
FALLBACK_PAR_YIELDS = np.array([0.0320, 0.0350, 0.0385, 0.0410, 0.0435])

_curve_source = "non chargee"


def fetch_bdf_curve():
    """
    Recupere le dernier tableau "Taux indicatifs des bons du Tresor et
    OAT" publie par la Banque de France, et renvoie (tenors_en_annees,
    rendements_par) pour les points trouves.

    NON TESTE EN REEL — logique best-effort a valider/debugger localement.
    """
    url = "https://www.banque-france.fr/fr/statistiques/taux-et-cours/taux-indicatifs-des-bons-du-tresor-et-oat"
    # En-tetes realistes : sans ca, urllib/pandas envoie un User-Agent
    # generique ("Python-urllib/x.x") que certains sites bloquent d'office.
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/128.0.0.0 Safari/537.36"),
        "Accept-Language": "fr-FR,fr;q=0.9",
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        html = response.read()

    # decimal="," : les nombres francais ("3,40") sinon pandas lit la virgule
    # comme separateur de milliers par defaut et "3,40" devient 340 (bug
    # trouve en testant le parsing sur une page HTML simulee).
    tables = pd.read_html(io.BytesIO(html), decimal=",", thousands=None)

    tenor_map = {
        "1 mois": 1 / 12, "3 mois": 3 / 12, "6 mois": 6 / 12, "9 mois": 9 / 12,
        "1 an": 1, "2 ans": 2, "5 ans": 5, "10 ans": 10, "30 ans": 30,
    }

    for table in tables:
        first_col = table.iloc[:, 0].astype(str)
        matches = {label: tenor for label, tenor in tenor_map.items()
                   if first_col.str.contains(label, case=False, regex=False).any()}
        if len(matches) >= 5:  # table plausible trouvee
            tenors, yields_pct = [], []
            for label, tenor in matches.items():
                row = table[first_col.str.contains(label, case=False, regex=False)].iloc[0]
                # derniere colonne numerique = date la plus recente publiee
                val = pd.to_numeric(row.iloc[1:].astype(str).str.replace(",", "."),
                                     errors="coerce").dropna().iloc[-1]
                tenors.append(tenor)
                yields_pct.append(val)
            order = np.argsort(tenors)
            return np.array(tenors)[order], np.array(yields_pct)[order] / 100

    raise RuntimeError("Table BdF non trouvee — structure de page a inspecter manuellement.")


def bootstrap_zero_curve(par_tenors, par_yields, max_tenor: int = MAX_TENOR):
    """
    Bootstrap standard, coupon annuel : convertit des rendements PAR
    (sur tenors possiblement epars) en taux zero-coupon annee par annee.
    Etape 1 : interpole le rendement par a chaque annee entiere.
    Etape 2 : bootstrap sequentiel (le taux zero de l'annee T est celui
    qui repriced a 100 le bond par de maturite T, sachant les taux zero
    deja trouves pour les annees < T).
    """
    years = np.arange(1, max_tenor + 1)
    par_at_year = np.interp(years, par_tenors, par_yields)
    zero = {}
    for T, par in zip(years, par_at_year):
        coupon = par * 100  # coupon annuel sur nominal 100
        if T == 1:
            zero[1] = par
        else:
            pv_coupons = sum(coupon / (1 + zero[t]) ** t for t in range(1, T))
            zero[T] = ((100 + coupon) / (100 - pv_coupons)) ** (1 / T) - 1
    return years.astype(float), np.array([zero[int(y)] for y in years])


@functools.lru_cache(maxsize=1)
def _load_par_curve():
    """Charge la courbe par aux pivots de risque (une seule fois par session)."""
    global _curve_source
    try:
        raw_tenors, raw_yields = fetch_bdf_curve()
        par_yields = np.interp(RISK_PIVOTS, raw_tenors, raw_yields)
        # Garde-fou : un parsing de page rate peut renvoyer des valeurs absurdes
        # (ex. 340 au lieu de 3,40). On prefere le repli a une courbe fausse.
        if not np.all((par_yields > -0.01) & (par_yields < 0.15)):
            raise ValueError(f"rendements hors plage plausible : {par_yields}")
        _curve_source = "Banque de France (donnees live)"
    except Exception as exc:
        warnings.warn(f"Courbe Banque de France indisponible ({type(exc).__name__}) : "
                      "repli sur la courbe par indicative.", RuntimeWarning, stacklevel=2)
        par_yields = FALLBACK_PAR_YIELDS.copy()
        _curve_source = "courbe indicative de repli (source BdF indisponible)"
    return RISK_PIVOTS.copy(), par_yields


@functools.lru_cache(maxsize=1)
def _load_zero_curve():
    """Courbe zero-coupon bootstrappee (une seule fois par session)."""
    return bootstrap_zero_curve(*_load_par_curve())


def get_par_curve():
    """(pivots, rendements par) — copies, donc modifiables sans risque."""
    tenors, yields = _load_par_curve()
    return tenors.copy(), yields.copy()


def get_curve():
    """(tenors, taux zero-coupon) — copies, donc modifiables sans risque."""
    tenors, rates = _load_zero_curve()
    return tenors.copy(), rates.copy()


def curve_source() -> str:
    """Dit d'ou vient la courbe utilisee : BdF live ou repli indicatif."""
    _load_par_curve()
    return _curve_source


def zero_rate(t: float, tenors=None, rates=None) -> float:
    """Taux zero-coupon a la maturite t (interpolation lineaire)."""
    if tenors is None or rates is None:
        tenors, rates = _load_zero_curve()
    return np.interp(t, tenors, rates)


def key_rate_01(cfs, par_tenors, par_yields, bump: float = 0.0001):
    """
    KR01 : sensibilite (EUR, perte de valeur pour un choc de +1bp) a
    chaque pivot de la courbe PAR, un a la fois, les autres pivots
    restant fixes. Le choc "tente" (plein au pivot, nul aux pivots
    voisins) sort naturellement de l'interpolation lineaire du par curve
    -- pas besoin de forme geometrique separee.
    Renvoie (kr01_par_pivot, prix_de_base).
    """
    from sensitivity import price_with_curve
    base_tenors, base_rates = bootstrap_zero_curve(par_tenors, par_yields)
    base_price = price_with_curve(cfs, base_tenors, base_rates)
    kr01 = []
    for i in range(len(par_tenors)):
        bumped = par_yields.copy()
        bumped[i] += bump
        b_tenors, b_rates = bootstrap_zero_curve(par_tenors, bumped)
        b_price = price_with_curve(cfs, b_tenors, b_rates)
        kr01.append(base_price - b_price)
    return np.array(kr01), base_price