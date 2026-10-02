"""
Radar — comparaison de stratégies simples (fixées à l'avance) contre le score actuel.

Aucune stratégie n'est ajustée sur ces données : ce sont des règles classiques de la littérature
(retour à la moyenne dans la tendance, cassures de range). On les juge sur deux périodes
séparées — les années d'avant (A) et la dernière année (B). Règle de choix décidée avant de
voir les résultats : on retient la stratégie dont le plus faible des deux t est le plus élevé,
avec au moins 30 signaux et un excès positif dans chaque période.

Signaux espacés d'au moins 10 bougies pour limiter le chevauchement des mesures à 20 séances.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screener as S  # noqa: E402
from backtest_classes import UNIVERSE  # même panier de 56 actifs  # noqa: E402

HS = (10, 20)
COOLDOWN = 10

BUY = {
    "B0": "Score Radar actuel ≥ 6",
    "B1": "RSI 14 sort de la survente (repasse au-dessus de 30), en tendance haussière",
    "B2": "RSI 2 sous 10 (repli très court), en tendance haussière",
    "B3": "Réintégration de la bande de Bollinger basse, en tendance haussière",
    "B4": "Retour à la moyenne combiné : B1 ou B2 ou B3",
    "B5": "Cassure du plus haut de 50 jours, en tendance haussière",
    "B6": "RSI 14 sort de la survente, sans filtre de tendance",
    "B7": "Order block haussier (cassure de structure + FVG), retour dans la zone",
    "B8": "Order block haussier (cassure de structure seule), retour dans la zone",
    "B9": "Order block haussier avec FVG, en tendance haussière",
    "B10": "Suivi de tendance : clôture au-dessus de la MM50 quand MM50 > MM200",
}
SELL = {
    "S0": "Score Radar actuel ≥ 5",
    "S1": "RSI 14 sort du surachat (repasse sous 70), en tendance baissière",
    "S2": "RSI 2 au-dessus de 90 (rebond très court), en tendance baissière",
    "S3": "Réintégration de la bande de Bollinger haute, en tendance baissière",
    "S4": "Rebond combiné : S1 ou S2 ou S3",
    "S5": "Cassure du plus bas de 50 jours, en tendance baissière",
    "S6": "Réintégration de la bande de Bollinger haute, sans filtre de tendance",
    "S7": "Order block baissier (cassure de structure + FVG), retour dans la zone",
    "S8": "Order block baissier (cassure de structure seule), retour dans la zone",
    "S9": "Order block baissier avec FVG, en tendance baissière",
    "S10": "Suivi de tendance : clôture sous la MM50 quand MM50 < MM200",
}


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    g, l = d.clip(lower=0), (-d).clip(lower=0)
    rs = g.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / l.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + rs)



def order_blocks(df: pd.DataFrame, need_fvg: bool, lookback: int = 15, life: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Order blocks « Smart Money » sur bougies jour, sans regarder le futur.

    Haussier : le cours clôture au-dessus du dernier sommet de swing confirmé (cassure de structure).
    L'order block est la dernière bougie baissière avant l'impulsion (au plus bas des `lookback`
    dernières bougies). Avec `need_fvg`, l'impulsion doit laisser un déséquilibre (FVG :
    plus bas d'une bougie au-dessus du plus haut de l'avant-veille). Signal d'achat au premier retour
    du cours dans la zone [plus bas ; plus haut] de l'order block, si la bougie clôture au-dessus du
    bas de zone, dans les `life` bougies. Zone invalidée par une clôture sous son plus bas.
    Baissier : symétrique.
    """
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    buy, sell = np.zeros(n, bool), np.zeros(n, bool)
    sw_hi = sw_lo = None
    zones_b, zones_s = [], []  # [zone_lo, zone_hi, créée_à, active]
    broke_hi = broke_lo = False
    for i in range(5, n):
        # swing confirmé à i-2 (fractal 5 bougies), connu seulement à la clôture de i
        k = i - 2
        if h[k] == max(h[k - 2:k + 3]):
            sw_hi, broke_hi = h[k], False
        if l[k] == min(l[k - 2:k + 3]):
            sw_lo, broke_lo = l[k], False
        # retours dans les zones existantes
        for z in zones_b:
            if not z[3]:
                continue
            if i - z[2] > life or c[i] < z[0]:
                z[3] = False
            elif l[i] <= z[1] and i > z[2]:
                buy[i] = True
                z[3] = False
        for z in zones_s:
            if not z[3]:
                continue
            if i - z[2] > life or c[i] > z[1]:
                z[3] = False
            elif h[i] >= z[0] and i > z[2]:
                sell[i] = True
                z[3] = False
        # cassures de structure
        if sw_hi is not None and not broke_hi and c[i] > sw_hi:
            broke_hi = True
            j0 = max(0, i - lookback)
            lo_idx = j0 + int(np.argmin(l[j0:i + 1]))
            ob_idx = next((j for j in range(lo_idx, max(j0, lo_idx - 5) - 1, -1) if c[j] < o[j]), None)
            if ob_idx is not None:
                fvg = any(l[m] > h[m - 2] for m in range(ob_idx + 2, i + 1))
                if fvg or not need_fvg:
                    zones_b.append([l[ob_idx], h[ob_idx], i, True])
        if sw_lo is not None and not broke_lo and c[i] < sw_lo:
            broke_lo = True
            j0 = max(0, i - lookback)
            hi_idx = j0 + int(np.argmax(h[j0:i + 1]))
            ob_idx = next((j for j in range(hi_idx, max(j0, hi_idx - 5) - 1, -1) if c[j] > o[j]), None)
            if ob_idx is not None:
                fvg = any(h[m] < l[m - 2] for m in range(ob_idx + 2, i + 1))
                if fvg or not need_fvg:
                    zones_s.append([l[ob_idx], h[ob_idx], i, True])
    return buy, sell


def signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c, p = df, df.shift(1)
    up = (c.close > c.sma200) & c.sma200.notna()
    dn = (c.close < c.sma200) & c.sma200.notna()
    r2 = rsi(df.close, 2)
    hh50 = df.high.rolling(50).max().shift(1)
    ll50 = df.low.rolling(50).min().shift(1)
    rsi_up = (p.rsi < 30) & (c.rsi >= 30)
    rsi_dn = (p.rsi > 70) & (c.rsi <= 70)
    bb_lo = (p.close < p.bb_lo) & (c.close > c.bb_lo)
    bb_up = (p.close > p.bb_up) & (c.close < c.bb_up)
    out = {
        "B1": rsi_up & up, "B2": (r2 < 10) & up, "B3": bb_lo & up,
        "B5": (c.close > hh50) & up, "B6": rsi_up,
        "S1": rsi_dn & dn, "S2": (r2 > 90) & dn, "S3": bb_up & dn,
        "S5": (c.close < ll50) & dn, "S6": bb_up,
    }
    out["B10"] = (p.close <= p.sma50) & (c.close > c.sma50) & (c.sma50 > c.sma200)
    out["S10"] = (p.close >= p.sma50) & (c.close < c.sma50) & (c.sma50 < c.sma200)
    out["B4"] = out["B1"] | out["B2"] | out["B3"]
    out["S4"] = out["S1"] | out["S2"] | out["S3"]
    d = {k: df[k].to_numpy(float) for k in df.columns}
    b0 = np.zeros(len(df), bool)
    s0 = np.zeros(len(df), bool)
    for i in range(201, len(df)):
        bs, _, be, ss, _, se = S.score_at(d, i)
        b0[i] = be and bs >= 6
        s0[i] = se and ss >= 5
    out = {k: v.fillna(False).to_numpy() if hasattr(v, "fillna") else v for k, v in out.items()}
    out["B0"], out["S0"] = b0, s0
    ob_b, ob_s = order_blocks(df, need_fvg=True)
    ob_b2, ob_s2 = order_blocks(df, need_fvg=False)
    upn, dnn = up.to_numpy(), dn.to_numpy()
    out["B7"], out["S7"] = ob_b, ob_s
    out["B8"], out["S8"] = ob_b2, ob_s2
    out["B9"], out["S9"] = ob_b & upn, ob_s & dnn
    return out


def run_asset(sym: str, cls: str):
    df, *_ = S.fetch(sym, "1d")
    df = S.indicators(df)
    close = df["close"].to_numpy(float)
    n = len(df)
    dates = df.index
    is_b = np.asarray(dates >= dates[-1] - timedelta(days=365))
    valid = df["sma200"].notna().to_numpy()
    fwd = {h: np.r_[close[h:] / close[:-h] - 1, [np.nan] * h] for h in HS}
    base = {}
    for per, m in (("A", valid & ~is_b), ("B", valid & is_b)):
        for h in HS:
            r = fwd[h][m]
            r = r[~np.isnan(r)]
            base[(per, h)] = float(r.mean()) if len(r) else 0.0
    rows = []
    for key, arr in signals(df).items():
        side = 1 if key.startswith("B") else -1
        last = -10**9
        for i in np.flatnonzero(arr):
            if i < 201 or not valid[i] or i - last < COOLDOWN:
                continue
            last = i
            per = "B" if is_b[i] else "A"
            row = {"sym": sym, "cls": cls, "strat": key, "per": per}
            for h in HS:
                r = fwd[h][i]
                row[f"g{h}"] = side * r if not np.isnan(r) else np.nan
                row[f"x{h}"] = side * (r - base[(per, h)]) if not np.isnan(r) else np.nan
            rows.append(row)
    years_a = max((~is_b & valid).sum() / 252, 0.5)
    return rows, years_a


def st(sub: pd.DataFrame, h: int):
    sub = sub[sub[f"x{h}"].notna()]
    n = len(sub)
    if n < 2:
        return {"n": n, "win": np.nan, "gain": np.nan, "x": np.nan, "t": np.nan}
    x = sub[f"x{h}"].to_numpy()
    sd = x.std(ddof=1)
    return {"n": n, "win": (sub[f"g{h}"] > 0).mean() * 100, "gain": sub[f"g{h}"].mean() * 100,
            "x": x.mean() * 100, "t": x.mean() / (sd / math.sqrt(n)) if sd > 0 else np.nan}


def cell(s):
    if not s["n"] or np.isnan(s["x"]):
        return "— | — | — | —"
    return f"{s['n']} | {s['win']:.0f} % | {s['x']:+.2f} % | {s['t']:+.1f}"


def main():
    rows, yrs, errors = [], [], []
    for cls, syms in UNIVERSE.items():
        for sym in syms:
            try:
                r, ya = run_asset(sym, cls)
                rows += r
                yrs.append(ya)
                print(f"{sym:10s} {len(r):5d}", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append(sym)
                print(f"{sym}: ERREUR {e}", flush=True)
    df = pd.DataFrame(rows)
    nsym = df.sym.nunique()
    ya = float(np.mean(yrs))

    L = []
    w = L.append
    w("# Stratégies simples contre le score actuel\n")
    w(f"{nsym} actifs, bougies jour. Période A : ~{ya:.1f} ans avant la dernière année. Période B : dernière année. "
      "Excès = gain au-delà de la tendance normale de l'actif. Signaux espacés d'au moins 10 bougies. "
      "Choix : meilleur « min(t A, t B) », avec ≥ 30 signaux et un excès positif dans chaque période.\n")
    if errors:
        w("Non chargés : " + ", ".join(errors) + "\n")

    best = {}
    for side, catalog in (("Achat", BUY), ("Vente", SELL)):
        w(f"\n## {side} — 20 séances\n")
        w("| | Stratégie | A : n | Réussite | Excès | t | B : n | Réussite | Excès | t | Signaux / actif / an | Robustesse |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        ranking = []
        for k, label in catalog.items():
            a = st(df[(df.strat == k) & (df.per == "A")], 20)
            b = st(df[(df.strat == k) & (df.per == "B")], 20)
            freq = (a["n"] + b["n"]) / max(nsym, 1) / (ya + 1)
            ok = a["n"] >= 30 and b["n"] >= 30 and a["x"] > 0 and b["x"] > 0
            rob = min(a["t"], b["t"]) if ok else np.nan
            ranking.append((k, rob))
            w(f"| {k} | {label} | {cell(a)} | {cell(b)} | {freq:.1f} | {'' if np.isnan(rob) else f'{rob:+.1f}'} |")
        ok = [x for x in ranking if not np.isnan(x[1])]
        best[side] = max(ok, key=lambda x: x[1])[0] if ok else None
        w(f"\n**Retenu ({side.lower()}) : {best[side] or 'aucune stratégie ne passe le critère'}**\n")

        w(f"\n### {side} — horizon 10 séances (contrôle)\n")
        w("| | A : n | Réussite | Excès | t | B : n | Réussite | Excès | t |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k in catalog:
            a = st(df[(df.strat == k) & (df.per == "A")], 10)
            b = st(df[(df.strat == k) & (df.per == "B")], 10)
            w(f"| {k} | {cell(a)} | {cell(b)} |")

        w(f"\n### {side} — par classe d'actif (20 séances, deux périodes réunies)\n")
        w("| | " + " | ".join(f"{c} : n · réussite · excès (t)" for c in UNIVERSE) + " |")
        w("|---|" + "---|" * len(UNIVERSE))
        for k in catalog:
            cells = []
            for c in UNIVERSE:
                s = st(df[(df.strat == k) & (df.cls == c)], 20)
                cells.append("—" if not s["n"] or np.isnan(s["x"]) else f"{s['n']} · {s['win']:.0f} % · {s['x']:+.2f} % ({s['t']:+.1f})")
            w(f"| {k} | " + " | ".join(cells) + " |")

    w("\n## Contrôle par classe et par période (20 séances)\n")
    w("| Stratégie | Classe | A : n · réussite · excès (t) | B : n · réussite · excès (t) |")
    w("|---|---|---|---|")
    for k in ("B0", "B1", "B6", "S0", "S1", "S3", "S6"):
        for c in UNIVERSE:
            cells = []
            for per in ("A", "B"):
                s_ = st(df[(df.strat == k) & (df.cls == c) & (df.per == per)], 20)
                cells.append("—" if not s_["n"] or np.isnan(s_["x"]) else f"{s_['n']} · {s_['win']:.0f} % · {s_['x']:+.2f} % ({s_['t']:+.1f})")
            w(f"| {k} | {c} | {cells[0]} | {cells[1]} |")
    report = "\n".join(L)
    out = Path(os.environ.get("BT_OUT", "bt-out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "strategies.md").write_text(report, encoding="utf-8")
    df.to_csv(out / "strategies.csv", index=False)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
