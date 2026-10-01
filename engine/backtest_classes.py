"""
Radar — réglages par classe d'actif : test hors échantillon (walk-forward).

Pour chaque classe (actions, indices/ETF, cryptos), on cherche le meilleur réglage sur les
années d'apprentissage, puis on le juge sur la dernière année, que la calibration n'a jamais vue.
On le compare au réglage commun actuel (achat ≥ 6, vente ≥ 5, RSI 30/70, tendance en bonus)
et à un réglage commun ré-optimisé de la même façon (pour isoler l'apport du « par classe »).

Paramètres explorés, volontairement peu nombreux pour limiter le surapprentissage :
  - seuil du score : 4, 5, 6, 7 ;
  - bornes du RSI : 30/70, 25/75, 35/65 ;
  - tendance de fond : en bonus (±1, actuel) ou en filtre strict (achat seulement au-dessus de la MM200,
    vente seulement en dessous).
Critère de choix : solidité statistique (t) de l'excès à 20 séances, avec au moins 30 signaux.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screener as S  # noqa: E402

UNIVERSE = {
    "action": ["MC.PA", "AI.PA", "OR.PA", "TTE.PA", "SAN.PA", "BNP.PA", "AIR.PA", "SU.PA", "RMS.PA", "KER.PA",
               "DG.PA", "CAP.PA", "STMPA.PA", "EL.PA", "RI.PA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
               "TSLA", "JPM", "XOM", "KO", "JNJ", "PG", "V"],
    "etf/indice": ["^FCHI", "CW8.PA", "^GSPC", "^IXIC", "^GDAXI", "^STOXX50E", "^FTSE", "^N225", "^RUT", "^HSI",
                   "GC=F", "SI=F", "EEM", "EWZ"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "ADA-USD", "DOGE-USD", "LINK-USD",
               "AVAX-USD", "DOT-USD", "LTC-USD", "TRX-USD", "XLM-USD", "ATOM-USD"],
}
H = 20
COOLDOWN = S.COOLDOWN_BARS
THRS = (4, 5, 6, 7)
RSI_BANDS = ((30, 70), (25, 75), (35, 65))
TREND = ("bonus", "strict")
CURRENT = {"buy": (6, (30, 70), "bonus"), "sell": (5, (30, 70), "bonus")}


def score(d, i, lo, hi, trend):
    """Score de production (screener.score_at) avec bornes RSI et mode de tendance paramétrables."""
    g = lambda k, j=i: float(d[k][j])  # noqa: E731
    ok = S._ok
    c0, c1, o1 = g("close", i - 1), g("close"), g("open")
    buy, sell = [], []
    r0, r1 = g("rsi", i - 1), g("rsi")
    if ok(r0, r1):
        if r0 < lo <= r1: buy.append((2, True))
        elif r1 < lo: buy.append((1, False))
        if r0 > hi >= r1: sell.append((2, True))
        elif r1 > hi: sell.append((1, False))
    m0, m1, s0, s1 = g("macd", i - 1), g("macd"), g("macd_sig", i - 1), g("macd_sig")
    if S.cross_up(m0, m1, s0, s1):
        buy.append((2, True))
        if m1 < 0: buy.append((1, False))
    if S.cross_down(m0, m1, s0, s1):
        sell.append((2, True))
        if m1 > 0: sell.append((1, False))
    lo0, lo1, up0, up1 = g("bb_lo", i - 1), g("bb_lo"), g("bb_up", i - 1), g("bb_up")
    if ok(lo0, lo1):
        if c0 < lo0 and c1 > lo1: buy.append((2, True))
        elif c1 < lo1: buy.append((1, False))
    if ok(up0, up1):
        if c0 > up0 and c1 < up1: sell.append((2, True))
        elif c1 > up1: sell.append((1, False))
    k0, k1, d0, d1 = g("stoch_k", i - 1), g("stoch_k"), g("stoch_d", i - 1), g("stoch_d")
    if S.cross_up(k0, k1, d0, d1) and min(k0, k1) < 20: buy.append((1, True))
    if S.cross_down(k0, k1, d0, d1) and max(k0, k1) > 80: sell.append((1, True))
    a0, a1, b0, b1 = g("sma50", i - 1), g("sma50"), g("sma200", i - 1), g("sma200")
    if S.cross_up(a0, a1, b0, b1): buy.append((3, True))
    if S.cross_down(a0, a1, b0, b1): sell.append((3, True))
    if S.cross_up(c0, c1, b0, b1): buy.append((2, True))
    if S.cross_down(c0, c1, b0, b1): sell.append((2, True))
    up = ok(b1) and c1 > b1
    if ok(b1):
        if up: buy.append((1, False)); sell.append((-1, False))
        else: buy.append((-1, False)); sell.append((1, False))
    vr = g("vol_ratio")
    if ok(vr) and vr >= 1.5:
        if c1 > o1 and any(e for _, e in buy): buy.append((1, False))
        if c1 < o1 and any(e for _, e in sell): sell.append((1, False))
    bs, be = sum(p for p, _ in buy), any(e for _, e in buy)
    ss, se = sum(p for p, _ in sell), any(e for _, e in sell)
    if trend == "strict":
        if not up: be = False
        if up or not ok(b1): se = False
    return bs, be, ss, se


def run_asset(sym, cls):
    df, *_ = S.fetch(sym, "1d")
    df = S.indicators(df)
    close = df["close"].to_numpy(float)
    n = len(df)
    fwd = np.r_[close[H:] / close[:-H] - 1, [np.nan] * H]
    dates = df.index
    test = np.asarray(dates >= dates[-1] - timedelta(days=365))
    valid = df["sma200"].notna().to_numpy()
    base = {}
    for split, mask in (("train", valid & ~test), ("test", valid & test)):
        r = fwd[mask]
        r = r[~np.isnan(r)]
        base[split] = float(r.mean()) if len(r) else 0.0
    d = {k: df[k].to_numpy(float) for k in df.columns}
    rows = []
    last = {}
    for i in range(201, n):
        if np.isnan(fwd[i]):
            continue
        split = "test" if test[i] else "train"
        for (lo, hi), trend in product(RSI_BANDS, TREND):
            bs, be, ss, se = score(d, i, lo, hi, trend)
            for side, sc, ev in (("buy", bs, be), ("sell", ss, se)):
                if not ev:
                    continue
                for t in THRS:
                    key = (side, t, lo, hi, trend)
                    if sc >= t and i - last.get(key, -10**9) >= COOLDOWN:
                        last[key] = i
                        sign = 1 if side == "buy" else -1
                        rows.append((sym, cls, split, side, t, f"{lo}/{hi}", trend,
                                     sign * fwd[i], sign * (fwd[i] - base[split])))
    return rows


def summarize(sub: pd.DataFrame):
    n = len(sub)
    if n == 0:
        return {"n": 0, "win": np.nan, "gain": np.nan, "excess": np.nan, "t": np.nan}
    ex = sub["excess"].to_numpy()
    sd = ex.std(ddof=1) if n > 1 else np.nan
    t = ex.mean() / (sd / math.sqrt(n)) if sd and sd > 0 else np.nan
    return {"n": n, "win": (sub["gain"] > 0).mean() * 100, "gain": sub["gain"].mean() * 100,
            "excess": ex.mean() * 100, "t": t}


def pick(train: pd.DataFrame, side: str, min_n=30):
    best, best_t = None, -1e9
    for (t, rsi, trend), g in train[train.side == side].groupby(["thr", "rsi", "trend"]):
        st = summarize(g)
        if st["n"] >= min_n and st["t"] > best_t:
            best, best_t = (int(t), rsi, trend), st["t"]
    return best


def sel(df, side, cfg):
    t, rsi, trend = cfg
    if isinstance(rsi, tuple):
        rsi = f"{rsi[0]}/{rsi[1]}"
    return df[(df.side == side) & (df.thr == t) & (df.rsi == rsi) & (df.trend == trend)]


def fmt(st):
    if not st["n"]:
        return "0 | — | — | — | —"
    mark = " ✅" if st["n"] >= 30 and st["t"] >= 2 else ""
    return f"{st['n']} | {st['win']:.0f} % | {st['gain']:+.2f} % | {st['excess']:+.2f} % | {st['t']:+.1f}{mark}"


def cfg_txt(cfg):
    t, rsi, trend = cfg
    rsi = rsi if isinstance(rsi, str) else f"{rsi[0]}/{rsi[1]}"
    return f"seuil {t}, RSI {rsi}, tendance {'filtre strict' if trend == 'strict' else 'en bonus'}"


def main():
    rows, errors = [], []
    for cls, syms in UNIVERSE.items():
        for sym in syms:
            try:
                r = run_asset(sym, cls)
                rows += r
                print(f"{sym:10s} {len(r):6d}", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{sym} ({e})")
                print(f"{sym}: ERREUR {e}", flush=True)
    df = pd.DataFrame(rows, columns=["sym", "cls", "split", "side", "thr", "rsi", "trend", "gain", "excess"])
    train, test = df[df.split == "train"], df[df.split == "test"]

    L = []
    w = L.append
    w(f"# Réglages par classe d'actif — test hors échantillon\n")
    w(f"{df.sym.nunique()} actifs. Calibration sur les années d'avant, verdict sur la **dernière année**, jamais vue pendant la calibration. "
      "Excès = gain au-delà de la tendance normale de l'actif, à 20 séances. ✅ = au moins 30 signaux et t ≥ 2.\n")
    if errors:
        w("Non chargés : " + ", ".join(errors) + "\n")

    glob = {side: pick(train, side) for side in ("buy", "sell")}
    chosen, totals = {}, {k: {"buy": [], "sell": []} for k in ("actuel", "commun", "classe")}
    for cls in UNIVERSE:
        tr, te = train[train.cls == cls], test[test.cls == cls]
        w(f"\n## {cls}\n")
        w("| Sens | Réglage | Paramètres | Test : n | Réussite | Gain | Excès | t |")
        w("|---|---|---|---|---|---|---|---|")
        chosen[cls] = {}
        for side in ("buy", "sell"):
            own = pick(tr, side) or CURRENT[side]
            chosen[cls][side] = own
            for lab, cfg in (("actuel", CURRENT[side]), ("commun ré-optimisé", glob[side] or CURRENT[side]), ("par classe", own)):
                sub = sel(te, side, cfg)
                totals[{"actuel": "actuel", "commun ré-optimisé": "commun", "par classe": "classe"}[lab]][side].append(sub)
                w(f"| {'Achat' if side == 'buy' else 'Vente'} | {lab} | {cfg_txt(cfg)} | {fmt(summarize(sub))} |")
            w(f"| | *apprentissage, par classe* | | {fmt(summarize(sel(tr, side, own)))} |")

    w("\n## Bilan toutes classes (dernière année)\n")
    w("| Sens | Approche | n | Réussite | Gain | Excès | t |")
    w("|---|---|---|---|---|---|---|")
    for side in ("buy", "sell"):
        for k, lab in (("actuel", "réglage actuel"), ("commun", "commun ré-optimisé"), ("classe", "par classe")):
            w(f"| {'Achat' if side == 'buy' else 'Vente'} | {lab} | {fmt(summarize(pd.concat(totals[k][side])))} |")

    out = Path(os.environ.get("BT_OUT", "bt-out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "class_params.json").write_text(json.dumps(
        {"global": {s: list(c) if c else None for s, c in glob.items()},
         "classes": {c: {s: list(v) for s, v in d.items()} for c, d in chosen.items()}}, indent=2), encoding="utf-8")
    report = "\n".join(L)
    (out / "classes.md").write_text(report, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
