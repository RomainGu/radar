"""
Radar — backtest des indicateurs.

Rejoue, bougie par bougie, chaque indicateur isolé et le score de confluence de Radar
sur un panier d'actifs, puis mesure ce qui s'est passé 5, 10 et 20 séances après chaque signal.
Un signal n'a de valeur que s'il fait mieux que la simple tendance de l'actif sur la même
période (« excès » par rapport à la référence).

Lancement : workflow « Radar — backtest » (onglet Actions). Résultat : résumé de l'exécution
+ fichiers report.md et signals.csv téléchargeables.
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

UNIVERSE = {
    # ta liste
    "BTC-EUR": "crypto", "ETH-EUR": "crypto", "CW8.PA": "etf/indice", "MC.PA": "action", "^FCHI": "etf/indice",
    # grandes valeurs françaises
    "AI.PA": "action", "OR.PA": "action", "TTE.PA": "action", "SAN.PA": "action", "BNP.PA": "action",
    "AIR.PA": "action", "SU.PA": "action", "RMS.PA": "action", "KER.PA": "action", "DG.PA": "action",
    "CAP.PA": "action", "STMPA.PA": "action",
    # grandes valeurs américaines
    "AAPL": "action", "MSFT": "action", "NVDA": "action", "AMZN": "action", "GOOGL": "action",
    "META": "action", "TSLA": "action", "JPM": "action", "XOM": "action", "KO": "action",
    # indices, ETF, or
    "^GSPC": "etf/indice", "^IXIC": "etf/indice", "^GDAXI": "etf/indice", "^STOXX50E": "etf/indice", "GC=F": "etf/indice",
    # cryptos
    "SOL-EUR": "crypto", "XRP-EUR": "crypto", "BNB-EUR": "crypto", "ADA-EUR": "crypto",
}
HORIZONS = (5, 10, 20)
COOLDOWN = S.COOLDOWN_BARS

COMP_LABEL = {
    "rsi": "RSI sort de survente / surachat",
    "macd": "MACD croise son signal",
    "bb": "Réintégration Bollinger",
    "stoch": "Stochastique en zone extrême",
    "cross": "Golden / death cross",
    "ma200": "Cours croise la MM200",
}


def components(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Déclencheurs isolés (achat, vente) — mêmes règles que screener.score_at."""
    c, p = df, df.shift(1)
    up = lambda a, b: (p[a] <= p[b]) & (c[a] > c[b])  # noqa: E731
    dn = lambda a, b: (p[a] >= p[b]) & (c[a] < c[b])  # noqa: E731
    return {
        "rsi": ((p.rsi < 30) & (c.rsi >= 30), (p.rsi > 70) & (c.rsi <= 70)),
        "macd": (up("macd", "macd_sig"), dn("macd", "macd_sig")),
        "bb": ((p.close < p.bb_lo) & (c.close > c.bb_lo), (p.close > p.bb_up) & (c.close < c.bb_up)),
        "stoch": (up("stoch_k", "stoch_d") & (np.minimum(p.stoch_k, c.stoch_k) < 20),
                  dn("stoch_k", "stoch_d") & (np.maximum(p.stoch_k, c.stoch_k) > 80)),
        "cross": (up("sma50", "sma200"), dn("sma50", "sma200")),
        "ma200": (up("close", "sma200"), dn("close", "sma200")),
    }


def score_v2(d: dict[str, np.ndarray], i: int):
    """Score révisé d'après le backtest v1 :
    - MACD : fort (3) seulement quand il croise loin de la tendance (sous zéro à l'achat,
      au-dessus de zéro à la vente) ; sinon 1 point (il était nuisible en suivi de tendance) ;
    - death cross retiré (signal de vente contre-productif sur 4 ans) ;
    - croisement cours / MM200 ramené à 1 point (bruit) ;
    - stochastique : simple confirmation, ne déclenche plus seul.
    """
    g = lambda k, j=i: float(d[k][j])  # noqa: E731
    ok = S._ok
    c0, c1, o1 = g("close", i - 1), g("close"), g("open")
    buy, sell = [], []
    r0, r1 = g("rsi", i - 1), g("rsi")
    if ok(r0, r1):
        if r0 < 30 <= r1: buy.append((2, True))
        elif r1 < 30: buy.append((1, False))
        if r0 > 70 >= r1: sell.append((2, True))
        elif r1 > 70: sell.append((1, False))
    m0, m1, s0, s1 = g("macd", i - 1), g("macd"), g("macd_sig", i - 1), g("macd_sig")
    if S.cross_up(m0, m1, s0, s1): buy.append((3 if m1 < 0 else 1, True))
    if S.cross_down(m0, m1, s0, s1): sell.append((3 if m1 > 0 else 1, True))
    lo0, lo1, up0, up1 = g("bb_lo", i - 1), g("bb_lo"), g("bb_up", i - 1), g("bb_up")
    if ok(lo0, lo1):
        if c0 < lo0 and c1 > lo1: buy.append((2, True))
        elif c1 < lo1: buy.append((1, False))
    if ok(up0, up1):
        if c0 > up0 and c1 < up1: sell.append((2, True))
        elif c1 > up1: sell.append((1, False))
    k0, k1, d0, d1 = g("stoch_k", i - 1), g("stoch_k"), g("stoch_d", i - 1), g("stoch_d")
    if S.cross_up(k0, k1, d0, d1) and min(k0, k1) < 20: buy.append((1, False))
    if S.cross_down(k0, k1, d0, d1) and max(k0, k1) > 80: sell.append((1, False))
    a0, a1, b0, b1 = g("sma50", i - 1), g("sma50"), g("sma200", i - 1), g("sma200")
    if S.cross_up(a0, a1, b0, b1): buy.append((2, True))
    if S.cross_up(c0, c1, b0, b1): buy.append((1, True))
    if S.cross_down(c0, c1, b0, b1): sell.append((1, True))
    if ok(b1):
        if c1 > b1: buy.append((1, False)); sell.append((-1, False))
        else: buy.append((-1, False)); sell.append((1, False))
    vr = g("vol_ratio")
    if ok(vr) and vr >= 1.5:
        if c1 > o1 and any(e for _, e in buy): buy.append((1, False))
        if c1 < o1 and any(e for _, e in sell): sell.append((1, False))
    return (sum(p for p, _ in buy), any(e for _, e in buy), sum(p for p, _ in sell), any(e for _, e in sell))


def with_cooldown(idx: list[int]) -> list[int]:
    out, last = [], -10**9
    for i in idx:
        if i - last >= COOLDOWN:
            out.append(i)
            last = i
    return out


def run_asset(sym: str, cls: str) -> tuple[list[dict], dict]:
    df, cur, name, qtype = S.fetch(sym, "1d")
    df = S.indicators(df)
    df = df[df.index <= df.index[-1]]  # la dernière bougie peut être en cours : on la garde, sans retour futur
    close = df["close"].to_numpy(float)
    n = len(df)
    dates = df.index
    fwd = {h: np.r_[close[h:] / close[:-h] - 1, [np.nan] * h] for h in HORIZONS}
    above = (df["close"] > df["sma200"]).to_numpy()
    valid = df["sma200"].notna().to_numpy()
    end = dates[-1]
    one_year = dates >= end - timedelta(days=365)

    rows = []

    def add(i, side, variant):
        rows.append({"sym": sym, "cls": cls, "date": dates[i], "side": side, "variant": variant,
                     "trend": "avec" if (above[i] if side == "buy" else not above[i]) else "contre",
                     "y1": bool(one_year[i]), **{f"r{h}": fwd[h][i] for h in HORIZONS}})

    # indicateurs isolés
    for key, (b, s) in components(df).items():
        for side, ser in (("buy", b), ("sell", s)):
            idx = [i for i in np.flatnonzero(ser.fillna(False).to_numpy()) if valid[i]]
            for i in with_cooldown(idx):
                add(i, side, f"comp:{key}")

    # score de confluence de Radar (logique de production)
    d = {k: df[k].to_numpy(float) for k in df.columns}
    last = {(side, t): -10**9 for side in ("buy", "sell") for t in (3, 4, 5, 6)}
    for i in range(201, n):
        bs, _, be, ss, _, se = S.score_at(d, i)
        for side, sc, ev in (("buy", bs, be), ("sell", ss, se)):
            if not ev:
                continue
            for t in (3, 4, 5, 6):
                if sc >= t and i - last[(side, t)] >= COOLDOWN:
                    last[(side, t)] = i
                    add(i, side, f"conf>={t}")

    last2 = {(side, t): -10**9 for side in ("buy", "sell") for t in (4, 5, 6, 7)}
    for i in range(201, n):
        bs, be, ss, se = score_v2(d, i)
        for side, sc, ev in (("buy", bs, be), ("sell", ss, se)):
            if not ev:
                continue
            for t in (4, 5, 6, 7):
                if sc >= t and i - last2[(side, t)] >= COOLDOWN:
                    last2[(side, t)] = i
                    add(i, side, f"v2>={t}")

    base = {}
    for scope, mask in (("y1", one_year & valid), ("all", valid)):
        for h in HORIZONS:
            r = fwd[h][mask]
            r = r[~np.isnan(r)]
            base[(scope, h)] = (float(r.mean()) if len(r) else np.nan, float((r > 0).mean()) if len(r) else np.nan)
    return rows, base


def stats(sub: pd.DataFrame, h: int, base: dict, scope: str) -> dict:
    col = f"r{h}"
    sub = sub[sub[col].notna()]
    if sub.empty:
        return {"n": 0}
    sign = np.where(sub.side == "buy", 1.0, -1.0)
    gain = sub[col].to_numpy() * sign  # gain si on suit le signal (vente = éviter la baisse)
    b = np.array([base[(s, scope, h)][0] for s in sub.sym])
    excess = gain - b * sign
    win = (gain > 0).mean()
    sd = excess.std(ddof=1) if len(excess) > 1 else np.nan
    t = excess.mean() / (sd / math.sqrt(len(excess))) if sd and sd > 0 else np.nan
    return {"n": len(sub), "win": win * 100, "gain": gain.mean() * 100, "excess": excess.mean() * 100, "t": t}


def fmt(st: dict) -> str:
    if not st.get("n"):
        return "— | — | — | — | —"
    verdict = "✅" if st["n"] >= 30 and st["t"] >= 2 else ("⚠️" if st["t"] <= -2 and st["n"] >= 30 else "·")
    return (f"{st['n']} | {st['win']:.0f} % | {st['gain']:+.2f} % | {st['excess']:+.2f} % | "
            f"{st['t']:+.1f} {verdict}")


def main():
    all_rows, bases, errors = [], {}, []
    for sym, cls in UNIVERSE.items():
        try:
            rows, base = run_asset(sym, cls)
            all_rows += rows
            for (scope, h), v in base.items():
                bases[(sym, scope, h)] = v
            print(f"{sym:10s} {len(rows):5d} signaux", flush=True)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sym}: {e}")
            print(f"{sym}: ERREUR {e}", flush=True)
    df = pd.DataFrame(all_rows)
    out = Path(os.environ.get("BT_OUT", "bt-out"))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "signals.csv", index=False)

    L = []
    w = L.append
    nsym = df.sym.nunique()
    w(f"# Backtest Radar — {nsym} actifs, bougies jour\n")
    w("Lecture : **Gain** = variation moyenne si l'on suit le signal (pour une vente : baisse évitée). "
      "**Excès** = gain au-delà de la tendance normale de l'actif sur la même période — c'est la vraie valeur ajoutée du signal. "
      "**t** = solidité statistique de l'excès (≥ 2 : probablement réel ✅ ; entre −2 et 2 : indiscernable du hasard ·). "
      "Signaux espacés d'au moins 3 bougies.\n")
    if errors:
        w("Actifs non chargés : " + ", ".join(errors) + "\n")

    w("\n## Comparaison : score actuel vs score révisé (v2)\n")
    w("| Période | Horizon | Score | Sens | n | Réussite | Gain | Excès | t |")
    w("|---|---|---|---|---|---|---|---|---|")
    for scope, title in (("y1", "1 an"), ("all", "4 ans")):
        sc = df[df.y1] if scope == "y1" else df
        bf = {(s, scope, h): bases[(s, scope, h)] for s in df.sym.unique() for h in HORIZONS if (s, scope, h) in bases}
        for h in (10, 20):
            for v in ("conf>=4", "conf>=5", "conf>=6", "v2>=4", "v2>=5", "v2>=6", "v2>=7"):
                for side in ("buy", "sell"):
                    sub = sc[(sc.variant == v) & (sc.side == side)]
                    lab = ("actuel " if v.startswith("conf") else "v2 ") + v.split(">=")[1]
                    w(f"| {title} | {h} | {lab} | {'Achat' if side == 'buy' else 'Vente'} | {fmt(stats(sub, h, bf, scope))} |")

    for scope, title in (("y1", "Dernière année"), ("all", "Contrôle sur ~4 ans")):
        sc = df[df.y1] if scope == "y1" else df
        bb = {(s, h): bases[(s, scope, h)] for s in df.sym.unique() for h in HORIZONS if (s, scope, h) in bases}
        base_full = {(s, scope, h): v for (s, h), v in bb.items()}
        w(f"\n## {title}\n")
        bavg = np.nanmean([v[0] for (s, h), v in bb.items() if h == 20]) * 100
        bwin = np.nanmean([v[1] for (s, h), v in bb.items() if h == 20]) * 100
        w(f"Référence sans signal, à 20 séances : variation moyenne {bavg:+.2f} %, hausse dans {bwin:.0f} % des cas.\n")
        for h in (10, 20):
            w(f"\n### Horizon {h} séances\n")
            w("| Signal | Sens | Tendance | n | Réussite | Gain | Excès | t |")
            w("|---|---|---|---|---|---|---|---|")
            variants = [f"comp:{k}" for k in COMP_LABEL] + ["conf>=3", "conf>=4", "conf>=5", "conf>=6"]
            for v in variants:
                label = COMP_LABEL.get(v.split(":")[1], v) if v.startswith("comp:") else f"Score Radar ≥ {v[-1]}"
                for side in ("buy", "sell"):
                    sub = sc[(sc.variant == v) & (sc.side == side)]
                    for tr in ("tous", "avec", "contre"):
                        s2 = sub if tr == "tous" else sub[sub.trend == tr]
                        w(f"| {label} | {'Achat' if side == 'buy' else 'Vente'} | {tr} | {fmt(stats(s2, h, base_full, scope))} |")
        w(f"\n### Score Radar ≥ 4 par classe d'actif (20 séances)\n")
        w("| Classe | Sens | Tendance | n | Réussite | Gain | Excès | t |")
        w("|---|---|---|---|---|---|---|---|")
        for cls in sorted(df.cls.unique()):
            for side in ("buy", "sell"):
                for tr in ("tous", "avec"):
                    sub = sc[(sc.variant == "conf>=4") & (sc.side == side) & (sc.cls == cls)]
                    if tr == "avec":
                        sub = sub[sub.trend == "avec"]
                    w(f"| {cls} | {'Achat' if side == 'buy' else 'Vente'} | {tr} | {fmt(stats(sub, 20, base_full, scope))} |")

    w("\n## Tes 5 actifs — score Radar ≥ 4, dernière année (20 séances)\n")
    w("| Actif | Sens | n | Réussite | Gain | Excès |")
    w("|---|---|---|---|---|---|")
    base_y1 = {(s, "y1", h): bases[(s, "y1", h)] for s in df.sym.unique() for h in HORIZONS if (s, "y1", h) in bases}
    for sym in list(UNIVERSE)[:5]:
        for side in ("buy", "sell"):
            st = stats(df[(df.y1) & (df.variant == "conf>=4") & (df.side == side) & (df.sym == sym)], 20, base_y1, "y1")
            row = "— | — | — | —" if not st.get("n") else f"{st['n']} | {st['win']:.0f} % | {st['gain']:+.2f} % | {st['excess']:+.2f} %"
            w(f"| {sym} | {'Achat' if side == 'buy' else 'Vente'} | {row} |")

    report = "\n".join(L)
    (out / "report.md").write_text(report, encoding="utf-8")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        Path(summ).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
