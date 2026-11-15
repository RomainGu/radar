"""
Radar — backtest des règles de SORTIE (gestion du trade après l'entrée).

L'entrée est fixée : le RSI 14 qui sort de la survente (la seule règle d'achat validée
sur les deux périodes au backtest du 2 octobre 2026). On compare ensuite des règles de
sortie classiques, déclarées à l'avance :

  E20  : sortie à 20 séances (la mesure actuelle de Radar) ;
  E10  : sortie à 10 séances ;
  E40  : sortie à 40 séances ;
  TR2  : stop suiveur à 2 × ATR sous le plus haut de clôture depuis l'entrée (max 40 séances) ;
  TR3  : idem à 3 × ATR ;
  RSI60: sortie quand le RSI 14 atteint 60 — le rebond a eu lieu (max 40 séances) ;
  BRKT : stop à l'entrée − 2 × ATR et objectif à l'entrée + 3 × ATR, premier touché
         (en clôture), sinon 40 séances ;
  STOPO: stop à l'entrée − 2 × ATR seulement, sinon 40 séances.

Mesure : rendement du trade, comparé à ce qu'aurait donné la simple détention de l'actif
pendant le même nombre de jours (« excès », en rythme quotidien ramené au trade). Jugement
sur deux périodes séparées : A (années d'avant) et B (dernière année), critère pré-déclaré
« meilleur min(t A, t B), au moins 30 trades et un excès positif dans chaque période ».
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
from backtest_classes import UNIVERSE  # noqa: E402

MAXH = 40
COOLDOWN = 10
EXITS = {
    "E20": "Sortie à 20 séances (règle actuelle)",
    "E10": "Sortie à 10 séances",
    "E40": "Sortie à 40 séances",
    "TR2": "Stop suiveur 2 × ATR (max 40)",
    "TR3": "Stop suiveur 3 × ATR (max 40)",
    "RSI60": "Sortie quand le RSI atteint 60 (max 40)",
    "BRKT": "Stop −2 ATR / objectif +3 ATR (max 40)",
    "STOPO": "Stop −2 ATR seul (max 40)",
}


def simulate(close: np.ndarray, rsi: np.ndarray, atr0: float, i: int, rule: str) -> tuple[float, int] | None:
    """Rendement et durée (en bougies) d'un trade ouvert à la clôture i. None si la fin
    de l'historique coupe le trade avant sa sortie."""
    n = len(close)
    entry = close[i]
    if rule in ("E10", "E20", "E40"):
        h = int(rule[1:])
        return (close[i + h] / entry - 1, h) if i + h < n else None
    hi = entry
    for k in range(i + 1, min(i + MAXH, n - 1) + 1):
        c = close[k]
        hi = max(hi, c)
        d = k - i
        if rule in ("TR2", "TR3") and c <= hi - (2 if rule == "TR2" else 3) * atr0:
            return c / entry - 1, d
        if rule == "RSI60" and rsi[k] >= 60:
            return c / entry - 1, d
        if rule in ("BRKT", "STOPO") and c <= entry - 2 * atr0:
            return c / entry - 1, d
        if rule == "BRKT" and c >= entry + 3 * atr0:
            return c / entry - 1, d
        if d >= MAXH:
            return c / entry - 1, d
    return None  # historique trop court pour conclure


def run_asset(sym: str, cls: str):
    df, *_ = S.fetch(sym, "1d")
    df = S.indicators(df)
    close = df["close"].to_numpy(float)
    rsi = df["rsi"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    n = len(df)
    dates = df.index
    is_b = np.asarray(dates >= dates[-1] - timedelta(days=365))
    valid = df["sma200"].notna().to_numpy()
    # rythme quotidien moyen de l'actif par période (référence de détention)
    daily = {}
    for per, m in (("A", valid & ~is_b), ("B", valid & is_b)):
        r = np.diff(np.log(close))[m[1:]]
        daily[per] = float(r.mean()) if len(r) else 0.0
    r0, r1 = np.r_[np.nan, rsi[:-1]], rsi
    entries = np.flatnonzero((r0 < 30) & (r1 >= 30) & valid)
    rows, last = [], -10**9
    for i in entries:
        if i < 201 or i - last < COOLDOWN or not math.isfinite(atr[i]) or atr[i] <= 0:
            continue
        last = i
        per = "B" if is_b[i] else "A"
        for rule in EXITS:
            out = simulate(close, rsi, atr[i], i, rule)
            if out is None:
                continue
            ret, days = out
            hold = math.exp(daily[per] * days) - 1  # détention passive de même durée
            rows.append({"sym": sym, "cls": cls, "per": per, "rule": rule,
                         "ret": ret, "days": days, "x": ret - hold})
    return rows


def st(sub: pd.DataFrame):
    n = len(sub)
    if n < 2:
        return {"n": n}
    x = sub["x"].to_numpy()
    sd = x.std(ddof=1)
    return {"n": n, "win": (sub["ret"] > 0).mean() * 100, "ret": sub["ret"].mean() * 100,
            "days": sub["days"].mean(), "x": x.mean() * 100,
            "t": x.mean() / (sd / math.sqrt(n)) if sd > 0 else np.nan}


def cell(s):
    if s.get("n", 0) < 2:
        return "— | — | — | — | — | —"
    return (f"{s['n']} | {s['win']:.0f} % | {s['ret']:+.2f} % | {s['days']:.0f} j | "
            f"{s['x']:+.2f} % | {s['t']:+.1f}")


def main():
    rows, errors = [], []
    for cls, syms in UNIVERSE.items():
        for sym in syms:
            try:
                r = run_asset(sym, cls)
                rows += r
                print(f"{sym:10s} {len(r):5d}", flush=True)
            except Exception as e:  # noqa: BLE001
                errors.append(sym)
                print(f"{sym}: ERREUR {e}", flush=True)
    df = pd.DataFrame(rows)
    L = []
    w = L.append
    w("# Règles de sortie après une entrée « RSI sort de la survente »\n")
    w(f"{df.sym.nunique()} actifs, bougies jour. A : années d'avant · B : dernière année. "
      "Rendement par trade, comparé à la détention passive de l'actif sur la même durée (« Excès »). "
      "Critère pré-déclaré : meilleur min(t A, t B), ≥ 30 trades et excès > 0 dans chaque période.\n")
    if errors:
        w("Non chargés : " + ", ".join(errors) + "\n")

    for scope, sub0 in (("Toutes classes", df), *[(c, df[df.cls == c]) for c in UNIVERSE]):
        w(f"\n## {scope}\n")
        w("| Sortie | A : n | Réussite | Rdt/trade | Durée | Excès | t | B : n | Réussite | Rdt/trade | Durée | Excès | t | Robustesse |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        best, bname = -1e9, None
        for rule, label in EXITS.items():
            a = st(sub0[(sub0.rule == rule) & (sub0.per == "A")])
            b = st(sub0[(sub0.rule == rule) & (sub0.per == "B")])
            ok = a.get("n", 0) >= 30 and b.get("n", 0) >= 30 and a.get("x", -1) > 0 and b.get("x", -1) > 0
            rob = min(a["t"], b["t"]) if ok else np.nan
            if ok and rob > best:
                best, bname = rob, rule
            w(f"| {label} | {cell(a)} | {cell(b)} | {'' if np.isnan(rob) else f'{rob:+.1f}'} |")
        w(f"\n**Retenue ({scope.lower()}) : {EXITS.get(bname, 'aucune règle ne passe le critère')}**\n")

    report = "\n".join(L)
    out = Path(os.environ.get("BT_OUT", "bt-out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "exits.md").write_text(report, encoding="utf-8")
    df.to_csv(out / "exits.csv", index=False)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
