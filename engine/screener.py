"""
Radar — moteur d'analyse technique et d'alertes.

Tourne dans GitHub Actions (toutes les 30 min). Pour chaque actif de watchlist.json :
  1. récupère l'historique de cours (Yahoo Finance) ;
  2. calcule les indicateurs (MM50/200, RSI, MACD, Bollinger, Stochastique, ATR, volume) ;
  3. score de confluence achat / vente sur la dernière bougie CLÔTURÉE ;
  4. backtest du même signal sur l'historique de l'actif (fiabilité passée) ;
  5. envoie les notifications (ntfy et/ou Telegram) selon le mode choisi par actif ;
  6. écrit status.json, lu par l'appli iPhone.

Aucun indicateur ne prédit le marché : le score signale des zones statistiquement
intéressantes, pas des certitudes. Ce n'est pas un conseil en investissement.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

PARIS = ZoneInfo("Europe/Paris")
NOW = datetime.now(timezone.utc)

ROOT = Path(__file__).resolve().parent.parent
WATCHLIST = Path(os.environ.get("WATCHLIST_PATH", ROOT / "watchlist.json"))
STATE_IN = Path(os.environ.get("STATE_IN", ROOT / "data-branch" / "status.json"))
STATUS_OUT = Path(os.environ.get("STATUS_OUT", ROOT / "out" / "status.json"))

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").strip().rstrip("/")
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
APP_URL = os.environ.get("APP_URL", "").strip()
TEST_NOTIF = os.environ.get("TEST_NOTIF", "").lower() in ("1", "true", "yes")
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

# Seuils de déclenchement (achat, vente) selon la sensibilité choisie pour l'actif.
# Calibrés par backtest (36 actifs, 1 an + 4 ans) : en dessous de 6 à l'achat et de 5 à la vente,
# les signaux ne faisaient pas mieux que la tendance normale des actifs.
THRESHOLDS = {"prudent": (6, 6), "normal": (6, 5), "reactif": (5, 4)}
# Nombre de bougies minimum entre deux alertes de même sens sur un actif
COOLDOWN_BARS = 3
# Horizon du backtest (en bougies) selon l'unité de temps
HORIZON = {"4h": 30, "1d": 20, "1wk": 8}
HORIZON_LABEL = {"4h": "5 j", "1d": "20 séances", "1wk": "8 sem."}
TF_LABEL = {"4h": "4 h", "1d": "jour", "1wk": "semaine"}
TF_DELTA = {"4h": timedelta(hours=4), "1d": timedelta(days=1), "1wk": timedelta(days=7)}

CURRENCY_SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£", "GBp": "p", "CHF": "CHF", "JPY": "¥"}


# --------------------------------------------------------------------------- utils

def log(*a):
    print(*a, flush=True)


def fnum(x: float | None, dec: int | None = None) -> str:
    """Nombre au format français (espace fine pour les milliers, virgule décimale)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if dec is None:
        ax = abs(x)
        dec = 2 if ax >= 1 else (4 if ax >= 0.01 else 6)
        if ax >= 10000:
            dec = 0
    s = f"{x:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",")


def fprice(x, cur):
    sym = CURRENCY_SYMBOL.get(cur or "", cur or "")
    return f"{fnum(x)} {sym}".strip()


def pct(x, dec=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return ("+" if x > 0 else "") + fnum(x, dec) + " %"


def clean(v):
    """Convertit pour JSON (NaN -> None, numpy -> python)."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 8)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# --------------------------------------------------------------------------- données

def fetch_yfinance(symbol: str, tf: str):
    import yfinance as yf

    interval, period = {"4h": ("1h", "730d"), "1d": ("1d", "5y"), "1wk": ("1wk", "15y")}[tf]
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval, auto_adjust=True, actions=False)
    if df is None or df.empty:
        raise ValueError("aucune donnée")
    meta = getattr(t, "history_metadata", {}) or {}
    cur = meta.get("currency")
    name = meta.get("longName") or meta.get("shortName")
    qtype = meta.get("instrumentType")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df, cur, name, qtype


def fetch_chart_api(symbol: str, tf: str):
    """Repli : API chart de Yahoo appelée directement."""
    interval, rng = {"4h": ("1h", "730d"), "1d": ("1d", "5y"), "1wk": ("1wk", "max")}[tf]
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"interval": interval, "range": rng},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    tz = res["meta"].get("exchangeTimezoneName") or "UTC"
    idx = idx.tz_convert(tz)
    df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]}, index=idx)
    adj = res["indicators"].get("adjclose")
    if adj:
        ratio = pd.Series(adj[0]["adjclose"], index=idx) / df["close"]
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c] * ratio
    meta = res["meta"]
    return df, meta.get("currency"), meta.get("longName") or meta.get("shortName"), meta.get("instrumentType")


def fetch(symbol: str, tf: str):
    last = None
    for fn in (fetch_yfinance, fetch_chart_api):
        for attempt in range(2):
            try:
                df, cur, name, qtype = fn(symbol, tf)
                df = df.dropna(subset=["close"])
                if len(df) < 60:
                    raise ValueError(f"historique trop court ({len(df)} bougies)")
                if tf == "4h":
                    df = resample_4h(df)
                return df, cur, name, qtype
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Données indisponibles pour {symbol} : {last}")


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
    d = df.copy()
    d.index = idx
    out = d.resample("4h", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


def bar_is_closed(ts: pd.Timestamp, tf: str, is_crypto: bool) -> bool:
    start = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    end = start + TF_DELTA[tf]
    if tf == "1d" and not is_crypto:
        # Bourses européennes et US fermées à 21h30 UTC : la séance du jour est close.
        local_day = ts.date()
        session_close = datetime(local_day.year, local_day.month, local_day.day, 21, 30, tzinfo=timezone.utc)
        end = min(end, pd.Timestamp(session_close))
    if tf == "1wk" and not is_crypto:
        end = start + timedelta(days=4, hours=22)
    return NOW >= end


# --------------------------------------------------------------------------- indicateurs

def rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"].fillna(0)
    out = df.copy()
    out["sma20"] = c.rolling(20).mean()
    out["sma50"] = c.rolling(50).mean()
    out["sma200"] = c.rolling(200).mean()

    delta = c.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = rma(gain, 14) / rma(loss, 14)
    out["rsi"] = 100 - 100 / (1 + rs)

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_sig"]

    std = c.rolling(20).std(ddof=0)
    out["bb_up"] = out["sma20"] + 2 * std
    out["bb_lo"] = out["sma20"] - 2 * std
    out["pctb"] = (c - out["bb_lo"]) / (out["bb_up"] - out["bb_lo"])

    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k_raw = 100 * (c - ll) / (hh - ll)
    out["stoch_k"] = k_raw.rolling(3).mean()
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr"] = rma(tr, 14)

    vavg = v.rolling(20).mean()
    out["vol_ratio"] = np.where(vavg > 0, v / vavg, np.nan)
    return out


def _ok(*vals) -> bool:
    return all(x is not None and not (isinstance(x, float) and math.isnan(x)) for x in vals)


def cross_up(a0, a1, b0, b1) -> bool:
    return _ok(a0, a1, b0, b1) and a0 <= b0 and a1 > b1


def cross_down(a0, a1, b0, b1) -> bool:
    return _ok(a0, a1, b0, b1) and a0 >= b0 and a1 < b1


def score_at(d: dict[str, np.ndarray], i: int):
    """Score de confluence achat et vente à la bougie i.

    Retourne (buy_score, buy_reasons, buy_has_event, sell_score, sell_reasons, sell_has_event).
    Les « événements » sont des croisements apparus sur cette bougie ; les « états » ne font
    que renforcer. Une alerte exige au moins un événement : sinon on serait alerté à chaque bougie.
    """
    g = lambda k, j=i: float(d[k][j])  # noqa: E731
    c0, c1, o1 = g("close", i - 1), g("close"), g("open")
    buy, sell = [], []  # (points, texte, est_un_événement)

    # RSI 14
    r0, r1 = g("rsi", i - 1), g("rsi")
    if _ok(r0, r1):
        if r0 < 30 <= r1:
            buy.append((2, f"RSI sort de la survente ({fnum(r1, 0)})", True))
        elif r1 < 30:
            buy.append((1, f"RSI en survente ({fnum(r1, 0)})", False))
        if r0 > 70 >= r1:
            sell.append((2, f"RSI sort du surachat ({fnum(r1, 0)})", True))
        elif r1 > 70:
            sell.append((1, f"RSI en surachat ({fnum(r1, 0)})", False))

    # MACD 12/26/9
    m0, m1, s0, s1 = g("macd", i - 1), g("macd"), g("macd_sig", i - 1), g("macd_sig")
    if cross_up(m0, m1, s0, s1):
        buy.append((2, "MACD croise sa ligne de signal à la hausse", True))
        if m1 < 0:
            buy.append((1, "… sous zéro : retournement précoce", False))
    if cross_down(m0, m1, s0, s1):
        sell.append((2, "MACD croise sa ligne de signal à la baisse", True))
        if m1 > 0:
            sell.append((1, "… au-dessus de zéro : essoufflement", False))

    # Bandes de Bollinger 20/2
    lo0, lo1, up0, up1 = g("bb_lo", i - 1), g("bb_lo"), g("bb_up", i - 1), g("bb_up")
    if _ok(lo0, lo1):
        if c0 < lo0 and c1 > lo1:
            buy.append((2, "Réintègre la bande de Bollinger basse", True))
        elif c1 < lo1:
            buy.append((1, "Clôture sous la bande de Bollinger basse", False))
    if _ok(up0, up1):
        if c0 > up0 and c1 < up1:
            sell.append((2, "Réintègre la bande de Bollinger haute", True))
        elif c1 > up1:
            sell.append((1, "Clôture au-dessus de la bande de Bollinger haute", False))

    # Stochastique 14/3/3
    k0, k1, d0, d1 = g("stoch_k", i - 1), g("stoch_k"), g("stoch_d", i - 1), g("stoch_d")
    if cross_up(k0, k1, d0, d1) and min(k0, k1) < 20:
        buy.append((1, "Stochastique croise à la hausse en zone basse", True))
    if cross_down(k0, k1, d0, d1) and max(k0, k1) > 80:
        sell.append((1, "Stochastique croise à la baisse en zone haute", True))

    # Moyennes mobiles 50 / 200
    a0, a1, b0, b1 = g("sma50", i - 1), g("sma50"), g("sma200", i - 1), g("sma200")
    if cross_up(a0, a1, b0, b1):
        buy.append((3, "Golden cross : MM50 passe au-dessus de la MM200", True))
    if cross_down(a0, a1, b0, b1):
        sell.append((3, "Death cross : MM50 passe sous la MM200", True))
    if cross_up(c0, c1, b0, b1):
        buy.append((2, "Le cours repasse au-dessus de la MM200", True))
    if cross_down(c0, c1, b0, b1):
        sell.append((2, "Le cours casse la MM200 à la baisse", True))

    # Contexte de tendance de fond
    if _ok(b1):
        if c1 > b1:
            buy.append((1, "Tendance de fond haussière (au-dessus de la MM200)", False))
            sell.append((-1, "Contre-tendance : le fond reste haussier", False))
        else:
            buy.append((-1, "Contre-tendance : sous la MM200", False))
            sell.append((1, "Tendance de fond baissière (sous la MM200)", False))

    # Confirmation par le volume (uniquement s'il y a un autre événement)
    vr = g("vol_ratio")
    if _ok(vr) and vr >= 1.5:
        if c1 > o1 and any(e for _, _, e in buy):
            buy.append((1, f"Volume élevé ({fnum(vr, 1)}× la moyenne)", False))
        if c1 < o1 and any(e for _, _, e in sell):
            sell.append((1, f"Volume élevé ({fnum(vr, 1)}× la moyenne)", False))

    bs = sum(p for p, _, _ in buy)
    ss = sum(p for p, _, _ in sell)
    return (bs, [t for _, t, _ in buy], any(e for _, _, e in buy),
            ss, [t for _, t, _ in sell], any(e for _, _, e in sell))


def bias_of(row) -> tuple[str, int]:
    pts = 0
    for cond in (
        (row["close"], row["sma200"]),
        (row["sma50"], row["sma200"]),
        (row["macd"], row["macd_sig"]),
        (row["rsi"], 50.0),
    ):
        a, b = cond
        if _ok(float(a), float(b)):
            pts += 1 if a > b else -1
    label = "haussier" if pts >= 2 else "baissier" if pts <= -2 else "neutre"
    return label, pts


def backtest(d: dict[str, np.ndarray], n: int, tf: str, thresholds: tuple[int, int]):
    """Rejoue le signal sur l'historique : que s'est-il passé H bougies après chaque signal ?"""
    H = HORIZON[tf]
    close = d["close"]
    res = {"buy": [], "sell": []}
    last = {"buy": -999, "sell": -999}
    start = 201 if n > 260 else 30
    for i in range(start, n):
        bs, _, be, ss, _, se = score_at(d, i)
        for side, sc, ev, threshold in (("buy", bs, be, thresholds[0]), ("sell", ss, se, thresholds[1])):
            if ev and sc >= threshold and i - last[side] >= COOLDOWN_BARS:
                last[side] = i
                if i + H < n:
                    res[side].append(close[i + H] / close[i] - 1)

    fwd = close[H:] / close[:-H] - 1 if n > H else np.array([])
    base = float(np.nanmean(fwd)) * 100 if len(fwd) else None
    out = {"horizon": H, "horizon_label": HORIZON_LABEL[tf], "baseline": base}
    for side, arr in res.items():
        arr = np.array(arr)
        if len(arr):
            wins = (arr > 0) if side == "buy" else (arr < 0)
            out[side] = {"n": int(len(arr)), "win": float(wins.mean() * 100), "avg": float(arr.mean() * 100)}
        else:
            out[side] = {"n": 0, "win": None, "avg": None}
    return out


# --------------------------------------------------------------------------- notifications

def notify(title: str, body: str, side: str = "info", priority: int = 3, sym: str | None = None) -> bool:
    tags = {"buy": ["green_circle"], "sell": ["red_circle"], "target": ["dart"],
            "stop": ["warning"], "info": ["bar_chart"], "test": ["white_check_mark"]}.get(side, [])
    sent = False
    if DRY_RUN:
        log(f"[DRY] {title}\n{body}\n")
        return True
    if NTFY_TOPIC:
        payload = {"topic": NTFY_TOPIC, "title": title, "message": body, "tags": tags, "priority": priority}
        if APP_URL:
            payload["click"] = APP_URL + (f"#asset/{sym}" if sym else "")
        try:
            r = requests.post(NTFY_SERVER, data=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type": "application/json"}, timeout=15)
            r.raise_for_status()
            sent = True
        except Exception as e:  # noqa: BLE001
            log("ntfy : échec", e)
    if TG_TOKEN and TG_CHAT:
        try:
            r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                              json={"chat_id": TG_CHAT, "text": f"{title}\n\n{body}",
                                    "disable_web_page_preview": True}, timeout=15)
            r.raise_for_status()
            sent = True
        except Exception as e:  # noqa: BLE001
            log("Telegram : échec", e)
    if not NTFY_TOPIC and not (TG_TOKEN and TG_CHAT):
        log("Aucun canal de notification configuré :", title)
    return sent


def in_quiet_hours(settings) -> bool:
    qs, qe = settings.get("quiet_start"), settings.get("quiet_end")
    if qs is None or qe is None or qs == qe:
        return False
    h = NOW.astimezone(PARIS).hour
    return (qs <= h or h < qe) if qs > qe else (qs <= h < qe)


# --------------------------------------------------------------------------- analyse d'un actif

def analyse(asset: dict, prev_state: dict, settings: dict):
    sym = asset["symbol"].strip().upper()
    tf = asset.get("timeframe", "1d")
    if tf not in TF_DELTA:
        tf = "1d"
    sens = asset.get("sensitivity", "normal")
    thr_b, thr_s = THRESHOLDS.get(sens, THRESHOLDS["normal"])
    mode = asset.get("alert", "both")
    st = dict(prev_state or {})
    alerts = []

    used = sym
    try:
        df, cur, long_name, qtype = fetch(sym, tf)
    except RuntimeError:
        # Yahoo ne cote pas toutes les cryptos en euros : on se rabat sur la paire en dollars.
        if not sym.endswith("-EUR"):
            raise
        used = sym[:-4] + "-USD"
        df, cur, long_name, qtype = fetch(used, tf)
    is_crypto = (qtype or "").upper() == "CRYPTOCURRENCY" or asset.get("type") == "crypto"
    df = indicators(df)

    closed_mask = [bar_is_closed(ts, tf, is_crypto) for ts in df.index[-3:]]
    last_closed = len(df) - 1 if closed_mask[-1] else len(df) - 2
    live = df.iloc[-1]
    price = float(live["close"])

    d = {k: df[k].to_numpy(dtype=float) for k in df.columns}
    n_closed = last_closed + 1
    i = last_closed
    bs, br, be, ss, sr, se = score_at(d, i)
    row = df.iloc[i]
    bias, bias_pts = bias_of(row)
    bt = backtest(d, n_closed, tf, (thr_b, thr_s))

    bar_ts = df.index[i].isoformat()
    atr = float(row["atr"]) if _ok(float(row["atr"])) else None

    def bars_since(ts_iso):
        if not ts_iso:
            return 10**6
        try:
            t = pd.Timestamp(ts_iso)
            return int((df.index[: i + 1] > t).sum())
        except Exception:  # noqa: BLE001
            return 10**6

    name = asset.get("name") or long_name or sym
    head = f"{name} ({sym})" if name != sym else sym

    def fiab(side):
        b = bt[side]
        if not b["n"]:
            return "Pas encore de signal comparable dans l'historique."
        return (f"Historique : {fnum(b['win'], 0)} % de réussite à {bt['horizon_label']} "
                f"sur {b['n']} signaux (moyenne {pct(b['avg'])}).")

    # --- signaux techniques (bougie clôturée)
    for side, sc, reasons, ev in (("buy", bs, br, be), ("sell", ss, sr, se)):
        if mode not in (side, "both"):
            continue
        key = f"last_{side}"
        thr = thr_b if side == "buy" else thr_s
        if ev and sc >= thr and st.get(key) != bar_ts and bars_since(st.get(key)) >= COOLDOWN_BARS:
            st[key] = bar_ts
            strength = "fort" if sc >= thr + 2 else "modéré"
            cl = float(row["close"])
            lines = [f"Signal {strength} · score {sc} · bougie {TF_LABEL[tf]}",
                     f"Clôture {fprice(cl, cur)}"]
            lines += [f"• {r}" for r in reasons]
            if atr:
                if side == "buy":
                    lines.append(f"Stop indicatif : {fprice(cl - 2 * atr, cur)} (2 × ATR)")
                else:
                    lines.append(f"Invalidation si retour au-dessus de {fprice(cl + 2 * atr, cur)}")
            lines.append(fiab(side))
            alerts.append({
                "symbol": sym, "name": name, "side": side, "kind": "signal", "score": sc,
                "strength": strength, "price": cl, "currency": cur, "bar": bar_ts, "tf": tf,
                "title": f"{'ACHAT' if side == 'buy' else 'VENTE'} · {head}",
                "body": "\n".join(lines), "reasons": reasons,
                "priority": 4 if strength == "fort" else 3,
            })

    # --- niveaux de prix personnels (cours en direct)
    levels = [("target_buy", "below", "target", "Zone d'achat atteinte", "buy"),
              ("stop", "below", "stop", "Stop atteint", "sell"),
              ("target_sell", "above", "target", "Objectif de vente atteint", "sell")]
    for field, direction, tag, label, side in levels:
        lvl = asset.get(field)
        if lvl in (None, "", 0) or mode == "off":
            continue
        lvl = float(lvl)
        fkey = f"fired_{field}"
        hit = price <= lvl if direction == "below" else price >= lvl
        rearm = price > lvl * 1.02 if direction == "below" else price < lvl * 0.98
        if hit and not st.get(fkey):
            st[fkey] = NOW.isoformat()
            alerts.append({
                "symbol": sym, "name": name, "side": side, "kind": tag, "price": price, "currency": cur,
                "title": f"{label} · {head}",
                "body": f"Cours {fprice(price, cur)} · niveau fixé {fprice(lvl, cur)}",
                "priority": 5 if tag == "stop" else 4, "tag": tag,
            })
        elif rearm and st.get(fkey):
            st.pop(fkey, None)

    # --- données pour l'appli
    look = 252 if tf == "1d" else (52 if tf == "1wk" else 6 * 7 * 5)
    win = df.iloc[-look:]
    hi, lo = float(win["high"].max()), float(win["low"].min())
    prev_close = float(df["close"].iloc[-2])
    entry = asset.get("entry")
    spark = df["close"].iloc[-90:].to_list()

    def rv(k):
        return float(row[k])

    status = {
        "ok": True, "symbol": sym, "source_symbol": used, "name": name, "long_name": long_name, "currency": cur,
        "type": "crypto" if is_crypto else (asset.get("type") or (qtype or "").lower()),
        "tf": tf, "sensitivity": sens, "threshold": thr_b, "threshold_buy": thr_b, "threshold_sell": thr_s, "alert": mode,
        "price": price, "change": (price / prev_close - 1) * 100,
        "bar": bar_ts, "bar_closed": bool(closed_mask[-1]),
        "bias": bias, "bias_pts": bias_pts,
        "score_buy": bs, "score_sell": ss, "event_buy": be, "event_sell": se,
        "reasons_buy": br, "reasons_sell": sr,
        "ind": {
            "rsi": rv("rsi"), "macd": rv("macd"), "macd_sig": rv("macd_sig"), "macd_hist": rv("macd_hist"),
            "sma50": rv("sma50"), "sma200": rv("sma200"), "pctb": rv("pctb"),
            "stoch_k": rv("stoch_k"), "stoch_d": rv("stoch_d"), "atr": rv("atr"),
            "atr_pct": rv("atr") / rv("close") * 100 if _ok(rv("atr")) else None,
            "vol_ratio": rv("vol_ratio"),
            "dist_sma200": (rv("close") / rv("sma200") - 1) * 100 if _ok(rv("sma200")) else None,
        },
        "range": {"high": hi, "low": lo, "pos": (price - lo) / (hi - lo) * 100 if hi > lo else None,
                  "from_high": (price / hi - 1) * 100},
        "position": ({"entry": float(entry), "pnl": (price / float(entry) - 1) * 100} if entry else None),
        "backtest": bt,
        "spark": spark,
        "updated": NOW.isoformat(),
    }
    return status, st, alerts


# --------------------------------------------------------------------------- programme principal

def main():
    wl = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    settings = wl.get("settings", {})
    assets = [a for a in wl.get("assets", []) if a.get("symbol")]

    prev = {}
    if STATE_IN.exists():
        try:
            prev = json.loads(STATE_IN.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
    state = prev.get("state", {})
    history = prev.get("history", [])
    pending = prev.get("pending", [])
    prev_assets = prev.get("assets", {})

    if TEST_NOTIF:
        ok = notify("Radar est connecté", "Les alertes arriveront ici. Bonne surveillance !", "test", 3)
        log("Notification de test envoyée" if ok else "Notification de test : échec")

    out_assets, new_alerts = {}, []
    for a in assets:
        sym = a["symbol"].strip().upper()
        try:
            s, st, al = analyse(a, state.get(sym, {}), settings)
            out_assets[sym] = s
            state[sym] = st
            new_alerts += al
            log(f"{sym:12s} {s['bias']:9s} achat {s['score_buy']:>2} vente {s['score_sell']:>2} · {len(al)} alerte(s)")
        except Exception as e:  # noqa: BLE001
            log(f"{sym}: ERREUR {e}")
            traceback.print_exc(limit=1)
            old = prev_assets.get(sym)
            out_assets[sym] = {**(old or {}), "ok": False, "symbol": sym, "name": a.get("name") or sym,
                               "error": str(e)[:200], "stale": bool(old), "alert": a.get("alert", "both")}

    # Envoi (ou mise en attente pendant les heures de silence)
    quiet = in_quiet_hours(settings)
    for al in new_alerts:
        al["at"] = NOW.isoformat()
    if quiet:
        pending += new_alerts
        log(f"Heures de silence : {len(new_alerts)} alerte(s) mises en attente")
    else:
        for al in pending:
            al["body"] += "\n(détectée pendant les heures de silence)"
        queue = pending + new_alerts
        pending = []
        for al in queue[:8]:
            notify(al["title"], al["body"], al.get("tag") or al["side"], al.get("priority", 3), al["symbol"])
        if len(queue) > 8:
            notify(f"{len(queue) - 8} autres alertes", "Ouvre Radar pour les voir toutes.", "info", 3)
    history = (new_alerts[::-1] + history)[:150]

    # Résumé quotidien
    digest_hour = settings.get("digest_hour")
    today = NOW.astimezone(PARIS).date().isoformat()
    digest_sent = prev.get("digest_sent")
    if digest_hour not in (None, "", -1) and NOW.astimezone(PARIS).hour == int(digest_hour) \
            and digest_sent != today and out_assets:
        lines = []
        for s in sorted(out_assets.values(), key=lambda x: -(x.get("score_buy", 0) - x.get("score_sell", 0))):
            if not s.get("ok"):
                lines.append(f"⚠︎ {s['symbol']} : données indisponibles")
                continue
            arrow = {"haussier": "↗", "baissier": "↘"}.get(s["bias"], "→")
            lines.append(f"{arrow} {s['name']} {fprice(s['price'], s.get('currency'))} ({pct(s['change'])}) "
                         f"· RSI {fnum(s['ind']['rsi'], 0)} · achat {s['score_buy']} / vente {s['score_sell']}")
        notify("Résumé du matin", "\n".join(lines), "info", 2)
        digest_sent = today

    status = {
        "version": 1,
        "updated": NOW.isoformat(),
        "quiet": quiet,
        "assets": out_assets,
        "history": history,
        "pending": pending,
        "state": state,
        "digest_sent": digest_sent,
        "channels": {"ntfy": bool(NTFY_TOPIC), "telegram": bool(TG_TOKEN and TG_CHAT)},
    }
    STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATUS_OUT.write_text(json.dumps(clean(status), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log(f"status.json écrit : {len(out_assets)} actifs, {len(new_alerts)} nouvelle(s) alerte(s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
