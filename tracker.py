"""
Daily Stock Prediction Tracker
==============================
Logs a forecast BEFORE the market moves, scores it AFTER the close,
and charts how far off it was — against a no-skill baseline.

Commands
--------
  python tracker.py predict            # run before market open: log today's forecasts
  python tracker.py evaluate           # run after market close: fill in actual prices
  python tracker.py report             # build the comparison charts + summary
  python tracker.py verify             # check the log hasn't been edited after the fact
  python tracker.py backtest --days 250  # simulated history (NOT live proof — labelled as such)

Add --demo to any command to use simulated prices (no internet needed).

Why the baseline matters
------------------------
"Tomorrow's close = today's close" is the naive forecast. Stock prices are
close to a random walk, so the naive forecast is very hard to beat. A model
only shows real skill if it beats the naive forecast AND its direction
accuracy is statistically above 50% over many days.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE / "config.json").read_text())
LOG = BASE / CONFIG.get("log_file", "predictions.csv")
REPORT_DIR = BASE / "report"

LOG_COLUMNS = [
    "id", "logged_at_utc", "target_date", "ticker", "last_close",
    "pred_close", "pred_direction", "baseline_close",
    "actual_close", "actual_direction", "evaluated_at_utc",
    "source", "prev_hash", "row_hash",
]


# ---------------------------------------------------------------- data

def fetch_history(ticker: str, demo: bool, end: date | None = None, days: int = 900) -> pd.Series:
    """Daily closing prices, oldest first. `end` is exclusive."""
    if demo:
        return _demo_prices(ticker, end, days)
    import yfinance as yf
    start = (end or date.today()) - timedelta(days=int(days * 1.6))
    df = yf.download(ticker, start=start.isoformat(),
                     end=(end.isoformat() if end else None),
                     progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).date
    return close.dropna()


def _demo_prices(ticker: str, end: date | None, days: int) -> pd.Series:
    """Deterministic simulated random-walk prices for offline testing."""
    seed = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % 2**32
    rng = np.random.default_rng(seed)
    all_days = pd.bdate_range(end=date.today(), periods=1500).date
    rets = rng.normal(0.0003, 0.015, len(all_days))
    prices = pd.Series(1000 * np.exp(np.cumsum(rets)), index=all_days)
    if end:
        prices = prices[prices.index < end]
    return prices.iloc[-days:]


# ---------------------------------------------------------------- model

def make_features(close: pd.Series) -> pd.DataFrame:
    r = np.log(close).diff()
    f = pd.DataFrame(index=close.index)
    for k in (1, 2, 3, 5):
        f[f"ret_lag{k}"] = r.shift(k - 1)
    f["mom_5"] = r.rolling(5).sum()
    f["mom_20"] = r.rolling(20).sum()
    f["vol_10"] = r.rolling(10).std()
    f["vol_20"] = r.rolling(20).std()
    ma20 = close.rolling(20).mean()
    f["dist_ma20"] = close / ma20 - 1
    f["target"] = r.shift(-1)  # next day's log return
    return f


def predict_next(close: pd.Series) -> float:
    """Train on everything up to the last known close; return predicted next close."""
    feats = make_features(close)
    train = feats.dropna()
    X, y = train.drop(columns="target"), train["target"]
    model = GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=0)
    model.fit(X.values, y.values)
    x_last = feats.drop(columns="target").iloc[[-1]]
    pred_ret = float(model.predict(x_last.values)[0])
    return float(close.iloc[-1] * math.exp(pred_ret))


# ---------------------------------------------------------------- log (append-only, hash-chained)

def load_log() -> pd.DataFrame:
    # Read everything as text so values (and hashes) round-trip exactly.
    if LOG.exists():
        log = pd.read_csv(LOG, dtype=str, keep_default_na=False)
    else:
        log = pd.DataFrame(columns=LOG_COLUMNS, dtype=str)
    log["id"] = pd.to_numeric(log["id"]).astype(int)
    return log


def is_blank(v) -> bool:
    return pd.isna(v) or str(v).strip() == ""


def _hash_row(row: dict, prev_hash: str) -> str:
    # Only the prediction fields are hashed — the actual-price fields are filled in later.
    payload = "|".join(str(row[c]) for c in (
        "id", "logged_at_utc", "target_date", "ticker", "last_close",
        "pred_close", "pred_direction", "baseline_close", "source")) + "|" + prev_hash
    return hashlib.sha256(payload.encode()).hexdigest()


def append_prediction(log: pd.DataFrame, **fields) -> pd.DataFrame:
    prev_hash = log["row_hash"].iloc[-1] if len(log) else "GENESIS"
    row = {c: "" for c in LOG_COLUMNS}
    row.update({k: str(v) for k, v in fields.items()})
    row["id"] = int(log["id"].max()) + 1 if len(log) else 1
    row["prev_hash"] = prev_hash
    row["row_hash"] = _hash_row(row, prev_hash)
    return pd.concat([log, pd.DataFrame([row])], ignore_index=True)


def save_log(log: pd.DataFrame) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log[LOG_COLUMNS].to_csv(LOG, index=False)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- commands

def market_now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(CONFIG.get("market_tz", "Asia/Kolkata")))


def cmd_predict(args) -> None:
    log = load_log()
    now = market_now()
    target = date.fromisoformat(args.date) if args.date else now.date()
    if target.weekday() >= 5:
        print(f"{target} is a weekend — no market, nothing to predict.")
        return
    # Honesty guard: a live prediction made after the market opens is not a real forecast.
    open_h, open_m = map(int, CONFIG.get("market_open", "09:15").split(":"))
    if not args.demo and target == now.date() and (now.hour, now.minute) >= (open_h, open_m):
        print(f"Market already opened at {CONFIG.get('market_open', '09:15')} — refusing to log "
              "late predictions for today. They would not count as genuine forecasts.")
        return
    for t in CONFIG["tickers"]:
        done = log[(log.ticker == t) & (log.target_date == target.isoformat())]
        if len(done):
            print(f"  {t}: already predicted for {target} — skipping (no overwriting)")
            continue
        close = fetch_history(t, args.demo, end=target)
        last = float(close.iloc[-1])
        pred = predict_next(close)
        log = append_prediction(
            log, logged_at_utc=now_utc(), target_date=target.isoformat(), ticker=t,
            last_close=round(last, 4), pred_close=round(pred, 4),
            pred_direction="UP" if pred > last else "DOWN",
            baseline_close=round(last, 4), source="demo" if args.demo else "live")
        print(f"  {t}: last {last:,.2f} -> predicted {pred:,.2f} ({'UP' if pred > last else 'DOWN'})")
    save_log(log)
    print(f"Logged to {LOG.name}. Run `evaluate` after the market closes.")


def cmd_evaluate(args) -> None:
    log = load_log()
    pending = log[log.actual_close.map(is_blank)]
    if pending.empty:
        print("Nothing to evaluate.")
        return
    filled = 0
    for idx, row in pending.iterrows():
        target = date.fromisoformat(row.target_date)
        today = market_now().date()
        close = fetch_history(row.ticker, args.demo, end=today + timedelta(days=1),
                              days=max(15, (today - target).days + 15))
        if target not in close.index:
            if len(close) and close.index[-1] > target:
                # A later trading day exists but this date has no close: market holiday.
                log.loc[idx, "actual_close"] = "NA"
                log.loc[idx, "actual_direction"] = "NO_TRADING"
                log.loc[idx, "evaluated_at_utc"] = now_utc()
                print(f"  {row.ticker} {target}: market holiday — excluded from scoring")
            else:
                print(f"  {row.ticker} {target}: no close yet (market still open?)")
            continue
        actual = float(close[target])
        log.loc[idx, "actual_close"] = str(round(actual, 4))
        log.loc[idx, "actual_direction"] = "UP" if actual > float(row.last_close) else "DOWN"
        log.loc[idx, "evaluated_at_utc"] = now_utc()
        err = (float(row.pred_close) - actual) / actual * 100
        print(f"  {row.ticker} {target}: predicted {float(row.pred_close):,.2f}, "
              f"actual {actual:,.2f} ({err:+.2f}% off)")
        filled += 1
    save_log(log)
    print(f"Evaluated {filled} prediction(s). Run `report` to update the charts.")


def cmd_verify(args) -> bool:
    log = load_log()
    prev = "GENESIS"
    for _, row in log.iterrows():
        r = {c: row[c] for c in LOG_COLUMNS}
        r["id"] = int(r["id"])
        if row.prev_hash != prev or _hash_row(r, prev) != row.row_hash:
            print(f"TAMPERED: row {row.id} ({row.ticker} {row.target_date}) does not match its hash.")
            return False
        prev = row.row_hash
    print(f"OK: all {len(log)} predictions intact. Latest hash: {prev[:16]}…")
    print("Publish this hash (e.g. a dated tweet/LinkedIn post or git commit) so anyone can check later.")
    return True


def cmd_backtest(args) -> None:
    """Walk-forward simulation: each day's model only sees data before that day."""
    rows = []
    for t in CONFIG["tickers"]:
        full = fetch_history(t, args.demo, days=args.days + 600)
        test_days = full.index[-args.days:]
        for d in test_days:
            hist = full[full.index < d]
            pred = predict_next(hist)  # retrained daily on past data only — no lookahead
            last, actual = float(hist.iloc[-1]), float(full[d])
            rows.append(dict(target_date=d.isoformat(), ticker=t, last_close=last,
                             pred_close=pred, baseline_close=last, actual_close=actual,
                             pred_direction="UP" if pred > last else "DOWN",
                             actual_direction="UP" if actual > last else "DOWN"))
        print(f"  {t}: backtested {len(test_days)} days")
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "backtest.csv", index=False)
    build_report(out, REPORT_DIR / "backtest", label="BACKTEST (simulated — not live proof)")


def cmd_report(args) -> None:
    log = load_log()
    done = log[~log.actual_close.map(is_blank) & (log.actual_direction != "NO_TRADING")]
    if done.empty:
        print("No evaluated predictions yet. Run `predict` then `evaluate` first.")
        return
    cmd_verify(args)
    build_report(done, REPORT_DIR / "live", label="LIVE — predictions logged before the outcome")


# ---------------------------------------------------------------- report

def summarize(df: pd.DataFrame) -> dict:
    df = df.copy()
    for c in ("last_close", "pred_close", "baseline_close", "actual_close"):
        df[c] = df[c].astype(float)
    n = len(df)
    model_err = (df.pred_close - df.actual_close).abs() / df.actual_close * 100
    base_err = (df.baseline_close - df.actual_close).abs() / df.actual_close * 100
    hits = int((df.pred_direction == df.actual_direction).sum())
    # One-sided binomial test vs a 50/50 coin flip (normal approximation, fine for n >= 30).
    z = (hits - 0.5 * n) / math.sqrt(0.25 * n) if n else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return dict(n=n, model_mape=model_err.mean(), base_mape=base_err.mean(),
                beats_baseline_pct=float((model_err < base_err).mean() * 100),
                hit_rate=hits / n * 100 if n else 0.0, hits=hits, p_value=p)


def verdict(s: dict) -> str:
    if s["n"] < 60:
        return (f"Too early to judge: only {s['n']} predictions. Collect at least 60–100 trading "
                "days before drawing any conclusion.")
    skill = s["model_mape"] < s["base_mape"] and s["p_value"] < 0.05
    if skill:
        return ("Evidence of skill: lower error than the naive baseline AND direction accuracy "
                f"significantly above 50% (p = {s['p_value']:.3f}). Keep tracking — edges often fade.")
    return ("No evidence of skill yet: the model does not reliably beat the naive "
            "'tomorrow = today' forecast. Do not trade real money on it.")


def build_report(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["target_date"] = pd.to_datetime(df.target_date)
    for c in ("last_close", "pred_close", "baseline_close", "actual_close"):
        df[c] = df[c].astype(float)
    df["err_pct"] = (df.pred_close - df.actual_close) / df.actual_close * 100
    df["base_err_pct"] = (df.baseline_close - df.actual_close) / df.actual_close * 100
    df["hit"] = (df.pred_direction == df.actual_direction).astype(int)

    INK, MUTED, GRID = "#1f2328", "#6e7781", "#d8dee4"
    BLUE, ORANGE, GREEN, RED = "#2f6fdb", "#d9822b", "#2e9e5b", "#cf3e3e"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})

    summaries = {}
    for t, g in df.groupby("ticker"):
        g = g.sort_values("target_date")
        s = summarize(g)
        summaries[t] = s
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), height_ratios=[3, 2, 2], sharex=True)
        fig.suptitle(f"{t} — predicted vs actual   [{label}]", fontsize=13, color=INK,
                     x=0.01, ha="left")

        ax = axes[0]
        ax.plot(g.target_date, g.actual_close, color=INK, lw=1.6, label="Actual close")
        ax.plot(g.target_date, g.pred_close, color=BLUE, lw=1.2, ls="--", label="Model prediction")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left", frameon=False)

        ax = axes[1]
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.bar(g.target_date, g.err_pct, color=np.where(g.err_pct >= 0, BLUE, ORANGE), width=1.0)
        ax.set_ylabel("How far off (%)")
        ax.set_title("Daily prediction error: positive = predicted too high", loc="left",
                     fontsize=10, color=MUTED)

        ax = axes[2]
        roll = g.hit.rolling(20, min_periods=5).mean() * 100
        cum = g.hit.expanding().mean() * 100
        ax.axhline(50, color=RED, lw=1, ls=":", label="Coin flip (50%)")
        ax.plot(g.target_date, roll, color=GREEN, lw=1.2, alpha=0.6, label="20-day rolling")
        ax.plot(g.target_date, cum, color=GREEN, lw=2, label="Cumulative")
        ax.set_ylabel("Direction correct (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="lower left", frameon=False, ncol=3)

        fig.text(0.01, 0.005,
                 f"n={s['n']} days | model avg error {s['model_mape']:.2f}% vs naive baseline "
                 f"{s['base_mape']:.2f}% | direction hit rate {s['hit_rate']:.1f}% "
                 f"(p={s['p_value']:.3f})", fontsize=9, color=MUTED)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        fig.savefig(out_dir / f"{t.replace('.', '_')}.png", dpi=130)
        plt.close(fig)

    # Overall scorecard
    s_all = summarize(df)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Model", "Naive baseline\n(tomorrow = today)"], [s_all["model_mape"], s_all["base_mape"]],
           color=[BLUE, MUTED], width=0.5)
    for i, v in enumerate([s_all["model_mape"], s_all["base_mape"]]):
        ax.text(i, v, f"{v:.3f}%", ha="center", va="bottom", color=INK)
    ax.set_ylabel("Average error (% of price)")
    ax.set_title(f"Overall: lower is better   [{label}]", loc="left", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "scorecard.png", dpi=130)
    plt.close(fig)

    lines = [f"# Prediction report — {label}", "",
             f"Generated {now_utc()}", "",
             "| Ticker | Days | Model avg error | Baseline avg error | Model beats baseline (days) "
             "| Direction hit rate | p-value vs coin flip |",
             "|---|---|---|---|---|---|---|"]
    for t, s in list(summaries.items()) + [("ALL", s_all)]:
        lines.append(f"| {t} | {s['n']} | {s['model_mape']:.3f}% | {s['base_mape']:.3f}% | "
                     f"{s['beats_baseline_pct']:.0f}% | {s['hit_rate']:.1f}% | {s['p_value']:.3f} |")
    lines += ["", f"**Verdict:** {verdict(s_all)}", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nCharts saved in {out_dir}")


# ---------------------------------------------------------------- cli

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["predict", "evaluate", "report", "verify", "backtest"])
    p.add_argument("--demo", action="store_true", help="use simulated prices (offline)")
    p.add_argument("--date", help="target date for predict (YYYY-MM-DD), default today")
    p.add_argument("--days", type=int, default=250, help="backtest length in trading days")
    args = p.parse_args()
    {"predict": cmd_predict, "evaluate": cmd_evaluate, "report": cmd_report,
     "verify": cmd_verify, "backtest": cmd_backtest}[args.command](args)


if __name__ == "__main__":
    main()
