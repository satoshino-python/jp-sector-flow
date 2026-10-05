"""株価データを取得して data/prices.parquet に差分保存する。

使い方:
    python src/fetch.py           # yfinanceから取得(差分更新)
    python src/fetch.py --demo    # 合成データを生成(動作確認用)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "universe.csv"
PRICES = ROOT / "data" / "prices.parquet"

INITIAL_DAYS = 420      # 初回に遡る日数(60日指標+余裕)
OVERLAP_DAYS = 7        # 差分更新時に重ねて取り直す日数(訂正対策)
RETRIES = 3


def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE, dtype={"code": str})


def download(codes: list[str], start: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    tickers = [f"{c}.T" for c in codes]
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            raw = yf.download(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                auto_adjust=False,   # 売買代金は未調整の終値×出来高で計算する
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if raw is not None and not raw.empty:
                break
        except Exception as e:  # noqa: BLE001
            last_err = e
        print(f"取得失敗(試行{attempt}/{RETRIES}) {last_err or '空の結果'}", file=sys.stderr)
        time.sleep(5 * attempt)
    else:
        return pd.DataFrame()

    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t][["Close", "Volume"]].dropna()
        if df.empty:
            continue
        df = df.reset_index()
        df.columns = ["date", "close", "volume"]
        df["code"] = t[:-2]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["turnover"] = out["close"] * out["volume"]
    return out[["date", "code", "close", "volume", "turnover"]]


def make_demo(codes_sectors: pd.DataFrame, days: int = 300) -> pd.DataFrame:
    """業種ごとにトレンドと出来高の増減を持たせた合成データ(動作確認専用)。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    sectors = sorted(codes_sectors["sector"].unique())
    sector_drift = {s: rng.normal(0, 0.0006) for s in sectors}
    sector_vol_trend = {s: rng.normal(0, 0.004) for s in sectors}
    market = rng.normal(0.0002, 0.008, days)
    rows = []
    for _, r in codes_sectors.iterrows():
        s = r["sector"]
        beta = rng.uniform(0.6, 1.3)
        drift = sector_drift.get(s, 0.0)
        ret = beta * market + drift + rng.normal(0, 0.009, days)
        close = rng.uniform(800, 9000) * np.exp(np.cumsum(ret))
        base_vol = rng.uniform(3e5, 5e6) if r["type"] != "etf" else rng.uniform(2e3, 3e4)
        trend = np.exp(np.linspace(0, sector_vol_trend.get(s, 0.0) * days, days))
        spike = np.exp(rng.normal(0, 0.25, days))
        volume = base_vol * trend * spike
        rows.append(pd.DataFrame({
            "date": dates, "code": r["code"], "close": close,
            "volume": volume, "turnover": close * volume,
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="合成データで生成する")
    args = ap.parse_args()

    uni = load_universe()
    PRICES.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        df = make_demo(uni)
        df.to_parquet(PRICES, index=False)
        print(f"デモデータを保存: {len(df)}行")
        return 0

    old = pd.read_parquet(PRICES) if PRICES.exists() else pd.DataFrame()
    if old.empty:
        start = pd.Timestamp.today().normalize() - pd.Timedelta(days=INITIAL_DAYS)
    else:
        start = old["date"].max() - pd.Timedelta(days=OVERLAP_DAYS)

    new = download(uni["code"].tolist(), start)
    if new.empty:
        print("取得結果が空のため、既存データを維持して終了します。", file=sys.stderr)
        return 1

    got = set(new["code"].unique())
    missing = sorted(set(uni["code"]) - got)
    if missing:
        print(f"取得できなかった銘柄({len(missing)}): {', '.join(missing)}", file=sys.stderr)

    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    merged = (merged.drop_duplicates(["date", "code"], keep="last")
                    .sort_values(["code", "date"]).reset_index(drop=True))
    # 保持期間を制限してリポジトリの肥大化を防ぐ
    cutoff = merged["date"].max() - pd.Timedelta(days=INITIAL_DAYS + 30)
    merged = merged[merged["date"] >= cutoff]
    merged.to_parquet(PRICES, index=False)
    print(f"保存: {len(merged)}行 / 最新日 {merged['date'].max().date()} / 銘柄数 {merged['code'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
