"""株価データを取得して data/prices.parquet に差分保存する。

使い方:
    python src/fetch.py           # yfinanceから取得(差分更新)
    python src/fetch.py --demo    # 合成データを生成(動作確認用)

保存する列:
    close     ... 終値(分割調整のみ)。売買代金の計算に使う
    adj_close ... 調整後終値(分割+配当調整)。リターンの計算に使う
    volume, turnover(= close × volume)

差分更新の注意:
    Yahooは分割や配当があると「過去の」値まで遡って書き換える。直近だけ取り直して
    古い行に継ぎ足すと、書き換え前と後の値が混ざって偽の急騰・急落ができる。
    そこで、取り直した期間と保存済みの値を突き合わせ、食い違う銘柄は全期間を取り直す。
"""
import argparse
import sys
import time
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "universe.csv"
PRICES = ROOT / "data" / "prices.parquet"

INITIAL_DAYS = 420      # 初回に遡る日数(60日指標+余裕)
OVERLAP_DAYS = 7        # 差分更新時に重ねて取り直す日数(訂正対策)
RETRIES = 3
REVISION_TOL = 0.001    # 重複期間で保存済みの値と0.1%超ずれたら「過去が書き換わった」とみなす
COLUMNS = ["date", "code", "close", "adj_close", "volume", "turnover"]


def drop_unfinished_day(df: pd.DataFrame, now=None) -> pd.DataFrame:
    """東証の取引が終わる前(15:45 JST前)は、当日分の行を捨てる。

    場中に取得すると、当日の出来高・売買代金が途中経過のまま入り、
    売買代金の指標(シェアz、当日/20日比など)が大きく歪むため。
    """
    if df.empty:
        return df
    now = now or datetime.now(ZoneInfo("Asia/Tokyo"))
    if now.time() >= dtime(15, 45):
        return df
    return df[df["date"] < pd.Timestamp(now.date())]


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
                auto_adjust=False,   # Close(分割調整のみ)と Adj Close(配当も調整)を両方受け取る
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
        return pd.DataFrame(columns=COLUMNS)

    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        sub = raw[t]
        if "Adj Close" not in sub.columns:
            sub = sub.assign(**{"Adj Close": sub["Close"]})
        df = sub[["Close", "Adj Close", "Volume"]].dropna(subset=["Close", "Volume"])
        if df.empty:
            continue
        df = df.reset_index()
        df.columns = ["date", "close", "adj_close", "volume"]
        df["code"] = t[:-2]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["adj_close"] = out["adj_close"].fillna(out["close"])
    out["turnover"] = out["close"] * out["volume"]
    return out[COLUMNS]


def revised_codes(old: pd.DataFrame, new: pd.DataFrame, tol: float = REVISION_TOL) -> list[str]:
    """重複する日付で、保存済みと新規取得の値が食い違う銘柄を返す。

    分割・配当で過去の値が書き換わった場合や、保存済みデータに誤った値が残っている場合に
    検出される。こうした銘柄は直近分を継ぎ足すと履歴の途中に段差ができるため、全期間を取り直す。
    """
    if old.empty or new.empty:
        return []
    m = old.merge(new, on=["date", "code"], suffixes=("_old", "_new"))
    if m.empty:
        return []
    bad = pd.Series(False, index=m.index)
    for col in ["close", "adj_close"]:
        a, b = m[f"{col}_old"], m[f"{col}_new"]
        ok = a.notna() & b.notna() & (a != 0)
        bad |= ok & ((b / a - 1).abs() > tol)
    return sorted(m.loc[bad, "code"].unique())


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
            "date": dates, "code": r["code"], "close": close, "adj_close": close,
            "volume": volume, "turnover": close * volume,
        }))
    return pd.concat(rows, ignore_index=True)[COLUMNS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="合成データで生成する")
    ap.add_argument("--full", action="store_true", help="保存済みデータを使わず全期間を取り直す")
    args = ap.parse_args()

    uni = load_universe()
    codes = uni["code"].tolist()
    PRICES.parent.mkdir(parents=True, exist_ok=True)

    if args.demo:
        df = make_demo(uni)
        df.to_parquet(PRICES, index=False)
        print(f"デモデータを保存: {len(df)}行")
        return 0

    old = pd.read_parquet(PRICES) if PRICES.exists() else pd.DataFrame(columns=COLUMNS)
    if not old.empty and "adj_close" not in old.columns:
        print("保存済みデータに調整後終値の列がないため、全期間を取り直します。")
        old = pd.DataFrame(columns=COLUMNS)
    if args.full:
        old = pd.DataFrame(columns=COLUMNS)

    full_start = pd.Timestamp.today().normalize() - pd.Timedelta(days=INITIAL_DAYS)
    start = full_start if old.empty else old["date"].max() - pd.Timedelta(days=OVERLAP_DAYS)

    new = download(codes, start)
    if new.empty:
        print("取得結果が空のため、既存データを維持して終了します。", file=sys.stderr)
        return 1

    if not old.empty:
        # 保存済みにない銘柄(新しく追加した銘柄)も全期間が必要
        refetch = sorted(set(revised_codes(old, new)) | (set(new["code"]) - set(old["code"])))
        if refetch:
            print(f"履歴を全期間取り直す銘柄({len(refetch)}): {', '.join(refetch)}")
            full = download(refetch, full_start)
            got = set(full["code"].unique())
            if got:
                old = old[~old["code"].isin(got)]
                new = pd.concat([new[~new["code"].isin(got)], full], ignore_index=True)
            failed = sorted(set(refetch) - got)
            if failed:
                print(f"全期間の取り直しに失敗(直近分のみ反映、段差が残る可能性): {', '.join(failed)}",
                      file=sys.stderr)

    got = set(new["code"].unique())
    missing = sorted(set(codes) - got)
    if missing:
        print(f"取得できなかった銘柄({len(missing)}): {', '.join(missing)}", file=sys.stderr)

    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    # 場中の途中経過を含めない(過去に混入した当日分もここで除去される)
    merged = drop_unfinished_day(merged)
    merged = (merged.drop_duplicates(["date", "code"], keep="last")
                    .sort_values(["code", "date"]).reset_index(drop=True))
    # 保持期間を制限してリポジトリの肥大化を防ぐ
    cutoff = merged["date"].max() - pd.Timedelta(days=INITIAL_DAYS + 30)
    merged = merged[merged["date"] >= cutoff][COLUMNS]
    merged.to_parquet(PRICES, index=False)
    print(f"保存: {len(merged)}行 / 最新日 {merged['date'].max().date()} / 銘柄数 {merged['code'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
