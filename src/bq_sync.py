"""data/prices.parquet と指標の計算結果を BigQuery に反映する。

使い方:
    python src/bq_sync.py --pull      # BigQuery から直近分の株価を data/prices.parquet に読み出す(実行の最初)
    python src/bq_sync.py             # 認証情報がなければ何もせず終了(手元での実行を壊さない)
    python src/bq_sync.py --require   # 認証情報がなければ失敗(Actionsで使う)
    python src/bq_sync.py --dry-run   # BigQueryには触らず、送る内容を表示する

反映するテーブル(データセット sector_flow):
    prices             ... 株価。parquetの中身を (date, code) で MERGE(過去値の書き換えにも対応)
    universe           ... 銘柄リスト(全置換)
    sector_summary     ... 業種ごとの最新指標(as_of × 業種で積み上げ)
    sector_timeseries  ... 直近120営業日の推移(全置換)

株価の保存先の正は BigQuery。data/prices.parquet は1回の実行の中だけで使う作業用ファイル
(--pull で作り、fetch.py が差分を足し、このスクリプトで書き戻す)で、リポジトリにはコミットしない。
反映後に、送った全行が同じ値で BigQuery にあることを確かめ、食い違えば失敗する。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bq_io  # noqa: E402
from metrics import Result, compute  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICES = ROOT / "data" / "prices.parquet"
UNIVERSE = ROOT / "data" / "universe.csv"
NAV = ROOT / "data" / "nav.parquet"

def _names(schema) -> list[str]:
    return [n for n, _ in schema]


def _to_date(s: pd.Series) -> pd.Series:
    """日付列を python の date 型(オブジェクト)にする。NaT は None。"""
    d = pd.to_datetime(s, errors="coerce").dt.date
    return d.astype(object).where(pd.notna(d), None)


def _str_or_none(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), None)


def build_prices(prices: pd.DataFrame, now: datetime) -> pd.DataFrame:
    df = prices.copy()
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["date"] = _to_date(df["date"])
    df["code"] = df["code"].astype(str)
    for c in ["open", "high", "low"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["close", "adj_close", "volume", "turnover", "open", "high", "low"]:
        df[c] = df[c].astype("float64")
    df["loaded_at"] = now
    return df[_names(bq_io.PRICES_SCHEMA)]


def build_universe(uni: pd.DataFrame, now: datetime) -> pd.DataFrame:
    df = uni.copy()
    for n, _ in bq_io.UNIVERSE_SCHEMA:
        if n not in df.columns:
            df[n] = None
    df["code"] = df["code"].astype(str)
    for c in ["name", "sector", "type", "sector33", "size"]:
        df[c] = _str_or_none(df[c])
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").astype("float64")
    df["weight_date"] = _to_date(df["weight_date"])
    df["loaded_at"] = now
    return df[_names(bq_io.UNIVERSE_SCHEMA)]


def build_summary(res: Result, now: datetime) -> pd.DataFrame:
    df = res.summary.reset_index()           # summary の index は業種名(列名 sector)
    df["as_of"] = res.as_of.date()
    df["rank"] = np.arange(1, len(df) + 1)   # summary は総合スコアの降順
    df["loaded_at"] = now
    df["etf_code"] = df["etf_code"].astype(str)
    df["nav_date"] = _to_date(df["nav_date"]) if "nav_date" in df.columns else None
    for n, t in bq_io.SUMMARY_SCHEMA:
        if n not in df.columns:
            df[n] = None
        if t == "FLOAT64":
            df[n] = pd.to_numeric(df[n], errors="coerce").astype("float64")
        elif t == "STRING":
            df[n] = _str_or_none(df[n])
    df["n_stocks"] = df["n_stocks"].astype("int64")
    df["rank"] = df["rank"].astype("int64")
    return df[_names(bq_io.SUMMARY_SCHEMA)]


def build_timeseries(res: Result, now: datetime) -> pd.DataFrame:
    def long(frame: pd.DataFrame | None, name: str, scale: float = 1.0) -> pd.DataFrame:
        if frame is None:
            return pd.DataFrame(columns=["date", "sector", name])
        f = frame.copy() * scale
        f.index.name = "date"
        f.columns.name = "sector"
        return f.stack().rename(name).reset_index()   # NaN は stack で落ちる

    parts = [
        long(res.rs_line, "rs_etf"),
        long(res.rs_line_ew, "rs_ew"),
        long(res.share5, "share5_pct", 100.0),
        long(res.share_z, "share_z"),
        long(res.rs_line_cw, "rs_cw"),
    ]
    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=["date", "sector"], how="outer")
    df["date"] = _to_date(df["date"])
    df["as_of"] = res.as_of.date()
    df["loaded_at"] = now
    for c in ["rs_etf", "rs_ew", "share5_pct", "share_z", "rs_cw"]:
        df[c] = df[c].astype("float64")
    df = df.sort_values(["sector", "date"]).reset_index(drop=True)
    return df[_names(bq_io.TIMESERIES_SCHEMA)]


PULL_DAYS = 450   # fetch.py の保持期間(420日+余裕)に合わせる


def pull(require: bool) -> int:
    """BigQuery から直近分の株価を作業用の data/prices.parquet に書き出す。"""
    if not bq_io.credentials_available():
        msg = "Google Cloud の認証情報がありません(GOOGLE_APPLICATION_CREDENTIALS)。"
        if require:
            print(msg, file=sys.stderr)
            return 1
        print(msg + " 手元の data/prices.parquet をそのまま使います。")
        return 0
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    since = (pd.Timestamp.today().normalize() - pd.Timedelta(days=PULL_DAYS)).date()
    client = bq_io.get_client()
    bq_io.ensure_tables(client)
    df = bq_io.read_prices(client, since, uni["code"].tolist())
    if df.empty:
        print("BigQuery に株価がありません。fetch.py が全期間を取得します。")
        PRICES.unlink(missing_ok=True)
        return 0
    df.to_parquet(PRICES, index=False)
    print(f"BigQuery から読み出し: {len(df)}行 / {df['code'].nunique()}銘柄 / "
          f"{df['date'].min().date()}〜{df['date'].max().date()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="BigQuery から株価を data/prices.parquet に読み出す")
    ap.add_argument("--require", action="store_true", help="認証情報がなければ失敗する")
    ap.add_argument("--dry-run", action="store_true", help="BigQueryには触らず内容だけ表示する")
    args = ap.parse_args()
    if args.pull:
        return pull(args.require)

    now = datetime.now(timezone.utc)
    prices_raw = pd.read_parquet(PRICES)
    uni_raw = pd.read_csv(UNIVERSE, dtype={"code": str})
    nav = pd.read_parquet(NAV) if NAV.exists() else None
    res = compute(prices_raw, uni_raw, nav)

    frames = {
        "prices": build_prices(prices_raw, now),
        "universe": build_universe(uni_raw, now),
        "sector_summary": build_summary(res, now),
        "sector_timeseries": build_timeseries(res, now),
    }
    for name, f in frames.items():
        print(f"{name}: {len(f)}行 x {len(f.columns)}列")

    if args.dry_run:
        return 0
    if not bq_io.credentials_available():
        msg = "Google Cloud の認証情報がありません(GOOGLE_APPLICATION_CREDENTIALS)。"
        if args.require:
            print(msg, file=sys.stderr)
            return 1
        print(msg + " BigQueryへの反映をスキップします。")
        return 0

    client = bq_io.get_client()
    bq_io.ensure_tables(client)
    bq_io.merge_table(client, frames["prices"], "prices", keys=["date", "code"],
                      compare=["close", "adj_close", "volume", "turnover", "open", "high", "low"], verify=True)
    bq_io.replace_table(client, frames["universe"], "universe")
    bq_io.merge_table(client, frames["sector_summary"], "sector_summary",
                      keys=["as_of", "sector"])
    bq_io.replace_table(client, frames["sector_timeseries"], "sector_timeseries")
    print(f"BigQuery反映完了: 基準日 {res.as_of.date()} / 業種 {len(frames['sector_summary'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
