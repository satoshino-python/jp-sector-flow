"""過去の株価を遡って BigQuery の prices に入れる(一回限り用)。

日次更新(fetch.py / bq_sync.py)の設定は変えない。日次は直近450日だけを読み書きするので、
ここで入れた古い行は日次の計算には入らず、日次の動作にも影響しない。

全期間を1回の取得でそろえて取る。配当落ちのたびに過去の adj_close が書き換わるため、
期間を分けて取ると継ぎ目に段差ができる。(date, code) をキーに MERGE するので、
既存の直近分も同じ取得の値に置き換わり、全期間で調整の基準日がそろう。

使い方:
    python src/backfill.py --dry-run                 # 取得だけして内容を表示(BigQueryには触らない)
    python src/backfill.py --start 2021-01-01        # 取得して BigQuery に反映
"""
import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

import bq_io
import bq_sync
import fetch

DEFAULT_START = "2021-01-01"
MAX_DAILY_MOVE = 0.45     # 日次の変化がこれを超える行は、誤データの疑いとして件数を表示する
MIN_COVERAGE = 0.95       # 取得できた銘柄がこの割合を下回ったら反映しない


def suspicious(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["code", "date"])
    chg = d.groupby("code")["adj_close"].pct_change().abs()
    return d[chg > MAX_DAILY_MOVE]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START, help="遡る開始日(既定 2021-01-01)")
    ap.add_argument("--dry-run", action="store_true", help="取得と検査だけ行い、BigQueryには反映しない")
    args = ap.parse_args()

    uni = fetch.load_universe()
    codes = uni["code"].astype(str).tolist()
    start = pd.Timestamp(args.start)
    print(f"対象 {len(codes)}銘柄 / {start.date()}以降を取得します")

    df = fetch.download(codes, start)
    if df.empty:
        print("取得できませんでした", file=sys.stderr)
        return 1
    df = fetch.drop_unfinished_day(df)
    df = df[df["code"].isin(codes)].drop_duplicates(["date", "code"], keep="last")

    got = df["code"].nunique()
    print(f"取得: {len(df)}行 / {got}/{len(codes)}銘柄 / {df['date'].min().date()}〜{df['date'].max().date()}")
    missing = sorted(set(codes) - set(df["code"]))
    if missing:
        print(f"取得できなかった銘柄 {len(missing)}件: {', '.join(missing[:30])}{' …' if len(missing) > 30 else ''}")
    first = df.groupby("code")["date"].min()
    late = first[first > start + pd.Timedelta(days=30)]
    if len(late):
        print(f"{start.date()}より後に始まる銘柄(上場が新しい等) {len(late)}件")
    sus = suspicious(df)
    if len(sus):
        print(f"日次±{MAX_DAILY_MOVE:.0%}超の行 {len(sus)}件(銘柄 {sus['code'].nunique()})。例:")
        print(sus[["date", "code", "close", "adj_close"]].head(10).to_string(index=False))

    if got < len(codes) * MIN_COVERAGE:
        print(f"取得率が{MIN_COVERAGE:.0%}未満のため反映しません", file=sys.stderr)
        return 1
    if args.dry_run:
        print("dry-run: BigQueryには反映しません")
        return 0
    if not bq_io.credentials_available():
        print("Google Cloud の認証情報がありません(GOOGLE_APPLICATION_CREDENTIALS)", file=sys.stderr)
        return 1

    frame = bq_sync.build_prices(df, datetime.now(timezone.utc))
    client = bq_io.get_client()
    bq_io.ensure_tables(client)
    bq_io.merge_table(client, frame, "prices", keys=["date", "code"],
                      compare=["close", "adj_close", "volume", "turnover", "open", "high", "low"], verify=True)
    print(f"BigQuery反映完了: {len(frame)}行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
