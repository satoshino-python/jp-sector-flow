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
MAX_PRICE = 1e7           # 国内株でありえない水準(これ以上は壊れた値として捨てる)
ADJ_RATIO_RANGE = (0.1, 10)  # adj_close / close がこの範囲外の行は壊れた値として捨てる


def drop_broken(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo の壊れた値(負・桁違い・調整係数の暴走)の行を捨てる。

    例: 8303 は adj_close が 2e13 や -2e8、close が 5.5e10 の行がある。
    日次側は日次±45%超を指数計算から除外するが、BigQuery に壊れた行を残さないよう反映前に除く。
    """
    ratio = df["adj_close"] / df["close"]
    bad = ((df[["close", "adj_close"]] <= 0).any(axis=1)
           | (df[["close", "adj_close"]] > MAX_PRICE).any(axis=1)
           | (ratio < ADJ_RATIO_RANGE[0]) | (ratio > ADJ_RATIO_RANGE[1]))
    return df[~bad]


# Yahoo が分割を過去に反映していない銘柄: {コード: (分割後の最初の補正済み日付, 分割比)}
# 8227 しまむら: 1:3 分割(2026-02-19権利落ち)。2/18の行だけ ÷3 済みで、2/17以前が未調整
MISSING_SPLITS = {"8227": (pd.Timestamp("2026-02-18"), 3)}


def apply_missing_splits(df: pd.DataFrame) -> pd.DataFrame:
    """MISSING_SPLITS の銘柄について、分割前の行を分割比で調整する(価格 ÷ 比、出来高 × 比)。

    すでに Yahoo が直っている場合に二重で調整しないよう、境目の終値の比が分割比に近いときだけ行う。
    """
    df = df.copy()
    for code, (since, ratio) in MISSING_SPLITS.items():
        sub = df[df["code"] == code].sort_values("date")
        before, after = sub[sub["date"] < since], sub[sub["date"] >= since]
        if before.empty or after.empty:
            continue
        jump = before["close"].iloc[-1] / after["close"].iloc[0]
        if not (ratio * 0.7 < jump < ratio * 1.3):
            print(f"{code}: 境目の比 {jump:.2f} が分割比 {ratio} と合わないため調整しません(Yahoo側が直った可能性)")
            continue
        m = (df["code"] == code) & (df["date"] < since)
        for c in ["close", "adj_close", "open", "high", "low"]:
            df.loc[m, c] = df.loc[m, c] / ratio
        df.loc[m, "volume"] = df.loc[m, "volume"] * ratio
        print(f"{code}: {since.date()}より前の{int(m.sum())}行を1:{ratio}分割で調整")
    return df


# Yahoo の不具合で価格が桁違いになっている行(反映前に除き、BigQuery 側の既存行も削除する)
# 1306 / 1629: 2026-03-30〜31 だけ終値が約1/10・1/500(4/1に元に戻る。分割・併合の事実なし)
BAD_ROWS = {"1306": ["2026-03-30", "2026-03-31"], "1629": ["2026-03-30", "2026-03-31"]}


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    bad = pd.Series(False, index=df.index)
    for code, dates in BAD_ROWS.items():
        bad |= (df["code"] == code) & df["date"].isin(pd.to_datetime(dates))
    if bad.any():
        print(f"既知の不具合行を除外: {int(bad.sum())}行")
    return df[~bad]


def delete_bad_rows(client) -> None:
    from google.cloud import bigquery

    n = 0
    for code, dates in BAD_ROWS.items():
        params = [bigquery.ScalarQueryParameter("code", "STRING", code),
                  bigquery.ArrayQueryParameter("dates", "DATE", [pd.Timestamp(d).date() for d in dates])]
        job = bq_io._run(client, f"DELETE FROM {bq_io.fqn('prices')} WHERE code = @code AND date IN UNNEST(@dates)",
                         params)
        n += job.num_dml_affected_rows or 0
    print(f"BigQuery から既知の不具合行を削除: {n}行")


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
    n0 = len(df)
    df = drop_broken(df)
    if n0 != len(df):
        print(f"壊れた値の行を除外: {n0 - len(df)}行")
    df = drop_bad_rows(df)
    df = apply_missing_splits(df)

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
    delete_bad_rows(client)
    print(f"BigQuery反映完了: {len(frame)}行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
