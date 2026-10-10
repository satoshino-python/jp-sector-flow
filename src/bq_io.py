"""BigQuery への書き込み補助(ロードジョブ + MERGE)。

方針:
  - 書き込みは「ロードジョブ」だけを使う(無料)。ストリーミング挿入は有料なので使わない
  - 差分の反映は「一時テーブルへロード → MERGE」。Yahooが過去の値を書き換えても、
    (date, code) をキーに上書きされるので、全期間の取り直しにもそのまま対応できる
  - クエリには maximum_bytes_billed を付け、想定外の大量スキャンを防ぐ
"""
from __future__ import annotations

import os
import uuid

import pandas as pd

PROJECT = os.environ.get("BQ_PROJECT", "jp-sector-flow")
DATASET = os.environ.get("BQ_DATASET", "sector_flow")
LOCATION = os.environ.get("BQ_LOCATION", "asia-northeast1")
MAX_BYTES_BILLED = 1_000_000_000  # 1クエリあたり1GBまで(これを超える見込みなら失敗する)

# 列定義: (名前, 型)
PRICES_SCHEMA = [
    ("date", "DATE"), ("code", "STRING"),
    ("close", "FLOAT64"), ("adj_close", "FLOAT64"),
    ("volume", "FLOAT64"), ("turnover", "FLOAT64"),
    ("loaded_at", "TIMESTAMP"),
    # 以下はローソク足のために追加(既存テーブルには ensure_tables が末尾に足す。過去の行は fetch.py が埋め戻す)
    ("open", "FLOAT64"), ("high", "FLOAT64"), ("low", "FLOAT64"),
]

UNIVERSE_SCHEMA = [
    ("code", "STRING"), ("name", "STRING"), ("sector", "STRING"), ("type", "STRING"),
    ("loaded_at", "TIMESTAMP"),
    # 以下は TOPIX 全銘柄化で追加(既存テーブルには ensure_tables が末尾に足す)
    ("sector33", "STRING"), ("size", "STRING"), ("weight", "FLOAT64"), ("weight_date", "DATE"),
]

_SUMMARY_FLOATS = [
    "ret_5d", "ret_20d", "ret_60d", "rel_5d", "rel_20d", "rel_60d", "med_ret_20d",
    "turn_ratio_1d", "share_pct", "share_z", "up_turn_ratio", "breadth_ma25", "breadth_pos20",
    "ew_ret_5d", "ew_rel_5d", "ew_ret_20d", "ew_rel_20d", "ew_ret_60d", "ew_rel_60d",
    "dev_20d", "nav_noise", "rel_20d_nav", "rel_20d_same", "score",
    "cw_ret_5d", "cw_rel_5d", "cw_ret_20d", "cw_rel_20d", "cw_ret_60d", "cw_rel_60d",
]
SUMMARY_SCHEMA = (
    [("as_of", "DATE"), ("sector", "STRING"), ("rank", "INT64"),
     ("etf_code", "STRING"), ("n_stocks", "INT64")]
    + [(c, "FLOAT64") for c in _SUMMARY_FLOATS]
    + [("label", "STRING"), ("ew_label", "STRING"), ("ew_flag", "STRING"),
       ("nav_date", "DATE"), ("nav_flag", "STRING"), ("loaded_at", "TIMESTAMP"),
       # 時価総額加重を主軸にしたことで追加(既存テーブルには ensure_tables が末尾に足す)
       ("cw_label", "STRING"), ("etf_flag", "STRING")]
)

TIMESERIES_SCHEMA = [
    ("as_of", "DATE"), ("date", "DATE"), ("sector", "STRING"),
    ("rs_etf", "FLOAT64"), ("rs_ew", "FLOAT64"),
    ("share5_pct", "FLOAT64"), ("share_z", "FLOAT64"),
    ("loaded_at", "TIMESTAMP"), ("rs_cw", "FLOAT64"),
]

TABLES = {
    "prices": dict(schema=PRICES_SCHEMA, partition="date", cluster=["code"],
                   desc="株価(日次)。キーは (date, code)"),
    "universe": dict(schema=UNIVERSE_SCHEMA, partition=None, cluster=[],
                     desc="対象銘柄リスト(毎回全置換)"),
    "sector_summary": dict(schema=SUMMARY_SCHEMA, partition=None, cluster=["sector"],
                           desc="業種ごとの最新指標。キーは (as_of, sector)。日ごとに積み上げ"),
    "sector_timeseries": dict(schema=TIMESERIES_SCHEMA, partition=None, cluster=["sector"],
                              desc="直近120営業日の相対強度・シェアの推移(毎回全置換。相対強度は期首=100に再基準化)"),
}


def fqn(table: str) -> str:
    return f"`{PROJECT}.{DATASET}.{table}`"


def credentials_available() -> bool:
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    return bool(path) and os.path.exists(path)


def get_client():
    from google.cloud import bigquery

    return bigquery.Client(project=PROJECT, location=LOCATION)


def _ddl(table: str) -> str:
    spec = TABLES[table]
    cols = ",\n  ".join(f"{n} {t}" for n, t in spec["schema"])
    extra = ""
    if spec["partition"]:
        extra += f"\nPARTITION BY {spec['partition']}"
    if spec["cluster"]:
        extra += f"\nCLUSTER BY {', '.join(spec['cluster'])}"
    desc = spec["desc"].replace('"', "'")
    return (f"CREATE TABLE IF NOT EXISTS {fqn(table)} (\n  {cols}\n){extra}\n"
            f'OPTIONS (description = "{desc}")')


def _run(client, sql: str, params=None):
    from google.cloud import bigquery

    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES_BILLED,
                                  query_parameters=params or [])
    job = client.query(sql, job_config=cfg)
    job.result()
    return job


def ensure_tables(client) -> None:
    """テーブルがなければ作り、あれば足りない列を追加する。DDLはクエリ料金がかからない。

    列の追加は末尾に足すだけ(既存の列は消さない・型も変えない)。新しい列は過去の行では NULL。
    """
    for table, spec in TABLES.items():
        _run(client, _ddl(table))
        adds = ", ".join(f"ADD COLUMN IF NOT EXISTS {n} {t}" for n, t in spec["schema"])
        _run(client, f"ALTER TABLE {fqn(table)} {adds}")


def read_prices(client, since, codes: list[str] | None = None) -> pd.DataFrame:
    """価格テーブルから since 以降(対象銘柄だけ)を読み出す。"""
    from google.cloud import bigquery

    where = "date >= @since" + (" AND code IN UNNEST(@codes)" if codes else "")
    params = [bigquery.ScalarQueryParameter("since", "DATE", since)]
    if codes:
        params.append(bigquery.ArrayQueryParameter("codes", "STRING", list(codes)))
    job = _run(client, f"SELECT date, code, close, adj_close, volume, turnover, open, high, low FROM {fqn('prices')} "
                       f"WHERE {where}", params)
    df = job.to_dataframe(create_bqstorage_client=False)
    if df.empty:
        return pd.DataFrame(columns=["date", "code", "close", "adj_close", "volume", "turnover", "open", "high", "low"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["close", "adj_close", "volume", "turnover", "open", "high", "low"]:
        df[c] = df[c].astype("float64")
    df["code"] = df["code"].astype(str)
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def _bq_schema(schema):
    from google.cloud import bigquery

    return [bigquery.SchemaField(n, t) for n, t in schema]


def _load(client, df: pd.DataFrame, table_id: str, schema, truncate: bool) -> None:
    from google.cloud import bigquery

    cfg = bigquery.LoadJobConfig(
        schema=_bq_schema(schema),
        write_disposition=(bigquery.WriteDisposition.WRITE_TRUNCATE if truncate
                           else bigquery.WriteDisposition.WRITE_APPEND),
    )
    client.load_table_from_dataframe(df, table_id, job_config=cfg).result()


def replace_table(client, df: pd.DataFrame, table: str) -> None:
    """テーブルの中身を全置換する(ロードジョブ。テーブルの定義・説明は維持される)。"""
    schema = TABLES[table]["schema"]
    _load(client, df[[n for n, _ in schema]], f"{PROJECT}.{DATASET}.{table}", schema, truncate=True)


def merge_table(client, df: pd.DataFrame, table: str, keys: list[str],
                compare: list[str] | None = None, verify: bool = False) -> None:
    """df を一時テーブル経由で table に MERGE する。

    keys    ... 一致判定に使う列。一致すれば更新、なければ挿入
    compare ... 指定すると、その列のどれかが変わった行だけ更新する(loaded_at を無駄に動かさない)
    verify  ... MERGE 後に、送った全行が同じ値でテーブルにあるかを確かめ、なければ失敗する
    """
    schema = TABLES[table]["schema"]
    names = [n for n, _ in schema]
    stage = f"_stg_{table}_{uuid.uuid4().hex[:8]}"
    stage_id = f"{PROJECT}.{DATASET}.{stage}"
    try:
        _load(client, df[names], stage_id, schema, truncate=True)
        on = " AND ".join(f"T.{k} = S.{k}" for k in keys)
        updatable = [n for n in names if n not in keys]
        cond = ""
        if compare:
            cond = " AND (" + " OR ".join(f"T.{c} IS DISTINCT FROM S.{c}" for c in compare) + ")"
        sql = (
            f"MERGE {fqn(table)} T USING {fqn(stage)} S ON {on}\n"
            f"WHEN MATCHED{cond} THEN UPDATE SET {', '.join(f'{c} = S.{c}' for c in updatable)}\n"
            f"WHEN NOT MATCHED THEN INSERT ({', '.join(names)}) "
            f"VALUES ({', '.join(f'S.{c}' for c in names)})"
        )
        _run(client, sql)
        if verify:
            cols = compare or [n for n in names if n not in keys]
            diff = " OR ".join(f"T.{c} IS DISTINCT FROM S.{c}" for c in cols)
            on_t = " AND ".join(f"T.{k} = S.{k}" for k in keys)
            row = list(_run(client,
                f"SELECT COUNT(*) AS n, COUNTIF(T.{keys[0]} IS NULL OR {diff}) AS bad "
                f"FROM {fqn(stage)} S LEFT JOIN {fqn(table)} T ON {on_t}").result())[0]
            if row["n"] != len(df) or row["bad"]:
                raise RuntimeError(f"{table}: 反映後の照合に失敗(送った{len(df)}行 / 一時テーブル{row['n']}行 / "
                                   f"不一致{row['bad']}行)")
            print(f"照合OK: {table} {len(df)}行")
    finally:
        client.delete_table(stage_id, not_found_ok=True)
