"""アーティファクト(スナップショット版)に埋め込むデータを data/snapshot.json に書き出す。

BigQuery にはつながない。bq_sync.py の反映と同じ時点の prices.parquet / universe.csv から計算し直す
(実行順は bq_sync.py --require の後)。中身は、リアルタイム版アーティファクトが BigQuery から取っているものと同じ。

    rows    ... 業種ごとの最新指標(sector_summary の最新日)
    series  ... 業種ごとの直近120営業日の推移(sector_timeseries)
    stocks  ... 業種ごとの構成銘柄(直近20日の売買代金が大きい順に上位 STOCK_LIMIT 銘柄、直近 STOCK_DAYS 営業日)
              点は [日付の番号, 始値, 高値, 安値, 終値, 売買代金(百万円)]。日付の番号は dates の添字。
              始値・高値・安値・終値はすべて配当調整後の水準(終値 = adj_close、始値・高値・安値は
              分割調整のみの値に adj_close / close を掛けたもの)。始値などがない行は null
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bq_sync import NAV, PRICES, UNIVERSE, build_summary, build_timeseries  # noqa: E402
from metrics import compute  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "snapshot.json"
STOCK_LIMIT = 30
STOCK_DAYS = 180   # ローソク足の拡大縮小で遡れる営業日数(初期表示は直近60営業日)


def _num(v, nd=None):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    f = float(v)
    return round(f, nd) if nd is not None else f


def _px(v):
    """価格は 1000 以上なら小数1桁、未満なら2桁に丸める(ファイルを小さくするため)。"""
    if v is None or pd.isna(v):
        return None
    v = float(v)
    return round(v, 1 if abs(v) >= 1000 else 2)


def build(prices: pd.DataFrame, uni: pd.DataFrame, nav: pd.DataFrame | None, now: datetime) -> dict:
    res = compute(prices, uni, nav)

    summ = build_summary(res, now)
    rows = []
    for rec in summ.to_dict("records"):
        o = {}
        for k, v in rec.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                v = v.isoformat()
            elif hasattr(v, "isoformat"):          # date
                v = v.isoformat()
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = _num(v, 4)
            elif v is not None and pd.isna(v):
                v = None
            o[k] = v
        rows.append(o)

    ts = build_timeseries(res, now)
    series: dict[str, list] = {}
    for sector, g in ts.sort_values("date").groupby("sector"):
        series[sector] = [{"d": d.isoformat(), "c": _num(c, 3), "e": _num(e, 3), "w": _num(w, 3),
                           "s": _num(s, 3), "z": _num(z, 3)}
                          for d, c, e, w, s, z in zip(g["date"], g["rs_cw"], g["rs_etf"], g["rs_ew"],
                                                      g["share5_pct"], g["share_z"])]

    # 構成銘柄
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    if "adj_close" not in p.columns:
        p["adj_close"] = p["close"]
    stk = uni[uni["type"] == "stock"]
    for c in ("open", "high", "low"):
        if c not in p.columns:
            p[c] = np.nan
    f = (p["adj_close"] / p["close"]).where(p["close"] > 0)      # 配当調整の倍率
    for c in ("open", "high", "low"):
        p[c] = p[c] * f
    p = p[p["code"].isin(stk["code"])]
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=STOCK_DAYS * 7 // 5 + 30)
    p = p[p["date"] >= cutoff].sort_values(["code", "date"])
    p["rn"] = p.groupby("code").cumcount(ascending=False) + 1      # 1 = 最新日
    dates = sorted(p[p["rn"] <= STOCK_DAYS]["date"].unique())
    didx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    avg20 = p[p["rn"] <= 20].groupby("code")["turnover"].mean()
    stocks: dict[str, dict] = {}
    by_code = {c: g for c, g in p[p["rn"] <= STOCK_DAYS].groupby("code")}
    for sector, g in stk.groupby("sector"):
        codes = [c for c in g["code"] if c in by_code]
        codes.sort(key=lambda c: -(avg20.get(c) if pd.notna(avg20.get(c)) else -1))
        names = dict(zip(g["code"], g["name"]))
        lst = []
        for c in codes[:STOCK_LIMIT]:
            d = by_code[c]
            pts = [[didx[pd.Timestamp(dt)], _px(o), _px(h), _px(lo), _px(a), None if pd.isna(t) else int(round(t / 1e6))]
                   for dt, o, h, lo, a, t in zip(d["date"], d["open"], d["high"], d["low"], d["adj_close"], d["turnover"])]
            lst.append({"code": c, "name": names.get(c, c), "pts": pts})
        stocks[sector] = {"total": len(g), "list": lst}

    return {
        "as_of": res.as_of.date().isoformat(),
        "generated_at": now.isoformat(),
        "dates": [pd.Timestamp(d).date().isoformat() for d in dates],
        "rows": rows,
        "series": series,
        "stocks": stocks,
    }


def main() -> int:
    now = datetime.now(timezone.utc)
    prices = pd.read_parquet(PRICES)
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    nav = pd.read_parquet(NAV) if NAV.exists() else None
    snap = build(prices, uni, nav, now)
    OUT.write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n_st = sum(len(v["list"]) for v in snap["stocks"].values())
    print(f"snapshot: 基準日 {snap['as_of']} / 業種 {len(snap['rows'])} / 銘柄 {n_st} / {OUT.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
