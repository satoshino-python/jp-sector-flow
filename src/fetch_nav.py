"""業種別ETFの基準価額(分配金再投資)を取得し、data/nav.parquet に保存する。

終値との突き合わせ専用の補助データ。取得に失敗しても本体の処理は止めない(終値のみで継続)。
データは毎回CSV全体を取り直すため蓄積はせず、nav.parquet はリポジトリにも含めない(.gitignore)。

使い方:
    python src/fetch_nav.py           # 野村アセットのCSVから取得
    python src/fetch_nav.py --demo    # 合成データ(動作確認用。prices.parquetが必要)
"""
import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "universe.csv"
PRICES = ROOT / "data" / "prices.parquet"
NAV = ROOT / "data" / "nav.parquet"

URL = "https://www.nomura-am.co.jp/fund/etf/history/ETF_{code}.csv"
UA = "Mozilla/5.0 (compatible; jp-sector-flow/1.0)"
KEEP_DAYS = 450
RETRIES = 3


def parse_csv(raw: bytes) -> pd.Series | None:
    """CSVから「分配金再投資の基準価額」を日付index・float値のSeriesで返す。

    ヘッダ行は 'Date' で始まる行を探して特定する(前後にメタ情報が付いても読めるように)。
    分配金再投資列が見つからない場合は、権利落ちで値が不連続になるため None を返す。
    """
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.lstrip('"').startswith("Date")), None)
    if start is None:
        return None
    rows = list(csv.reader(lines[start:]))
    header = rows[0]
    col = next((i for i, h in enumerate(header) if "Dividend-included" in h), None)
    if col is None:
        return None
    dates, vals = [], []
    for r in rows[1:]:
        if len(r) <= col or not (len(r[0]) == 8 and r[0].isdigit()):
            continue   # メタ情報などの行は読み飛ばす
        try:
            v = float(r[col])
        except ValueError:
            continue
        dates.append(pd.to_datetime(r[0], format="%Y%m%d", errors="coerce"))
        vals.append(v)
    if not dates:
        return None
    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).dropna()
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.tail(KEEP_DAYS)


def fetch_one(code: str) -> pd.Series | None:
    req = urllib.request.Request(URL.format(code=code), headers={"User-Agent": UA})
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                s = parse_csv(r.read())
            if s is not None and len(s) > 0:
                return s
            last = "CSVの形式が想定と違います"
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 * attempt)
    print(f"  {code}: 取得失敗 ({last})", file=sys.stderr)
    return None


def make_demo() -> pd.DataFrame:
    """ETF終値にノイズを乗せた合成NAV。流動性が低い業種ほどノイズを大きくする。"""
    rng = np.random.default_rng(7)
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    etf_codes = uni.loc[uni["type"] == "etf", "code"].tolist()
    p = pd.read_parquet(PRICES)
    rows = []
    for i, c in enumerate(etf_codes):
        g = p[p["code"] == c].sort_values("date")
        noise_sd = 0.0015 if i % 4 else 0.006
        drift = np.cumsum(rng.normal(0, noise_sd, len(g)))
        rows.append(pd.DataFrame({"date": g["date"].values, "code": c,
                                  "nav": g["close"].values * np.exp(-drift * 0.3)}))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    if "--demo" in sys.argv:
        df = make_demo()
        df.to_parquet(NAV, index=False)
        print(f"デモNAVを保存: {len(df)}行")
        return 0

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    codes = uni.loc[uni["type"] == "etf", "code"].tolist()
    frames = []
    for c in codes:
        s = fetch_one(c)
        if s is not None:
            frames.append(pd.DataFrame({"date": s.index, "code": c, "nav": s.values}))
    if not frames:
        print("基準価額を1本も取得できませんでした(終値のみで続行します)。", file=sys.stderr)
        NAV.unlink(missing_ok=True)   # 古い値を使い回さない
        return 1
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(NAV, index=False)
    print(f"基準価額: {df['code'].nunique()}/{len(codes)}本 / 最新日 {df['date'].max().date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
