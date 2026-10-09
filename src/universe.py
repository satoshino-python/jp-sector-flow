"""TOPIX 構成銘柄(JPX公表のウエイト一覧)から対象銘柄リスト data/universe.csv を作る。

使い方:
    python src/universe.py                      # 取得して作成(範囲は環境変数 UNIVERSE_SCOPE、既定 topix500)
    python src/universe.py --scope all          # TOPIX 全銘柄
    python src/universe.py --offline            # 取得せず、保存済みの data/topix_weights.csv から作る
    python src/universe.py --parse FILE.csv     # 手元のCSVを読んで内容を確認する(何も書き込まない)

流れ:
    1. JPX の topixweight_j.csv(Shift-JIS、月1回更新)を取得し、列名を推定して読み取る
    2. 読み取った結果を data/topix_weights.csv に保存(取得に失敗したら前回の保存分を使う)
    3. 東証33業種を TOPIX-17 業種に対応付け、範囲(topix500 / all)で絞り込む
    4. ETF・TOPIX連動ETF(data/etfs.csv、手作業の固定リスト)と合わせて data/universe.csv に書く

JPX のCSVは列構成が変わる可能性があるので、列は名前の一部(「コード」「業種」「ウエイト」など)で探す。
見つからない・件数が想定外・業種が対応表にない場合は、universe.csv を書き換えずに失敗する。
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "data" / "topix_weights.csv"
ETFS = ROOT / "data" / "etfs.csv"
UNIVERSE = ROOT / "data" / "universe.csv"

JPX_URL = "https://www.jpx.co.jp/automation/markets/indices/topix/files/topixweight_j.csv"
UA = "Mozilla/5.0 (compatible; jp-sector-flow; +https://github.com/satoshino-python/jp-sector-flow)"

UNIVERSE_COLUMNS = ["code", "name", "sector", "type", "sector33", "size", "weight", "weight_date"]
WEIGHT_COLUMNS = ["weight_date", "code", "name", "sector33", "size", "weight"]

# 東証33業種 → TOPIX-17 業種(JPXの定義どおり。右側の名前は data/etfs.csv の sector と同じ表記)
SECTOR33_TO_17 = {
    "水産・農林業": "食品", "食料品": "食品",
    "鉱業": "エネルギー資源", "石油・石炭製品": "エネルギー資源",
    "建設業": "建設・資材", "ガラス・土石製品": "建設・資材", "金属製品": "建設・資材",
    "繊維製品": "素材・化学", "パルプ・紙": "素材・化学", "化学": "素材・化学",
    "医薬品": "医薬品",
    "ゴム製品": "自動車・輸送機", "輸送用機器": "自動車・輸送機",
    "鉄鋼": "鉄鋼・非鉄", "非鉄金属": "鉄鋼・非鉄",
    "機械": "機械",
    "電気機器": "電機・精密", "精密機器": "電機・精密",
    "その他製品": "情報通信・サービスその他", "情報・通信業": "情報通信・サービスその他",
    "サービス業": "情報通信・サービスその他",
    "電気・ガス業": "電力・ガス",
    "陸運業": "運輸・物流", "海運業": "運輸・物流", "空運業": "運輸・物流",
    "倉庫・運輸関連業": "運輸・物流",
    "卸売業": "商社・卸売",
    "小売業": "小売",
    "銀行業": "銀行",
    "証券、商品先物取引業": "金融(除く銀行)", "保険業": "金融(除く銀行)",
    "その他金融業": "金融(除く銀行)",
    "不動産業": "不動産",
}

# 範囲ごとの「ニューインデックス区分」と、想定する銘柄数の範囲(外れたら読み取りを疑って失敗する)
SCOPES = {
    "topix500": dict(sizes=("core30", "large70", "mid400"), expect=(430, 540)),
    "all": dict(sizes=None, expect=(1000, 2300)),
}
STALE_DAYS = 70       # JPXは月1回更新。これより古いウエイトしかなければ失敗扱い
CODE_RE = re.compile(r"^[0-9][0-9A-Z]{3}$")


def _norm(s) -> str:
    """表記ゆれ(全角/半角、空白、読点の種類)を吸収した比較用の文字列。"""
    t = unicodedata.normalize("NFKC", str(s)).strip()
    t = re.sub(r"\s+", "", t)
    return t.replace("、", ",").replace("､", ",").replace("・", "･").lower()


_SECTOR_KEYS = {_norm(k): v for k, v in SECTOR33_TO_17.items()}


def sector17(sector33: str) -> str | None:
    return _SECTOR_KEYS.get(_norm(sector33))


def _decode(raw: bytes) -> str:
    for enc in ("cp932", "utf-8-sig"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace")


def _find(header: list[str], *keys: str, required: bool = True) -> int | None:
    nh = [_norm(h) for h in header]
    for k in keys:
        nk = _norm(k)
        for i, h in enumerate(nh):
            if nk in h:
                return i
    if required:
        raise ValueError(f"列が見つかりません(探した名前: {', '.join(keys)})。ヘッダー: {header}")
    return None


def _code(v: str) -> str:
    c = unicodedata.normalize("NFKC", str(v)).strip().upper()
    c = re.sub(r"\.0+$", "", c)       # 数値として書かれた "1301.0"
    return c


def _weight(v: str) -> float | None:
    t = unicodedata.normalize("NFKC", str(v)).strip().replace("%", "").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _date(v: str) -> str | None:
    t = unicodedata.normalize("NFKC", str(v)).strip()
    m = re.search(r"(\d{4})[/\-年.](\d{1,2})[/\-月.](\d{1,2})", t)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def parse_jpx(raw: bytes) -> pd.DataFrame:
    """JPX のウエイト一覧CSVを読み、WEIGHT_COLUMNS の DataFrame にする。"""
    rows = list(csv.reader(io.StringIO(_decode(raw))))
    hi = next((i for i, r in enumerate(rows) if any("コード" in unicodedata.normalize("NFKC", c) for c in r)),
              None)
    if hi is None:
        raise ValueError("ヘッダー行(「コード」を含む行)が見つかりません。先頭: " + repr(rows[:3]))
    header = rows[hi]
    ic = _find(header, "コード")
    iname = _find(header, "銘柄名", "銘柄")
    isec = _find(header, "業種")
    iw = _find(header, "ウエイト", "ウェイト", "構成比")
    isize = _find(header, "ニューインデックス区分", "インデックス区分", "規模", "区分", required=False)
    idate = _find(header, "日付", "基準日", required=False)

    out = []
    for r in rows[hi + 1:]:
        if len(r) <= max(ic, iname, isec, iw):
            continue
        code = _code(r[ic])
        if not CODE_RE.match(code):
            continue          # 合計行・注記行など
        out.append({
            "weight_date": _date(r[idate]) if idate is not None else None,
            "code": code,
            "name": unicodedata.normalize("NFKC", r[iname]).strip(),
            "sector33": r[isec].strip(),
            "size": r[isize].strip() if isize is not None else "",
            "weight": _weight(r[iw]),
        })
    df = pd.DataFrame(out, columns=WEIGHT_COLUMNS)
    if df.empty:
        raise ValueError("銘柄の行が1件も読み取れませんでした。ヘッダー: " + repr(header))
    if df["code"].duplicated().any():
        dup = df.loc[df["code"].duplicated(), "code"].tolist()[:10]
        raise ValueError(f"コードが重複しています: {dup}")
    if df["weight"].isna().mean() > 0.01:
        raise ValueError("ウエイトを数値として読めない行が多すぎます。例: "
                         + repr(df.loc[df["weight"].isna()].head(3).to_dict("records")))
    total = df["weight"].sum()
    if 0.9 < total < 1.1:          # 比率(合計1)で書かれていた場合は%にそろえる
        df["weight"] = df["weight"] * 100
        total *= 100
    if len(df) >= 1000 and not (95 < total < 105):
        raise ValueError(f"ウエイトの合計が100%付近になりません({total:.2f})。列の読み違いの可能性があります。")
    dates = df["weight_date"].dropna()
    df["weight_date"] = dates.max() if len(dates) else None
    unmapped = sorted({s for s in df["sector33"] if sector17(s) is None})
    if unmapped:
        raise ValueError(f"17業種への対応表にない業種名があります: {unmapped}")
    return df


def download(url: str = JPX_URL, retries: int = 3) -> bytes:
    last = None
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"JPXのCSV取得に失敗(試行{i}/{retries}): {e}", file=sys.stderr)
            time.sleep(5 * i)
    raise RuntimeError(f"JPXのCSVを取得できませんでした: {last}")


def _in_scope(size: pd.Series, scope: str) -> pd.Series:
    sizes = SCOPES[scope]["sizes"]
    if sizes is None:
        return pd.Series(True, index=size.index)
    ns = size.map(_norm)
    return ns.apply(lambda v: any(k in v for k in sizes))


def build_universe(weights: pd.DataFrame, etfs: pd.DataFrame, scope: str) -> pd.DataFrame:
    w = weights[_in_scope(weights["size"].fillna(""), scope)].copy()
    lo, hi = SCOPES[scope]["expect"]
    if not lo <= len(w) <= hi:
        sizes = weights["size"].value_counts().to_dict()
        raise ValueError(f"範囲 {scope} の銘柄数が想定外です({len(w)}銘柄。想定 {lo}〜{hi})。"
                         f"区分の内訳: {sizes}")
    w["sector"] = w["sector33"].map(sector17)
    w["type"] = "stock"
    etf_codes = set(etfs["code"])
    w = w[~w["code"].isin(etf_codes)]
    stocks = w.sort_values(["sector", "weight"], ascending=[True, False])
    e = etfs.copy()
    for c in UNIVERSE_COLUMNS:
        if c not in e.columns:
            e[c] = None
    missing = set(e.loc[e["type"] == "etf", "sector"]) - set(stocks["sector"])
    if missing:
        raise ValueError(f"構成銘柄が0になる業種があります: {sorted(missing)}")
    return pd.concat([e[UNIVERSE_COLUMNS], stocks[UNIVERSE_COLUMNS]], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default=os.environ.get("UNIVERSE_SCOPE", "topix500"), choices=sorted(SCOPES))
    ap.add_argument("--offline", action="store_true", help="取得せず保存済みのウエイトから作る")
    ap.add_argument("--parse", metavar="CSV", help="手元のCSVを読んで要約を表示するだけ")
    args = ap.parse_args()

    if args.parse:
        df = parse_jpx(Path(args.parse).read_bytes())
        print(df.head(10).to_string())
        print(f"{len(df)}銘柄 / 基準日 {df['weight_date'].iloc[0]} / ウエイト合計 {df['weight'].sum():.3f}")
        print(df["size"].value_counts().to_string())
        return 0

    weights = None
    if not args.offline:
        try:
            weights = parse_jpx(download())
            weights.to_csv(WEIGHTS, index=False)
            print(f"TOPIXウエイトを更新: {len(weights)}銘柄 / 基準日 {weights['weight_date'].iloc[0]}")
        except Exception as e:  # noqa: BLE001
            print(f"TOPIXウエイトの取得・読み取りに失敗: {e}", file=sys.stderr)
    if weights is None:
        if not WEIGHTS.exists():
            print("保存済みのウエイトもないため、universe.csv は変更しません。", file=sys.stderr)
            return 1
        weights = pd.read_csv(WEIGHTS, dtype={"code": str, "size": str})
        print(f"保存済みのウエイトを使用: 基準日 {weights['weight_date'].iloc[0]}", file=sys.stderr)

    etfs = pd.read_csv(ETFS, dtype={"code": str})
    uni = build_universe(weights, etfs, args.scope)
    uni.to_csv(UNIVERSE, index=False)
    n = (uni["type"] == "stock").sum()
    print(f"universe.csv を作成: 範囲 {args.scope} / 個別株 {n}銘柄 + ETF {len(etfs)}本")
    print(uni[uni["type"] == "stock"].groupby("sector").size().to_string())
    # 取得に失敗しても保存分で続行するが、ウエイトが古すぎる(2か月超)なら失敗にして気づけるようにする
    wd = pd.to_datetime(uni["weight_date"].dropna().max(), errors="coerce")
    if pd.notna(wd) and (pd.Timestamp.today() - wd).days > STALE_DAYS:
        print(f"TOPIXウエイトの基準日 {wd.date()} が古すぎます。JPXのCSVの取得・形式を確認してください。",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
