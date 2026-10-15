"""ミネルヴィニのトレンドテンプレートで候補を絞り、VCP(ボラティリティ収縮パターン)の買いポイントと、
銘柄ごとのボラティリティ(ATR)を加味した損切り位置を出す。

使い方:
    python src/minervini.py                       # data/prices.parquet の最新日で判定し data/minervini.csv に書き出す
    python src/minervini.py --as-of 2026-09-30    # 過去のある日時点で判定(その日より後のデータは使わない)
    python src/minervini.py --equity 5000000      # 資金額を渡すと、1トレードのリスクから株数(100株単位)も出す
    python src/minervini.py --bq                  # parquet を使わず BigQuery の prices から直接読む(要認証)

手順:
  1. トレンドテンプレート(8条件。全部満たす銘柄が候補)
       1) 株価 > 150日線 かつ > 200日線      2) 150日線 > 200日線
       3) 200日線が1か月(21営業日)前より上   4) 50日線 > 150日線 かつ > 200日線
       5) 株価 > 50日線                       6) 株価が52週安値の +30% 以上
       7) 株価が52週高値の -25% 以内          8) RS(相対力)レーティング 70 以上
     RS は IBD の独自指標の代わりに、直近3・6・9・12か月の騰落率を 2:1:1:1 で加重した値を、
     対象銘柄(universe.csv の個別株)の中でパーセンタイル順位にしたもの(1〜99)。
     対象が TOPIX500 の間は「TOPIX500の中での相対力」になる点に注意。
  2. VCP の検出(候補銘柄のみ)
       ・ベースの起点 = 直近 BASE_LOOKBACK 日(最後の BREAKOUT_LOOKBACK 日を除く)の最高値
       ・起点から今日までを ZigZag で高値・安値に分け、高値→次の安値の下落率を「収縮」とする。
         ZigZag の反転幅は銘柄の ATR に比例させる(値動きの荒い銘柄で小さな揺れを収縮と数えないため)
       ・収縮が MIN_CONTRACTIONS〜MAX_CONTRACTIONS 回、最初が MAX_FIRST_DEPTH 以内、最後が MAX_FINAL_DEPTH 以内で
         ベースの中で最も浅く(途中で一度深くなってもよい)、最初の FINAL_VS_FIRST 倍以下、
         直近の出来高が50日平均より細っていれば VCP とみなす
       ・ピボット = 最後の収縮の起点の高値。終値がこれを出来高を伴って上抜けたら買い
  3. 損切り(銘柄ごとのボラティリティを加味)
       ・構造上の損切り = 最後の収縮の安値 − STOP_ATR_BUFFER × ATR
       ・ノイズで刈られない最低幅 = エントリー − STOP_ATR_MIN × ATR
       ・この2つの「遠い方」を損切りにする。損切り幅が MAX_RISK_PCT(ミネルヴィニの上限10%)を超える銘柄は
         「リスク過大」として見送り扱い(損切りを勝手に浅くはしない)
       ・利益目標の目安 2R・3R、建値へストップを引き上げる目安(BREAKEVEN_R)も出す

業績(EPS・売上の加速)は取得していないため判定に入っていない。候補は決算で別途確かめること。
本ツールは投資判断の材料の一つであり、売買の推奨ではない。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
PRICES = ROOT / "data" / "prices.parquet"
UNIVERSE = ROOT / "data" / "universe.csv"
SUMMARY = ROOT / "data" / "summary.csv"
OUT = ROOT / "data" / "minervini.csv"
MIN_SAVE_TT = 6            # BigQuery に保存する銘柄: テンプレートをこの条件数以上満たしたもの(惜しい銘柄も確かめられるように)
BQ_DAYS = 420              # --bq で読む暦日数(52週・200日線・12か月RSに足りる約290営業日)

# --- トレンドテンプレート ---
MA200_RISING_DAYS = 21     # 200日線が上向き: この日数前より上(1か月)
MIN_ABOVE_LOW = 0.30       # 52週安値から +30% 以上
MAX_BELOW_HIGH = 0.25      # 52週高値から -25% 以内
MIN_RS = 70                # RSレーティングの下限
WEEKS52 = 252
RS_PERIODS = [(63, 2.0), (126, 1.0), (189, 1.0), (252, 1.0)]   # (営業日, 重み)

# --- VCP ---
ATR_DAYS = 20
BASE_LOOKBACK = 130        # ベースの起点を探す範囲(約6か月)
BREAKOUT_LOOKBACK = 5      # 起点探しから除く直近の日数(ブレイクした日が起点にならないように)
MIN_BASE_DAYS = 15         # ベースの最短期間(約3週間)
ZZ_MIN_PCT = 0.05          # ZigZag の反転幅の下限
ZZ_ATR_MULT = 2.5          # ZigZag の反転幅 = max(ZZ_MIN_PCT, ZZ_ATR_MULT × ATR%)。大きな収縮だけを数えるため粗めにする
ZZ_FINE_MIN_PCT = 0.03     # 最後の大きな転換点より後(最後の収縮のあたり)だけは、細かい反転幅で見る
ZZ_FINE_ATR_MULT = 1.5
MIN_CONTRACTIONS = 2
MAX_CONTRACTIONS = 6
MAX_FIRST_DEPTH = 0.35     # 最初の収縮の深さの上限
MAX_FINAL_DEPTH = 0.10     # 最後の収縮の深さの上限(タイトな値動き)
FINAL_TOL = 1.10           # 最後の収縮は、それまでの収縮で最も浅いもののこの倍率以下(ほぼ同じ深さは許容)。
                           # 途中で一度深くなっても、最後が最もタイトなら可
MID_TOL = 1.10             # 途中の収縮は、最初の収縮のこの倍率以下(最初より深い収縮が途中にあるものは VCP としない)
FINAL_VS_FIRST = 0.5       # 最後の収縮は最初の収縮のこの倍率以下
VOL_AVG_DAYS = 50
DRYUP_DAYS = 10            # 出来高の細りを見る直近日数
MAX_DRYUP_RATIO = 0.85     # 直近の平均出来高 / 50日平均 がこれ以下なら「細っている」
BREAKOUT_VOL_RATIO = 1.4   # ブレイク当日の出来高 / 50日平均 がこれ以上なら出来高を伴うブレイク
MAX_CHASE = 0.05           # ピボットからこれ以上離れたら追いかけない(伸びすぎ)
NEAR_PIVOT = 0.05          # ピボットの手前この範囲なら「ピボット接近」

# --- 損切り・資金管理 ---
STOP_ATR_BUFFER = 0.5      # 最後の収縮の安値から、さらに ATR のこの倍だけ下に置く
STOP_ATR_MIN = 1.5         # 損切り幅の下限(ATR の倍数)。日々の値動きで刈られないように
MAX_RISK_PCT = 0.10        # 損切り幅の上限(ミネルヴィニの「最大10%」)
BREAKEVEN_R = 2.0          # 含み益がこの R に達したらストップを建値へ(目安)
RISK_PER_TRADE = 0.0125    # 1トレードで資金の何割までを損失として許すか(--equity 指定時)
MAX_POSITION_PCT = 0.25    # 1銘柄の建玉の上限(資金比)
LOT = 100


def _pivot(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="code", values=col).sort_index()


def load_panels(prices: pd.DataFrame, codes: list[str]) -> dict[str, pd.DataFrame]:
    """配当調整後の水準にそろえた始値・高値・安値・終値と出来高。

    始値・高値・安値がない行(古いデータ)は終値で埋める(その分 ATR は小さめに出る)。
    """
    close = _pivot(prices, "close").reindex(columns=codes)
    adj = (_pivot(prices, "adj_close") if "adj_close" in prices.columns else close).reindex(columns=codes)
    adj = adj.fillna(close)
    f = (adj / close).where(close > 0)
    out = {"close": adj, "volume": _pivot(prices, "volume").reindex(columns=codes)}
    for c in ["open", "high", "low"]:
        raw = _pivot(prices, c).reindex(columns=codes) if c in prices.columns else close * np.nan
        out[c] = (raw * f).fillna(adj)
    return out


def rs_rating(close: pd.DataFrame) -> pd.Series:
    """3・6・9・12か月騰落率の加重値を、銘柄間のパーセンタイル順位(1〜99)にする。

    上場から日が浅く期間が足りない銘柄は、取れる期間だけで重みを付け直す。
    """
    last = close.ffill().iloc[-1]
    num = pd.Series(0.0, index=close.columns)
    den = pd.Series(0.0, index=close.columns)
    for n, w in RS_PERIODS:
        if len(close) <= n:
            continue
        r = last / close.iloc[-1 - n] - 1
        ok = r.notna()
        num[ok] += w * r[ok]
        den[ok] += w
    score = (num / den).where(den > 0)
    return (score.rank(pct=True) * 98 + 1).round().clip(1, 99)


def trend_template(p: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close = p["close"]
    c = close.iloc[-1]
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    m50, m150, m200 = ma50.iloc[-1], ma150.iloc[-1], ma200.iloc[-1]
    m200_prev = ma200.iloc[-1 - MA200_RISING_DAYS] if len(ma200) > MA200_RISING_DAYS else ma200.iloc[0] * np.nan
    hi52 = p["high"].tail(WEEKS52).max()
    lo52 = p["low"].tail(WEEKS52).min()
    rs = rs_rating(close)
    df = pd.DataFrame({
        "close": c, "ma50": m50, "ma150": m150, "ma200": m200,
        "high52": hi52, "low52": lo52, "rs": rs,
        "pct_from_high52": (c / hi52 - 1) * 100,
        "pct_from_low52": (c / lo52 - 1) * 100,
    })
    df["tt1"] = (c > m150) & (c > m200)
    df["tt2"] = m150 > m200
    df["tt3"] = m200 > m200_prev
    df["tt4"] = (m50 > m150) & (m50 > m200)
    df["tt5"] = c > m50
    df["tt6"] = c >= lo52 * (1 + MIN_ABOVE_LOW)
    df["tt7"] = c >= hi52 * (1 - MAX_BELOW_HIGH)
    df["tt8"] = rs >= MIN_RS
    tt = [f"tt{i}" for i in range(1, 9)]
    df["tt_score"] = df[tt].sum(axis=1)
    df["tt_pass"] = df["tt_score"] == 8
    return df


def trend_template_panel(p: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """trend_template を全営業日について計算する(バックテスト用)。値は [日付 × 銘柄] の表。

    ある日の行は、その日までのデータで trend_template を計算した結果と同じになるようにしている。
    """
    close = p["close"]
    ma50, ma150, ma200 = (close.rolling(n).mean() for n in (50, 150, 200))
    hi52 = p["high"].rolling(WEEKS52, min_periods=1).max()
    lo52 = p["low"].rolling(WEEKS52, min_periods=1).min()
    last = close.ffill()
    num = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    den = num.copy()
    for n, w in RS_PERIODS:
        r = last / close.shift(n) - 1
        ok = r.notna()
        num = num + (w * r).where(ok, 0.0)
        den = den + ok * w
    score = (num / den).where(den > 0)
    rs = (score.rank(axis=1, pct=True) * 98 + 1).round().clip(1, 99)
    flags = [
        (close > ma150) & (close > ma200),
        ma150 > ma200,
        ma200 > ma200.shift(MA200_RISING_DAYS),
        (ma50 > ma150) & (ma50 > ma200),
        close > ma50,
        close >= lo52 * (1 + MIN_ABOVE_LOW),
        close >= hi52 * (1 - MAX_BELOW_HIGH),
        rs >= MIN_RS,
    ]
    tt_pass = flags[0]
    for f in flags[1:]:
        tt_pass = tt_pass & f
    return {"tt_pass": tt_pass & close.notna(), "rs": rs, "ma50": ma50}


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = ATR_DAYS) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(5, n // 2)).mean()


def zigzag(high: np.ndarray, low: np.ndarray, th: float, first: str = "H") -> list[tuple[str, int, float]]:
    """先頭から始まる ZigZag。確定した転換点 ('H'/'L', 位置, 値) の列を返す。

    先頭は first で指定した種類の転換点(既定は高値 = ベースの起点)。最後に、まだ確定していない進行中の極値を付ける。
    起点の日の安値(高値)は使わず、翌日から追う。1日の中で高値と安値のどちらが先かは分からず、
    起点の日の値幅を「高値の後の下落」と数えてしまうため(決算などで値幅の大きい日に起きる)。
    """
    if len(high) < 2:
        return [(first, 0, high[0] if first == "H" else low[0])] * 2
    if first == "H":
        pts = [("H", 0, high[0])]
        mode, ext_i, ext_v = "down", 1, low[1]    # 起点の高値から下落を追う
    else:
        pts = [("L", 0, low[0])]
        mode, ext_i, ext_v = "up", 1, high[1]     # 起点の安値から上昇を追う
    for i in range(2, len(high)):
        if mode == "down":
            if low[i] < ext_v:
                ext_i, ext_v = i, low[i]
            elif high[i] >= ext_v * (1 + th):     # 安値から th 以上戻した → 安値を確定
                pts.append(("L", ext_i, ext_v))
                mode, ext_i, ext_v = "up", i, high[i]
        else:
            if high[i] > ext_v:
                ext_i, ext_v = i, high[i]
            elif low[i] <= ext_v * (1 - th):      # 高値から th 以上下げた → 高値を確定
                pts.append(("H", ext_i, ext_v))
                mode, ext_i, ext_v = "down", i, low[i]
    pts.append(("L" if mode == "down" else "H", ext_i, ext_v))   # 進行中(未確定)
    return pts


@dataclass
class Vcp:
    ok: bool = False
    reason: str = ""
    base_days: int = 0
    depths: list[float] = field(default_factory=list)
    pivot: float = np.nan
    final_low: float = np.nan
    dryup: float = np.nan
    start: int = -1                                   # ベースの起点の位置(系列の先頭からの添字)
    points: list = field(default_factory=list)        # ZigZag の転換点 ('H'/'L', 添字, 値)。最後は進行中
    dry_end: int = -1                                 # 出来高の細りを見た期間の終わり(この添字の手前まで)


def detect_vcp(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
               atr_pct: float) -> Vcp:
    h, l, c = high.to_numpy(float), low.to_numpy(float), close.to_numpy(float)
    n = len(c)
    if n < BASE_LOOKBACK or not np.isfinite(atr_pct):
        return Vcp(reason="データ不足")
    lo_i = n - BASE_LOOKBACK
    hi_i = n - BREAKOUT_LOOKBACK
    start = lo_i + int(np.nanargmax(h[lo_i:hi_i]))
    base_days = n - 1 - start
    if base_days < MIN_BASE_DAYS:
        return Vcp(reason="ベースが短い", base_days=base_days, start=start)

    # 大きな収縮は粗い ZigZag で数え、最後の大きな転換点より後だけを細かい ZigZag で見直す
    # (最後のタイトな収縮は粗い反転幅より浅いことが多く、戻した後だと粗い方では拾えないため)
    th = max(ZZ_MIN_PCT, ZZ_ATR_MULT * atr_pct)
    coarse = zigzag(h[start:], l[start:], th)[:-1]
    k0 = coarse[-1][1]
    fine = zigzag(h[start + k0:], l[start + k0:], max(ZZ_FINE_MIN_PCT, ZZ_FINE_ATR_MULT * atr_pct), first=coarse[-1][0])
    pts = coarse[:-1] + [(t, k0 + i, val) for t, i, val in fine]
    confirmed, last = pts[:-1], pts[-1]
    # 高値→安値 の組を収縮とする。最後が「確定した高値+進行中の安値」なら、それも収縮に数える
    seq = confirmed + ([last] if last[0] == "L" else [])
    highs = [p for p in seq if p[0] == "H"]
    depths, lows = [], []
    for k in range(len(seq) - 1):
        if seq[k][0] == "H" and seq[k + 1][0] == "L":
            depths.append(1 - seq[k + 1][2] / seq[k][2])
            lows.append(seq[k + 1])
    v = Vcp(base_days=base_days, depths=depths, start=start,
            points=[(k, start + i, val) for k, i, val in pts])
    if not depths:
        v.reason = "収縮なし"
        return v
    v.pivot = highs[len(depths) - 1][2]           # 最後の収縮の起点の高値
    v.final_low = lows[-1][2]
    vol = volume.to_numpy(float)
    avg = np.nanmean(vol[-VOL_AVG_DAYS - DRYUP_DAYS:-DRYUP_DAYS]) if n > VOL_AVG_DAYS + DRYUP_DAYS else np.nan
    # ブレイク後は出来高が膨らむので、細りは最後の収縮の安値までの直近で見る
    end = start + lows[-1][1] + 1 if last[0] == "H" else n
    v.dryup = float(np.nanmean(vol[max(end - DRYUP_DAYS, 0):end]) / avg) if avg and avg > 0 else np.nan
    v.dry_end = end

    reasons = []
    if not MIN_CONTRACTIONS <= len(depths) <= MAX_CONTRACTIONS:
        reasons.append(f"収縮{len(depths)}回")
    if depths[0] > MAX_FIRST_DEPTH:
        reasons.append("最初の収縮が深い")
    if depths[-1] > MAX_FINAL_DEPTH:
        reasons.append("最後の収縮が深い")
    if len(depths) >= 3 and max(depths[1:-1]) > depths[0] * MID_TOL:
        reasons.append("途中の収縮が最初より深い")
    if len(depths) >= 2 and depths[-1] > min(depths[:-1]) * FINAL_TOL:
        reasons.append("最後の収縮が最も浅くない")
    if len(depths) >= 2 and depths[-1] > depths[0] * FINAL_VS_FIRST:
        reasons.append("収縮の縮み不足")
    if not (v.dryup <= MAX_DRYUP_RATIO):
        reasons.append("出来高が細っていない")
    v.ok = not reasons
    v.reason = " / ".join(reasons)
    return v


def entry_status(v: Vcp, close: pd.Series, volume: pd.Series) -> tuple[str, float, int]:
    """(状態, ブレイク日の出来高倍率, ブレイクからの日数)"""
    c = close.to_numpy(float)
    vol = volume.to_numpy(float)
    avg = pd.Series(vol).rolling(VOL_AVG_DAYS).mean().shift(1).to_numpy()
    last = c[-1]
    if last <= v.pivot:
        dist = v.pivot / last - 1
        return ("ピボット接近" if dist <= NEAR_PIVOT else "ベース形成中"), np.nan, -1
    # 終値が初めてピボットを上抜けた日(直近 BREAKOUT_LOOKBACK 日の中で、それより前は下にいた日)
    above = c > v.pivot
    k = len(c) - 1
    while k > 0 and above[k - 1] and len(c) - 1 - k < BREAKOUT_LOOKBACK:
        k -= 1
    days = len(c) - 1 - k
    vr = vol[k] / avg[k] if avg[k] and avg[k] > 0 else np.nan
    if last > v.pivot * (1 + MAX_CHASE):
        return "伸びすぎ(見送り)", vr, days
    if not (vr >= BREAKOUT_VOL_RATIO):
        return "ブレイク(出来高不足)", vr, days
    return ("ブレイク" if days == 0 else f"ブレイク{days}日目"), vr, days


def plan_trade(v: Vcp, last: float, atr_v: float, broke_out: bool, equity: float | None,
               stop_atr_min: float = STOP_ATR_MIN) -> dict:
    entry = last if broke_out else v.pivot
    structural = v.final_low - STOP_ATR_BUFFER * atr_v
    noise_floor = entry - stop_atr_min * atr_v
    stop = min(structural, noise_floor)
    r = entry - stop
    risk_pct = r / entry
    plan = {
        "entry": entry, "stop": stop,
        "stop_basis": "収縮安値" if structural <= noise_floor else f"{stop_atr_min}ATR",
        "risk_pct": risk_pct * 100, "risk_atr": r / atr_v,
        "target_2r": entry + 2 * r, "target_3r": entry + 3 * r,
        "breakeven_at": entry + BREAKEVEN_R * r,
        "risk_ok": risk_pct <= MAX_RISK_PCT,
    }
    if equity:
        by_risk = equity * RISK_PER_TRADE / r
        by_cap = equity * MAX_POSITION_PCT / entry
        plan["shares"] = int(min(by_risk, by_cap) // LOT * LOT)
    return plan


def screen(prices: pd.DataFrame, uni: pd.DataFrame,
           equity: float | None = None) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """(基準日, テンプレート通過銘柄の判定結果, 全銘柄のテンプレート判定) を返す。"""
    stocks = uni[uni["type"] == "stock"].set_index("code")
    codes = [c for c in stocks.index if c in set(prices["code"])]
    p = load_panels(prices, codes)
    as_of = p["close"].index[-1]
    tt = trend_template(p)
    tt = tt[tt["close"].notna()]

    rows = []
    for code in tt.index[tt["tt_pass"]]:
        hi, lo, cl, vo = (p[k][code].dropna() for k in ["high", "low", "close", "volume"])
        idx = cl.index
        hi, lo, vo = hi.reindex(idx).fillna(cl), lo.reindex(idx).fillna(cl), vo.reindex(idx)
        a = atr(hi, lo, cl).iloc[-1]
        atr_pct = a / cl.iloc[-1]
        v = detect_vcp(hi, lo, cl, vo, atr_pct)
        day = lambda i: idx[i].strftime("%Y-%m-%d")   # noqa: E731
        row = {"code": code, "atr_pct": atr_pct * 100, "vcp": v.ok, "vcp_note": v.reason,
               "base_days": v.base_days, "n_contractions": len(v.depths),
               "depths": " → ".join(f"{d * 100:.0f}%" for d in v.depths),
               "pivot": v.pivot, "final_low": v.final_low, "dryup": v.dryup,
               # 検証用ページでチャートに重ねるための位置(日付)
               "base_start": day(v.start) if v.start >= 0 else None,
               "zigzag": json.dumps([[k, day(i), round(float(val), 2)] for k, i, val in v.points]) if v.points else None,
               "dry_from": day(max(v.dry_end - DRYUP_DAYS, 0)) if v.dry_end > 0 else None,
               "dry_to": day(v.dry_end - 1) if v.dry_end > 0 else None}
        if v.ok:
            status, vr, _ = entry_status(v, cl, vo)
            row.update({"status": status, "breakout_vol": vr,
                        "to_pivot_pct": (v.pivot / cl.iloc[-1] - 1) * 100})
            row.update(plan_trade(v, cl.iloc[-1], a, cl.iloc[-1] > v.pivot, equity))
            if not row["risk_ok"] and not status.startswith("伸びすぎ"):
                row["status"] = "リスク過大(見送り)"
        rows.append(row)

    out = tt[tt["tt_pass"]].join(pd.DataFrame(rows).set_index("code") if rows else pd.DataFrame())
    out = out.join(stocks[["name", "sector"]])
    if SUMMARY.exists():   # 業種の資金動向(ダッシュボードの判定)を添える
        sm = pd.read_csv(SUMMARY, encoding="utf-8-sig").set_index("sector")
        cols = [c for c in ["cw_label", "cw_rel_20d"] if c in sm.columns]
        out = out.join(sm[cols].rename(columns={"cw_label": "sector_label", "cw_rel_20d": "sector_rel_20d"}),
                       on="sector")
    order = {"ブレイク": 0, "ピボット接近": 1}
    out["_o"] = out.get("status", pd.Series(index=out.index, dtype=object)).map(
        lambda s: 0 if isinstance(s, str) and s.startswith("ブレイク") and "不足" not in s
        else order.get(s, 2) if isinstance(s, str) else 3)
    out = out.sort_values(["_o", "rs"], ascending=[True, False]).drop(columns="_o")
    out.index.name = "code"
    return as_of, out, tt.join(stocks[["name", "sector"]])


def results_frame(as_of, out: pd.DataFrame, tt_all: pd.DataFrame) -> pd.DataFrame:
    """BigQuery の minervini_screen に入れる行。テンプレートを MIN_SAVE_TT 条件以上満たした銘柄。

    テンプレート通過銘柄には VCP の判定とトレード計画が付く(株数は資金額しだいなので入れない)。
    """
    import bq_io

    keep = tt_all[tt_all["tt_score"] >= MIN_SAVE_TT].copy()
    extra = out.drop(columns=[c for c in out.columns if c in keep.columns])
    df = keep.join(extra).reset_index().rename(columns={"index": "code"})
    df["as_of"] = pd.Timestamp(as_of).date()
    df["loaded_at"] = pd.Timestamp.now(tz="UTC")
    names = [n for n, _ in bq_io.MINERVINI_SCHEMA]
    for n, t in bq_io.MINERVINI_SCHEMA:
        if n not in df.columns:
            df[n] = None
        if t == "DATE" and n != "as_of":
            d = pd.to_datetime(df[n], errors="coerce").dt.date
            df[n] = d.astype(object).where(pd.notna(d), None)
        elif t == "BOOL":
            df[n] = df[n].astype(object).where(df[n].notna(), None).map(lambda v: None if v is None else bool(v))
        elif t == "INT64":
            df[n] = pd.to_numeric(df[n], errors="coerce").round().astype("Int64")
        elif t == "FLOAT64":
            df[n] = pd.to_numeric(df[n], errors="coerce").astype("float64")
        elif t == "STRING":
            df[n] = df[n].astype(object).where(df[n].notna(), None)
    return df[names]


def save_results(df: pd.DataFrame) -> None:
    """(as_of, code) をキーに minervini_screen へ MERGE する。同じ日を再実行すれば上書きされる。

    その日の行はいったん全部入れ直す(前回はテンプレート通過だったが今回は外れた銘柄の行を残さないため)。
    """
    import bq_io
    from google.cloud import bigquery

    client = bq_io.get_client()
    bq_io.ensure_tables(client)
    days = sorted({d.isoformat() for d in df["as_of"]})
    bq_io._run(client, f"DELETE FROM {bq_io.fqn('minervini_screen')} WHERE as_of IN UNNEST(@days)",
               [bigquery.ArrayQueryParameter("days", "DATE", days)])
    bq_io.merge_table(client, df, "minervini_screen", keys=["as_of", "code"], verify=True)


def _print(as_of, out: pd.DataFrame, n_universe: int) -> None:
    print(f"基準日 {as_of.date()}  トレンドテンプレート通過 {len(out)} / {n_universe} 銘柄、"
          f"うちVCP {int(out.get('vcp', pd.Series(dtype=bool)).fillna(False).sum())} 銘柄")
    if "vcp_note" in out.columns:   # VCP にならなかった理由の内訳(条件の調整用)
        notes = out["vcp_note"].fillna("").str.split(" / ").explode()
        counts = notes[notes != ""].value_counts()
        if len(counts):
            print("VCPにならなかった理由: " + "、".join(f"{k} {v}" for k, v in counts.items()))
    vcp = out[out.get("vcp", False) == True]  # noqa: E712
    if vcp.empty:
        return
    cols = ["name", "sector", "rs", "status", "close", "pivot", "to_pivot_pct", "depths", "dryup",
            "atr_pct", "stop", "risk_pct", "risk_atr", "stop_basis", "target_2r"]
    cols += ["shares"] if "shares" in vcp.columns else []
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 24,
                           "display.float_format", lambda x: f"{x:,.1f}"):
        print(vcp[cols].to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--as-of", help="この日(YYYY-MM-DD)までのデータで判定する")
    ap.add_argument("--equity", type=float, help="資金額(円)。指定すると株数を出す")
    ap.add_argument("--bq", action="store_true", help="BigQuery の prices から直接読む")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--save", action="store_true", help="結果を BigQuery の minervini_screen に保存する(要認証)")
    args = ap.parse_args()

    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    if args.bq or args.save:
        import bq_io

        if not bq_io.credentials_available():
            print("Google Cloud の認証情報がありません(GOOGLE_APPLICATION_CREDENTIALS)")
            return 1
    if args.bq:
        end = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
        since = (end - pd.Timedelta(days=BQ_DAYS)).date()
        prices = bq_io.read_prices(bq_io.get_client(), since, uni.loc[uni["type"] == "stock", "code"].tolist())
    elif PRICES.exists():
        prices = pd.read_parquet(PRICES)
    else:
        print(f"{PRICES} がありません。--bq を付けるか、python src/bq_sync.py --pull で作ってください")
        return 1
    prices["date"] = pd.to_datetime(prices["date"])
    prices["code"] = prices["code"].astype(str)
    if args.as_of:
        prices = prices[prices["date"] <= pd.Timestamp(args.as_of)]
    as_of, out, tt_all = screen(prices, uni, args.equity)
    out.drop(columns=["zigzag"], errors="ignore").round(4).to_csv(args.out, encoding="utf-8-sig")
    _print(as_of, out, int((uni["type"] == "stock").sum()))
    print(f"→ {args.out}")
    if args.save:
        df = results_frame(as_of, out, tt_all)
        save_results(df)
        print(f"→ BigQuery minervini_screen ({as_of.date()}, {len(df)}行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
