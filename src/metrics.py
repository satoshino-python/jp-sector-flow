"""業種別の資金動向指標を計算する。

指標の考え方:
  価格側 ... 業種ETF(TOPIX-17)のTOPIX対比リターン(相対強度)
  売買側 ... 構成銘柄の売買代金を業種別に合算したもの
            (業種ETF自体の売買代金は薄くノイズが大きいため使わない)
  広がり ... 業種内の上昇銘柄比率、25日線上の銘柄比率
  銘柄指数 ... 構成銘柄の調整後終値から作る業種指数
            時価総額加重(cw_*) ... 【主軸】JPX公表のTOPIXウエイト(浮動株調整済み)を値動きで動かして使う。
                                   銘柄リスト(universe.csv)にウエイト(weight列)が必要
            均等加重(ew_*) ... 全銘柄では実質的に中小型株の動きを表す(補助)

判定(cw_label)・スコア・順位は時価総額加重の相対強度(cw_rel_20d)を使う。
業種ETFをTOPIX連動ETF(1306)と比べた値(rel_*、label)は、食い違いを見る参考値として残す。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

BENCH_CODE = "1306"
MIN_COVERAGE = 0.8   # その日にデータがある銘柄が全体の何割以上なら有効日とするか
DEV_WARN_PT = 0.5    # 終値と基準価額の20日リターン差がこれ(%pt)を超えたら「乖離大」
SIGN_EPS = 0.3       # 相対強度の符号比較で、絶対値がこれ未満なら「ゼロ近傍」として無視(%pt)
MAX_DAILY_MOVE = 0.45  # 日次リターンの絶対値がこれを超えたら誤データとみなして除外(値幅制限を超える動き)
MIN_MEMBER_RATIO = 0.5  # 業種指数: その日にリターンがある銘柄が業種の何割以上なら計算するか


@dataclass
class Result:
    as_of: pd.Timestamp
    summary: pd.DataFrame       # 業種ごとの最新指標
    share5: pd.DataFrame        # 売買代金シェア(5日平均)の時系列 [date x sector]
    share_z: pd.DataFrame       # 同zスコア
    rs_line: pd.DataFrame       # 業種ETF/TOPIX を直近で100に揃えたもの
    n_missing: int              # 取得できなかった銘柄数
    nav_as_of: pd.Timestamp | None = None   # 基準価額の最新の突き合わせ日(取得できなければNone)
    rs_line_ew: pd.DataFrame | None = None  # 均等加重の業種指数/全銘柄均等加重 を直近で100に揃えたもの
    outliers: list | None = None            # 誤データとして除外した (日付, 銘柄, 日次リターン)
    rs_line_cw: pd.DataFrame | None = None  # 時価総額加重の業種指数/全銘柄時価総額加重 を直近で100に揃えたもの


def _stock_returns(adj: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """銘柄の日次リターン。値幅制限を超えるような動きは誤データとして除外する。"""
    r = adj.pct_change(fill_method=None)
    bad = r.abs() > MAX_DAILY_MOVE
    rows, cols = np.where(bad.values)
    outliers = [(r.index[i].date(), r.columns[j], float(r.iat[i, j])) for i, j in zip(rows, cols)]
    return r.mask(bad), outliers


def _ew_index(r: pd.DataFrame, min_ratio: float = MIN_MEMBER_RATIO) -> pd.Series:
    """均等加重指数(初日=1)。日々の構成銘柄リターンの単純平均を積み上げる。

    その日にリターンがある銘柄が少なすぎる日は、リターン0(据え置き)として扱う。
    """
    n_valid = r.notna().sum(axis=1)
    daily = r.mean(axis=1).where(n_valid >= max(1, int(np.ceil(r.shape[1] * min_ratio))), 0.0)
    return (1 + daily.fillna(0.0)).cumprod()


def _cw_index(r: pd.DataFrame, adj: pd.DataFrame, w: pd.Series, base_date) -> pd.Series:
    """時価総額加重指数(初日=1)。

    w は基準日 base_date 時点のウエイト(JPX公表値)。その他の日のウエイトは、
    株数が変わらないものとして値動きで動かす: w_i(t) = w_i × P_i(t) / P_i(基準日)。
    日々のリターン = 前日のウエイトで加重したリターンの平均(その日にリターンがある銘柄だけ)。
    """
    w = w.reindex(r.columns).fillna(0.0)
    p = adj[r.columns].ffill()
    if base_date is None:
        base = p.iloc[-1]
    else:   # 基準日以前の最終営業日。データが基準日より後からしかなければ初日
        pos = p.index.searchsorted(pd.Timestamp(base_date), side="right") - 1
        base = p.iloc[max(pos, 0)]
    rel = p.div(base.where(base > 0))
    wt = (rel.mul(w, axis=1)).shift(1)                      # 前日終値時点のウエイト
    valid = r.notna() & wt.notna() & (wt > 0)
    num = (wt * r).where(valid).sum(axis=1)
    den = wt.where(valid).sum(axis=1)
    daily = (num / den).where(den > 0, 0.0)
    return (1 + daily.fillna(0.0)).cumprod()


def _sign_conflict(a: float, b: float) -> bool:
    return (pd.notna(a) and pd.notna(b) and abs(a) > SIGN_EPS and abs(b) > SIGN_EPS
            and np.sign(a) != np.sign(b))


def _nav_stats(px: pd.Series, bench: pd.Series, nv: pd.Series) -> dict:
    """ETF終値と基準価額(分配金再投資)を、両方そろった日だけで突き合わせる。

    水準(1口/10口など単位)には依存せず、リターン同士で比較する。
    """
    d = pd.concat({"px": px, "nv": nv, "bm": bench}, axis=1).dropna()
    if len(d) < 62:
        return {}

    def r20(s: pd.Series) -> float:
        return float(s.iloc[-1] / s.iloc[-21] - 1) * 100

    daily_diff = (d["px"].pct_change() - d["nv"].pct_change()).dropna().tail(60) * 100
    return {
        "nav_date": d.index[-1],
        "dev_20d": r20(d["px"]) - r20(d["nv"]),                 # 終値20日騰落 − 基準価額20日騰落
        "nav_noise": float(daily_diff.std()),                   # 日次の差の標準偏差(%)
        "rel_20d_nav": r20(d["nv"]) - r20(d["bm"]),             # 基準価額ベースの相対強度
        "rel_20d_same": r20(d["px"]) - r20(d["bm"]),            # 同じ日付区間での終値ベースの相対強度
    }


def _nav_flag(row: dict) -> str:
    notes = []
    dev = row.get("dev_20d")
    if dev is not None and pd.notna(dev) and abs(dev) > DEV_WARN_PT:
        notes.append("乖離大")
    a, b = row.get("rel_20d_same"), row.get("rel_20d_nav")
    if (a is not None and b is not None and pd.notna(a) and pd.notna(b)
            and abs(a) > SIGN_EPS and abs(b) > SIGN_EPS and np.sign(a) != np.sign(b)):
        notes.append("基準価額と判定相違")
    return " / ".join(notes)


def _pivot(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="code", values=col).sort_index()


def _label(rel20: float, z: float) -> str:
    if pd.isna(rel20) or pd.isna(z):
        return "データ不足"
    if rel20 > 0 and z > 0:
        return "資金流入"
    if rel20 > 0:
        return "買い細り"
    if z > 0:
        return "売り圧力/転換前"
    return "低調"


def compute(prices: pd.DataFrame, uni: pd.DataFrame, nav: pd.DataFrame | None = None) -> Result:
    close = _pivot(prices, "close")
    turn = _pivot(prices, "turnover")
    # 調整後終値がない古いデータでも動くように、なければ終値で代用する
    adj = _pivot(prices, "adj_close") if "adj_close" in prices.columns else close.copy()
    adj = adj.reindex(columns=close.columns).fillna(close)
    nav_pv = (nav.pivot(index="date", columns="code", values="nav").sort_index()
              if nav is not None and not nav.empty else None)

    etf = uni[uni["type"] == "etf"].set_index("sector")["code"]
    stocks = uni[uni["type"] == "stock"]
    stock_codes = [c for c in stocks["code"] if c in close.columns]
    n_missing = int((uni["type"] == "stock").sum() - len(stock_codes))

    # 銘柄カバレッジが低い日(取得途中の日など)を除外する
    # 分母は「その日までに取引が始まっている銘柄」(上場から日が浅い銘柄で過去の日が除外されないように)
    started = close[stock_codes].notna().cummax()
    coverage = close[stock_codes].notna().sum(axis=1) / started.sum(axis=1).clip(lower=1)
    valid_dates = coverage[coverage >= MIN_COVERAGE].index
    close, turn, adj = close.loc[valid_dates], turn.loc[valid_dates], adj.loc[valid_dates]
    if len(close) < 70:
        raise ValueError(f"有効な営業日が少なすぎます({len(close)}日)。60日指標に最低70日分が必要です。")

    sec_of = stocks.set_index("code")["sector"].to_dict()
    sectors = list(etf.index)

    # --- 売買代金シェア ---
    t = turn[stock_codes].fillna(0.0)
    sec_turn = t.T.groupby(t.columns.map(sec_of)).sum().T[sectors]
    share = sec_turn.div(sec_turn.sum(axis=1), axis=0)
    share5 = share.rolling(5).mean()
    roll = share5.rolling(60)
    share_z = (share5 - roll.mean()) / roll.std()
    turn_ratio_1d = sec_turn / sec_turn.rolling(20).mean().shift(1)

    # --- 価格(ETF・TOPIX対比) ---
    bench = close[BENCH_CODE]

    def ret(s: pd.Series, n: int) -> float:
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1) * 100

    # --- 価格(個別銘柄の均等加重指数・全銘柄均等加重対比) ---
    stock_ret, outliers = _stock_returns(adj[stock_codes])
    ew_all = _ew_index(stock_ret)
    ew_sec = pd.DataFrame({
        s: _ew_index(stock_ret[[c for c in stock_codes if sec_of[c] == s]])
        for s in sectors if any(sec_of[c] == s for c in stock_codes)
    })

    # --- 価格(時価総額加重。JPXのウエイトがあるときだけ) ---
    if "weight" not in stocks.columns or not pd.to_numeric(stocks["weight"], errors="coerce").notna().any():
        raise ValueError("universe.csv に TOPIX ウエイト(weight列)がありません。"
                         "python src/universe.py で銘柄リストを作り直してください。")
    wts = pd.to_numeric(stocks.set_index("code")["weight"], errors="coerce").reindex(stock_codes)
    wdate = (pd.to_datetime(stocks["weight_date"], errors="coerce").max()
             if "weight_date" in stocks.columns else None)
    wdate = None if wdate is None or pd.isna(wdate) else wdate
    cw_all = _cw_index(stock_ret, adj, wts, wdate)
    cw_sec = pd.DataFrame({
        s: _cw_index(stock_ret[m], adj, wts[m], wdate)
        for s in sectors
        if (m := [c for c in stock_codes if sec_of[c] == s]) and wts[m].fillna(0).sum() > 0
    })

    # --- 銘柄単位の指標 ---
    sc = close[stock_codes]
    daily_ret = sc.pct_change()
    up_turn = (t.where(daily_ret > 0, 0.0)).tail(5).sum()
    down_turn = (t.where(daily_ret < 0, 0.0)).tail(5).sum()
    above_ma25 = sc.iloc[-1] > sc.rolling(25).mean().iloc[-1]
    pos20 = (sc.iloc[-1] / sc.iloc[-21] - 1) > 0
    ret20_stock = (sc.iloc[-1] / sc.iloc[-21] - 1) * 100

    rows = []
    for s in sectors:
        ec = etf[s]
        members = [c for c in stock_codes if sec_of[c] == s]
        if ec not in close.columns or not members:
            continue
        e = close[ec].dropna()
        u, d = up_turn[members].sum(), down_turn[members].sum()
        row = {
            "sector": s,
            "etf_code": ec,
            "n_stocks": len(members),
            "ret_5d": ret(e, 5), "ret_20d": ret(e, 20), "ret_60d": ret(e, 60),
            "rel_5d": ret(e, 5) - ret(bench, 5),
            "rel_20d": ret(e, 20) - ret(bench, 20),
            "rel_60d": ret(e, 60) - ret(bench, 60),
            "med_ret_20d": float(ret20_stock[members].median()),
            "turn_ratio_1d": float(turn_ratio_1d[s].iloc[-1]),
            "share_pct": float(share5[s].iloc[-1] * 100),
            "share_z": float(share_z[s].iloc[-1]),
            "up_turn_ratio": float(u / (u + d)) if (u + d) > 0 else np.nan,
            "breadth_ma25": float(above_ma25[members].mean() * 100),
            "breadth_pos20": float(pos20[members].mean() * 100),
        }
        row["label"] = _label(row["rel_20d"], row["share_z"])
        if s in ew_sec.columns:
            ix = ew_sec[s]
            for n in (5, 20, 60):
                row[f"ew_ret_{n}d"] = ret(ix, n)
                row[f"ew_rel_{n}d"] = ret(ix, n) - ret(ew_all, n)
            row["ew_label"] = _label(row["ew_rel_20d"], row["share_z"])
        if s in cw_sec.columns:
            ix = cw_sec[s]
            for n in (5, 20, 60):
                row[f"cw_ret_{n}d"] = ret(ix, n)
                row[f"cw_rel_{n}d"] = ret(ix, n) - ret(cw_all, n)
            row["cw_label"] = _label(row["cw_rel_20d"], row["share_z"])
            # 主軸(時価総額加重)と、均等加重・ETFの向きが逆の業種
            row["ew_flag"] = ("均等加重と方向相違"
                              if _sign_conflict(row["cw_rel_20d"], row.get("ew_rel_20d", np.nan)) else "")
            row["etf_flag"] = "ETFと方向相違" if _sign_conflict(row["cw_rel_20d"], row["rel_20d"]) else ""
        stats = (_nav_stats(close[ec], bench, nav_pv[ec])
                 if nav_pv is not None and ec in nav_pv.columns else {})
        row.update({k: stats.get(k, np.nan) for k in
                    ["nav_date", "dev_20d", "nav_noise", "rel_20d_nav", "rel_20d_same"]})
        row["nav_flag"] = _nav_flag(row)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("sector")
    # 総合スコア: 4指標のパーセンタイル順位の平均(0-100)。順位付け用の簡易指標。
    # 価格の指標は時価総額加重の相対強度(ETFは使わない)
    rank_cols = ["cw_rel_20d", "share_z", "up_turn_ratio", "breadth_ma25"]
    summary["score"] = summary[rank_cols].rank(pct=True).mean(axis=1) * 100
    summary = summary.sort_values("score", ascending=False)

    # --- 相対強度ライン(直近120営業日、期首=100) ---
    rs = pd.DataFrame({s: close[etf[s]] / bench for s in summary.index if etf[s] in close.columns})
    rs = rs.tail(120)
    rs_line = rs / rs.iloc[0] * 100

    rs_ew = ew_sec[[s for s in summary.index if s in ew_sec.columns]].div(ew_all, axis=0).tail(120)
    rs_line_ew = rs_ew / rs_ew.iloc[0] * 100

    rs_cw = cw_sec[[s for s in summary.index if s in cw_sec.columns]].div(cw_all, axis=0).tail(120)
    rs_line_cw = rs_cw / rs_cw.iloc[0] * 100

    nav_dates = pd.to_datetime(summary["nav_date"], errors="coerce").dropna()
    return Result(
        nav_as_of=nav_dates.max() if len(nav_dates) else None,
        as_of=close.index[-1],
        summary=summary,
        share5=share5.tail(120),
        share_z=share_z.tail(120),
        rs_line=rs_line,
        n_missing=n_missing,
        rs_line_ew=rs_line_ew,
        outliers=outliers,
        rs_line_cw=rs_line_cw,
    )
