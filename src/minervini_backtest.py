"""ミネルヴィニのスクリーナー(src/minervini.py)のルールで、過去の売買を再現して成績を確かめる。

使い方(BigQuery の株価を使うので、認証のある GitHub Actions の minervini-backtest で動かす):
    python src/minervini_backtest.py --start 2021-01-01 --end 2025-12-31 --save

売買のルール(VARIANTS の3通りを比べる):
  シグナル ... その日にトレンドテンプレートを満たし、
                vcp_pivot       ... VCP と判定され、終値が出来高を伴ってピボットを初めて上抜けた(daily の「ブレイク」)日
                vcp_pivot_anyvol ... 上の出来高の条件を外したもの(daily の「ブレイク(出来高不足)」の当日も含む)
                high52_8pct     ... 終値が前日までの52週高値を初めて上抜けた日
                vcp_high52_8pct ... 上の52週高値抜けのうち、その日に VCP と判定されたもの
              判定は daily の screen と同じ関数を使う
  買い     ... シグナルの翌営業日の始値。次の場合は見送る:
                翌日の始値が損切り以下(下に窓を開けた)/ ピボット(52週高値)から MAX_CHASE 超上(伸びすぎ)/
                損切り幅が MAX_RISK_PCT 超(リスク過大)/ その銘柄を保有中
  損切り   ... vcp_pivot 系: シグナル日に plan_trade で決めた価格(最後の収縮の安値 − 0.5ATR と 終値 − 1.5ATR の遠い方)
              52週高値の2つ: 買値の8%下
              安値が損切りに触れたら損切り価格で売る(始値が損切りより下なら始値で売る)
  建値へ   ... 高値が 買値 + BREAKEVEN_R × R に達したら、翌日から損切りを買値に上げる
  利益確定 ... 終値が50日線を割ったら、翌営業日の始値で売る
  R        ... 買値 − 当初の損切り。成績は R の倍数(損益 ÷ R)と騰落率で見る

注意:
  - 対象は現在の universe.csv(現在の TOPIX500)。過去に上場廃止・指数除外された銘柄は入っていないため、
    成績は良く出やすい(生存者バイアス)
  - 手数料・スリッページは入れていない
  - 1トレードずつの成績で、資金の制約(同時に持てる銘柄数)は考えていない
  - シグナルの期間外(--end の後)も、手仕舞うまでは株価を追う
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minervini as mv  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    """経過時間付きで出す(どの処理に時間がかかったかをログで追えるように)。"""
    print(f"[{time.time() - T0:6.1f}秒] {msg}", flush=True)


# 比べる売買ルール: (variant, シグナルのルール, 損切り)。損切りは ("atr", 損切り幅の下限) か ("pct", 買値からの割合)
# (以前は ATR の下限 1.0/1.5/2.0 を比べたが、損切りは全件で収縮の安値の方が遠く、結果が同じだったので 1.5 だけ残す)
VARIANTS = [
    ("vcp_pivot", "vcp_pivot", ("atr", mv.STOP_ATR_MIN)),
    ("vcp_pivot_anyvol", "vcp_pivot_anyvol", ("atr", mv.STOP_ATR_MIN)),
    ("high52_8pct", "high52", ("pct", 0.08)),
    ("vcp_high52_8pct", "vcp_high52", ("pct", 0.08)),
]
WARMUP_DAYS = 420                 # 指標の計算に使う、開始日より前の暦日数(200日線・52週・12か月RS)
SLICE = 260                       # detect_vcp に渡す直近の営業日数(ベース130日 + 出来高の平均 + ATR に足りる)


def load_prices(start: pd.Timestamp, uni: pd.DataFrame) -> pd.DataFrame:
    import bq_io

    if not bq_io.credentials_available():
        raise SystemExit("Google Cloud の認証情報がありません(GOOGLE_APPLICATION_CREDENTIALS)")
    since = (start - pd.Timedelta(days=WARMUP_DAYS)).date()
    codes = uni.loc[uni["type"] == "stock", "code"].tolist()
    log("株価の読み込み開始")
    df = bq_io.read_prices(bq_io.get_client(), since, codes)
    log(f"株価を読み込み: {len(df):,}行 {df['code'].nunique()}銘柄 {df['date'].min().date()}〜{df['date'].max().date()}")
    return df


class Stock:
    """1銘柄の、取引のあった日だけの系列(screen が使う cl.dropna() と同じ並び)。"""

    def __init__(self, p: dict[str, pd.DataFrame], code: str):
        cl = p["close"][code].dropna()
        idx = cl.index
        self.code = code
        self.dates = idx
        self.c = cl
        self.h = p["high"][code].reindex(idx).fillna(cl)
        self.l = p["low"][code].reindex(idx).fillna(cl)
        self.o = p["open"][code].reindex(idx).fillna(cl)
        self.v = p["volume"][code].reindex(idx)
        self.atr = mv.atr(self.h, self.l, self.c)
        self.ma50 = cl.rolling(50).mean()
        self.hi52p = self.h.rolling(mv.WEEKS52).max().shift(1)     # 前日までの52週高値
        # 売買の再現は numpy で速く回す
        self.na = {k: getattr(self, k).to_numpy(float) for k in ["o", "h", "l", "c", "ma50", "hi52p"]}

    def pos(self, d: pd.Timestamp) -> int:
        """日付 d の位置(その日に取引がなければ -1)。"""
        i = self.dates.searchsorted(d)
        return i if i < len(self.dates) and self.dates[i] == d else -1


def find_signals(p, panel, stocks: dict[str, Stock], start, end) -> pd.DataFrame:
    """期間中のシグナルを、ルールごとに集める(rule 列)。

      vcp_pivot  ... VCP と判定され、終値が出来高を伴ってピボットを初めて上抜けた(daily の「ブレイク」)
      vcp_pivot_anyvol ... 出来高を問わず、VCP と判定され、終値がピボットを初めて上抜けた(breakout_vol で出来高を見られる)
      high52     ... 終値が前日までの52週高値(直近252営業日の高値)を初めて上抜けた
      vcp_high52 ... high52 のうち、その日に VCP と判定されたもの
    どのルールも、その日にトレンドテンプレートを満たしていることが前提。
    """
    tt = panel["tt_pass"]
    days = tt.index[(tt.index >= start) & (tt.index <= end)]
    rows, n_checked, t0 = [], 0, time.time()
    for d in days:
        for code in tt.columns[tt.loc[d].to_numpy()]:
            s = stocks[code]
            i = s.pos(d)
            if i < mv.BASE_LOOKBACK or not np.isfinite(s.na["c"][i]):
                continue
            c = s.na["c"]
            # どのルールも「当日の終値が前日より上」でないと起きない(前日の終値は前日までの高値・ピボット以下)ので判定を省く
            if c[i] <= c[i - 1]:
                continue
            hp = s.na["hi52p"]
            is52 = np.isfinite(hp[i]) and c[i] > hp[i] and np.isfinite(hp[i - 1]) and c[i - 1] <= hp[i - 1]
            a, b = max(0, i + 1 - SLICE), i + 1
            hi, lo, cl, vo = s.h.iloc[a:b], s.l.iloc[a:b], s.c.iloc[a:b], s.v.iloc[a:b]
            atr_v = s.atr.iloc[i]
            n_checked += 1
            v = mv.detect_vcp(hi, lo, cl, vo, atr_v / cl.iloc[-1])
            status, vr, bdays = None, np.nan, -1
            if v.ok:
                status, vr, bdays = mv.entry_status(v, cl, vo)
            vol_avg = s.v.iloc[max(0, i - 50):i].mean()
            base = {"code": code, "signal_date": d, "i": i, "vcp_ok": v.ok,
                    "atr": atr_v, "atr_pct": atr_v / cl.iloc[-1] * 100, "close": cl.iloc[-1],
                    "rs": float(panel["rs"].at[d, code]), "vcp_obj": v}
            if v.ok:   # 検証用ページでチャートに重ねるための位置(日付)。daily の minervini_screen と同じ形
                day = lambda k: s.dates[a + k].strftime("%Y-%m-%d")   # noqa: E731  slice 内の位置 → 日付
                base.update({"final_low": v.final_low, "base_start": day(v.start),
                             "zigzag": json.dumps([[t, day(k), round(float(val), 2)] for t, k, val in v.points]),
                             "dry_from": day(max(v.dry_end - mv.DRYUP_DAYS, 0)), "dry_to": day(v.dry_end - 1),
                             "depths": " → ".join(f"{x * 100:.0f}%" for x in v.depths),
                             "n_contractions": len(v.depths), "base_days": v.base_days, "dryup": v.dryup})
            if v.ok and status == "ブレイク":
                rows.append({**base, "rule": "vcp_pivot", "level": v.pivot, "breakout_vol": vr})
            if v.ok and bdays == 0 and status in ("ブレイク", "ブレイク(出来高不足)"):
                rows.append({**base, "rule": "vcp_pivot_anyvol", "level": v.pivot, "breakout_vol": vr})
            if is52:
                vr52 = s.v.iloc[i] / vol_avg if vol_avg and vol_avg > 0 else np.nan
                rows.append({**base, "rule": "high52", "level": hp[i], "breakout_vol": vr52})
                if v.ok:
                    rows.append({**base, "rule": "vcp_high52", "level": hp[i], "breakout_vol": vr52})
    df = pd.DataFrame(rows)
    counts = df["rule"].value_counts().to_dict() if len(df) else {}
    log(f"シグナル探索: {len(days)}営業日、VCP判定 {n_checked:,}回、シグナル {counts} ({time.time() - t0:.0f}秒)")
    return df


def simulate(sig: pd.DataFrame, stocks: dict[str, Stock], variant: str, rule: str, stop: tuple) -> pd.DataFrame:
    """rule のシグナルを順に売買する。見送ったシグナルも理由付きで返す。

    stop ... ("atr", k): plan_trade の損切り(最後の収縮の安値 − 0.5ATR と 終値 − k×ATR の遠い方。シグナル日に決める)
             ("pct", x): 買値の x 下(買った日に決める)
    """
    out = []
    busy_until: dict[str, int] = {}          # 銘柄ごとに、保有中のトレードの手仕舞い位置
    for r in sig[sig["rule"] == rule].sort_values("signal_date").itertuples():
        s = stocks[r.code]
        a = s.na
        i = r.i
        if stop[0] == "atr":
            plan = mv.plan_trade(r.vcp_obj, r.close, r.atr, True, None, stop_atr_min=stop[1])
            stop0, basis = plan["stop"], plan["stop_basis"]
        else:
            stop0, basis = None, f"買値の{stop[1] * 100:.0f}%下"
        rec = {"variant": variant, "rule": rule, "code": r.code, "signal_date": r.signal_date.date(), "pivot": r.level,
               "vcp_ok": bool(r.vcp_ok), "stop": stop0, "stop_basis": basis, "atr_pct": r.atr_pct, "rs": r.rs,
               "breakout_vol": r.breakout_vol}
        for k in ["final_low", "base_start", "zigzag", "dry_from", "dry_to", "depths", "n_contractions", "base_days", "dryup"]:
            val = getattr(r, k, None)
            rec[k] = None if val is None or (isinstance(val, float) and np.isnan(val)) else val
        if r.code in busy_until and busy_until[r.code] >= i:
            out.append({**rec, "result": "見送り: 保有中"})
            continue
        if i + 1 >= len(a["o"]):
            out.append({**rec, "result": "見送り: 翌日のデータなし"})
            continue
        e = i + 1
        entry = a["o"][e]
        if stop0 is None:
            stop0 = entry * (1 - stop[1])
            rec["stop"] = stop0
        if not np.isfinite(entry) or entry <= stop0:
            out.append({**rec, "entry": entry, "result": "見送り: 翌日始値が損切り以下"})
            continue
        if entry > r.level * (1 + mv.MAX_CHASE):
            out.append({**rec, "entry": entry, "result": "見送り: 伸びすぎ"})
            continue
        risk = entry - stop0
        if risk / entry > mv.MAX_RISK_PCT:
            out.append({**rec, "entry": entry, "result": "見送り: リスク過大"})
            continue

        stop_px, be_done, mfe, be_i = stop0, False, entry, None
        exit_i, exit_px, reason = None, None, None
        j = e
        n = len(a["o"])
        while j < n:
            o, h, lo, c = a["o"][j], a["h"][j], a["l"][j], a["c"][j]
            if j > e and o <= stop_px:                       # 窓を開けて損切りより下で始まった
                exit_i, exit_px, reason = j, o, "建値ストップ" if be_done else "損切り"
                break
            if lo <= stop_px:
                exit_i, exit_px, reason = j, stop_px, "建値ストップ" if be_done else "損切り"
                break
            mfe = max(mfe, h)
            if not be_done and h >= entry + mv.BREAKEVEN_R * risk:   # 翌日から損切りを買値へ
                stop_px, be_done, be_i = max(stop_px, entry), True, j + 1
            if np.isfinite(a["ma50"][j]) and c < a["ma50"][j]:      # 50日線割れ → 翌日の始値で売る
                if j + 1 < n:
                    exit_i, exit_px, reason = j + 1, a["o"][j + 1], "50日線割れ"
                else:
                    exit_i, exit_px, reason = j, c, "保有中(データ終わり)"
                break
            j += 1
        if exit_i is None:
            exit_i, exit_px, reason = n - 1, a["c"][n - 1], "保有中(データ終わり)"
        busy_until[r.code] = exit_i
        out.append({**rec, "result": "売買", "entry_date": s.dates[e].date(), "entry": entry,
                    "risk_pct": risk / entry * 100, "exit_date": s.dates[exit_i].date(), "exit": exit_px,
                    "exit_reason": reason, "r_mult": (exit_px - entry) / risk, "ret_pct": (exit_px / entry - 1) * 100,
                    "hold_days": exit_i - e + 1, "mfe_r": (mfe - entry) / risk,
                    # 損切りを買値に上げた日(この日から有効)
                    "breakeven_date": s.dates[be_i].date() if be_i is not None and be_i < len(s.dates) else None})
    return pd.DataFrame(out)


def summarize(trades: pd.DataFrame) -> None:
    t = trades[trades["result"] == "売買"].copy()
    t["year"] = pd.to_datetime(t["signal_date"]).dt.year

    def stats(g: pd.DataFrame) -> pd.Series:
        win, loss = g[g["r_mult"] > 0], g[g["r_mult"] <= 0]
        gross_loss = -loss["r_mult"].sum()
        return pd.Series({
            "件数": len(g), "勝率%": len(win) / len(g) * 100 if len(g) else np.nan,
            "平均R": g["r_mult"].mean(), "勝ち平均R": win["r_mult"].mean(), "負け平均R": loss["r_mult"].mean(),
            "PF": win["r_mult"].sum() / gross_loss if gross_loss > 0 else np.nan,
            "ペイオフ": win["r_mult"].mean() / -loss["r_mult"].mean() if len(win) and len(loss) else np.nan,
            "合計R": g["r_mult"].sum(), "平均騰落%": g["ret_pct"].mean(), "平均保有日": g["hold_days"].mean(),
            "損切り幅%": g["risk_pct"].mean(),
        })

    fmt = {"display.width": 220, "display.max_columns": 30, "display.float_format": lambda x: f"{x:,.2f}"}
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        print("\n=== 売買ルールごと(シグナル全期間) ===")
        print(t.groupby("variant").apply(stats, include_groups=False).to_string())
        print("\n=== 年別 ===")
        print(t.groupby(["variant", "year"]).apply(stats, include_groups=False).to_string())
        print("\n=== 手仕舞いの理由 ===")
        print(t.groupby(["variant", "exit_reason"]).size().unstack(fill_value=0).to_string())
        print("\n=== 見送り ===")
        print(trades.groupby(["variant", "result"]).size().unstack(fill_value=0).to_string())


def save(trades: pd.DataFrame, start, end) -> None:
    import bq_io

    df = trades.copy()
    df["period"] = f"{pd.Timestamp(start).date()}〜{pd.Timestamp(end).date()}"
    df["loaded_at"] = pd.Timestamp.now(tz="UTC")
    for n, t in bq_io.BACKTEST_SCHEMA:
        if n not in df.columns:
            df[n] = None
        if t == "DATE":
            d = pd.to_datetime(df[n], errors="coerce").dt.date
            df[n] = d.astype(object).where(pd.notna(d), None)
        elif t == "INT64":
            df[n] = pd.to_numeric(df[n], errors="coerce").round().astype("Int64")
        elif t == "FLOAT64":
            df[n] = pd.to_numeric(df[n], errors="coerce").astype("float64")
        elif t == "BOOL":
            df[n] = df[n].astype(object).where(df[n].notna(), None).map(lambda v: None if v is None else bool(v))
        elif t == "STRING":
            df[n] = df[n].astype(object).where(df[n].notna(), None)
    client = bq_io.get_client()
    bq_io.ensure_tables(client, ["minervini_backtest"])
    bq_io.replace_table(client, df, "minervini_backtest")
    log(f"→ BigQuery minervini_backtest ({len(df)}行、全置換)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--save", action="store_true", help="売買の一覧を BigQuery の minervini_backtest に保存(全置換)")
    args = ap.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    uni = pd.read_csv(mv.UNIVERSE, dtype={"code": str})
    prices = load_prices(start, uni)
    stocks_u = uni[uni["type"] == "stock"].set_index("code")
    have = set(prices["code"].unique())   # 銘柄ごとに set を作り直すと、80万行で約3分かかる
    codes = [c for c in stocks_u.index if c in have]
    p = mv.load_panels(prices, codes)
    panel = mv.trend_template_panel(p)
    log("指標の計算")

    # 確認: 全営業日版のトレンドテンプレートが、daily の trend_template と同じ結果になるか(期間の最終日で)
    last = panel["tt_pass"].index[panel["tt_pass"].index <= end][-1]
    window = {k: v.loc[(v.index > last - pd.Timedelta(days=mv.BQ_DAYS)) & (v.index <= last)] for k, v in p.items()}
    daily = mv.trend_template(window)
    a = set(daily.index[daily["tt_pass"] & daily["close"].notna()])
    b = set(panel["tt_pass"].columns[panel["tt_pass"].loc[last].to_numpy()])
    print(f"確認 {last.date()}: テンプレート通過 daily {len(a)} / 全営業日版 {len(b)} / 食い違い {len(a ^ b)}")

    stocks = {c: Stock(p, c) for c in codes}
    log("銘柄ごとの系列")
    sig = find_signals(p, panel, stocks, start, end)
    if sig.empty:
        print("シグナルがありません")
        return 0
    trades = pd.concat([simulate(sig, stocks, v, rule, stop) for v, rule, stop in VARIANTS], ignore_index=True)
    trades = trades.join(stocks_u[["name", "sector"]], on="code")
    log("売買の再現")
    summarize(trades)
    if args.save:
        save(trades, start, end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
