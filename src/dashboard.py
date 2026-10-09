"""data/prices.parquet から指標を計算し、docs/index.html を生成する。"""
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import DEV_WARN_PT, compute  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICES = ROOT / "data" / "prices.parquet"
UNIVERSE = ROOT / "data" / "universe.csv"
NAV = ROOT / "data" / "nav.parquet"
OUT_HTML = ROOT / "docs" / "index.html"
OUT_CSV = ROOT / "data" / "summary.csv"

LABEL_COLORS = {
    "資金流入": "#d62728",
    "買い細り": "#ff9f1c",
    "売り圧力/転換前": "#1f77b4",
    "低調": "#8c8c8c",
    "データ不足": "#cccccc",
}
FONT = "Hiragino Sans, Noto Sans JP, Meiryo, sans-serif"


def fig_heatmap(summary: pd.DataFrame) -> go.Figure:
    cols = [
        ("rel_5d", "相対強度\n5日", lambda v: f"{v:+.1f}"),
        ("rel_20d", "相対強度\n20日", lambda v: f"{v:+.1f}"),
        ("rel_60d", "相対強度\n60日", lambda v: f"{v:+.1f}"),
        ("ew_rel_20d", "銘柄均等\n相対20日", lambda v: f"{v:+.1f}"),
        ("share_z", "売買代金\nシェアz", lambda v: f"{v:+.1f}"),
        ("turn_ratio_1d", "売買代金\n当日/20日", lambda v: f"{v:.2f}x"),
        ("up_turn_ratio", "上昇日の\n売買比率", lambda v: f"{v * 100:.0f}%"),
        ("breadth_ma25", "25日線上\n銘柄比率", lambda v: f"{v:.0f}%"),
    ]
    ranks = summary[[c for c, _, _ in cols]].rank(pct=True)
    z = ranks.values
    text = [[fmt(summary.iloc[i][c]) if pd.notna(summary.iloc[i][c]) else "-"
             for c, _, fmt in cols] for i in range(len(summary))]
    fig = go.Figure(go.Heatmap(
        z=z, x=[n for _, n, _ in cols], y=list(summary.index),
        text=text, texttemplate="%{text}", textfont={"size": 11},
        colorscale="RdBu_r", zmin=0, zmax=1, showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickfont={"size": 11})
    fig.update_xaxes(side="top", tickfont={"size": 10})
    fig.update_layout(height=60 + 34 * len(summary) + 40, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def fig_scatter(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, color in LABEL_COLORS.items():
        d = summary[summary["label"] == label]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["rel_20d"], y=d["share_z"], mode="markers+text", name=label,
            text=d.index, textposition="top center", textfont={"size": 10},
            marker=dict(size=np.clip(d["share_pct"] * 2.2, 8, 40), color=color, opacity=0.8),
            hovertemplate="%{text}<br>相対強度20日 %{x:+.1f}pt<br>シェアz %{y:+.2f}<extra></extra>",
        ))
    fig.add_vline(x=0, line_width=1, line_color="#999")
    fig.add_hline(y=0, line_width=1, line_color="#999")
    fig.update_xaxes(title="相対強度(ETF 20日リターン − TOPIX, %pt)", zeroline=False)
    fig.update_yaxes(title="売買代金シェアのzスコア(直近5日平均)", zeroline=False)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def fig_compare(summary: pd.DataFrame) -> go.Figure:
    """業種ETFと、構成銘柄の均等加重指数の相対強度(20日)を並べる。"""
    d = summary.sort_values("ew_rel_20d")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d.index, x=d["rel_20d"], orientation="h", name="業種ETF − TOPIX",
                         marker_color="#9aa5b1",
                         hovertemplate="%{y}<br>ETF %{x:+.1f}pt<extra></extra>"))
    fig.add_trace(go.Bar(y=d.index, x=d["ew_rel_20d"], orientation="h",
                         name="銘柄均等加重 − 全銘柄均等加重", marker_color="#d62728",
                         hovertemplate="%{y}<br>銘柄均等 %{x:+.1f}pt<extra></extra>"))
    fig.add_vline(x=0, line_width=1, line_color="#999")
    fig.update_xaxes(title="相対強度 20日(%pt)")
    fig.update_layout(barmode="group", height=60 + 30 * len(d) + 60,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.12))
    return fig


def fig_lines(df: pd.DataFrame, order: list[str], ytitle: str, top_n: int = 5) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(order):
        if s not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[s], mode="lines", name=s,
            visible=True if i < top_n else "legendonly", line={"width": 2},
        ))
    fig.update_yaxes(title=ytitle)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def table_html(summary: pd.DataFrame) -> str:
    rows = []
    for s, r in summary.iterrows():
        color = LABEL_COLORS.get(r["label"], "#999")
        if pd.notna(r.get("dev_20d")):
            flag = r.get("nav_flag") or ""
            mark = f" <span class='warn'>⚠ {html.escape(flag)}</span>" if flag else ""
            nav_cell = f"{r['dev_20d']:+.2f}{mark}"
            noise_cell = f"{r['nav_noise']:.2f}"
        else:
            nav_cell, noise_cell = "-", "-"
        ew_label = r.get("ew_label")
        if isinstance(ew_label, str):
            ew_color = LABEL_COLORS.get(ew_label, "#999")
            ew_flag = r.get("ew_flag") or ""
            ew_mark = f" <span class='warn'>⚠ {html.escape(ew_flag)}</span>" if ew_flag else ""
            ew_cells = (f"<td>{r['ew_rel_20d']:+.1f}</td>"
                        f"<td class='l'><span class='tag' style='background:{ew_color}'>"
                        f"{html.escape(ew_label)}</span>{ew_mark}</td>")
        else:
            ew_cells = "<td>-</td><td>-</td>"
        rows.append(
            f"<tr><td>{html.escape(s)}</td>"
            f"<td><span class='tag' style='background:{color}'>{html.escape(r['label'])}</span></td>"
            f"<td>{r['score']:.0f}</td><td>{r['rel_20d']:+.1f}</td>{ew_cells}"
            f"<td>{r['share_z']:+.2f}</td>"
            f"<td>{r['breadth_ma25']:.0f}%</td><td>{r['n_stocks']}</td>"
            f"<td class='l'>{nav_cell}</td><td>{noise_cell}</td></tr>"
        )
    return (
        "<table><thead><tr><th>業種</th><th>判定</th><th>スコア</th><th>相対20日<br>(ETF)</th>"
        "<th>相対20日<br>(銘柄均等)</th><th>判定<br>(銘柄均等)</th>"
        "<th>シェアz</th><th>25日線上</th><th>銘柄数</th>"
        "<th>終値−基準価額<br>20日(pt)</th><th>日次ずれ<br>σ(%)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def build_html(res, demo: bool) -> str:
    s = res.summary
    order = list(s.index)
    figs = [
        ("業種ヒートマップ", "色は各列の順位(赤=強い、青=弱い)。数値は実測値。上から総合スコア順。",
         fig_heatmap(s)),
        ("相対強度 × 売買代金シェア", "右上=価格も強く売買も増えている(資金流入)。円の大きさは売買代金シェア。",
         fig_scatter(s)),
        ("相対強度の推移(対TOPIX、期首=100)", "凡例クリックで業種を追加表示。初期表示は総合スコア上位5業種。",
         fig_lines(res.rs_line, order, "ETF ÷ TOPIX")),
        ("ETFと個別銘柄の比較(相対強度20日)",
         "灰=業種ETFのTOPIX対比、赤=構成銘柄の均等加重指数の全銘柄均等加重対比。"
         "ETFは時価総額加重なので、差が大きい業種は一部の大型株に動きが偏っています。",
         fig_compare(s)),
        ("相対強度の推移(銘柄均等加重、期首=100)",
         "構成銘柄の調整後終値から作った均等加重指数 ÷ 全銘柄の均等加重指数。",
         fig_lines(res.rs_line_ew, order, "業種 ÷ 全銘柄")),
        ("売買代金シェアの推移(5日平均)", "構成銘柄の売買代金を業種別に合算したシェア(%)。",
         fig_lines(res.share5 * 100, order, "シェア(%)")),
    ]
    parts = []
    for i, (title, note, fig) in enumerate(figs):
        fig.update_layout(template="plotly_white", font={"family": FONT, "size": 12},
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        div = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                          config={"displayModeBar": False, "responsive": True})
        parts.append(f"<section><h2>{title}</h2><p class='note'>{note}</p>{div}</section>")

    demo_banner = ("<div class='demo'>デモデータ(合成)です。実データではありません。</div>"
                   if demo else "")
    missing = (f"<p class='note'>取得できなかった銘柄: {res.n_missing}件(集計から除外)</p>"
               if res.n_missing else "")
    if res.outliers:
        items = ", ".join(f"{d} {c}({v * 100:+.0f}%)" for d, c, v in res.outliers[-10:])
        missing += (f"<p class='note'>誤データの疑いで除外した日次リターン: {len(res.outliers)}件"
                    f"(直近: {html.escape(items)})</p>")
    if res.nav_as_of is not None:
        nav_note = (f"基準価額(分配金再投資)との突き合わせ基準日: {res.nav_as_of.date()}。"
                    f"終値の20日騰落と基準価額の20日騰落の差が{DEV_WARN_PT}ptを超える、または"
                    "相対強度の符号が食い違う業種に⚠を付けています(薄いETFの終値ノイズの目安)。")
    else:
        nav_note = "基準価額を取得できなかったため、終値のみで算出しています(突き合わせなし)。"
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>日本株 業種別 資金動向</title>
<style>
body{{font-family:{FONT};margin:0 auto;max-width:960px;padding:12px 14px 40px;color:#222;background:#fff}}
h1{{font-size:20px;margin:8px 0 2px}} h2{{font-size:16px;margin:28px 0 2px}}
.note{{color:#666;font-size:12px;margin:2px 0 6px}}
.demo{{background:#fff3cd;border:1px solid #e0c36a;padding:8px;border-radius:6px;margin:8px 0;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border-bottom:1px solid #e5e5e5;padding:5px 4px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;white-space:normal}}
.tag{{color:#fff;border-radius:4px;padding:1px 6px;font-size:11px}}
.wrap{{overflow-x:auto}}
td.l{{white-space:nowrap}} .warn{{color:#b45309;font-size:11px}}
</style></head><body>
<h1>日本株 業種別 資金動向</h1>
<p class="note">データ基準日: {res.as_of.date()} / 業種ETF(TOPIX-17)+主要銘柄の売買代金・均等加重指数 / 投資判断の材料の一つであり、売買の推奨ではありません</p>
{demo_banner}
{''.join(parts)}
<section><h2>業種一覧</h2>
<p class="note">判定は相対強度(20日)と売買代金シェアzの符号による簡易分類。スコアは4指標の順位平均(0-100)。
「判定(銘柄均等)」は相対強度を銘柄均等加重に置き換えた場合の判定で、検証用です(スコア・並び順はETFベースのまま)。
ETFと銘柄均等で相対強度の符号が食い違う業種に⚠を付けています。</p>
<div class="wrap">{table_html(s)}</div>
<p class="note">{nav_note}</p>{missing}</section>
<section><h2>読み方の注意</h2>
<ul class="note">
<li>売買代金は買いと売りの合計で、純流入ではありません。方向は価格とセットで見てください。</li>
<li>集計対象は大型株中心です。不動産・小売・サービスなど中小型が多い業種は精度が落ちます。</li>
<li>先物・オプションSQ、指数リバランス、決算期は売買代金が歪みます。</li>
<li>純フローは外国人投資家の週次売買(JPX)などで裏取りしてください。</li>
</ul></section>
</body></html>"""


def main() -> int:
    demo = "--demo" in sys.argv
    prices = pd.read_parquet(PRICES)
    uni = pd.read_csv(UNIVERSE, dtype={"code": str})
    nav = pd.read_parquet(NAV) if NAV.exists() else None
    res = compute(prices, uni, nav)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(res, demo), encoding="utf-8")
    out = res.summary.copy()
    num_cols = out.select_dtypes("number").columns
    out[num_cols] = out[num_cols].round(4)
    out.to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"生成: {OUT_HTML} (基準日 {res.as_of.date()}, 業種 {len(res.summary)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
