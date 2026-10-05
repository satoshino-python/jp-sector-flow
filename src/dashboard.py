"""data/prices.parquet から指標を計算し、docs/index.html を生成する。"""
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICES = ROOT / "data" / "prices.parquet"
UNIVERSE = ROOT / "data" / "universe.csv"
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
        rows.append(
            f"<tr><td>{html.escape(s)}</td>"
            f"<td><span class='tag' style='background:{color}'>{html.escape(r['label'])}</span></td>"
            f"<td>{r['score']:.0f}</td><td>{r['rel_20d']:+.1f}</td><td>{r['share_z']:+.2f}</td>"
            f"<td>{r['breadth_ma25']:.0f}%</td><td>{r['n_stocks']}</td></tr>"
        )
    return (
        "<table><thead><tr><th>業種</th><th>判定</th><th>スコア</th><th>相対20日</th>"
        "<th>シェアz</th><th>25日線上</th><th>銘柄数</th></tr></thead><tbody>"
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
</style></head><body>
<h1>日本株 業種別 資金動向</h1>
<p class="note">データ基準日: {res.as_of.date()} / 業種ETF(TOPIX-17)+主要銘柄の売買代金集計 / 投資判断の材料の一つであり、売買の推奨ではありません</p>
{demo_banner}
{''.join(parts)}
<section><h2>業種一覧</h2>
<p class="note">判定は相対強度(20日)と売買代金シェアzの符号による簡易分類。スコアは4指標の順位平均(0-100)。</p>
<div class="wrap">{table_html(s)}</div>{missing}</section>
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
    res = compute(prices, uni)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(res, demo), encoding="utf-8")
    res.summary.round(4).to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"生成: {OUT_HTML} (基準日 {res.as_of.date()}, 業種 {len(res.summary)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
