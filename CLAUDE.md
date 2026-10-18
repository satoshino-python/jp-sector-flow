# CLAUDE.md

日本株の業種別資金動向ダッシュボードと、ミネルヴィニ流の個別銘柄スクリーナー。仕組みと指標の説明は README.md。
ユーザーとのやり取り・コメント・コミットメッセージ以外の文章(README など)は日本語で、平易に書く。

## データの置き場所

- 株価の正は BigQuery `jp-sector-flow.sector_flow.prices`(2020年〜)。`data/prices.parquet` は1回の実行の中だけで使う作業用ファイルで、コミットしない
- スクリーニング結果は BigQuery `sector_flow.minervini_screen`(`as_of` × 銘柄)。daily-update の中で `src/minervini.py --save` が書く
- テーブル定義は `src/bq_io.py` の `*_SCHEMA` と `TABLES`。列を足すと `ensure_tables` が既存テーブルに足す
- 結果やチャート用のデータを、リポジトリやデータ用ブランチに置かない(以前に実行ごとのブランチを作る案を試してやめた)

## クラウドの作業環境での制約(Claude Code on the web)

- **Google Cloud の認証がない**(`GOOGLE_APPLICATION_CREDENTIALS` なし)。`bq_sync.py --pull` や `minervini.py --bq` は手元では動かない
  - BigQuery を使う処理は GitHub Actions で動かす(Secret `GCP_SA_KEY`)。`minervini-screen` ワークフローは、作業ブランチに `src/minervini.py` か自分自身の変更を push すると動き、その時点のコードで当日分の `minervini_screen` を上書きする
  - 結果の確認は Google Cloud BigQuery コネクタ(`execute_sql_readonly`、projectId `jp-sector-flow`)で読む。読み取りのみ
  - Actions のログは GitHub MCP の `get_job_logs` で読める(実行中は 404 になる)
- **Yahoo Finance に接続できない**(プロキシで 403)。`fetch.py` の実データ取得は手元では動かない。動作確認は `fetch.py --demo` か合成データで行う
- **強制 push とリモートブランチの削除はできない**(強制 push は自動モードで拒否、削除はプロキシで切断)。必要ならユーザーに GitHub 上で操作してもらう
- Playwright のブラウザは `/opt/pw-browsers/chromium-1194/chrome-linux/chrome` を `executable_path` で指定する(`playwright install` はしない)

## スクリーナー(src/minervini.py)

- 流れ: トレンドテンプレート(8条件) → VCP 検出 → エントリー状態 → ATR を加味した損切り。詳細と、条件を決めた経緯は README の「個別銘柄スクリーナー」
- 収縮の区切りは2段階の ZigZag(粗い反転幅で全体 → 最後に確定した転換点から今日までを細かい反転幅で)。起点の日の値幅は数えない
- しきい値はファイル冒頭の定数。変えたら `artifact/minervini.html` の `P`(表示用の写し)と README も合わせる
- 合成データでの確認: 上昇 → 収縮が浅くなるベース → ブレイク の系列を作り `screen()` / `detect_vcp()` を呼ぶ(テストファイルはリポジトリにない)
- 判定を変えたら、`minervini-screen` を動かして `minervini_screen` の件数・理由の内訳をコネクタで確かめ、気になる銘柄は検証ページのチャートで見る

## バックテスト(src/minervini_backtest.py)

- `minervini-backtest` ワークフローで動かす(作業ブランチに `src/minervini_backtest.py` かワークフロー自身の変更を push しても動く)。全体で1分弱(株価の読み込みは BigQuery Storage API で数秒)
- 結果は `minervini_backtest` を全置換。成績はジョブのログ(`get_job_logs`)か、コネクタで `minervini_backtest` を集計して見る
- 売買ルールは `VARIANTS`。ルールを足したら、`artifact/minervini.html` の `VLABEL` / `VNOTE` と README の表も合わせる
- 判定の関数(`detect_vcp` など)は daily の `src/minervini.py` と共通。全営業日のトレンドテンプレートは `trend_template_panel`(daily の `trend_template` と一致することを実行時に確かめている)

## 検証用ページ(artifact/minervini.html)

- Claude のアーティファクト <https://claude.ai/artifact/2rRdaecUy6vdJw6ziDaMji> (非公開)。このファイルをそのまま公開する(埋め込むデータはない)
- `mcp` capability で、閲覧者の Google Cloud BigQuery コネクタ(`execute_sql_readonly`, `get_query_results`)を使って `minervini_screen`・`minervini_backtest`・`prices` を読む。再公開時は capabilities を省略すると引き継がれる
- 「日々のスクリーニング」と「バックテスト」の2つの表示がある。バックテストは件数が多いので、ルールごとに一覧の列だけを読み、選んだトレードの詳細は後から読む
- 動作確認は、`window.claude.use("mcp")` を差し替える模擬スクリプトを Playwright の `add_init_script` で入れ、`page.route` で配信したページを開いて行う(`set_content` では差し替えが効かない)
- 業種ダッシュボードの共有版は別物: `artifact/snapshot.template.html` + `artifact/build.py`(データは孤立ブランチ `snapshot` の JSON を埋め込む)

## 売買シミュレーション(artifact/minervini_sim.html)

- Claude のアーティファクト <https://claude.ai/artifact/JEMSAY8QrhXoj398ShtSP4> (非公開)。検証用ページと同じく `mcp` capability で BigQuery を読む。このファイルをそのまま公開する
- 日々の「ブレイク」(`minervini_screen`)はページの中で売買を再現する(`replayDaily` は `minervini_backtest.py` の `simulate` と同じ手順)。バックテストのシグナルは `minervini_backtest` の売買に資金の制約だけを加える
- `simulate` 側の売買ルールや `src/minervini.py` の資金管理の定数を変えたら、ページの `P`・`replayDaily` と README も合わせる
- 動作確認は検証用ページと同じ方法(模擬スクリプト + `page.route`)

## 作業の進め方

- 作業ブランチで作業し、内容が確定したら main にマージする
- main への push は daily-update を動かさない(動くのは `trigger/run.txt` の変更だけ)。daily-update は平日 16:37 / 16:57 JST に main の内容で動く
