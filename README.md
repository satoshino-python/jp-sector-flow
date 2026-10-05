# jp-sector-flow

日本株の業種別に、どこへ資金が向かっているかを毎営業日に更新するダッシュボードです。

## 仕組み

- **価格・相対強度**: NEXT FUNDS TOPIX-17業種ETF(1617〜1633)をTOPIX連動ETF(1306)と比較
- **売買代金**: 業種ETFは薄くノイズが大きいため使わず、構成銘柄の売買代金を業種別に合算
- **広がり**: 業種内の25日線上銘柄比率、上昇日の売買代金比率
- データ取得は `yfinance`(非公式)。GitHub Actionsが平日16:30 JSTに差分取得し、`docs/index.html` を更新します

## 主な指標

| 指標 | 意味 |
|---|---|
| 相対強度(5/20/60日) | 業種ETFのリターン − TOPIXのリターン(%pt) |
| 売買代金シェアz | 業種の売買代金シェア(5日平均)が、過去60日の平均からどれだけ離れているか |
| 売買代金 当日/20日 | 当日の業種売買代金 ÷ 直近20日平均(当日除く) |
| 上昇日の売買比率 | 直近5日の売買代金のうち、上昇銘柄の分の割合 |
| 25日線上比率 | 業種内で25日移動平均を上回る銘柄の割合 |
| 判定 | 相対強度20日とシェアzの符号で4分類(資金流入/買い細り/売り圧力・転換前/低調) |

売買代金は買いと売りの合計で、純流入ではありません。価格とセットで読み、外国人投資家の週次売買(JPX公表)などで裏取りしてください。

## セットアップ(初回のみ)

1. **Actionsの書き込み権限**: Settings → Actions → General → Workflow permissions → *Read and write permissions*
2. **Pagesの有効化**: Settings → Pages → Source: *Deploy from a branch* → Branch: `main` / `/docs`
3. **初回実行**: Actions タブ → *daily-update* → *Run workflow*

公開URLは `https://<ユーザー名>.github.io/jp-sector-flow/` です。

## ローカル実行

```bash
pip install -r requirements.txt
python src/fetch.py --demo      # 合成データで動作確認(実データではない)
python src/dashboard.py --demo  # docs/index.html を生成
python src/fetch.py             # 実データ(差分取得)
python src/dashboard.py
```

## 銘柄リストの保守

`data/universe.csv` が対象銘柄です。**これは主要銘柄をもとにした初期リストで、日経225の完全な構成ではありません。**
上場廃止・社名/コード変更・指数の構成変更があるため、定期的に見直してください。
取得できなかった銘柄は実行ログに出力され、集計からは自動で除外されます。

## 注意

- 大型株中心の集計のため、中小型が多い業種は精度が落ちます
- 先物・オプションSQ、指数リバランス、決算期は売買代金が歪みます
- GitHub Actionsのcronは遅延することがあります。リポジトリに60日間動きがないとスケジュールが止まる場合があるため、止まっていたら手動実行してください
- 本ツールは投資判断の材料の一つであり、売買の推奨ではありません
