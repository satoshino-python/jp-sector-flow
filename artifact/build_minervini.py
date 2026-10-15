"""VCPスクリーニングの検証ページ(1枚のHTML)を作る。

データは src/minervini.py --charts が書き出す JSON(GitHub Actions の minervini-screen が
minervini-data/run-<実行番号> ブランチに置く)。

    python artifact/build_minervini.py <minervini_charts.json> [出力先.html]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "minervini.template.html"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "minervini-page.html"
    # </script> でページが壊れないように "<" を逃がす
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__MINERVINI_JSON__", payload)
    out.write_text(html, encoding="utf-8")
    print(f"{out} ({len(html) / 1e6:.2f} MB, {len(data['stocks'])} 銘柄)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
