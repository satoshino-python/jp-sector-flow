"""data/snapshot.json (+ 任意で comments.json) を snapshot.template.html に埋め込み、公開用の1枚のHTMLを作る。

    python artifact/build.py [--snapshot data/snapshot.json] [--comments comments.json] [--out out.html]

comments.json は [{"as_of": "...", "headline": "...", "body": "..."}, ...] (新しい順でなくてよい)。
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(ROOT / "data" / "snapshot.json"))
    ap.add_argument("--comments")
    ap.add_argument("--out", default=str(ROOT / "snapshot-page.html"))
    a = ap.parse_args()
    snap = json.loads(Path(a.snapshot).read_text(encoding="utf-8"))
    if a.comments:
        snap["comments"] = json.loads(Path(a.comments).read_text(encoding="utf-8"))
    # <script> の中に埋め込むので、</script> や <!-- で閉じられないようにする
    js = json.dumps(snap, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/").replace("<!--", "<\\!--")
    tpl = (HERE / "snapshot.template.html").read_text(encoding="utf-8")
    assert tpl.count("__SNAPSHOT_JSON__") == 1
    Path(a.out).write_text(tpl.replace("__SNAPSHOT_JSON__", js), encoding="utf-8")
    print(f"{a.out}: {Path(a.out).stat().st_size/1024:.0f} KB / 基準日 {snap.get('as_of')}")


if __name__ == "__main__":
    main()
