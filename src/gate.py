"""スケジュール起動の重複実行を避けるための確認。

cron を2回(16:37 と 16:57 JST)かけているので、2回目の起動時点で当日分が BigQuery に入っていれば、
本体の更新をスキップする。手動実行(workflow_dispatch)と trigger/run.txt の push では、常に実行する。

GITHUB_OUTPUT に skip=true/false を書く。確認に失敗したときは skip=false(= 実行する)にする。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bq_io  # noqa: E402


def decide(event: str, latest_as_of, today) -> bool:
    """True ならスキップ(当日分がすでに入っている)。"""
    return event == "schedule" and latest_as_of is not None and latest_as_of == today


def main() -> int:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    today = (datetime.now(timezone.utc) + timedelta(hours=9)).date()   # 日本時間の今日
    skip = False
    if event == "schedule":
        try:
            client = bq_io.get_client()
            rows = list(bq_io._run(client, f"SELECT MAX(as_of) AS d FROM {bq_io.fqn('sector_summary')}").result())
            latest = rows[0]["d"] if rows else None
            skip = decide(event, latest, today)
            print(f"event={event} 日本時間の今日={today} BigQueryの最新基準日={latest} → {'スキップ' if skip else '実行'}")
        except Exception as e:   # 確認できなければ実行する(取りこぼさない側に倒す)
            print(f"::warning::当日分の確認に失敗したので実行します: {e}")
    else:
        print(f"event={event or '(手元)'} → 常に実行")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"skip={'true' if skip else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
