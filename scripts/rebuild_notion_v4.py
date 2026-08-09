"""Read-only preflight or checkpointed Notion layout v4 migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.notion_migration import rebuild_notion_workspace_v4
from modules.notion_sync import NotionClient, get_notion_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Back up and rebuild the connected Notion workspace into five study databases."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Perform the migration. Without this flag only a read-only preflight runs.",
    )
    args = parser.parse_args()
    settings = get_notion_settings()
    if not settings.configured or not settings.token:
        raise SystemExit("Notion chưa được cấu hình trong môi trường này.")
    result = rebuild_notion_workspace_v4(
        NotionClient(settings.token), settings, confirm=args.confirm
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
