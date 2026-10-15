#!/usr/bin/env python3
"""
CLI: Wipe all local runtime state — SQLite DBs, uploaded user documents,
conversation logs, and dev tool caches — back to a clean checkout.

Nothing here is version-controlled (data/, logs/, and the cache dirs are all
untracked), so this is destructive with no git-backed undo. Defaults to a
dry run; pass --yes to actually delete.

Usage:
    uv run python scripts/reset_state.py            # dry run, shows what would go
    uv run python scripts/reset_state.py --yes       # actually delete everything
    uv run python scripts/reset_state.py --yes --keep-users   # wipe db/logs/cache, keep uploaded docs
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _targets(*, keep_db: bool, keep_users: bool, keep_logs: bool, keep_cache: bool) -> list[Path]:
    targets: list[Path] = []

    if not keep_db:
        targets += sorted((ROOT / "data").glob("*.db"))

    if not keep_users:
        users_dir = ROOT / "data" / "users"
        if users_dir.exists():
            targets += sorted(users_dir.iterdir())

    if not keep_logs:
        logs_dir = ROOT / "logs"
        if logs_dir.exists():
            targets += sorted(logs_dir.glob("*.json")) + sorted(logs_dir.glob("*.log"))

    if not keep_cache:
        for name in (".pytest_cache", ".ruff_cache", ".mypy_cache"):
            p = ROOT / name
            if p.exists():
                targets.append(p)
        targets += [
            p for p in ROOT.glob("**/__pycache__")
            if ".venv" not in p.relative_to(ROOT).parts
        ]

    return [t for t in targets if t.exists()]


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Actually delete (default is dry run)")
    parser.add_argument("--keep-db", action="store_true", help="Keep data/*.db (sqlite memory)")
    parser.add_argument("--keep-users", action="store_true", help="Keep data/users/* (uploaded docs)")
    parser.add_argument("--keep-logs", action="store_true", help="Keep logs/* (conversation audit trail)")
    parser.add_argument("--keep-cache", action="store_true", help="Keep dev tool caches (pytest/ruff/mypy/__pycache__)")
    args = parser.parse_args()

    targets = _targets(
        keep_db=args.keep_db,
        keep_users=args.keep_users,
        keep_logs=args.keep_logs,
        keep_cache=args.keep_cache,
    )

    if not targets:
        print("Nothing to remove — already clean.")
        return

    print(f"{'Deleting' if args.yes else 'Would delete (dry run)'}:")
    for t in targets:
        kind = "dir " if t.is_dir() else "file"
        print(f"  [{kind}] {t.relative_to(ROOT)}")

    if not args.yes:
        print(f"\n{len(targets)} item(s). Re-run with --yes to actually delete.")
        return

    for t in targets:
        _remove(t)

    # Recreate empty structure so the app doesn't need to re-init dirs on next run
    (ROOT / "data" / "users").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    print(f"\nDeleted {len(targets)} item(s). State is clean.")


if __name__ == "__main__":
    if "-h" not in sys.argv and "--help" not in sys.argv and "--yes" not in sys.argv:
        print("(dry run — pass --yes to actually delete)\n")
    main()
