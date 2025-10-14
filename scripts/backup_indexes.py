from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Bundle indexes and receipts into a timestamped archive")
    p.add_argument("--source", required=True, help="Directory containing index files and receipts")
    p.add_argument("--out", required=True, help="Directory to write the archive to")
    args = p.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = out / f"oscillink_backup_{ts}"
    archive = shutil.make_archive(str(base), "gztar", root_dir=str(src))
    print(archive)


if __name__ == "__main__":
    main()
