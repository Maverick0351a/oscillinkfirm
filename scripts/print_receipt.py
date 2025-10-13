from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Print a receipt and optionally validate it")
    p.add_argument("receipt", help="Path to receipt JSON file")
    p.add_argument(
        "--schema",
        help="Optional path to JSON schema for validation",
        default=str(
            Path(__file__).resolve().parents[1]
            / "oscillink"
            / "assets"
            / "schemas"
            / "ingest_receipt.schema.json"
        ),
    )
    args = p.parse_args()

    data = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    print(json.dumps(data, indent=2, sort_keys=True))

    # Optional validation if jsonschema is available
    try:
        import jsonschema  # type: ignore

        schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)
        print("\nValidation: OK")
    except ModuleNotFoundError:
        print("\nValidation skipped (jsonschema not installed)")
    except Exception as e:  # pragma: no cover - utility script
        print(f"\nValidation error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
