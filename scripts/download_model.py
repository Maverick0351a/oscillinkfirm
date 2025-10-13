from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:  # pragma: no cover - runtime utility
    print("huggingface_hub is required. Install with: pip install huggingface-hub", file=sys.stderr)
    raise


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_weight_file(root: Path) -> Path | None:
    candidates = []
    for ext in (".safetensors", ".bin"):
        for fp in root.rglob(f"*{ext}"):
            # Prioritize typical HF weight filenames
            score = 0
            name = fp.name.lower()
            if name in {"model.safetensors", "pytorch_model.bin"}:
                score += 10
            # Larger files likely hold the weights
            try:
                size = fp.stat().st_size
            except OSError:
                size = 0
            candidates.append((score, size, fp))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


def main() -> int:
    ap = argparse.ArgumentParser(description="Download HF model to local models/ and update models_registry.json")
    ap.add_argument("--repo-id", default="BAAI/bge-small-en-v1.5", help="Hugging Face repo id")
    ap.add_argument("--model-key", default="bge-small-en-v1.5", help="Key in models_registry.json to update")
    ap.add_argument("--local-dir", default="models/bge-small-en-v1.5", help="Destination folder under repo root")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dest = (repo_root / args.local_dir).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} to {dest} ...")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {local_path}")

    weights = pick_weight_file(dest)
    if weights is None:
        print("Warning: Could not locate a weights file to hash.")
        sha = None
    else:
        sha = sha256_file(weights)
        print(f"Weights file: {weights} -> sha256={sha}")

    reg_path = repo_root / "models_registry.json"
    registry = {} if not reg_path.exists() else json.loads(reg_path.read_text(encoding="utf-8"))

    entry = registry.get(args.model_key, {})
    entry.setdefault("license", "Apache-2.0")
    entry.setdefault("dim", 384)
    entry["path"] = args.local_dir.replace("\\", "/")
    if sha:
        entry["sha256"] = sha
        entry["sha256_weights"] = sha
    registry[args.model_key] = entry

    reg_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Updated registry at {reg_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - utility
    raise SystemExit(main())
