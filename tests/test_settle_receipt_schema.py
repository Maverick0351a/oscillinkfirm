from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oscillink.core.lattice import OscillinkLattice


def _make_small_lattice(N: int = 6, D: int = 8) -> OscillinkLattice:
    rng = np.random.RandomState(0)
    Y = rng.randn(N, D).astype(np.float32)
    lat = OscillinkLattice(
        Y,
        kneighbors=min(3, max(1, N - 1)),
        lamG=1.0,
        lamC=0.5,
        lamQ=4.0,
        deterministic_k=True,
    )
    psi = rng.randn(D).astype(np.float32)
    lat.set_query(psi)
    lat.settle(tol=1e-3)
    return lat


def test_settle_receipt_validates_against_schema(tmp_path: Path) -> None:
    lat = _make_small_lattice()
    rec = lat.receipt()
    # Write to tmp file to reuse existing print_receipt validation flow if desired
    receipt_path = tmp_path / "settle_receipt.json"
    receipt_path.write_text(json.dumps(rec), encoding="utf-8")

    # Try schema validation only when jsonschema is installed
    try:
        import jsonschema  # type: ignore

        schema_path = (
            Path(__file__).resolve().parents[1]
            / "oscillink"
            / "assets"
            / "schemas"
            / "settle_receipt.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=rec, schema=schema)
    except ModuleNotFoundError:
        # Optional dependency not installed in minimal envs
        pass