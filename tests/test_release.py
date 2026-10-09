"""Focused release checks runnable without pytest."""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_release import check_inputs, verify_claims  # noqa: E402


def main() -> int:
    if not compileall.compile_dir(ROOT / "satbeam_study", quiet=1):
        raise AssertionError("satbeam_study compilation failed")
    if not compileall.compile_dir(ROOT / "satbeam", quiet=1):
        raise AssertionError("satbeam helper compilation failed")
    inputs = check_inputs()
    claims = verify_claims()
    if len(inputs) < 10:
        raise AssertionError("Frozen-data input coverage is incomplete")
    if claims["main_scenarios"] != 30:
        raise AssertionError("Expected 30 paired scenarios")
    print("SateBeam focused release tests PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
