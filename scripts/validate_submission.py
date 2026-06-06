"""Validate Cora GCN warm-up submission files."""

import argparse
import json
import pickle
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cora.pkl"
RESULT_PATH = ROOT / "outputs" / "results" / "result.json"
ZIP_PATH = ROOT / "outputs" / "results" / "result.zip"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to cora.pkl.")
    parser.add_argument(
        "--result",
        type=Path,
        default=RESULT_PATH,
        help="Path to result.json.",
    )
    parser.add_argument("--zip", type=Path, default=ZIP_PATH, help="Path to result.zip.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        with args.data.open("rb") as f:
            raw = pickle.load(f)
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise SystemExit(
                "NumPy is required to load data/cora.pkl. "
                "Run this script inside the project environment."
            ) from exc
        raise

    with args.result.open(encoding="utf-8") as f:
        result = json.load(f)

    test_keys = {str(i) for i, value in enumerate(raw["test_mask"]) if value}
    result_keys = set(result)
    values = list(result.values())

    checks = {
        "result_count_matches_test_mask": len(result) == len(test_keys),
        "result_keys_match_test_mask": result_keys == test_keys,
        "values_are_int": all(isinstance(value, int) for value in values),
        "values_are_in_0_6": all(0 <= value <= 6 for value in values),
    }

    if args.zip.exists():
        with zipfile.ZipFile(args.zip) as zf:
            checks["zip_root_files"] = sorted(zf.namelist()) == [
                "gcn.py",
                "result.json",
            ]
    else:
        checks["zip_root_files"] = False

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
