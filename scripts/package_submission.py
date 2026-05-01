"""Package the Cora GCN warm-up submission archive."""

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "release"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_RELEASE / "gcn.py",
        help="Path to gcn.py.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RELEASE / "result.json",
        help="Path to result.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RELEASE / "result.zip",
        help="Output zip path.",
    )
    return parser.parse_args()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main():
    args = parse_args()
    require_file(args.source)
    require_file(args.result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.source, arcname="gcn.py")
        zf.write(args.result, arcname="result.json")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
