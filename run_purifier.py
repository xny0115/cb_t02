"""CLI for dataset purifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from purifier.purifier import clean_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset purifier")
    parser.add_argument("src", type=Path, help="raw dataset file or directory")
    parser.add_argument("dst", type=Path, nargs="?", default=Path("datas"), help="output directory")
    args = parser.parse_args()

    if args.src.is_dir():
        files = sorted(args.src.glob("*"))
    else:
        files = [args.src]

    args.dst.mkdir(parents=True, exist_ok=True)

    for fp in files:
        if fp.is_file() and fp.suffix.lower() in {".json", ".txt"}:
            out = clean_file(fp, args.dst)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
