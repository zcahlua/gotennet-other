from __future__ import annotations

import argparse

from scripts.download_transition1x import main as download_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    import sys

    sys.argv = [sys.argv[0], "--cache-dir", args.cache_dir, "--dataset-name", "SN2RXN"]
    download_main()


if __name__ == "__main__":
    main()
