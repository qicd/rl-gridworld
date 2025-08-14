from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/downloads/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = args.url.split("/")[-1] or "downloaded_file"
    out_path = out_dir / file_name

    print(f"Downloading {args.url} -> {out_path}")
    resp = requests.get(args.url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()


