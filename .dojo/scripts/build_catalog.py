#!/usr/bin/env python3
"""Generate catalog.json from wiki HTML pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from catalog_builder import build_catalog, write_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = build_catalog(args.root)
    write_catalog(catalog, args.output)
    print(
        f"generated {len(catalog['pages'])} pages, "
        f"{len(catalog['edges'])} edges, "
        f"{len(catalog['warnings'])} warnings -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
