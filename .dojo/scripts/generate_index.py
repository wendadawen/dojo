#!/usr/bin/env python3
"""Retired compatibility entrypoint."""


def main() -> int:
    print(
        "generate_index.py is retired and will not rewrite index.html. "
        "Use build_catalog.py to generate catalog.json."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
