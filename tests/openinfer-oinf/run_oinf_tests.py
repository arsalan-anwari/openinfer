#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import unittest


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


def _load_suite(start_dir: Path) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    return loader.discover(
        start_dir=str(start_dir),
        pattern="test_*.py",
        top_level_dir=str(start_dir),
    )


def _list_tests(suite: unittest.TestSuite, filter_substr: str | None) -> list[str]:
    tests = []
    for test in _iter_tests(suite):
        name = test.id()
        if filter_substr and filter_substr not in name:
            continue
        tests.append(name)
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Run openinfer-oinf tests")
    parser.add_argument("--list", action="store_true", help="List tests and exit")
    parser.add_argument("--filter", help="Substring filter for test ids")
    args = parser.parse_args()

    start_dir = Path(__file__).resolve().parent
    suite = _load_suite(start_dir)
    tests = _list_tests(suite, args.filter)

    if args.list:
        for name in tests:
            print(name)
        return 0

    if args.filter and not tests:
        print(f"error: no tests match filter '{args.filter}'", file=sys.stderr)
        return 1

    if args.filter:
        suite = unittest.TestSuite(
            [test for test in _iter_tests(suite) if test.id() in tests]
        )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
