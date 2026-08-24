"""
CLI entry point for ddg.scan.

    python -m ddg.scan build   --sequence <SEQ> --name <ID>
    python -m ddg.scan predict --config experiment_configs/scan_<ID>.yaml

`build` writes the mutation table + experiment config; the pipeline then extracts
Boltz features for it; `predict` fits the regressor on the labelled corpora and
scores the scan.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

USAGE = """usage: python -m ddg.scan {build|predict} [options]

  build    generate the L*19 mutation CSV + experiment config for a sequence
  predict  score a scan's feature table with regimes A/B/D

Run `python -m ddg.scan <command> --help` for each command's options."""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "build":
        from ddg.scan.build import main as run
    elif command == "predict":
        from ddg.scan.predict import main as run
    else:
        print(f"unknown command '{command}'\n\n{USAGE}", file=sys.stderr)
        return 2
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
