"""Enable `python -m ddg ...` as an alias for the CLI."""
import sys

from ddg.cli import main

if __name__ == "__main__":
    sys.exit(main())
