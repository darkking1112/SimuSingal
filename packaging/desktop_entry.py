"""Entry used by PyInstaller; worker subcommand shares the same executable."""

from simusignal.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
