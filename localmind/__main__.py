"""Позволяет запускать пакет как `python -m localmind ...`."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
