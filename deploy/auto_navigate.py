"""Automatic navigation: START once, execute continuously, advance on stationary."""
from pathlib import Path
from jetson_navigate import main


if __name__ == '__main__':
    main(automatic=True, entry_script=Path(__file__))
