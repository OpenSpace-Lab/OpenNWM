"""Fixed-time reference sampling and current-and-future target-window navigation."""
from pathlib import Path
from jetson_navigate import main

if __name__ == '__main__':
    main(automatic=True, window_navigation=True, entry_script=Path(__file__))
