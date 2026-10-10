"""Run the 4-dynamic-path CSI simulation."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cornersense.simulation.run import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
