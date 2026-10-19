"""Yellow River annual water yield project."""

import os

# Set before NumPy/SciPy load. Multiprocessing supplies project-level parallelism.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

__version__ = "0.1.0"
