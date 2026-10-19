"""Apply thread limits before importing numerical libraries."""

import os

from ecologyhydro.config import Resources


def configure_threads(resources: Resources) -> None:
    threads = str(resources.threads_per_worker)
    # The conda-forge SciPy 1.18.1/MKL 2026.1 combination can raise a Windows
    # delay-load exception in repeated SVD solves with the threaded MKL layer.
    # Processes provide parallelism for this project, so use MKL's stable
    # sequential layer inside each worker.
    os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GDAL_NUM_THREADS",
    ):
        os.environ[variable] = threads
