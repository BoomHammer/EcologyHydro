"""Apply thread limits before importing numerical libraries."""

import os

from ecologyhydro.config import Resources


def configure_threads(resources: Resources) -> None:
    threads = str(resources.threads_per_worker)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GDAL_NUM_THREADS",
    ):
        os.environ[variable] = threads
