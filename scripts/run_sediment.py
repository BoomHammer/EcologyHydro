"""Prepare and execute the official InVEST SDR model on the current fine map."""

import argparse
import multiprocessing
from pathlib import Path

from ecologyhydro.sediment import run_sediment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/sediment.yaml"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    print(
        run_sediment(args.config, prepare_only=args.prepare_only, resume_run=args.resume_run),
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
