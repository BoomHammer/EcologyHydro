"""Fetch bounded provider subsets for independent checks, never replace inputs."""

import argparse
import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from ecologyhydro.cache import fingerprint
from ecologyhydro.config import project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", choices=("pet", "ppt", "aet"), default="pet")
    parser.add_argument("--year", type=int, choices=range(2019, 2024), default=2019)
    parser.add_argument("--full-basin", action="store_true")
    args = parser.parse_args()
    name = "climate_reference_full" if args.full_basin else "climate_reference"
    if args.year == 2023:
        name += "_holdout"
    output = project_root() / "project/repairs" / name / "provider"
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"terraclimate_current_{args.variable}_{args.year}.nc"
    if target.exists():
        record = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        if fingerprint(target)["sha256"] != record["file"]["sha256"]:
            raise ValueError("Downloaded reference checksum mismatch")
        print(f"Already downloaded: {target}")
        return
    query = urlencode(
        {
            "var": args.variable,
            "north": 42.5 if args.full_basin else 37,
            "south": 31.5 if args.full_basin else 32,
            "west": 95 if args.full_basin else 95.4,
            "east": 120 if args.full_basin else 104,
            "horizStride": 1,
            "time_start": f"{args.year}-01-01T00:00:00Z",
            "time_end": f"{args.year}-12-31T23:59:59Z",
            "timeStride": 1,
            "accept": "netcdf",
        }
    )
    url = (
        "https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/"
        f"agg_terraclimate_{args.variable}_1950_CurrentYear_GLOBE.nc?{query}"
    )
    record = {"url": url, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        with urlopen(url, timeout=45) as response:
            data = response.read(30 * 1024**2 + 1)
        if len(data) > 30 * 1024**2 or not data.startswith((b"CDF", b"\x89HDF")):
            raise ValueError("Not a bounded NetCDF response")
        target.write_bytes(data)
        record.update({"status": "downloaded", "file": fingerprint(target)})
    except Exception as error:
        record.update({"status": "failed", "error": str(error)})
    target.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    if record["status"] != "downloaded":
        raise RuntimeError("Provider retrieval failed; see request record")


if __name__ == "__main__":
    main()
