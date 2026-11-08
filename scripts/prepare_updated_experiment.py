"""Prepare fresh four-map samples and historical selection for the current map."""

import argparse
from pathlib import Path

import yaml
from experiment_independent_calibration import prepare

from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.simulation import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "preparation.json").exists():
        raise FileExistsError("This experiment has already been prepared")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = Path("config/landcover_comparison.yaml")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    historical = yaml.safe_load(Path("config/historical_refinement.yaml").read_text("utf-8"))
    sample_recipe = read_json(Path(historical["reference"]) / "sample_recipe.json")
    prepared = Path(protocol["paths"]["prepared"])
    manifest = read_json(prepared / "samples/manifest.json")
    if manifest["sample"] != fingerprint(prepared / "samples/samples_fine.npz"):
        raise ValueError("Fine sample cache is incomplete or changed")
    sample_recipe["sample_sources"]["fine"] = str(prepared / "samples")
    sample_recipe["fine_sample_manifest"] = str(prepared / "samples/manifest.json")
    sample_path = args.output / "sample_recipe.json"
    write_json(sample_path, sample_recipe)
    recipe = yaml.safe_load(Path("config/independent_calibration.yaml").read_text("utf-8"))
    recipe.update(sample_recipe=str(sample_path), output_root=str(args.output / "reference"))
    recipe_path = args.output / "sample_preparation.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, allow_unicode=True), encoding="utf-8")
    reference, _, _ = prepare(recipe_path)
    historical.update(
        reference=str(reference),
        samples=str(reference / "samples"),
        output_root=str(args.output / "water"),
    )
    historical_path = args.output / "historical_refinement.yaml"
    historical_path.write_text(
        yaml.safe_dump(historical, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    sources = [Path(protocol["products"]["fine"][key]) for key in ("source", "classes", "aligned")]
    sources += [protocol_path, Path("config/shared_biophysical.yaml"), Path(__file__)]
    write_json(
        args.output / "preparation.json",
        {
            "reference": str(reference),
            "historical_config": str(historical_path),
            "sources": [fingerprint(path) for path in sources],
            "policy": "Fresh fine and level-one samples; unchanged declared historical candidates",
        },
    )
    print(historical_path, flush=True)


if __name__ == "__main__":
    main()
