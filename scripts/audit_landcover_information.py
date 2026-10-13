"""Audit which fine labels can affect AWY, without fitting or selecting on runoff."""

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from experiment_fine_focus_routing import cached_natural, parameters
from experiment_independent_calibration import sample_response
from report_independent_calibration import markdown_table
from run_landcover_comparison import worker_setup

from ecologyhydro.accuracy_goals import PRODUCTS
from ecologyhydro.baseline import read_json
from ecologyhydro.cache import fingerprint
from ecologyhydro.calibration_followup import seasonal_data
from ecologyhydro.class_response import pixel_response
from ecologyhydro.long_record import ENDPOINTS, REACH_MAP, STATIONS
from ecologyhydro.routing_robustness import forward
from ecologyhydro.shared_biophysical import load_registry, parameter_table
from ecologyhydro.simulation import write_json


def audit(reference, accepted, output):
    config = worker_setup(reference / "execution_protocol.yaml")
    registry = load_registry(reference / "biophysical.yaml")
    frozen = read_json(accepted / "current_parameters.json")
    detail = {r["lucode"]: r for r in parameter_table(registry, "fine", "detail")}
    parent = {r["lucode"]: r for r in parameter_table(registry, "fine", "parent")}
    data = sample_response(reference, "fine", compressed=False)
    first = data["groups"] // 11 == 0
    changed = seasonal_data(data, frozen["seasonal_factors"])
    changed["kc"] = data["kc"].copy()
    head = (REACH_MAP[data["zone"]] == 0) & (data["veg"] == 1)
    changed["kc"][head] *= frozen["head_multiplier"]
    x = np.array(frozen["seed"])
    x[13:] = 0
    omega_capped = 1.25 + x[REACH_MAP[data["zone"]]] * changed["awc"] / data["p"] >= 5
    parent_awc = data["awc"].copy()
    rows, signatures = [], {}
    for code, row in detail.items():
        mask = data["code"] == code
        single = mask & first
        weight = data["weights"][single]
        area = float(weight.sum() * 1e5)
        if not area:
            continue
        root = row["root_depth"]
        generic_root = parent[code]["root_depth"]
        different = (row["lulc_veg"] == 1) and not np.isclose(root, generic_root)
        survives = different & ~np.isclose(
            np.minimum(root, data["soil"][single]),
            np.minimum(generic_root, data["soil"][single]),
        )
        parent_awc[mask] = np.minimum(generic_root, data["soil"][mask]) * data["pawc"][mask]
        signature = (row["lulc_veg"], row["kc"], root if row["lulc_veg"] else None)
        if signature not in signatures:
            signatures[signature] = len(signatures) + 1
        rows.append(
            dict(
                code=code,
                profile=row["profile"],
                area_km2=area,
                signature=signatures[signature],
                kc=row["kc"],
                root_mm=root,
                parent_root_mm=generic_root,
                root_differs_from_profile_parent=bool(different),
                root_contrast_survives_soil_area_km2=float(weight[survives].sum() * 1e5),
                soil_limited_fraction=float(
                    weight[data["soil"][single] < root].sum() / weight.sum()
                )
                if row["lulc_veg"]
                else None,
                omega_capped_area_year_fraction=float(
                    data["weights"][mask & omega_capped].sum() / data["weights"][mask].sum()
                )
                if row["lulc_veg"]
                else None,
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "class_information.csv", index=False)
    root_parent_data = changed.copy()
    root_parent_data["awc"] = parent_awc
    upper = REACH_MAP[data["zone"]] == 0
    root_parent_data["awc"][upper] *= np.asarray(frozen["seasonal_factors"])[
        data["groups"][upper] // 11
    ]
    delta_depth = pixel_response(x, changed) - pixel_response(x, root_parent_data)
    delta = np.bincount(data["groups"], weights=delta_depth * data["weights"]).reshape(-1, 11)
    cumulative = delta.cumsum(axis=1)[:, ENDPOINTS]
    pd.DataFrame(cumulative, index=config["forcing_years"], columns=STATIONS).to_csv(
        output / "root_detail_natural_delta.csv", index_label="year", encoding="utf-8-sig"
    )
    # Use every inherited source, never choose a source by the sign of its map contrast.
    contrasts = []
    for source in PRODUCTS:
        routing = parameters(read_json(reference / f"freeze_{source}.json"))
        fine = forward(cached_natural(reference, "fine", source, config["forcing_years"]), routing)
        broad = forward(
            cached_natural(reference, "fine_level1", source, config["forcing_years"]), routing
        )
        for yi, year in enumerate(config["forcing_years"]):
            for si, station in enumerate(STATIONS):
                contrasts.append(
                    dict(
                        source=source,
                        year=year,
                        station=station,
                        fine_minus_level1=fine[yi, si] - broad[yi, si],
                    )
                )
    pd.DataFrame(contrasts).to_csv(
        output / "fixed_model_map_contrasts.csv", index=False, encoding="utf-8-sig"
    )
    groups = table.groupby("signature").agg(
        codes=("code", lambda s: ",".join(s.astype(str))),
        area_km2=("area_km2", "sum"),
        count=("code", "size"),
    )
    groups.to_csv(output / "equivalent_labels.csv")
    total = table.area_km2.sum()
    summary = dict(
        present_fine_labels=len(table),
        distinct_base_responses=len(groups),
        area_km2=total,
        area_fraction_in_multi_label_response_groups=groups.loc[
            groups["count"] > 1, "area_km2"
        ].sum()
        / total,
        area_fraction_with_root_detail=table.loc[
            table.root_differs_from_profile_parent, "area_km2"
        ].sum()
        / total,
        area_fraction_root_contrast_survives_soil=table.root_contrast_survives_soil_area_km2.sum()
        / total,
        max_absolute_root_detail_natural_change=float(np.max(np.abs(cumulative))),
        flow_units="1e8 m3",
        calibrated_parameters_unchanged=True,
        interpretation="weighted sample diagnosis, no new fit and no accuracy claim",
        profile_parent_is_not_actual_level1_hierarchy=True,
    )
    write_json(output / "summary.json", summary)
    lines = [
        "# 精细地类信息进入模型的诊断",
        "",
        "未拟合新参数，未用径流评价选择方案。原主表及冻结模型不变。",
        "",
        f"出现 {len(table)} 个精细类别，对应 {len(groups)} 种不同基础 AWY 响应参数组合。",
        "同组的类别在相同气候、土壤和区域参数下不能仅凭标签产生不同响应。"
        "这不意味着它们所在位置的产水相同，也不意味着整个地图没有信息。",
        "",
        markdown_table(groups[groups["count"] > 1].set_index("codes")),
        "",
        "这些多标签组合覆盖面积的 "
        f"{100 * summary['area_fraction_in_multi_label_response_groups']:.3f}%。",
        f"细类根深与功能型父类不同的面积占 {100 * summary['area_fraction_with_root_detail']:.3f}%；"
        "经土壤深度截断后仍保留根深差异的面积占 "
        f"{100 * summary['area_fraction_root_contrast_survives_soil']:.3f}%。",
        "本项父类是参数库的功能型父类，不等于实际一级类，不能把这个比例称为全部精细信息贡献。",
        "`root_detail_natural_delta.csv` 只消融细类根深，其余参数不变；"
        "`fixed_model_map_contrasts.csv` 才是实际一级类与细类在所有原参数来源下的完整换图差异。",
        "",
        "共同加减的管理账户和固定损失会在两图预测差中抵消，不能凭调整这些项增加两图的物理区分度。"
        "但共同偏移仍可改变相对于观测的误差排名，所以不能用有利排名选管理参数。",
        "这里只诊断模型的信息表达能力，不能据此捏造差异参数或承诺精细分类一定获胜。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("experiments/calibration_followup/20261011T154702_556627Z"),
    )
    parser.add_argument(
        "--accepted",
        type=Path,
        default=Path("experiments/scientific_robustness_final/20261011T172623_968568Z"),
    )
    args = parser.parse_args()
    output = Path("experiments/landcover_information") / datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True)
    inputs = [Path(__file__), args.accepted / "current_parameters.json"]
    inputs += [
        args.reference / name
        for name in (
            "biophysical.yaml",
            "hierarchy.json",
            "sample_recipe.json",
            "execution_protocol.yaml",
        )
    ]
    inputs += sorted(args.reference.glob("natural_*.json"))
    inputs += sorted(args.reference.glob("freeze_*.json"))
    inputs += [
        Path("src/ecologyhydro") / name
        for name in (
            "shared_biophysical.py",
            "class_response.py",
            "routing_robustness.py",
            "calibration_followup.py",
        )
    ]
    config = worker_setup(args.reference / "execution_protocol.yaml")
    recipe = read_json(args.reference / "sample_recipe.json")
    inputs.append(Path(recipe["sample_sources"]["fine"]) / "samples_fine.npz")
    frozen_sources = [fingerprint(p) for p in inputs]
    write_json(output / "sources.json", frozen_sources)
    shutil.copyfile(__file__, output / "executed_audit_landcover_information.py")
    print("OUTPUT", output, flush=True)
    try:
        audit(args.reference, args.accepted, output)
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise
    if frozen_sources != [fingerprint(Path(item["path"])) for item in frozen_sources]:
        raise ValueError("Source changed while auditing")
    write_json(
        output / "status.json", dict(status="complete", forcing_years=config["forcing_years"])
    )


if __name__ == "__main__":
    main()
