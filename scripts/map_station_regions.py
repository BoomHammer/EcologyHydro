"""Export the declared gauge-based three/seven regions; no official polygon claim."""

import argparse

import matplotlib
import numpy as np
from osgeo import gdal

from ecologyhydro.config import project_root
from ecologyhydro.long_record import MACRO_MAP, REACH_MAP
from ecologyhydro.spatial import OPTIONS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="project/diagnostics/long_record_v2")
    args = parser.parse_args()
    root, output = project_root(), project_root() / args.output
    if not (output / "summary.json").is_file():
        raise ValueError("A completed water-account audit is required")
    gdal.UseExceptions()
    with gdal.Open(str(root / "project/repairs/closed_routing_v3/zones.tif")) as source:
        values = source.ReadAsArray()
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, name, mapping, title in zip(
            axes,
            ("macro3", "reach7"),
            (MACRO_MAP, REACH_MAP),
            ("上中下游代理：头道拐／花园口分界", "七个连续测站汇水区间"),
            strict=True,
        ):
            target_path = output / f"{name}.tif"
            if target_path.exists():
                raise ValueError("Region maps already exist; do not overwrite")
            mapped = np.r_[0, mapping + 1][values].astype(np.uint8)
            with gdal.GetDriverByName("GTiff").CreateCopy(
                str(target_path), source, options=OPTIONS
            ) as target:
                target.GetRasterBand(1).WriteArray(mapped)
            cmap = ListedColormap(
                [
                    "white",
                    "#667caa",
                    "#e4ab60",
                    "#84ae8c",
                    "#d78381",
                    "#a292bd",
                    "#7dbfc4",
                    "#b4b36a",
                ][: int(mapping.max()) + 2]
            )
            preview = mapped[::8, ::8]
            ax.imshow(preview, cmap=cmap, vmin=0, vmax=mapping.max() + 1, interpolation="nearest")
            for group in range(1, int(mapping.max()) + 2):
                yy, xx = np.nonzero(preview == group)
                ax.text(
                    np.median(xx),
                    np.median(yy),
                    str(group),
                    fontsize=12,
                    ha="center",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                )
            ax.set(title=title)
            ax.axis("off")
        fig.suptitle(
            "固定汇流范围上的测站代理分区\n七区依次：贵得以上、贵得—兰州、兰州—头道拐、头道拐—龙门、龙门—三门峡、三门峡—花园口、花园口—利津",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output / "station_regions.png", dpi=180)
        fig.savefig(output / "station_regions.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
