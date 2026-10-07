"""Small, offline native-library and Windows multiprocessing smoke checks."""

import importlib
import importlib.metadata
import os
import platform
import tempfile
import time
from pathlib import Path

from ecologyhydro.config import ProjectConfig


def _double_raster(source: str, target: str) -> None:
    import pygeoprocessing
    from osgeo import gdal

    gdal.UseExceptions()
    pygeoprocessing.raster_calculator([(source, 1)], _double, target, gdal.GDT_Float32, -9999)


def _double(values):
    import numpy as np

    return np.where(values == -9999, -9999, values * 2)


def check_environment(config: ProjectConfig) -> dict:
    import numpy as np
    import psutil
    import taskgraph
    from osgeo import gdal, ogr, osr

    started = time.perf_counter()
    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    for module in (
        "natcap.invest.annual_water_yield",
        "pygeoprocessing.routing",
        "scipy",
        "pandas",
        "shapely",
        "rtree",
        "fiona",
        "geopandas",
        "matplotlib",
    ):
        importlib.import_module(module)
    config.paths.cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m0_", dir=config.paths.cache) as directory:
        work = Path(directory)
        source = work / "source.tif"
        raster = gdal.GetDriverByName("GTiff").Create(str(source), 3, 2, 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32648)  # Synthetic fixture only; not the project's research CRS.
        raster.SetProjection(srs.ExportToWkt())
        raster.SetGeoTransform((500000, 100, 0, 4000000, 0, -100))
        values = np.array([[1, 2, -9999], [3, 4, 5]], dtype=np.float32)
        raster.GetRasterBand(1).SetNoDataValue(-9999)
        raster.GetRasterBand(1).WriteArray(values)
        raster = None

        workers = min(2, config.resources.max_workers)
        graph = taskgraph.TaskGraph(str(work / "tasks"), workers)
        targets = [work / f"result_{index}.tif" for index in range(2)]
        try:
            for target in targets:
                graph.add_task(
                    func=_double_raster,
                    args=(str(source), str(target)),
                    target_path_list=[str(target)],
                    task_name=target.stem,
                )
        finally:
            graph.close()
            graph.join()
        for target in targets:
            result = gdal.Open(str(target))
            np.testing.assert_array_equal(result.ReadAsArray(), _double(values))
            if result.GetRasterBand(1).GetNoDataValue() != -9999:
                raise RuntimeError("Raster NoData did not round-trip")
            if result.GetGeoTransform() != (500000, 100, 0, 4000000, 0, -100):
                raise RuntimeError("Raster transform did not round-trip")
            if not srs.IsSame(osr.SpatialReference(wkt=result.GetProjection())):
                raise RuntimeError("Raster CRS did not round-trip")
            result = None

        vector_path = work / "station.gpkg"
        vector = ogr.GetDriverByName("GPKG").CreateDataSource(str(vector_path))
        layer = vector.CreateLayer("stations", srs=srs, geom_type=ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn("station_id", ogr.OFTString))
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("station_id", "synthetic_station")
        feature.SetGeometry(ogr.CreateGeometryFromWkt("POINT (500050 3999950)"))
        layer.CreateFeature(feature)
        feature = layer = vector = None
        vector = ogr.Open(str(vector_path))
        layer = vector.GetLayer(0)
        feature = layer.GetNextFeature()
        if layer.GetFeatureCount() != 1 or feature.GetField("station_id") != "synthetic_station":
            raise RuntimeError("Vector attributes did not round-trip")
        if feature.GetGeometryRef().GetX() != 500050 or not srs.IsSame(layer.GetSpatialRef()):
            raise RuntimeError("Vector geometry/CRS did not round-trip")
        feature = layer = vector = None

    packages = (
        "natcap.invest",
        "GDAL",
        "numpy",
        "pandas",
        "scipy",
        "pygeoprocessing",
        "taskgraph",
        "PyYAML",
        "pydantic",
        "psutil",
    )
    return {
        "status": "passed",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "gdal_runtime": gdal.VersionInfo("RELEASE_NAME"),
        "checks": [
            "native_imports",
            "geotiff_roundtrip",
            "geopackage_roundtrip",
            "pygeoprocessing_nodata",
            "taskgraph_process_workers",
        ],
        "smoke_workers": workers,
        "thread_limit": os.environ.get("OMP_NUM_THREADS"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "available_memory_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        "resource_configuration": config.resources.model_dump(),
        "scope": "M0 only; full-basin runtime and memory enforcement require M3 validation",
    }
