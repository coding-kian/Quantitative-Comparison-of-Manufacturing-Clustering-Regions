import pandas as pd
import ee
import glob, os
from scripts.database_interactions import database_read, database_write
from scripts.util_config import total_runtime, logger

def init_ee(project_id: str):
    """Make sure billing and configuration permissions are setup. permissions(Service Usage Consumer & Earth Engine Resource Viewer)"""
    try:
        ee.Initialize(project=project_id)
        logger.info(f"initialised {project_id}")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        logger.info(f"authenticated {project_id}")


def point_id(df: pd.DataFrame) -> ee.FeatureCollection:
    """Gives each geometric point (longitude, latitude) in google earth engine a grid id """
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]), {"grid_id": int(r.grid_id)})
        for r in df.itertuples(index=False)])


def export_viirs_year(grid_points: ee.FeatureCollection, year: int, drive_folder: str, desc: str, scale_m: int = 500) -> None:
    stats = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1))
        .select("avg_rad").mean()).reduceRegions(grid_points, ee.Reducer.mean(), scale_m)

    def format_features(f): # feature map
        d = ee.Dictionary(f.toDictionary())
        return ee.Feature(None, {"grid_id": d.get("grid_id"), "year": year, "light_mean": d.get("mean")})

    task = ee.batch.Export.table.toDrive(collection=stats.map(format_features), description=desc, folder=drive_folder, fileFormat="CSV")
    task.start()
    status = task.status()
    logger.info(f'VIIRS: {status.get("state", "UNKNOWN")} & {status.get("description", "")}')

def export_s1_year(grid_points: ee.FeatureCollection, year: int, drive_folder: str, desc: str, scale_m: int = 1_500) -> None:
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year + 1, 1, 1))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("resolution_meters", 10)).select(["VV", "VH"]))

    stats = s1.median().reduceRegions(grid_points, ee.Reducer.mean(), scale_m)

    def format_features(f):
        d = ee.Dictionary(f.toDictionary())
        return ee.Feature(None, {"grid_id": d.get("grid_id"), "year": year, "vv_mean": d.get("VV"),"vh_mean": d.get("VH")})

    task = ee.batch.Export.table.toDrive(collection=stats.map(format_features), description=desc, folder=drive_folder, fileFormat="CSV")
    task.start(); status = task.status()
    logger.info(f'Senintel1: {status.get("state", "UNKNOWN")} & {status.get("description", "")}') 


@total_runtime
def run_satellite_exports(project_id: str, db_path: str, year: int, drive_folder: str, chunk_size: int = 3_000):
    init_ee(project_id)
    offset, chunk = 0, 0
    total = int(database_read(db_path, "SELECT COUNT(*) FROM grid"   , ())[0][0])
    logger.info(f"currnt database has rows: {total}")

    while offset < total:
        sql = f"SELECT grid_id, longitude, latitude FROM GRID ORDER BY grid_id LIMIT ? OFFSET ?"
        df = pd.DataFrame( database_read(db_path, sql, (int(chunk_size), int(offset))), columns = ["grid_id", "longitude", "latitude"])
        if df.empty: break
        chunk += 1
        logger.info(f"\nchunk - {chunk} @ {len(df):,}/{offset:,} ") # this is just for logging purposes

        points = point_id(df) # uses feature collection to assign grid id to each point (longitude, latitude)
        export_viirs_year(points, year, drive_folder, f"viirs_{year}_c{chunk:04d}")
        export_s1_year(points, year, drive_folder, f"s1_{year}_c{chunk:04d}")
        offset += len(df)

    logger.info("Main GEE export done")


@total_runtime
def gee_to_db(db_path: str, exports_folder: str):
    viirs_files = sorted(glob.glob(os.path.join(exports_folder, "viirs_*.csv")))
    s1_files    = sorted(glob.glob(os.path.join(exports_folder, "s1_*.csv")))

    viirs_sql = "INSERT INTO gee_viirs (grid_id, light_mean) VALUES (?, ?)"
    s1_sql = "INSERT INTO gee_sentinel1 (grid_id, vv_mean, vh_mean) VALUES (?, ?, ?)"

    # Load VIIRS
    for f in viirs_files:
        df = pd.read_csv(f)
        params = [(int(r.grid_id), None if pd.isna(r.light_mean) else float(r.light_mean))
            for r in df[["grid_id", "light_mean"]].itertuples(index=False)]

        database_write(db_path, viirs_sql, params)
        logger.info(f"Loaded VIIRS: {os.path.basename(f)} ({len(params):,} rows)")

    # Load Sentinel-1
    for f in s1_files:
        df = pd.read_csv(f) # exported columns: grid_id, vv_mean, vh_mean
        params = [(int(r.grid_id), None if pd.isna(r.vv_mean) else float(r.vv_mean), None if pd.isna(r.vh_mean) else float(r.vh_mean))
            for r in df[["grid_id", "vv_mean", "vh_mean"]].itertuples(index=False)]

        database_write(db_path, s1_sql, params)
        logger.info(f"Loaded S1: {os.path.basename(f)} ({len(params):,} rows)")

