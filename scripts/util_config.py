import json, sqlite3, logging, time
from pathlib import Path

import pandas as pd, numpy as np, geopandas as gpd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="| %(message)s")

METHODS = ("umap", "pca", "satellite")
NOISE_LABEL = -1
EPSG_GEOGRAPHIC = 'EPSG:4326'
EPSG_PROJECTED = 'EPSG:3857'
EPS = 1e-9

SQL_GRID_FEATURES = """ SELECT 
    g.grid_id,
    g.longitude,
    g.latitude,
    COALESCE(cgmd.total_manufacturers, 0) AS total_manufacturers,
    COALESCE(cgmd.industrial_zone_count, 0) AS industrial_zone_count,
    COALESCE(osm.road_km, 0) AS road_km,
    COALESCE(osm.industrial_area_km2, 0) AS industrial_area_km2,
    COALESCE(osm.industrial_building_count, 0) AS industrial_building_count,
    COALESCE(osm.port_count, 0) AS port_count,
    COALESCE(wpi.distance, 0) AS distance,
    COALESCE(wpi.weighted_distance, 0) AS weighted_distance,
    COALESCE(viirs.light_mean, 0) AS light_mean,
    COALESCE(sentinel.vv_mean, 0) AS vv_mean,
    COALESCE(sentinel.vh_mean, 0) AS vh_mean
FROM grid AS g
LEFT JOIN cgmd_grid cgmd ON g.grid_id = cgmd.grid_id
LEFT JOIN osm_grid osm ON g.grid_id = osm.grid_id
LEFT JOIN wpi_nearest wpi ON g.grid_id = wpi.grid_id
LEFT JOIN gee_viirs viirs ON g.grid_id = viirs.grid_id
LEFT JOIN gee_sentinel1 sentinel ON g.grid_id = sentinel.grid_id
ORDER BY g.grid_id;
""" # COALESCE returns the first value that is not NULL.


def total_runtime(funct: object) -> object:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = funct(*args, **kwargs)
        wrapper.running = time.time() - start_time
        return result
    return wrapper


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_feature_frame(db_path: str) -> pd.DataFrame:
    """Loads the database joined together with all of the features needed for the research and fills missing values. CONCATENATE"""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(SQL_GRID_FEATURES, conn)
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

    
def add_density_fields(df: pd.DataFrame, cell_size_m: int) -> pd.DataFrame:
    """Adds manufacturer density to the dataset based on industrial area^2 and total manufacturers."""
    area = pd.to_numeric(df['industrial_area_km2'], errors='coerce').fillna(0.0)
    manu = pd.to_numeric(df['total_manufacturers'], errors='coerce').fillna(0.0)

    cell_area_km2 = (cell_size_m/1000)**2
    df['density'] = manu/cell_area_km2 # main densirt manufacturers per km2
    df['industrial_intensity'] = manu/(area+1) # +1 avoids exploding value, and manufacturers relative to mapped industrial land.
    return df


def point_frame(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Converts longitude and latitude columns into a point geodataframe."""
    return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs=EPSG_GEOGRAPHIC)


def polygon_frame(df: pd.DataFrame, cell_size: int) -> gpd.GeoDataFrame:
    """Converts grid-cell centre points into a square polygon using longitude and latitude."""
    points = point_frame(df)
    projected = points.to_crs(EPSG_PROJECTED)
    polygons = projected.geometry.buffer(cell_size / 2.0, cap_style=3)
    points['geometry'] = gpd.GeoSeries(polygons, crs=EPSG_PROJECTED).to_crs(EPSG_GEOGRAPHIC)
    return points
