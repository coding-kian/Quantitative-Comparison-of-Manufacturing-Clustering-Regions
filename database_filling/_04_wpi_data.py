import numpy as np
from sklearn.neighbors import BallTree
from scripts.database_interactions import database_write, database_read
from scripts.util_config import EPSG_GEOGRAPHIC, logger
import geopandas as gpd

EARTH_RADIUS_KM = 6371.0088

def load_wpi(db_path: str, china_union: object, wpi_path: str, batch_size: int) -> None:
    """Gets all of the ports from world port index."""
    sql = "INSERT INTO wpi_ports (port_name, longitude, latitude) VALUES (?, ?, ?);"
    gdf = gpd.read_file(wpi_path).to_crs(EPSG_GEOGRAPHIC)

    if not (gdf.geometry.geom_type == "Point").all():
        gdf["geometry"] = gdf.geometry.centroid

    rows = []
    inserted = 0

    for _, row in gdf[gdf.geometry.within(china_union)].iterrows():
        rows.append((row["main_port_name"], row.geometry.x, row.geometry.y))
        if len(rows) >= batch_size:
            inserted += len(rows)
            database_write(db_path, sql, rows); rows.clear()

    if rows:
        inserted += len(rows)
        database_write(db_path, sql, rows); rows.clear()

    logger.info(f"Inserted {inserted} WPI ports into wpi_ports.")


def build_wpi_nearest(db_path: str, cell_size: int, eps: float = 0.05) -> None:
    """Uses Ball tree with haversine distance (spherical distance) to find the shortest path between ports and gridcells."""
    grid = database_read(db_path, "SELECT * FROM GRID", ())
    ports = database_read(db_path, "SELECT port_id, longitude, latitude FROM WPI_PORTS", ())
    roads = database_read(db_path, "SELECT grid_id, road_km FROM OSM_GRID", ())

    road_km_by_id = {int(grid_id): float(road_km or 0.0) for grid_id, road_km in roads}
    ports_latlon = np.radians(np.array([(p[2], p[1]) for p in ports]))
    port_ids = np.array([int(p[0]) for p in ports])
    grid_latlon = np.radians(np.array([(g[2], g[1]) for g in grid]))

    d_rad, idx = BallTree(ports_latlon, metric="haversine").query(grid_latlon, k=1) # finds the nearest ports in each grid cell
    nearest_port_id = port_ids[idx[:, 0]]
    distance_km = (d_rad[:, 0] * EARTH_RADIUS_KM)

    rows = []
    for i, (grid_id, lon, lat) in enumerate(grid):
        road_km = road_km_by_id.get(int(grid_id), 0.0)
        weighted = float(distance_km[i])/max(road_km/((cell_size/1000)**2), eps) # divded by area km2

        rows.append((int(grid_id), int(nearest_port_id[i]), distance_km[i], weighted))
  
    sql = "INSERT INTO wpi_nearest (grid_id, port_id, distance, weighted_distance) VALUES (?, ?, ?, ?)"
    database_write(db_path, sql, rows)
    logger.info(f"wpi nearest inserted/updated {len(rows)} rows")

