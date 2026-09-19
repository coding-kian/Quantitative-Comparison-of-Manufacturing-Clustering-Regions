# grid_creation.py
import math
from pyproj import Transformer
from shapely.geometry import Point
from scripts.database_interactions import database_write
from scripts.util_config import total_runtime, logger, EPSG_GEOGRAPHIC, EPSG_PROJECTED

@total_runtime
def grid_creation(db_path: str, bbox: tuple, china_union: object, cell_size: int, batch_size: int) -> None:
    """
    4326 - geographical, spherical coordinate system. (-180 to 180)
    3857 - projected onto 2d uses spherical but has distortion since x-y position.
    Used for creating the grid, it extracts the bounding box for china for the longitude and latitude and the x and y coords
    """
    sql = "INSERT INTO grid (longitude, latitude) VALUES (?, ?);"
    south_lat, west_long, north_lat, east_long = bbox # bounding box

    xy_position = Transformer.from_crs(EPSG_GEOGRAPHIC, EPSG_PROJECTED, always_xy=True).transform
    ll_position = Transformer.from_crs(EPSG_PROJECTED, EPSG_GEOGRAPHIC, always_xy=True).transform

    (x_min, y_min), (x_max, y_max) = xy_position(west_long, south_lat), xy_position(east_long, north_lat)
    (x0, x1), (y0, y1) = sorted((x_min, x_max)), sorted((y_min, y_max))

    n_cols, n_rows = math.ceil((x1-x0) / cell_size), math.ceil((y1-y0) / cell_size)

    rows = []
    is_china = 0
    for row_index in range(n_rows):
        y = y0+(row_index+0.5) * cell_size
        for col_index in range(n_cols):
            x = x0+(col_index+0.5)*cell_size
            longitude, latitude = ll_position(x, y) # centroids of longitude and atitude
            
            if china_union.contains(Point(longitude, latitude)): # makes sure this point is part of china
                rows.append((longitude, latitude)); is_china += 1

            if len(rows) >= batch_size: database_write(db_path, sql, rows); rows.clear()

    if rows: database_write(db_path, sql, rows)

    logger.info(f"{is_china}/ {n_rows*n_cols} = ({n_rows}×{n_cols}) at {cell_size/1000:.0f}km")
