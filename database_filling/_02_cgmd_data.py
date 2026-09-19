# cgmd_data.py
import fiona
import math
from shapely.geometry import shape
from scripts.database_interactions import database_write, database_read

EARTH_MEAN_RADIUS = 6_378_137.0
mercator_x_axis = lambda longitude: EARTH_MEAN_RADIUS*math.radians(longitude) # longitude is simpler since lines evenly spread
mercator_y_axis = lambda latitude: EARTH_MEAN_RADIUS*math.log(math.tan(math.pi/4 + math.radians(max(min(latitude, 85.0), -85.0))/2.0))
# mercator projection is a cylinder map of the earth maintaining angles but distorting distances

def export_cgmd_to_db(db_path: str, bbox: tuple, cgmd_gdb: str, layer: str, cell_size: int, batch_size: int) -> None:
    """Gets CGMD into the database"""
    sql = """INSERT INTO cgmd_grid (grid_id, total_manufacturers, industrial_zone_count) VALUES (?, ?, ?)
    ON CONFLICT(grid_id) DO UPDATE SET
        total_manufacturers = total_manufacturers + excluded.total_manufacturers,
        industrial_zone_count = industrial_zone_count + excluded.industrial_zone_count;
    """

    south_lat, west_long, north_lat, east_long = bbox # bounding box
    
    x0, y0 = mercator_x_axis(west_long), mercator_x_axis(south_lat)

    rows = database_read(db_path,"SELECT grid_id, longitude, latitude FROM grid;",())
    grid_xy_to_id, insert_cgmd_data = dict(), dict()
    for grid_id, longitude, latitude in rows:
        x, y = mercator_x_axis(longitude), mercator_y_axis(latitude)
        col = int((x - x0) // cell_size)
        row = int((y - y0) // cell_size)
        grid_xy_to_id[(row, col)] = int(grid_id)

    matched, skipped = 0, 0
    
    with fiona.open(cgmd_gdb, layer=layer) as src:
        for total_scanned, feature in enumerate(src.filter(where="SUM19 > 0")):
            geometric_shape = feature.get("geometry")
            if not geometric_shape: continue
            centroid = shape(geometric_shape).centroid
            cent_longitude, cent_latitude = centroid.x, centroid.y

            if not (west_long <= cent_longitude <= east_long and south_lat <= cent_latitude <= north_lat): continue

            sum19 = float((feature.get("properties") or {}).get("SUM19") or 0)
            if sum19 <= 0: continue # total manufactures in this gridcell in cgmd
            col, row = int((mercator_x_axis(cent_longitude)-x0)//cell_size), int((mercator_y_axis(cent_latitude)-y0)//cell_size)

            grid_id = grid_xy_to_id.get((row, col))
            if grid_id is None: # does not exist in this grid. 
                skipped+=1; continue

            insert_cgmd_data.setdefault(grid_id, [0, 0])
            insert_cgmd_data[grid_id][0] += sum19
            insert_cgmd_data[grid_id][1] += 1
            matched+=1

            if len(insert_cgmd_data) >= batch_size:
                batch = [(id_, int(round(v[0])), int(v[1])) for id_, v in insert_cgmd_data.items()]
                database_write(db_path, sql, batch)
                print(f"CGMD & {len(batch):,} grids | scanned={total_scanned:,} matched={matched:,} skipped={skipped:,}")
                insert_cgmd_data.clear()

    if insert_cgmd_data:
        batch = [(id_, int(round(v[0])), int(v[1])) for id_, v in insert_cgmd_data.items()]
        database_write(db_path, sql, batch)
        print(f"CGMD & {len(batch):,} grids (final) | scanned={total_scanned:,} matched={matched:,} skipped={skipped:,}")
