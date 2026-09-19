# osm_data.py
import asyncio, random, math
from collections import defaultdict

import aiohttp
from tqdm import tqdm
from pyproj import Transformer
from scripts.database_interactions import database_write, database_read
from scripts.util_config import total_runtime, EPSG_GEOGRAPHIC, EPSG_PROJECTED

THREADS_AMOUNT, RETRIES = 3, 8
OVERPASS = ["https://overpass-api.de/api/interpreter","https://overpass.kumi.systems/api/interpreter","https://overpass.nchc.org.tw/api/interpreter"]

QUERIES = ('way["highway"~"motorway|trunk|primary|secondary"]',
'way["landuse"="industrial"]',
'way["building"~"industrial|warehouse"]','relation["building"~"industrial|warehouse"]',
'node["harbour"]','way["man_made"="pier"]','relation["man_made"="pier"]')

SQL_SELECT_PREFIX_OSM = "SELECT grid_id, road_km, industrial_area_km2, industrial_building_count, port_count FROM osm_grid WHERE grid_id IN"
SQL_INSERT_REPLACE = "INSERT OR REPLACE INTO osm_grid (grid_id, road_km, industrial_area_km2, industrial_building_count, port_count) VALUES (?, ?, ?, ?, ?);"

def tiles(bbox: tuple, tile_degrees: float = 4.0) -> object:
    """A generator that divides a geographic bounding box into smaller rectangular tiles up to tile_degrees in latitude and longitude"""
    south_lat, west_long, north_lat, east_long = bbox
    latitude = south_lat
    while latitude < north_lat: # goes through all directions
        longitude = west_long
        while longitude < east_long:
            yield (latitude, longitude, min(latitude + tile_degrees, north_lat), min(longitude + tile_degrees, east_long))
            longitude += tile_degrees
        latitude += tile_degrees


def queries(tile: tuple, timeout_server: int = 120) -> str:
    """Creates the string to query the post request"""
    s, w, n, e = tile
    block="\n".join(f"{query}({s},{w},{n},{e});" for query in QUERIES)
    return f"[out:json][timeout:{timeout_server}];({block});out body center;>;out skel qt;"


def grid_index(bbox: tuple, cell_size: int) -> tuple:
    """gets the x and y position of the grid based on the cell."""
    south_lat, west_long, north_lat, east_long = bbox
    xy_position = Transformer.from_crs(EPSG_GEOGRAPHIC, EPSG_PROJECTED, always_xy=True).transform
    (x0, y0), (x1, y1) = xy_position(west_long, south_lat), xy_position(east_long, north_lat)
    (x0, x1), (y0, y1) = sorted((x0, x1)), sorted((y0, y1))
    n_cols = int(math.ceil((x1-x0) / cell_size))
    return xy_position, x0, y0, n_cols


def xy_to_gridid(x: float, y: float, x0: float, y0: float, n_cols: int, cell_size: int) -> float:
    col = int((x - x0) // cell_size)
    row = int((y - y0) // cell_size)
    if row < 0 or col < 0:
        return None
    return row * n_cols + col + 1


def path_length_km(points_xy: list) -> float:
    """Calculates length the path/line from projected points then returns the distance in KM."""
    if len(points_xy) < 2: return 0.0
    return sum(math.hypot(x2-x1, y2-y1) for (x1, y1), (x2, y2) in zip(points_xy, points_xy[1:])) / 1_000


def poly_area_km2(points_xy: list) -> float:
    """Calculates area of a closed polygon from projected points then returns the area in square KM."""
    if len(points_xy) < 4 or points_xy[0] != points_xy[-1]: return 0.0
    return abs(sum(x1*y2 - x2*y1 for (x1, y1), (x2, y2) in zip(points_xy, points_xy[1:]))*0.5)/1_000_000


def fetch_existing(db_path: str, grid_ids, chunk_size: int = 1_000) -> dict:
    out = {}
    if grid_ids:
        grid_ids = list(grid_ids)
        for i in range(0, len(grid_ids), chunk_size):
            part = grid_ids[i:i + chunk_size]
            sql = SQL_SELECT_PREFIX_OSM + '(' + ','.join(['?'] * len(part)) + ');'
            for grid_id, road_km, area_km2, building_count, port_count in database_read(db_path, sql, tuple(part)):
                out[int(grid_id)] = (road_km, area_km2, building_count, port_count)
    return out


async def fetch_json(session: aiohttp.ClientSession, query: str, timeout_client: int = 300, wait_time: int = 90):
    last_error = None
    for attempt in range(RETRIES):
        try:
            url = random.choice(OVERPASS)
            async with session.post(url, data={'data': query}, timeout=aiohttp.ClientTimeout(total=timeout_client)) as response:
                if response.status in (429, 502, 503, 504): # common errors
                    await asyncio.sleep(min(wait_time, (2 ** attempt) + random.random() * 2))
                    continue
                response.raise_for_status()
                return await response.json()
        except Exception as exception:
            last_error = exception
            await asyncio.sleep(min(wait_time, (2 ** attempt) + random.random() * 2))
    raise RuntimeError(f'Overpass failed: {last_error}')


async def process_tile(tile: tuple, session: aiohttp.ClientSession, db_path: str, db_lock: asyncio.Lock, valid: set,
    xy_position: Transformer.from_crs, x0: float, y0: float, n_cols: int, cell_size: int, batch_size: int = 5_000):
    """
    Fetches the map elements for each tile, assigning to a grid cell, aggregating features with the result being merged into the existing database values
    and written in batches
    """
    data = await fetch_json(session, queries(tile))
    elements = data.get('elements', [])

    node_xy = {el['id']: (el['lon'], el['lat']) for el in elements if el.get('type') == 'node'} # gets the id, longitude and latitude for each element
    road_km, area_km2 = defaultdict(float), defaultdict(float)
    building_count, port_count = defaultdict(int), defaultdict(int)

    for el in elements:
        element_type = el.get("type"); tags=el.get("tags") or {} # assigns an empty dictionary if "tags" is not an element of the dictionary
        if element_type == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get('center')
            if not center: continue
            lat, lon = center['lat'], center['lon']
            
        x, y = xy_position(lon, lat) # center positions
        grid_id = xy_to_gridid(x, y, x0, y0, n_cols, cell_size)
        if grid_id not in valid: continue

        if element_type == "way":
            points = [xy_position(node_xy[i][0], node_xy[i][1]) for i in el.get("nodes", []) if i in node_xy]
            if "highway" in tags:
                road_km[grid_id] += path_length_km(points)
            if tags.get("landuse") == "industrial":
                area_km2[grid_id] += poly_area_km2(points)

        if element_type in ("way", "relation") and tags.get("building") in ("industrial", "warehouse"):
            building_count[grid_id] += 1
        if (element_type == "node" and "harbour" in tags) or tags.get("man_made") == "pier":
            port_count[grid_id] += 1
            
    touched = set(road_km)|set(area_km2)|set(building_count)|set(port_count) # set notation
    if not touched: return

    async with db_lock:
        existing = fetch_existing(db_path, touched) # i need to do this to see if there is an overlap in the gridcell then sum them together.. 
        rows = []
        for grid_id in touched:
            old = existing.get(int(grid_id), (0.0,0.0,0,0))
            rows.append((int(grid_id),
                old[0]+float(road_km.get(grid_id,0.0)),
                old[1]+float(area_km2.get(grid_id,0.0)),
                old[2]+int(building_count.get(grid_id,0)),
                old[3]+int(port_count.get(grid_id,0))))
            if len(rows) >= batch_size:
                database_write(db_path, SQL_INSERT_REPLACE, rows); rows.clear()
        if rows: database_write(db_path, SQL_INSERT_REPLACE, rows)

@total_runtime
async def build_osm_grid(db_path: str, bbox: tuple, cell_size: int):
    """Processes all OSM tiles within the bounding box and aggregate features into the gridded database"""
    valid = {i[0] for i in database_read(db_path,"SELECT grid_id FROM grid;",())}
    xy_position, x0, y0, n_cols = grid_index(bbox, cell_size)
    sem = asyncio.Semaphore(THREADS_AMOUNT)
    db_lock = asyncio.Lock()
    all_tiles = list(tiles(bbox))
    pbar = tqdm(total=len(all_tiles), desc="OSM tiles")

    async with aiohttp.ClientSession() as session:
        async def runner(tile):
            async with sem:
                await process_tile(tile, session, db_path, db_lock, valid, xy_position, x0, y0, n_cols, cell_size)
                pbar.update(1)

        await asyncio.gather(*[runner(i) for i in all_tiles])
    pbar.close()
