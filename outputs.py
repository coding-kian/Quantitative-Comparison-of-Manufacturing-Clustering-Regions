import asyncio, time
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler 

import pandas as pd, numpy as np, geopandas as gpd

from scripts.database_interactions import database_create
from database_filling._01_grid_creation import grid_creation
from database_filling._02_cgmd_data import export_cgmd_to_db
from database_filling._03_osm_data import build_osm_grid
from database_filling._04_wpi_data import build_wpi_nearest, load_wpi
from database_filling._05_gee_data import gee_to_db, run_satellite_exports

from scripts.evaluation_metrics import (add_lisa_columns, local_stability_percentage, local_concordance_percentage, final_labels_from_runs,
    pairwise_run_stability, representation_concordance)
from scripts.representation_clustering import umap_single_cluster, umap_full_runs, pca_cluster, satellite_cluster
from scripts.util_config import (total_runtime, add_density_fields, load_feature_frame, load_json, point_frame, 
    logger, NOISE_LABEL, EPSG_PROJECTED, EPSG_GEOGRAPHIC, METHODS)
from scripts.visualizations import save_kepler_dual_map, save_stability_breakdown, save_cluster_count_summary, save_lisa_breakdown

STRUCTURAL_FEATURES = ['total_manufacturers', 'industrial_zone_count', 'road_km',
    'industrial_area_km2', 'industrial_building_count', 'port_count', 'weighted_distance', 'density', 'industrial_intensity']
SATELLITE_FEATURES = ['vv_mean', 'vh_mean', 'light_mean']
BASE_COLUMNS = ['grid_id', 'longitude', 'latitude', *STRUCTURAL_FEATURES, *SATELLITE_FEATURES,
    'density_lisa_class', 'density_lisa_glyph', 'weighted_lisa_class', 'weighted_lisa_glyph']

def build_database(config: dict) -> None:
    """Builds the SQL database when the data generation is enabled in the config."""
    db_path, gee_config = config['db_path'], config['gee']
    bbox, cell_size, batch_size = tuple(config['bbox']), config['cell_size_m'], config['batch_size']

    # Loads China's boundary box and returns a single combined shape using ISO_A3 for the allowed regions
    world = gpd.read_file(config['china_boundary_path']).to_crs(EPSG_GEOGRAPHIC)
    china_union = world[world['ISO_A3'] == 'CHN'].geometry.union_all()

    database_create(db_path, config['schema_path'])
    grid_creation(db_path, bbox, china_union, cell_size, batch_size); logger.info(f"Grid creation in {grid_creation.running:.1f}")
    export_cgmd_to_db(db_path, bbox, config['cgmd_path'], config['cgmd_layer'], cell_size, batch_size)
    asyncio.run(build_osm_grid(db_path, bbox, cell_size));logger.info(f"OSM in {build_osm_grid.running:.1f}")
    load_wpi(db_path, china_union, config['wpi_path'], batch_size)
    build_wpi_nearest(db_path, cell_size)
    
    run_satellite_exports(gee_config['project_id'], db_path, gee_config['year'],
        gee_config['drive_folder'], gee_config['chunk_size']); logger.info(f"Remote-Sensing in {run_satellite_exports.running:.1f}")
    # print("Move Satelite Data to local storage from google drive once the task has finished completing, check google earth engine tasks.`")
    gee_to_db(db_path, config['gee_exports_folder']); logger.info(f"GEE database in {gee_to_db.running:.1f}")


def _cluster_labels(data: np.ndarray, method: str, config: dict, seed: int) -> np.ndarray:
    """Runs one clustering method and returns the labels associated to it"""
    min_cluster_size = config['min_cluster_size']
    if method == 'umap':
        return umap_single_cluster(data, seed, config['umap_dims'], config['n_neighbors'],
            config['min_dist'], config['min_cluster_size'], config['sample_size'], config['variance'])
    if method == 'pca':
        return pca_cluster(data, seed, min_cluster_size, config['variance'])
    return satellite_cluster(data, seed, min_cluster_size, config['variance'])


def _bootstrap_method(data: np.ndarray, method: str, config: dict) -> dict:
    """Runs repeated subsamples on the method, returning the label and the sample mask, it is bootstrapping method for stability."""
    n, n_runs = len(data), config["bootstrap_runs"]
    rng = np.random.default_rng(config["stability_seeds"][0])
    seeds = rng.choice(np.arange(1_000_000), size=n_runs, replace=False)
    sample_size = max(2, int(round(config["bootstrap_rate"] * n)))

    run_labels = np.full((n, n_runs), NOISE_LABEL, dtype=int)
    sample_masks = np.zeros((n, n_runs), dtype=bool)

    logger.info("Bootstrapping")
    for run_idx, seed in enumerate(tqdm(seeds, desc=f"{method} bootstrap")): # fixed number of seeds
        logger.info(f"Current bootstrap is {run_idx}")
        sample_id = np.sort(rng.choice(n, size=sample_size, replace=False))
        sample_masks[sample_id, run_idx] = True
        run_labels[sample_id, run_idx] = _cluster_labels(data[sample_id], method, config, int(seed)) # since it is a numpy int

    return {'run_labels': run_labels, 'sample_masks': sample_masks}


def _scaled_features(df: pd.DataFrame, cols: list, log_cols: list) -> np.ndarray:
    """log transforms features to prevent being skewed, and then standarises"""
    x = df[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors='coerce').fillna(0.0)
        if col in log_cols: x[col] = np.log1p(x[col].clip(lower=0.0))
    return StandardScaler().fit_transform(x.to_numpy(dtype=float))


@total_runtime
def run_analysis(config: dict) -> None:
    """
    Runs the full analysis and framework, generating, metrics, figures and interactive visualisation. Loads all grid level features
    calculating density, weighted distance and LISA diagnostics, performs PCA, UMAP and satelite clustering, evaluates stability concordance and spatial connectivity.
    """
    analysis_config = config['analysis']
    kepler_config = config['kepler']
    cell_size = config['cell_size_m']
    out_dir = config['out_dir']
    cluster_path = f'{out_dir}/cluster_results.csv'

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(kepler_config['out_dir']).mkdir(parents=True, exist_ok=True)

    df = add_density_fields(load_feature_frame(config['db_path']), config['cell_size_m'])
    lisa_gdf, density_moran = add_lisa_columns(point_frame(df).to_crs(EPSG_PROJECTED), 'density', 'density', analysis_config)
    lisa_gdf, weighted_moran = add_lisa_columns(lisa_gdf, 'weighted_distance', 'weighted', analysis_config)
    df = pd.DataFrame(lisa_gdf.drop(columns='geometry'))
    
    manifold_scaled = _scaled_features(df, STRUCTURAL_FEATURES, STRUCTURAL_FEATURES)
    satellite_scaled = _scaled_features(df, SATELLITE_FEATURES, STRUCTURAL_FEATURES+['light_mean'])

    umap_runs = umap_full_runs(manifold_scaled, analysis_config)
    pca_labels = pca_cluster(manifold_scaled, analysis_config['stability_seeds'][0], analysis_config['min_cluster_size'], analysis_config['variance'])
    satellite_labels = satellite_cluster(satellite_scaled, analysis_config['stability_seeds'][0], analysis_config['min_cluster_size'], analysis_config['variance'])

    umap_boot = _bootstrap_method(manifold_scaled, 'umap', analysis_config)
    pca_boot = _bootstrap_method(manifold_scaled, 'pca', analysis_config)
    satellite_boot = _bootstrap_method(satellite_scaled, 'satellite', analysis_config)

    method_data = { # used to find the percentage of stability
        'umap': {'labels': final_labels_from_runs(umap_runs), 'runs': umap_runs, 'stability': local_stability_percentage(umap_runs)},
        'pca': {'labels': pca_labels, 'runs': pca_boot['run_labels'], 'stability': local_stability_percentage(pca_boot['run_labels'])},
        'satellite': {'labels': satellite_labels, 'runs': satellite_boot['run_labels'], 'stability': local_stability_percentage(satellite_boot['run_labels'])}}
    # final labels from runs for the complete bootstrapped for full data clustering results, the stability oclumn shows how robust over repepeated runs
    cluster_results = df[BASE_COLUMNS].copy()

    for method_name, values in method_data.items():
        for idx in range(values['runs'].shape[1]):
            cluster_results[f'{method_name}_run_{idx + 1}'] = values['runs'][:, idx]

        cluster_results[f'{method_name}_cluster'] = values['labels']
        cluster_results[f'{method_name}_local_stability'] = values['stability']

    cluster_results['umap_vs_satellite_local_concordance_percentage'] = local_concordance_percentage(umap_runs, satellite_labels)
    cluster_results['pca_vs_satellite_local_concordance_percentage'] = local_concordance_percentage(pca_boot['run_labels'], satellite_labels)

    cluster_results.to_csv(cluster_path, index=False)
    save_cluster_count_summary(cluster_results, out_dir)
    save_lisa_breakdown(cluster_results, out_dir)
    save_stability_breakdown(cluster_results, out_dir)

    concordance_summary = []
    for method_name, run_labels in [('umap', umap_runs), ('pca', pca_boot['run_labels'])]:
        summary = representation_concordance(run_labels, satellite_labels, method_name, 'satellite')
        concordance_summary.append(summary)
    concordance_summary = pd.concat(concordance_summary, ignore_index=True)
    concordance_summary.to_csv(f'{out_dir}/representation_concordance.csv', index=False)

    stability_summary = []
    for method_name, run_labels in [('umap', umap_boot['run_labels']), ('pca', pca_boot['run_labels']), ('satellite', satellite_boot['run_labels'])]:
        _, summary = pairwise_run_stability(run_labels, method_name)
        stability_summary.append(summary)
    stability_summary = pd.concat(stability_summary, ignore_index=True)
    stability_summary.to_csv(f'{out_dir}/global_stability_summary.csv', index=False)

    pd.concat([density_moran, weighted_moran], ignore_index=True).to_csv(f'{out_dir}/feature_moran_lisa_summary.csv', index=False)

    logger.info('Saved main research outputs to %s', out_dir)

    save_kepler_dual_map(analysis_config, kepler_config, cluster_path, cell_size)


if __name__ == '__main__':
    config = load_json('config/project_config.json')
    # build_database(config)
    run_analysis(config); logger.info(run_analysis.running)
