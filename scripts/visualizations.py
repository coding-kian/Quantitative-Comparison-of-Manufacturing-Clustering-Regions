from keplergl import KeplerGl
import json, re
from pathlib import Path
from libpysal.weights import KNN

import numpy as np, pandas as pd, geopandas as gpd

from scripts.evaluation_metrics import add_lisa_columns, local_stability_percentage, local_concordance_percentage, final_labels_from_runs, _clustering_alignment
from scripts.util_config import NOISE_LABEL, EPSG_PROJECTED, METHODS, logger, add_density_fields, load_json, point_frame, polygon_frame


LAYERS = (
    ("umap_clusters", "UMAP clusters", "umap_cluster_group", "string", True, "ordinal"),
    ("pca_clusters", "PCA clusters", "pca_cluster_group", "string", False, "ordinal"),
    ("satellite_clusters", "Satellite clusters", "satellite_cluster_group", "string", True, "ordinal"),
    ("local_stability_layer", "Local stability", "local_stability", "real", False, "quantile"),
    ("local_concordance_layer", "Local concordance", "local_concordance_percentage", "real", False, "quantile"))


def save_stability_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    specs = {
        **{f"{method}_local_stability": (75, 90) for method in METHODS},
        **{f"{method}_vs_satellite_local_concordance_percentage": (50, 75)
           for method in ("umap", "pca")}}

    def summary_row(column, thresholds):
        values = pd.to_numeric(df[column], errors="coerce").fillna(0)
        return {"metric": column, "mean": values.mean(), "median": values.median(), "q25": values.quantile(.25), "q75": values.quantile(.75),
            **{f"pct_{cutoff}": values.ge(cutoff).mean() * 100 for cutoff in thresholds}}

    rows = [summary_row(column, thresholds) for column, thresholds in specs.items() if column in df]
    pd.DataFrame(rows).to_csv(f"{out_dir}/stability_summary.csv", index=False)


def save_cluster_count_summary(df: pd.DataFrame, out_dir: str) -> None:
    rows = []
    for method in ("umap", "pca", "satellite"):
        col = f"{method}_cluster"
        total = len(df)

        for cluster_id, count in df[col].value_counts().sort_index().items():
            rows.append({"method": method, "cluster_id": int(cluster_id),
                "cluster_label": "Noise (-1)" if int(cluster_id) == -1 else f"Cluster {int(cluster_id)}",
                "n_cells": int(count), "pct_cells": count/total*100})

    pd.DataFrame(rows).to_csv(f"{out_dir}/cluster_count_summary.csv", index=False)


def save_lisa_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    rows = []
    for metric in ("density", "weighted"):
        class_col = f"{metric}_lisa_class"
        glyph_col = f"{metric}_lisa_glyph"
        total = len(df)
        grouped = (df.groupby([class_col, glyph_col]).size().reset_index(name="n_cells"))

        for _, row in grouped.iterrows():
            rows.append({"metric": metric, "lisa_class": row[class_col],
                "glyph": row[glyph_col], "n_cells": int(row["n_cells"]),
                "pct_cells": row["n_cells"]/total*100})

    pd.DataFrame(rows).to_csv(f"{out_dir}/lisa_class_breakdown.csv", index=False)



## KEPLER.gl
def _range_filter_min_max(series: pd.Series) -> list:
    """Returns a safe minimum and maxmium range for kepler filter for the grid"""
    values = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty: return [0.0, 1.0]
    low, high = values.min(), values.max()
    return [low, high if high > low else low + 1.0]


def _build_kepler_config(gdf, template_path: str, dataset_name: str, tooltips: list, title: str, style_type: str) -> dict:
    """loads the kepler config template, and updates the visualisation information so that it aligns with the requirements nicely"""
    range_filters = {'light_filter': 'light_mean','vv_filter': 'vv_mean', 'vh_filter': 'vh_mean'} # this is for the range filter on the filtering

    polygon_layers = {'umap_clusters', 'pca_clusters', 'satellite_clusters'}
    glyph_layers = {'density_lisa_glyphs'}

    config = load_json(template_path) # all the configs preset
    config_vis = config['config']['visState']

    filters = {i['id']: i for i in config_vis['filters']}
    layers = {i['id']: i for i in config_vis['layers']}

    for filter_config in filters.values():
        if filter_config['id'] != 'selected_cell_filter':
            filter_config['dataId'] = [dataset_name]
            
    for layer_id, layer_config in layers.items():
        if layer_id in polygon_layers|glyph_layers: layer_config['config']['dataId'] = dataset_name

    for method in METHODS:
        layer_id = f'{method}_clusters'
        if layer_id in layers: layers[layer_id]['config']['label'] = f'{method.upper()} clusters'
    filters['stability_filter']['value'] = [0.0, 100.0]

    for filter_id, col in range_filters.items():
        if filter_id in filters and col in gdf.columns:
            filters[filter_id]['value'] = _range_filter_min_max(gdf[col])

    config_vis['interactionConfig']['tooltip']['fieldsToShow'] = {dataset_name: tooltips, 'selected_cell': tooltips}

    config['config']['mapState']['isSplit'] = True
    config['config']['mapStyle']['styleType'] = style_type
    config['config']['title'] = title
    return config

    
def _add_lisa_fields(df: pd.DataFrame, config: dict) -> pd.DataFrame: 
    """For the overlay for the glyph symbols"""
    gdf = point_frame(df).to_crs(EPSG_PROJECTED)
    out = df.copy()
    for value_col, prefix in [('density', 'density'), ('weighted_distance', 'weighted')]:
        gdf, summary = add_lisa_columns(gdf, value_col, prefix, config)
        out[f'{prefix}_lisa_class'] = gdf[f'{prefix}_lisa_class'].values
        out[f'{prefix}_lisa_glyph'] = gdf[f'{prefix}_lisa_glyph'].values
    return out

_label_text = lambda values: ['Noise (-1)' if int(v) == NOISE_LABEL else f'Cluster {int(v)}' for v in values]
_column_name = lambda df, method_name: sorted(i for i in df.columns if i.startswith(f"{method_name}_run_")
    ) or ([f"{method_name}_cluster"] if f"{method_name}_cluster" in df.columns else [])


def _method_stability(df: pd.DataFrame, method_name: str) -> np.ndarray:
    """
    Ensures dataframe has stability column for method then clearns values and returns percentage, 
    if it doesnt then it calculates the stability using the cluster label column instead. 
    """    
    for col in (f'{method_name}_local_stability', f'{method_name}_stability'): # checks as after the first run it will have a stability. 
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').fillna(0.0).to_numpy(float)
            return values*100 if values.max(initial=0.0) <= 1.0 else values

    return local_stability_percentage((df[_column_name(df, method_name)].apply(pd.to_numeric, errors="coerce").fillna(NOISE_LABEL).astype(int).to_numpy()))


# same colour
def _aligned_display_labels(df: pd.DataFrame, reference_df: pd.DataFrame, method_name: str) -> np.ndarray:
    """Align current cluster IDs to the default result so colours remain consistent."""
    cluster_col = f"{method_name}_cluster"

    current = pd.to_numeric(df[cluster_col], errors="coerce").fillna(NOISE_LABEL).astype(int).to_numpy()

    reference = df[["grid_id"]].merge(reference_df[["grid_id", cluster_col]], on="grid_id", how="left")[cluster_col]
    reference = pd.to_numeric(reference, errors="coerce").fillna(NOISE_LABEL).astype(int).to_numpy()
    aligned = _clustering_alignment(reference, current)

    valid_reference = reference[reference != NOISE_LABEL]
    next_label = int(valid_reference.max()) + 1 if valid_reference.size else 0
    unmatched = np.unique(current[(current != NOISE_LABEL) & (aligned == NOISE_LABEL)])

    for label in sorted(unmatched):
        aligned[current == label] = next_label
        next_label += 1

    return aligned


def _config_pair_frame(config: dict, kepler_config: dict, cluster_path: str, left_method: str, right_method: str, cell_size: int) -> gpd.GeoDataFrame:
    """
    Creates a map for comparison, loading cluster data and adding missing density info, calculate cluster labels and stabilitiy scores
    Allows comparison across representatoins, measuring amount of concordance, and marks the grid as so or not. readable cluster labels returned
    """
    df = pd.read_csv(cluster_path).sort_values('grid_id').reset_index(drop=True) # previously saved all clustering results in csv
    reference_df = (pd.read_csv(Path(kepler_config["cluster_results"])).sort_values("grid_id").reset_index(drop=True)) # same colours

    if 'density' not in df.columns: df = add_density_fields(df, cell_size)
    for method_name in METHODS:
        cols = _column_name(df, method_name)
        if cols:
            run_labels = df[cols].apply(pd.to_numeric, errors='coerce').fillna(NOISE_LABEL).astype(int).to_numpy()
            df[f'{method_name}_cluster'] = final_labels_from_runs(run_labels)
            df[f'{method_name}_local_stability'] = _method_stability(df, method_name)

        # if f'{method_name}_cluster' in df.columns: df[f'{method_name}_cluster_group'] = _label_text(df[f'{method_name}_cluster'])

        if f"{method_name}_cluster" in df.columns: df[f"{method_name}_cluster_group"] = _label_text(_aligned_display_labels(df, reference_df, method_name)) # same colours
            
    right_labels = pd.to_numeric(df[f'{right_method}_cluster'], errors='coerce').fillna(NOISE_LABEL).astype(int).to_numpy()
    left_runs = df[_column_name(df, left_method)].apply(pd.to_numeric, errors='coerce').fillna(NOISE_LABEL).astype(int).to_numpy()
    df['local_concordance_percentage'] = local_concordance_percentage(left_runs[:, 0] if left_runs.shape[1] == 1 else left_runs, right_labels)
    df['local_concordance_flag'] = np.where(df['local_concordance_percentage'] >= 50.0, 'Concordant', 'Not concordant')
    df['local_stability'] = np.minimum(df[f'{left_method}_local_stability'], df[f'{right_method}_local_stability'])
    df['diagnostic_state'] = np.select([df['local_stability'] < 75, df['local_concordance_flag'].eq('Not concordant')], ['Instability', 'Divergence'], default='Agreement')

    df['left_cluster_group'] = _label_text(df[f'{left_method}_cluster'])
    df['right_cluster_group'] = _label_text(df[f'{right_method}_cluster'])

    df = df.drop(columns=[col for col in df.columns if '_run_' in col]) # each run saved repeatedly

    return polygon_frame(_add_lisa_fields(df, config), cell_size)


def _add_template_appendix(source_html: str, appendix_path: str) -> None:
    html = Path(source_html).read_text(encoding="utf-8")
    appendix = Path(appendix_path).read_text(encoding="utf-8")
    appendix = appendix.replace("</body>", "").replace("</html>", "")
    position = html.lower().rfind("</body>")
    html = html[:position] + appendix + "\n" + html[position:]
    Path(source_html).write_text(html, encoding="utf-8")


def save_kepler_dual_map(config: dict, kepler_config: dict, cluster_path: str, cell_size: int) -> None:
    """Builds and saved the kepler.gl html map for the pair of representations"""
    dataset_name = kepler_config['dataset_name']

    gdf = _config_pair_frame(config, kepler_config, cluster_path, "umap", "satellite", cell_size)

    for col in gdf.select_dtypes(include='number').columns:
        if col != 'grid_id':
            gdf[col] = gdf[col].round(3)

    built_config = _build_kepler_config(gdf, kepler_config['template_path'], dataset_name, 
        kepler_config['tooltips'], kepler_config['title'], kepler_config['style_type'])

    out_html = f"{kepler_config['out_dir']}/{kepler_config['sub_path']}.html"

    map_widget = KeplerGl(height=900, config=built_config)
    map_widget.add_data(data=json.loads(gdf.to_json()), name=dataset_name)

    # map_widget.add_data(data=json.loads(gdf.to_json()), name='selected_cell') # makes it take alot of data
    tooltip_cols = [i['name'] for i in kepler_config['tooltips'] if i['name'] in gdf.columns]
    selected_gdf = gdf[list(dict.fromkeys(['grid_id', *tooltip_cols, 'geometry']))].copy()
    map_widget.add_data(data=json.loads(selected_gdf.to_json()), name='selected_cell')

    map_widget.save_to_html(file_name=out_html, read_only=False)

    _add_template_appendix(out_html, kepler_config["template_appendix"])

    logger.info(f'kepler.gl saved {out_html}')
