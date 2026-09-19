import itertools
import time
from collections import Counter

import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN
from scripts.util_config import NOISE_LABEL, logger


LISA_CLASS_LABEL = {0: 'Undefined', 1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
LISA_CLASS_GLYPH = {'HH': '■', 'LL': '✚', 'HL': '▲', 'LH': '▼', 'Undefined': ' '}

def moran_and_lisa(gdf: pd.DataFrame, value_col: str, config: dict) -> tuple:
    """
    Creates statical test for distance using euclidean.
    Calculate global and local spatial autocorrelation, then classify each feature into LISA categories
    """
    t1 = time.time()
    weights = KNN.from_array(np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values]), k=config["knn_k"]); logger.info(f"KNN {time.time()-t1}")
    weights.transform = 'R' # row standardize everything reweighted to 1

    moran = Moran(gdf[value_col], weights, permutations=config["moran_permutations"]); logger.info(f"MORAN {time.time()-t1}") # set to 999
    lisa = Moran_Local(gdf[value_col], weights, permutations=config["lisa_permutations"]); logger.info(f"LISA {time.time()-t1}") # takes longer, so set to 499

    return pd.DataFrame([{'metric': value_col, "moran_I": moran.I, "z_sim": moran.z_sim, "p_sim": moran.p_sim}]
        ), np.where(lisa.p_sim < config["lisa_significance"], lisa.q.astype(int), 0).astype(int) # lower probability then  stronger evidence but fewer clusters


def add_lisa_columns(gdf: pd.DataFrame, value_col: str, out_prefix: str, config: dict) -> tuple:
    """Append LISA quadrant, class, and glyph columns to a GeoDataFrame for mapping spatial clusters and outliers."""
    summary, quad = moran_and_lisa(gdf, value_col, config)
    out = gdf.copy() # copies to prevent changing the input geospatialdataframe
    out[f'{out_prefix}_lisa_quad'] = quad
    out[f'{out_prefix}_lisa_class'] = [LISA_CLASS_LABEL[int(v)] for v in quad]
    out[f'{out_prefix}_lisa_glyph'] = [LISA_CLASS_GLYPH[v] for v in out[f'{out_prefix}_lisa_class']]
    return out, summary
    

def _clustering_alignment(reference_labels: np.ndarray, target_labels: np.ndarray) -> np.ndarray:
    """Builds table showing overlap across representations allowing cluster labels to be found.
    Then algins targets to cluster IDs using maximum overlap matching (hungarian method). Just called in this file"""
    reference_labels = np.asarray(reference_labels, dtype=int)
    target_labels = np.asarray(target_labels, dtype=int)

    valid = (reference_labels != NOISE_LABEL) & (target_labels != NOISE_LABEL)
    valid_target = target_labels != NOISE_LABEL
    out = np.full(len(target_labels), NOISE_LABEL, dtype=int)

    if valid.any():
        table = pd.crosstab(pd.Series(reference_labels[valid], name='reference'),
            pd.Series(target_labels[valid], name='target')) # how many observations overlap for each refernce

        rows, cols = linear_sum_assignment(table.to_numpy().max() - table.to_numpy()) # finds one to one assignment, hungarian
        label_map = {int(table.columns[j]): int(table.index[i]) for i, j in zip(rows, cols)} # mapping
        out[valid_target] = [label_map.get(int(v), NOISE_LABEL) for v in target_labels[valid_target]] # replaces unmatched with the noise label

    return out


## BOOSTRAPPING LEADS TO STABILITY METRIC
def _label_to_matrix(labels: np.ndarray) -> np.ndarray:
    """convert 1d label array into 2d run matsrix."""
    labels = np.asarray(labels, dtype=int)
    return labels[:, None] if labels.ndim == 1 else labels # just going to be a label if 1 dimension


_most_common = lambda values: Counter(map(int, values)).most_common(1)[0][0]

def _run_reference_alignment(run_labels: np.ndarray) -> np.ndarray:
    """Aligns runs to the first run so labels are comparable."""
    run_labels = _label_to_matrix(run_labels)
    reference = run_labels[:, 0]
    return np.column_stack([reference] + [_clustering_alignment(reference, run) for run in run_labels[:, 1:].T]) # matrix tranformation is flipping array
    

def local_stability_percentage(run_labels: np.ndarray) -> np.ndarray:
    """Calculates percentage of runs where a grid cell keeps the same aligned cluster label."""
    aligned = _run_reference_alignment(run_labels)
    if aligned.shape[1] == 1:
        return np.where(aligned[:, 0] == NOISE_LABEL, 0.0, 100.0)

    scores = []
    for row in aligned:
        valid = row[row != NOISE_LABEL]
        scores.append(0 if len(valid) == 0 else (valid == _most_common(valid)).mean()*100)
    return np.asarray(scores, dtype=float)

def final_labels_from_runs(run_labels: np.ndarray) -> np.ndarray:
    """Final cluster label for each grid cell using the most common label across all runs."""
    return np.array([_most_common(valid) if len(valid := row[row != NOISE_LABEL]) else NOISE_LABEL for row in _run_reference_alignment(run_labels)], dtype=int)

    
### CONCORDANCE/AGREEMENT/ARI (ADJUSTED RAND INDEX)
def _calculate_ari(left: np.ndarray, right: np.ndarray) -> tuple:
    """Calculates ARI when both labels across representations are not noise labels"""
    valid = (left != NOISE_LABEL) & (right != NOISE_LABEL)
    score = adjusted_rand_score(left[valid], right[valid]) if valid.sum() >= 2 else np.nan
    return score, int(valid.sum())


def _ari_summary(df: pd.DataFrame, label_col: str, label_value: str, count_col: str) -> pd.DataFrame:
    """Summarises ARI results using mean, standard deviation, and comparison count."""
    scores = df.get('ari', pd.Series(dtype=float)).dropna()
    return pd.DataFrame([{label_col: label_value, 'mean_ari': float(scores.mean()), 'std_ari': float(scores.std(ddof=0)), count_col: int(scores.size)}])


def local_concordance_percentage(left_run_labels: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
    """Calculates how often each grid cell agrees with the reference cluster across runs (first clustering run)."""
    reference = np.asarray(reference_labels, dtype=int)

    scores = []
    for run in _label_to_matrix(left_run_labels).T: # matrix transformation
        aligned = _clustering_alignment(reference, run)
        valid = (aligned != NOISE_LABEL) & (reference != NOISE_LABEL)
        scores.append(np.where(valid, aligned == reference, 0.0))

    return np.mean(scores, axis=0) * 100.0
    

def pairwise_run_stability(run_labels: np.ndarray, method_name: str) -> tuple:
    """Compares every pair of clustering runs for one method using ARI to measure global stability."""
    run_labels = _label_to_matrix(run_labels)
    rows = [] # pairwise
    for left_idx, right_idx in itertools.combinations(range(run_labels.shape[1]), 2):
        score, n_overlap = _calculate_ari(run_labels[:, left_idx], run_labels[:, right_idx])

        rows.append({'left_run': left_idx + 1, 'right_run': right_idx + 1, 'ari': score, 'n_overlap': n_overlap})
    pairwise = pd.DataFrame(rows)
    return pairwise, _ari_summary(pairwise, 'method', method_name, 'n_pairs')


def representation_concordance(left_run_labels: np.ndarray, right_labels: np.ndarray, left_name: str, right_name: str) -> pd.DataFrame:
    """Compares each run from one representation against another representation using ARI."""
    left_run_labels = _label_to_matrix(left_run_labels)
    right_labels = np.asarray(right_labels, dtype=int)
    rows = []
    for run_idx in range(left_run_labels.shape[1]):
        score, n_overlap = _calculate_ari(left_run_labels[:, run_idx], right_labels)
        rows.append({'left_method': left_name, 'right_method': right_name, 'run': run_idx + 1, 'ari': score, 'n_overlap': n_overlap})
    detail = pd.DataFrame(rows)
    return _ari_summary(detail, 'comparison', f'{left_name} vs {right_name}', 'n_runs')
