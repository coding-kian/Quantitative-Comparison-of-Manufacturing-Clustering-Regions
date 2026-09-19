import numpy as np, pandas as pd
from tqdm import tqdm
import hdbscan
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

from scripts.util_config import logger


def _cluster_hdbscan(embedding: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Clusters the embedding using HDBSCAN a density based clustering method, returing the labels and membership probabilities"""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples = max(2, min_cluster_size // 4),
        metric='euclidean', prediction_data=False).fit(embedding)
    return clusterer.labels_.astype(int)


def pca_cluster(data: np.ndarray, seed: int, min_cluster_size: int, variance: float):
    """PCA reduces noise then cluster results with HDBSCAN"""
    embedding = PCA(n_components=variance, random_state=seed).fit_transform(data)# embedding is used to get the labels the mebedding does not matter
    labels = _cluster_hdbscan(embedding, min_cluster_size)
    return labels


def satellite_cluster(data: np.ndarray, seed: int, min_cluster_size: int, variance: float) -> np.ndarray:
    """
    PCA reducing features to retain the requested proportion of variance.
    Clusters satelite features after PCA has reduced to meet the variance
    """
    x_pca = PCA(n_components=variance, random_state=seed).fit_transform(data)
    labels = _cluster_hdbscan(x_pca, min_cluster_size)
    return labels


def umap_single_cluster(data: np.ndarray, seed: int, umap_dims: int, n_neighbors: int, min_dist: float, min_cluster_size: int,
    sample_size: int, variance: float) -> np.ndarray:
    """
    First preprocesses with PCA then UMAP embedding then finally clusters with HDBSCAN
    First it sets the seed, since UMAP is stochastic, then selects a random samplesize for UMAP to be trained on, then clustsers
    from there trustworthiness for how good the UMAP embeddings are
    """
    rng = np.random.default_rng(seed)
    x_pca = PCA(n_components=variance, random_state=seed).fit_transform(data)
    fit_size = max(2, min(sample_size, len(x_pca))) # maximum size 
    fit_ids = rng.choice(len(x_pca), size=fit_size, replace=False)

    reducer = umap.UMAP(n_components=umap_dims, n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean', random_state=seed)
    reducer.fit(x_pca[fit_ids])
    embedding = reducer.transform(x_pca)

    labels = _cluster_hdbscan(embedding, min_cluster_size)

    is_noise = labels == -1
    trust_ids = rng.choice(len(x_pca), size=min(int(sample_size*0.2), len(x_pca)), replace=False) # random sample to give idea of how good embeddings are
    logger.info('Trustworthiness: %.3f', trustworthiness(x_pca[trust_ids], embedding[trust_ids], n_neighbors=n_neighbors))
    logger.info(f"Clusters/Noise/Labels - {len(set(labels)) - int(is_noise.any())} / {is_noise.sum()} / {len(labels)} ")
    return labels
 

def umap_full_runs(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Takes a long time to run, uses analysis config, it runs UMAP clustering on each of the seeds, then returns all the of the run labels. - stability"""
    run_labels = []
    for seed in tqdm(config['stability_seeds']):
        labels = umap_single_cluster(data, seed, config['umap_dims'], config['n_neighbors'],
            config['min_dist'], config['min_cluster_size'], config['sample_size'], config['variance'])
        run_labels.append(labels)
    return pd.DataFrame(run_labels).T.to_numpy(dtype=int)
