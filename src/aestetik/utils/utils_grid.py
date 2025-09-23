import numpy as np
import lightning as L
import anndata
import logging
import random
import torch
import os

from joblib import Parallel, delayed
from torch.backends import cudnn
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler

from typing import Dict, List, Optional, Tuple

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

def fix_seed(seed: int) -> None:
    """
    Set all random seeds and configurations for reproducibility.
    
    Parameters
    ----------
    seed: int
        value for all random generators
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    L.seed_everything(seed, workers=True)



def create_st_grid(adata: anndata,
                   used_obsm: str,
                   window_size: int,
                   cpu_count: int,
                   used_obs_sample: Optional[str] = None) -> np.ndarray:
    """
    Creates a grid of features for each spot. Then it stores the result in adata.obsm['st_grid'].
    """
    x_array = adata.obs["x_array"].to_numpy()
    y_array = adata.obs["y_array"].to_numpy()
    embs = adata.obsm[used_obsm]
    embs = MinMaxScaler(feature_range=(0, 1)).fit_transform(embs)

    n_spots, dim_emb = embs.shape

    if used_obs_sample is not None and used_obs_sample in adata.obs.columns:
        sample_labels = adata.obs[used_obs_sample].astype("category").cat.codes.to_numpy()
    else:
        logging.info(
            "No sample label column specified or found in adata.obs. "
            "We treat all data as coming from a single tissue slice."
        )
        sample_labels = np.zeros(len(x_array), dtype=int)
    
    trees, sample_to_indices = _build_trees(x_array=x_array,
                                           y_array=y_array,
                                           sample_labels=sample_labels)
    
    half = window_size // 2
    offsets_flat = _compute_offsets_flat(start=-half, end=-half + window_size)

    x_array.setflags(write=False)
    y_array.setflags(write=False)
    embs.setflags(write=False)
    sample_labels.setflags(write=False)
    offsets_flat.setflags(write=False)

    sample_indices = np.array_split(range(n_spots), cpu_count)
    delayed_create_sample_grid = delayed(_create_sample_grid)

    spot_grid = Parallel(n_jobs=cpu_count,
                       prefer="threads")(
                        delayed_create_sample_grid(spot_indices=spot_indices,
                                                  x_array=x_array,
                                                  y_array=y_array,
                                                  sample_labels=sample_labels,
                                                  embs=embs,
                                                  trees=trees,
                                                  sample_to_indices=sample_to_indices,
                                                  offsets_flat=offsets_flat,
                                                  window_size=window_size) for spot_indices in sample_indices
                       )
    spot_grid = np.concatenate(spot_grid)
    spot_grid = np.moveaxis(spot_grid, 3, 1)
    return spot_grid # shape: (num_spots, dim_emb, window_size, window_size)

def _build_trees(x_array: np.ndarray,
                 y_array: np.ndarray,
                 sample_labels: np.ndarray) -> Tuple[Dict[int, KDTree], Dict[int, np.ndarray]]:
    sample_ids = np.unique(sample_labels)
    trees = dict()
    sample_to_indices = dict()

    for sample_id in sample_ids:
        spot_indices = np.where(sample_labels == sample_id)[0]
        coords = np.column_stack([x_array[spot_indices], y_array[spot_indices]])
        trees[sample_id] = KDTree(coords)
        sample_to_indices[sample_id] = spot_indices 
    
    return trees, sample_to_indices

def _create_sample_grid(spot_indices: np.ndarray,
                       x_array: np.ndarray,
                       y_array: np.ndarray,
                       sample_labels: np.ndarray,
                       embs: np.ndarray,
                       trees: Dict, 
                       sample_to_indices: Dict,
                       offsets_flat: np.ndarray,
                       window_size: int) -> List[np.ndarray]:
    sample_grids = []
    for spot_index in spot_indices:
        spot = _create_spot(spot_idx=spot_index,
                            x_array=x_array,
                            y_array=y_array,
                            sample_labels=sample_labels,
                            embs=embs,
                            trees=trees,
                            sample_to_indices=sample_to_indices,
                            offsets_flat=offsets_flat,
                            window_size=window_size)
        sample_grids.append(spot)
    return sample_grids

def _create_spot(spot_idx: int,
                 x_array: np.ndarray,
                 y_array: np.ndarray,
                 sample_labels: np.ndarray, 
                 embs:np.ndarray,
                 trees: Dict, 
                 sample_to_indices: Dict,
                 offsets_flat: np.ndarray,
                 window_size: int) -> np.ndarray:
    """
    Creates a grid for a single spot.
    """
    x_center, y_center, sample_id = x_array[spot_idx], y_array[spot_idx], sample_labels[spot_idx]
    center = np.array([x_center, y_center]) 
  
    grid = np.full((window_size, window_size, embs.shape[1]), 
                    fill_value=np.nan,
                    dtype=embs.dtype)
    indices_in_sample = sample_to_indices[sample_id]

    for offset_idx, (dx_offset, dy_offset) in enumerate(offsets_flat):
        position = center + np.array([dx_offset, dy_offset])
        distance, neighbor_idx = trees[sample_id].query(position)
        if distance > 0:
            continue
        grid_row, grid_column = np.unravel_index(offset_idx,
                                           shape=(window_size, window_size))

        grid[grid_row, grid_column] = embs[indices_in_sample[neighbor_idx]]
    
    median_spot = np.nanmedian(grid, axis=(0, 1))
    nan_indices = np.where(np.isnan(grid))
    grid[nan_indices] = np.take(median_spot, nan_indices[1])

    return grid

def _compute_offsets_flat(start:int, end:int) -> np.ndarray:
    offsets = np.arange(start, end)
    dx, dy = np.meshgrid(offsets, offsets)
    offsets_flat = np.stack([dx.ravel(), dy.ravel()], axis=1)
    return offsets_flat # shape: (N,2)