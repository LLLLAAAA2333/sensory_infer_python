import os
import numpy as np
import h5py
import json
from typing import Optional, Tuple, Union
from tqdm import tqdm

VolumeInput = Union[str, np.ndarray, list]


def _apply_zrange_and_depth(volume: np.ndarray, zrange: Optional[Tuple[int, int]], max_depth: Optional[int]) -> np.ndarray:
    """Apply zrange slicing and enforce an optional depth ceiling."""
    if zrange is not None:
        z_start, z_end = zrange
        z_start = max(z_start, 0)
        if z_end == -1 or z_end > volume.shape[2]:
            z_end = volume.shape[2]
        if z_start >= z_end:
            raise ValueError("Invalid zrange: start must be less than end")
        volume = volume[:, :, z_start:z_end]

    if max_depth is not None:
        if max_depth <= 0:
            raise ValueError("max_depth must be positive when provided")
        if volume.shape[2] > max_depth:
            volume = volume[:, :, :max_depth]
    return volume

def rescale_image(image, target_min, target_max, source_min=None, source_max=None):
    if target_min >= target_max:
        raise ValueError("target_min must be less than target_max")
    if source_min is None:
        source_min = np.min(image)
    if source_max is None:
        original_max = np.max(image)
        if original_max >= 1000:
            source_max = np.max(image[image < 1000])
        else:
            source_max = original_max
    image_float32 = image.astype(np.float32)
    if source_min >= source_max:
        return np.full(image.shape, target_min, dtype=image.dtype)
        
    return np.clip((image_float32 - source_min) / (source_max - source_min) * (target_max - target_min) + target_min,
                             target_min, target_max).astype(image.dtype)

def create_zephir_data(volume_input: VolumeInput, zephir_folder, chunk_size=100, zrange=None, max_depth: Optional[int] = None):
    """
    Create ZephIR `data.h5` files from a volume source.

    volume_input may be one of:
    - a list of .npy file paths (each containing a (Y, X, Z) volume)
    - a numpy ndarray of shape (T, Y, X, Z) or (Y, X, Z)
    - a string path to a directory that contains `aligned_volumes_mip.npy` or a set
      of .npy files (this is to support the MIP workflow where aligned volumes
      and `neuron_pt_tuple.npy` are saved together in one folder).

    Returns (num_volumes, shape) where shape is (Z, Y, X).
    """
    os.makedirs(zephir_folder, exist_ok=True)

    # If a directory path is provided, try to locate aligned_volumes_mip.npy
    if isinstance(volume_input, str):
        if os.path.isdir(volume_input):
            candidate = os.path.join(volume_input, 'aligned_volumes_mip.npy')
            if os.path.exists(candidate):
                try:
                    arr = np.load(candidate)
                    volume_input = arr
                except Exception:
                    # fall back to collecting .npy files in the folder
                    files = sorted([os.path.join(volume_input, f) for f in os.listdir(volume_input) if f.endswith('.npy')])
                    if not files:
                        raise ValueError(f"No aligned_volumes_mip.npy and no .npy files found in {volume_input}")
                    volume_input = files
            else:
                files = sorted([os.path.join(volume_input, f) for f in os.listdir(volume_input) if f.endswith('.npy')])
                if not files:
                    raise ValueError(f"No .npy files found in directory {volume_input}")
                volume_input = files
        elif os.path.isfile(volume_input):
            # single file path provided
            volume_input = np.load(volume_input)
        else:
            raise ValueError(f"Path {volume_input} is not a file or directory")

    is_list = isinstance(volume_input, list)
    if is_list:
        num_volumes = len(volume_input)
        # load first to determine shape
        v0 = np.load(volume_input[0])
        v0 = _apply_zrange_and_depth(v0, zrange, max_depth)
        shape = (v0.shape[2], v0.shape[0], v0.shape[1])  # (Z, Y, X)
    else:
        arr = np.asarray(volume_input)
        if arr.ndim == 3:
            # single (Y, X, Z) volume -> make it a single-frame sequence
            arr = arr[np.newaxis, ...]
        if arr.ndim != 4:
            raise ValueError(f"Unsupported ndarray shape for volume_input: {arr.shape}")
        # arr is (T, Y, X, Z)
        num_volumes = arr.shape[0]
        sample = _apply_zrange_and_depth(arr[0], zrange, max_depth)
        shape = (sample.shape[2], sample.shape[0], sample.shape[1])  # (Z, Y, X)

    print(f"Processing {num_volumes} volumes with shape {shape} (Z, Y, X)")

    num_chunks = (num_volumes + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_t = chunk_idx * chunk_size
        end_t = min(start_t + chunk_size, num_volumes)

        subfolder_name = f"vol_{start_t}_{end_t-1}"
        subfolder_path = os.path.join(zephir_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        chunk_vol_num = end_t - start_t
        img_data = np.empty((chunk_vol_num, 1, shape[0], shape[1], shape[2]), dtype=np.uint8)

        for i in range(chunk_vol_num):
            idx = start_t + i
            if is_list:
                vol = np.load(volume_input[idx])
            else:
                vol = arr[idx]

            vol = _apply_zrange_and_depth(vol, zrange, max_depth)

            # vol is (Y, X, Z) -> transpose to (Z, Y, X)
            vol = np.transpose(vol, (2, 0, 1))

            vol_scaled = rescale_image(vol, 0, 255).astype(np.uint8)
            img_data[i, 0] = vol_scaled

        data_path = os.path.join(subfolder_path, 'data.h5')
        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset('data', data=img_data, chunks=True)

    return num_volumes, shape

def create_zephir_annotations(neuron_pt_tuple, zephir_folder, shape, chunk_size=100, z_ratio=5.0):
    # neuron_pt_tuple: (T, N, F) or (N, F)
    if neuron_pt_tuple.ndim == 2:
        neuron_pt_tuple = neuron_pt_tuple[np.newaxis, ...]
        
    total_t = neuron_pt_tuple.shape[0]
    num_chunks = (total_t + chunk_size - 1) // chunk_size
    
    depth, height, width = shape # Z, Y, X
    
    for chunk_idx in range(num_chunks):
        start_t = chunk_idx * chunk_size
        end_t = min(start_t + chunk_size, total_t)
        
        subfolder_name = f"vol_{start_t}_{end_t-1}"
        subfolder_path = os.path.join(zephir_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        h5file_path = os.path.join(subfolder_path, 'annotations.h5')
        if os.path.exists(h5file_path):
            os.remove(h5file_path)
            
        with h5py.File(h5file_path, 'w') as f:
            max_shape = (None,)
            ds_x = f.create_dataset('/x', shape=(0,), maxshape=max_shape, dtype='float32')
            ds_y = f.create_dataset('/y', shape=(0,), maxshape=max_shape, dtype='float32')
            ds_z = f.create_dataset('/z', shape=(0,), maxshape=max_shape, dtype='float32')
            ds_id = f.create_dataset('/id', shape=(0,), maxshape=max_shape, dtype='uint32')
            ds_parent = f.create_dataset('/parent_id', shape=(0,), maxshape=max_shape, dtype='uint16')
            ds_worldline = f.create_dataset('/worldline_id', shape=(0,), maxshape=max_shape, dtype='uint16')
            ds_prov = f.create_dataset('/provenance', shape=(0,), maxshape=max_shape, dtype='S4')
            ds_t = f.create_dataset('/t_idx', shape=(0,), maxshape=max_shape, dtype='uint32')
            
            current_offset = 0
            
            for local_t in range(end_t - start_t):
                global_t = start_t + local_t
                points = neuron_pt_tuple[global_t]
                
                valid_mask = ~np.isnan(points[:, 0])
                valid_points = points[valid_mask]
                
                if valid_points.shape[0] == 0:
                    continue
                
                n_points = valid_points.shape[0]
                
                # Assuming coords are (X, Y, Z)
                # Wait, in pipeline `translation_matching_bruteforce_with_dist` returns (row, col) -> (Y, X).
                # And `shift_vector` is (shift_yx[1], shift_yx[0], 0) -> (X, Y, Z).
                # So coords are (X, Y, Z).
                
                x = valid_points[:, 0]
                y = valid_points[:, 1]
                z = valid_points[:, 2]
                
                # ZephIR expects normalized coordinates
                # x_rel = x / width
                # y_rel = y / height
                # z_rel = z / (z_ratio * depth)
                
                ds_x.resize((current_offset + n_points,))
                ds_y.resize((current_offset + n_points,))
                ds_z.resize((current_offset + n_points,))
                ds_id.resize((current_offset + n_points,))
                ds_parent.resize((current_offset + n_points,))
                ds_worldline.resize((current_offset + n_points,))
                ds_prov.resize((current_offset + n_points,))
                ds_t.resize((current_offset + n_points,))
                
                ds_x[current_offset:] = x / width
                ds_y[current_offset:] = y / height
                ds_z[current_offset:] = z / (z_ratio * depth)
                
                indices = np.where(valid_mask)[0]
                ds_worldline[current_offset:] = indices.astype(np.uint16)
                ds_id[current_offset:] = np.arange(current_offset + 1, current_offset + n_points + 1)
                ds_t[current_offset:] = local_t
                
                current_offset += n_points

def create_metadata_json(zephir_folder, num_volumes, shape, chunk_size=100):
    num_chunks = (num_volumes + chunk_size - 1) // chunk_size
    depth, height, width = shape
    
    for chunk_idx in range(num_chunks):
        start_t = chunk_idx * chunk_size
        end_t = min(start_t + chunk_size, num_volumes)
        chunk_vol_num = end_t - start_t
        
        subfolder_name = f"vol_{start_t}_{end_t-1}"
        subfolder_path = os.path.join(zephir_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        metadata = {
            "shape_t": chunk_vol_num,
            "shape_c": 1,
            "shape_z": depth,
            "shape_y": height,
            "shape_x": width,
            "dtype": "uint8"
        }
        
        with open(os.path.join(subfolder_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


def convert_npy_to_ZephIR_format(
    volume_input: VolumeInput,
    neuron_pt_tuple_source: Union[str, np.ndarray],
    zephir_folder: str,
    chunk_size: int = 100,
    zrange: Optional[Tuple[int, int]] = None,
    z_ratio: float = 5.0,
    max_depth: Optional[int] = 20,
):
    """Convert numpy volumes and neuron coordinates into ZephIR chunked datasets."""

    if isinstance(neuron_pt_tuple_source, str):
        if not os.path.exists(neuron_pt_tuple_source):
            raise FileNotFoundError(f"Cannot find neuron_pt_tuple source: {neuron_pt_tuple_source}")
        neuron_pt_tuple = np.load(neuron_pt_tuple_source)
    else:
        neuron_pt_tuple = np.asarray(neuron_pt_tuple_source)

    neuron_pt_tuple = np.copy(neuron_pt_tuple)
    if neuron_pt_tuple.ndim == 2:
        neuron_pt_tuple = neuron_pt_tuple[np.newaxis, ...]

    if max_depth is not None:
        upper = max_depth - 1
        if upper < 0:
            raise ValueError("max_depth must be >= 1 when provided")
        neuron_pt_tuple[..., 2] = np.clip(neuron_pt_tuple[..., 2], 0, upper)

    num_volumes, shape = create_zephir_data(
        volume_input,
        zephir_folder,
        chunk_size=chunk_size,
        zrange=zrange,
        max_depth=max_depth,
    )
    create_zephir_annotations(
        neuron_pt_tuple,
        zephir_folder,
        shape,
        chunk_size=chunk_size,
        z_ratio=z_ratio,
    )
    create_metadata_json(zephir_folder, num_volumes, shape, chunk_size=chunk_size)
    return shape
