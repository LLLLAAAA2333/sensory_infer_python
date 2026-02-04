import os
import sys
import time
import json
import re
import argparse
import numpy as np
import pandas as pd
import h5py
from glob import glob
from tqdm import tqdm
import torch
from torch.nn import functional as F
import src.plot_result.vis_trajectory as vis
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.infer_sequence import run_inference_on_volume_sequence
from src.infer_single import run_inference_on_single_volume
from src.inference.volume_alignment import translate_matrix, volume_alignment, compute_distance_fft, compute_3d_shift
from src.inference.blur import get_image4processing, pixel_threshold
from src.inference.intensity_extract import extract_neuron_intensities_torch
from src.merge_resize_inference import Treeformer_End2End
from src.comm_utils.prints import print_info_message, print_log_message, print_warning_message
from src.zephir import zephir_utils


def _compute_z_ratio(config_dict):
    xoy_unit = config_dict.get('xoy_unit')
    z_unit = config_dict.get('z_unit')
    if xoy_unit and z_unit:
        try:
            return float(z_unit) / float(xoy_unit)
        except (ZeroDivisionError, TypeError, ValueError):
            print_warning_message("Failed to compute z_ratio from config; falling back to default 5.0")
    return 5.0


def export_volumes_to_zephir(volume_source, neuron_pt_tuple_source, zephir_path, z_ratio, zrange, max_depth=20, denoise_range=(102, 1000)):
    os.makedirs(zephir_path, exist_ok=True)
    print_info_message(f"Converting inference outputs to ZephIR format at {zephir_path}...")
    zephir_utils.convert_npy_to_ZephIR_format(
        volume_source,
        neuron_pt_tuple_source,
        zephir_path,
        zrange=zrange,
        z_ratio=z_ratio,
        max_depth=max_depth,
        denoise_range=denoise_range,
    )
    print_info_message(f"ZephIR data saved to {zephir_path}")


def clamp_neuron_depth(array, depth_limit=20):
    if array is None or depth_limit is None:
        return array
    if depth_limit <= 0:
        raise ValueError("depth_limit must be positive")
    if array.ndim < 2 or array.shape[-1] < 6:
        return array
    np.clip(array[..., 5], 10, depth_limit, out=array[..., 5])
    return array


def clamp_neuron_depth_file(file_path, depth_limit=20):
    if not file_path or not os.path.exists(file_path):
        return None
    data = np.load(file_path)
    clamp_neuron_depth(data, depth_limit)
    np.save(file_path, data)
    return data


def apply_zrange(volume_np, zrange=None):
    """Slice volume along Z axis using the provided [start, end) range."""
    if zrange is None:
        return volume_np
    z_start, z_end = zrange
    z_start = max(z_start, 0)
    if z_end == -1 or z_end > volume_np.shape[2]:
        z_end = volume_np.shape[2]
    if z_start == 0 and z_end == volume_np.shape[2]:
        return volume_np
    return volume_np[:, :, z_start:z_end]

def load_datapath(folder_path):
    """
    Load several .npy files from several folders
    """
    if not os.path.isdir(folder_path):
        return []
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
    file_paths.sort(key=natural_sort_key)
    return file_paths


def translation_matching_fft_with_dist(binary_image1, binary_image2, shiftrange=(21, 21), device='cuda'):
    """
    Args:
        (Y, X, Z) format GPU Tensors.
        Return (row_shift, col_shift) and min_distance.
    """

    t1 = time.time()

    if binary_image1.shape != binary_image2.shape:
        print_warning_message(f"MIP shape mismatch {binary_image1.shape} vs {binary_image2.shape}. Skipping alignment.")
        return

    rows, cols = binary_image2.shape
    distance_matrix = compute_distance_fft(binary_image1, binary_image2, rows, cols, shiftrange)
    t2 = time.time()
    
    min_distance, min_idx = torch.min(distance_matrix.reshape(-1), 0)
    min_distance_index = np.unravel_index(min_idx.cpu().numpy(), distance_matrix.shape)

    shift_yx = (min_distance_index[0] - shiftrange[0] // 2, min_distance_index[1] - shiftrange[1] // 2)
    t3 = time.time()
    
    print_log_message(f"[FFT] Dist: {t2-t1:.4f}s, Post: {t3-t2:.4f}s")
    return shift_yx, min_distance.item()

def load_ex_vol_gpu(path, cache, device='cuda', zrange=None):
    if path not in cache:
        vol_data = np.load(path)
        
        # Filter out high intensity noise
        if vol_data.max() >=10000:
             print_warning_message(f"Found high intensity values (>= 10000) in {os.path.basename(path)}. Clamping to median.")
             vol_data[vol_data >= 10000] = np.median(vol_data).astype(vol_data.dtype)

        vol_data = apply_zrange(vol_data, zrange)
        if vol_data.dtype == np.uint16:
            vol_data = vol_data.astype(np.float32)
        cache[path] = torch.from_numpy(vol_data).to(device).float()
    return cache[path]

def interpolate_and_extract(ref_coords, ex_vol_folders, ref_vol_paths, output_dir,
                            mode='interpolate', device='cuda', shiftrange=(21, 21), zrange=None, align_preproc='dog', z_ratio=1.0):
    """
    Args:
        ref_coords: numpy array of shape (T_ref, N_neurons, F_features)
        ex_vol_folders: list of folders, each containing experimental .npy volumes
        output_dir: directory to save outputs
        mode: 'interpolate', 'align', or 'dual_propagate'
        shiftrange: tuple for bruteforce mode and align mode
        align_preproc: 'dog' (Difference of Gaussians) or 'afp' (Automatic Foreground Process)
    """
    T_ref, N_neurons, F_features = ref_coords.shape
    print_log_message(f"Loaded reference matrix: {T_ref} ref volumes, {N_neurons} unique neurons, {F_features} features.")
    print_info_message(f"Running in mode: '{mode}'")
    matching_func = translation_matching_fft_with_dist
    preproc_method = 'dog' if align_preproc == 'dog' else 'threshold'

    if len(ex_vol_folders) != T_ref - 1:
        print_warning_message(f"Mismatch! Found {T_ref} ref volumes but {len(ex_vol_folders)} experimental volume folders. Expected {T_ref - 1} folders.")

    if mode in ('align', 'dual_propagate') and len(ref_vol_paths) != T_ref:
        print_warning_message(f"Align/Propagate mode error: Need {T_ref} ref vol paths, but found {len(ref_vol_paths)}. Falling back to 'interpolate'.")
        mode = 'interpolate'
    
    # cache for phasecorr mode
    ref_vols_gpu_cache = {}
    def get_ref_vol_gpu(idx):
        if idx not in ref_vols_gpu_cache:
            try:
                vol_data = np.load(ref_vol_paths[idx])
                vol_data = apply_zrange(vol_data, zrange)
                if vol_data.dtype == np.uint16:
                    vol_data = vol_data.astype(np.float32)
                ref_vols_gpu_cache[idx] = torch.from_numpy(vol_data).to(device).float()
            except Exception as e:
                print_warning_message(f"Error loading ref_vol {ref_vol_paths[idx]}: {e}. Cannot use 'phasecorr'.")
                return None
        return ref_vols_gpu_cache[idx]
    
    ex_folder_files = [load_datapath(folder) for folder in ex_vol_folders]
    ex_seg_counts = np.array([len(files) for files in ex_folder_files], dtype=np.int32)

    # all_intensities_df = pd.DataFrame(index=range(N_neurons))
    all_intensities_list = []
    global_frame_counter = 0
    ex_tuples = []

    num_segments = min(T_ref - 1, len(ex_vol_folders))
    total_volumes = sum(len(ex_folder_files[idx]) for idx in range(num_segments))
    volume_progress = tqdm(total=total_volumes, desc="Processing volumes", unit="vol") if total_volumes else None

    for t in range(num_segments):
        ref_start_coords = ref_coords[t]
        ref_end_coords = ref_coords[t + 1]

        current_mode = mode

        ex_files = ex_folder_files[t]
        num_ex_vols = len(ex_files)
        if num_ex_vols == 0:
            print_warning_message(f"No .npy files found in {ex_vol_folders[t]}. Skipping this folder.")
            continue
        
        if current_mode == 'align' or current_mode == 'dual_propagate':
            ref_A_vol_gpu = get_ref_vol_gpu(t)
            ref_B_vol_gpu = get_ref_vol_gpu(t + 1)
            if ref_A_vol_gpu is None or ref_B_vol_gpu is None:
                print_warning_message(f"Segment {t}: Failed to load ref vols, falling back to 'interpolate'.")
                current_mode = 'interpolate'

        segment_coords = np.full((num_ex_vols, N_neurons, F_features), np.nan, dtype=np.float32)
        if current_mode == 'dual_propagate':
            print_log_message(f"Running dual propagation for segment {t} ({num_ex_vols} frames)...")
            t_start_align = time.time()

            print_log_message(f"  Aligning ref_A to ex_vol_0...")
            vol_k_minus_1_gpu = load_ex_vol_gpu(ex_files[0], {}, device, zrange=zrange)
            
            shift_xyz, dist = compute_3d_shift(ref_A_vol_gpu, vol_k_minus_1_gpu, matching_func, shiftrange, z_ratio, device, method=preproc_method)
            current_f_coords = ref_start_coords.copy()
            # shift_xyz is [dx, dy, dz]. Coords are [x, y, z].
            shift_vector = np.array([shift_xyz[0], shift_xyz[1], shift_xyz[2]], dtype=current_f_coords.dtype)
            current_f_coords[:, :3] -= shift_vector

            forward_coords = np.full((num_ex_vols, N_neurons, F_features), np.nan, dtype=np.float32)
            forward_coords[0] = current_f_coords.copy()

            for k in range(1, num_ex_vols):
                vol_k_gpu = load_ex_vol_gpu(ex_files[k], {}, device, zrange=zrange)

                shift_xyz, dist = compute_3d_shift(vol_k_minus_1_gpu, vol_k_gpu, matching_func, shiftrange, z_ratio, device, method=preproc_method)
                shift_vector = np.array([shift_xyz[0], shift_xyz[1], shift_xyz[2]], dtype=current_f_coords.dtype)
                current_f_coords[:, :3] -= shift_vector
                forward_coords[k] = current_f_coords.copy()
                vol_k_minus_1_gpu = vol_k_gpu

            del vol_k_gpu, vol_k_minus_1_gpu

            print_log_message(f"  Aligning ref_B to ex_vol_{num_ex_vols - 1}...")
            vol_k_plus_1_gpu = load_ex_vol_gpu(ex_files[-1], {}, device, zrange=zrange)
            # vol_k_plus_1 matched against ref_B
            # matching_func(ref_B, vol_k_plus_1) -> shift of vol_k_plus_1 relative to ref_B
            shift_xyz, dist = compute_3d_shift(ref_B_vol_gpu, vol_k_plus_1_gpu, matching_func, shiftrange, z_ratio, device, method=preproc_method)

            current_b_coords = ref_end_coords.copy()
            shift_vector = np.array([shift_xyz[0], shift_xyz[1], shift_xyz[2]], dtype=current_b_coords.dtype)
            current_b_coords[:, :3] -= shift_vector
            backward_coords = np.full((num_ex_vols, N_neurons, F_features), np.nan, dtype=np.float32)
            backward_coords[-1] = current_b_coords.copy()

            for k in range(num_ex_vols - 2, -1, -1):
                vol_k_gpu = load_ex_vol_gpu(ex_files[k], {}, device, zrange=zrange)
                # Align vol_k against vol_k_plus_1
                shift_xyz, dist = compute_3d_shift(vol_k_plus_1_gpu, vol_k_gpu, matching_func, shiftrange, z_ratio, device, method=preproc_method)
                shift_vector = np.array([shift_xyz[0], shift_xyz[1], shift_xyz[2]], dtype=current_b_coords.dtype)
                current_b_coords[:, :3] -= shift_vector
                backward_coords[k] = current_b_coords.copy()
                vol_k_plus_1_gpu = vol_k_gpu
            
            del vol_k_gpu, vol_k_plus_1_gpu
            torch.cuda.empty_cache()

            print_log_message(f"  Blending forward and backward propagation...")
            for k in range(num_ex_vols):
                ratio = k / (num_ex_vols - 1.0) if num_ex_vols > 1 else 0.5
                # get the mean of forward and backward coords
                f_coords = forward_coords[k]
                b_coords = backward_coords[k]
                f_valid = ~np.isnan(f_coords[:, 0])
                b_valid = ~np.isnan(b_coords[:, 0])
                blended_coords = np.full((N_neurons, F_features), np.nan, dtype=np.float32)
                both_valid = f_valid & b_valid
                blended_coords[both_valid] = (1.0 - ratio) * f_coords[both_valid] + ratio * b_coords[both_valid]
                only_f = f_valid & ~b_valid
                blended_coords[only_f] = f_coords[only_f]
                only_b = ~f_valid & b_valid
                blended_coords[only_b] = b_coords[only_b]
                segment_coords[k] = blended_coords
            
            t_end_align = time.time()
            print_log_message(f"Alignment (dual_propagate) for segment {t} took {t_end_align - t_start_align:.4f}s")


        for k, ex_file_path in enumerate(ex_files):
            interp_pt_tuple = None

            if current_mode == 'interpolate':
                interp_ratio = (k + 1.0) / (num_ex_vols + 1.0)
                coord_features = [0,1,2]
                interp_coords = ref_start_coords[:, coord_features] + (ref_end_coords[:, coord_features] - ref_start_coords[:, coord_features]) * interp_ratio
                interp_pt_tuple = ref_start_coords.copy()
                interp_pt_tuple[:, coord_features] = interp_coords

            elif current_mode == 'align':
                ex_vol_k_data = np.load(ex_file_path)
                ex_vol_k_data = apply_zrange(ex_vol_k_data, zrange)
                if ex_vol_k_data.dtype == np.uint16:
                    ex_vol_k_data = ex_vol_k_data.astype(np.float32)
                ex_vol_k_gpu = torch.from_numpy(ex_vol_k_data).to(device).float()
                
                t_start_align = time.time()
                # align with ref volumes (3D)
                shift_A_xyz, dist_A = compute_3d_shift(ref_A_vol_gpu, ex_vol_k_gpu, matching_func, shiftrange, z_ratio, device, method=preproc_method)
                shift_B_xyz, dist_B = compute_3d_shift(ref_B_vol_gpu, ex_vol_k_gpu, matching_func, shiftrange, z_ratio, device, method=preproc_method)
                t_end_align = time.time()
                print_log_message(f"Alignment (align mode) for frame {k} took {t_end_align - t_start_align:.4f}s")
                coords_A = ref_start_coords.copy()
                coords_B = ref_end_coords.copy()

                shift_vec_A = np.array([shift_A_xyz[0], shift_A_xyz[1], shift_A_xyz[2]], dtype=coords_A.dtype)
                coords_A[:, :3] -= shift_vec_A

                shift_vec_B = np.array([shift_B_xyz[0], shift_B_xyz[1], shift_B_xyz[2]], dtype=coords_B.dtype)
                coords_B[:, :3] -= shift_vec_B

                # dists = np.array([dist_A, dist_B], dtype=np.float64)
                # finite_mask = np.isfinite(dists)
                # if not finite_mask.any():
                #     interp_pt_tuple = coords_A
                # else:
                #     adjusted = np.zeros_like(dists)
                #     min_dist = np.nanmin(dists[finite_mask])
                #     adjusted[finite_mask] = np.exp(-(dists[finite_mask] - min_dist))
                #     weight_sum = np.sum(adjusted)

                #     if weight_sum <= 0.0:
                #         interp_pt_tuple = coords_A if dist_A <= dist_B else coords_B
                #     else:
                #         ratio_A = adjusted[0] / weight_sum
                #         ratio_B = adjusted[1] / weight_sum

                ratio = k / (num_ex_vols - 1.0) if num_ex_vols > 1 else 0.5
                interp_pt_tuple = np.full_like(coords_A, np.nan)
                valid_A = ~np.isnan(coords_A[:, 0])
                valid_B = ~np.isnan(coords_B[:, 0])
                both_valid = valid_A & valid_B
                if both_valid.any():
                    interp_pt_tuple[both_valid] = (1.0 - ratio) * coords_A[both_valid] + ratio * coords_B[both_valid]
                only_A = valid_A & ~valid_B
                if only_A.any():
                    interp_pt_tuple[only_A] = coords_A[only_A]
                only_B = ~valid_A & valid_B
                if only_B.any():
                    interp_pt_tuple[only_B] = coords_B[only_B]
                if np.isnan(interp_pt_tuple[:, 0]).all():
                    interp_pt_tuple = coords_A if dist_A <= dist_B else coords_B

            elif current_mode == 'dual_propagate':
                interp_pt_tuple = segment_coords[k]

            # handle nan situations
            mask_nan_interp = np.isnan(interp_pt_tuple[:,0])
            mask_not_nan_end = ~np.isnan(ref_end_coords[:,0])
            mask_not_nan_start = ~np.isnan(ref_start_coords[:,0])

            use_end = mask_nan_interp & mask_not_nan_end
            interp_pt_tuple[use_end] = ref_end_coords[use_end]
            use_start = mask_nan_interp & mask_not_nan_start
            interp_pt_tuple[use_start] = ref_start_coords[use_start]

            ex_tuples.append(interp_pt_tuple.astype(np.float32))
            # filter nan neurons in both start and end
            final_valid_mask = ~np.isnan(interp_pt_tuple[:, 0])
            valid_interp_tuple = interp_pt_tuple[final_valid_mask]

            ex_vol_data = np.load(ex_file_path)
            ex_vol_data = apply_zrange(ex_vol_data, zrange)

            t_start_extract = time.time()
            intensity_values, intensity_indices, _ = extract_neuron_intensities_torch(
                ex_vol_data,
                valid_interp_tuple,
                area_ratio=0.8,
                background_threshold=0,
                device=device,
            )
            t_end_extract = time.time()
            print_log_message(f"Extraction for frame {k} took {t_end_extract - t_start_extract:.4f}s")
            if isinstance(intensity_values, torch.Tensor):
                intensity_values = intensity_values.detach().cpu().numpy()
            if isinstance(intensity_indices, torch.Tensor):
                intensity_indices = intensity_indices.detach().cpu().numpy()

            intensity_values = np.asarray(intensity_values)
            intensity_indices = np.asarray(intensity_indices)
            if intensity_indices.size == 0:
                # all_intensities_df[global_frame_counter] = pd.Series(np.nan, index=range(N_neurons))
                all_intensities_list.append(pd.Series(np.nan, index=range(N_neurons)))
                global_frame_counter += 1
                continue
            
            if intensity_indices.dtype != np.int64 and intensity_indices.dtype != np.int32 and intensity_indices.dtype != np.bool_:
                intensity_indices = intensity_indices.astype(np.int64)
            
            valid_count = int(np.sum(final_valid_mask))
            if intensity_indices.dtype != np.bool_:
                in_range = (intensity_indices >= 0) & (intensity_indices < valid_count)
                intensity_indices = intensity_indices[in_range]
                intensity_values = intensity_values[in_range]
            
            vol_intensity = pd.Series(np.nan, index=range(N_neurons))
            original_indices = np.where(final_valid_mask)[0][intensity_indices]
            vol_intensity.iloc[original_indices] = intensity_values
            # all_intensities_df[global_frame_counter] = vol_intensity
            all_intensities_list.append(vol_intensity)
            global_frame_counter += 1

            if volume_progress is not None:
                volume_progress.set_postfix({"segment": f"{t+1}/{num_segments}", "frame": f"{k+1}/{num_ex_vols}"}, refresh=False)
                volume_progress.update(1)

    if volume_progress is not None:
        volume_progress.close()
        

    if all_intensities_list:
        all_intensities_df = pd.concat(all_intensities_list, axis=1)
        all_intensities_df.columns = range(len(all_intensities_list))
    else:
        all_intensities_df = pd.DataFrame(index=range(N_neurons))
    # save as CSV
    output_csv_path = os.path.join(output_dir, "neuron_intensities_extracted.csv")
    all_intensities_df.to_csv(output_csv_path, index_label="neuron_index")
    # save as h5
    match = re.search(r'w(\d+)', output_dir)
    if match:
        prefix = match.group(0)  
        file_name = prefix + '_trace.h5'
        file_path = os.path.join(output_dir, file_name)
    else:
        file_path = os.path.join(output_dir, 'trace.h5')
    
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset('intensity', data=all_intensities_df.values)
        h5f.create_dataset('n_seg', data=ex_seg_counts.astype(np.int64))
    print_log_message(f"Saved extracted intensities to {output_csv_path} and {file_path}.")

    # save all interpolated experimental coords (exclude ref)
    if len(ex_tuples) > 0:
        ex_neuron_pt_tuple = np.stack(ex_tuples, axis=0)  # shape: (T_ex, N_neurons, F_features)
        ex_coords_path = os.path.join(output_dir, "ex_neuron_pt_tuple.npy")
        np.save(ex_coords_path, ex_neuron_pt_tuple)
        print_log_message(f"Saved interpolated experimental coords to {ex_coords_path} with shape {ex_neuron_pt_tuple.shape}.")
    else:
        ex_neuron_pt_tuple = None
        print_warning_message("No experimental volumes processed; ex_neuron_pt_tuple not saved.")
    
    return ex_neuron_pt_tuple, all_intensities_df

def run_mip_inference_and_extract(
    ex_vol_folders,
    config_path,
    output_dir,
    zrange=None,
    transfer_zephir_path=None,
    skip_inference=False,
    neuron_depth_limit=20,
    zephir_z_ratio=5.0,
    **kwargs,
):
    """
    Args:
        ex_vol_folders: list of folders, each containing experimental .npy volumes
        config_path: path to model configuration JSON file
        output_dir: directory to save outputs
        kwargs: additional arguments for interpolate_and_extract
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    device = kwargs.get('device', 'cuda')
    
    # get experiment volume paths
    ex_folder_files = [load_datapath(folder) for folder in ex_vol_folders]
    ex_seg_counts = np.array([len(files) for files in ex_folder_files], dtype=np.int32)

    ex_vol_paths = []
    for files in ex_folder_files:
        ex_vol_paths.extend(files)
    
    if not ex_vol_paths:
        print_warning_message("No .npy files found in any experimental volume folders. Stopping MIP mode.")
        return None, None, []
    
    print_log_message(f"Found {len(ex_vol_paths)} total experimental volumes for MIP processing.")
    
    synthetic_volume_save_path = os.path.join(output_dir, "synthetic_mip_results")
    os.makedirs(synthetic_volume_save_path, exist_ok=True)

    shiftrange = kwargs.get('shiftrange', (101, 101))
    align_preproc = kwargs.get('align_preproc', 'dog')
    preproc_method = 'dog' if align_preproc == 'dog' else 'threshold'
    aligned_path = os.path.join(synthetic_volume_save_path, 'aligned_volumes_mip.npy')
    shift_path = os.path.join(synthetic_volume_save_path, 'shift_list.npy')
    neuron_tuple_path = os.path.join(synthetic_volume_save_path, 'neuron_pt_tuple.npy')

    aligned_volumes_mip = np.load(aligned_path)
    if skip_inference:
        if not os.path.exists(aligned_path) or not os.path.exists(neuron_tuple_path):
            raise FileNotFoundError("Missing aligned_volumes_mip.npy or neuron_pt_tuple.npy in synthetic_mip_results. Cannot skip inference.")
        aligned_volumes_mip = np.load(aligned_path)
        neuron_pt_tuple_np = np.load(neuron_tuple_path)
        shift_list = np.load(shift_path) if os.path.exists(shift_path) else None
        print_log_message("Loaded precomputed MIP inference artifacts.")
    else:
        print_log_message("Running volume alignment to create artificial MIP...")
        aligned_volumes_mip, shift_list = volume_alignment(
            ex_vol_paths,
            synthetic_volume_save_path,
            shiftrange=shiftrange,
            zrange=zrange,
            method=preproc_method,
        )
        np.save(aligned_path, aligned_volumes_mip)
        np.save(shift_path, shift_list)
        print_log_message(f"Artificial MIP and shift list saved to {synthetic_volume_save_path}")

        print_log_message("Running inference on artificial MIP...")
        neuron_pt_tuple_np = run_inference_on_single_volume(aligned_volumes_mip, config_path, synthetic_volume_save_path, **kwargs)
        print_log_message("Inference on artificial MIP completed.")

    clamp_neuron_depth(neuron_pt_tuple_np, neuron_depth_limit)
    np.save(neuron_tuple_path, neuron_pt_tuple_np)

    if transfer_zephir_path:
        export_volumes_to_zephir(
            aligned_volumes_mip,
            neuron_pt_tuple_np,
            transfer_zephir_path,
            z_ratio=zephir_z_ratio,
            zrange=None,
            max_depth=neuron_depth_limit,
        )
        return None, None, ex_vol_paths
    
    if shift_list is None:
        raise FileNotFoundError("Shift list not found; cannot extract intensities in MIP mode.")

    print_log_message("Extracting 3D intensities from all experimental volumes using shifts...")
    num_neurons = neuron_pt_tuple_np.shape[0]
    all_intensities_list = []
    ex_neuron_pt_tuple_list = []

    for index, file_path in enumerate(tqdm(ex_vol_paths, desc="Extracting 3D Intensities")):
        current_coords = neuron_pt_tuple_np.copy()
        if index > 0:
            # shift_list[index-1] corresponds to volume[index] vs volume[0]
            shift = shift_list[index - 1] 
            row_shift = -shift[1]
            col_shift = -shift[0]
            current_coords[:, 0] += row_shift
            current_coords[:, 1] += col_shift
        ex_neuron_pt_tuple_list.append(current_coords)

        ex_vol_data = np.load(file_path)
        ex_vol_data = apply_zrange(ex_vol_data, zrange)
        intensity_values, intensity_indices, _ = extract_neuron_intensities_torch(
            ex_vol_data,
            current_coords,
            area_ratio=0.8,
            background_threshold=0,
            device=device,
        )

        vol_intensity = pd.Series(np.nan, index=range(num_neurons))
        if intensity_values.size > 0:
            vol_intensity.iloc[intensity_indices] = intensity_values

        all_intensities_list.append(vol_intensity)
    
    all_intensities_df = pd.concat(all_intensities_list, axis=1)
    all_intensities_df.columns = range(len(ex_vol_paths))

    output_csv_path = os.path.join(output_dir, "neuron_intensities_extracted.csv")
    all_intensities_df.to_csv(output_csv_path, index_label="neuron_index")
    match = re.search(r'w(\d+)', output_dir)
    if match:
        prefix = match.group(0)  
        file_name = prefix + '_trace.h5'
        file_path = os.path.join(output_dir, file_name)
    else:
        file_path = os.path.join(output_dir, 'trace.h5')
    
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset('intensity', data=all_intensities_df.values)
        h5f.create_dataset('n_seg', data=ex_seg_counts.astype(np.int64))
    print_log_message(f"Saved extracted intensities to {output_csv_path} and {file_path}.")

    ex_neuron_pt_tuple = np.stack(ex_neuron_pt_tuple_list, axis=0)
    ex_coords_path = os.path.join(output_dir, "ex_neuron_pt_tuple.npy")
    np.save(ex_coords_path, ex_neuron_pt_tuple)
    print_log_message(f"Saved aligned experimental coords for video to {ex_coords_path}")

    return ex_neuron_pt_tuple, all_intensities_df, ex_vol_paths

def generate_experiment_volume_video(
    ex_volumes,
    ex_neuron_pt_tuple,
    save_dir,
    fps=1,
    z_ratio=5.0,
    source_min=102,
    source_max=200,
    default_bbox_size=(6.0, 6.0, 6.0),
    red_pseudo_color=False,
    zrange=None,
):
    """
    Render a volume video with neuron overlays from experimental volumes.
    Args:
        ex_volumes: sequence of np.ndarray (T, Y, X, Z) or list of .npy paths.
        ex_neuron_pt_tuple: np.ndarray (T, N, F) with at least XYZ columns.
        save_dir: output directory for frames/video produced by Plot3DResult.
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(ex_volumes, np.ndarray):
        total_frames = ex_volumes.shape[0]
        def load_frame(idx):
            return ex_volumes[idx]
    else:
        ex_volume_paths = list(ex_volumes)
        total_frames = len(ex_volume_paths)
        def load_frame(idx):
            volume = np.load(ex_volume_paths[idx])
            return apply_zrange(volume, zrange)

    ex_neuron_pt_tuple = np.asarray(ex_neuron_pt_tuple)
    if ex_neuron_pt_tuple.shape[0] != total_frames:
        raise ValueError(f"Frame mismatch: {total_frames} volumes vs {ex_neuron_pt_tuple.shape[0]} neuron frames.")

    def mip_fetcher(idx):
        raw_volume = load_frame(idx)
        if raw_volume.ndim != 3:
            raise ValueError(f"Volume at index {idx} must be 3-D, got shape {raw_volume.shape}.")
        volume_zyx = np.transpose(raw_volume, (2, 0, 1))
        volume_u8 = np.clip(volume_zyx, source_min, source_max)
        volume_u8 = ((volume_u8 - source_min) / (source_max - source_min) * 255).astype(np.uint8)
        return vis.get_mip_from_uint8_gray_volume(
            volume_u8,
            x_ratio=1.0,
            y_ratio=1.0,
            z_ratio=z_ratio,
            source_min=source_min,
            source_max=source_max,
            red_pseudo_color=red_pseudo_color,
        )

    def neuron_fetcher(idx):
        frame_pts = np.asarray(ex_neuron_pt_tuple[idx])
        if frame_pts.ndim != 2 or frame_pts.shape[1] < 3:
            raise ValueError(f"Neuron data at frame {idx} must be (N, F>=3), got {frame_pts.shape}.")
        valid_mask = ~np.isnan(frame_pts[:, 0])
        frame_pts = frame_pts[valid_mask]
        if frame_pts.size == 0:
            return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.int32)
        bbox = np.zeros((frame_pts.shape[0], 6), dtype=np.float32)
        bbox[:, :3] = frame_pts[:, :3]
        if frame_pts.shape[1] >= 6:
            bbox[:, 3:6] = frame_pts[:, 3:6]
        else:
            bbox[:, 3] = default_bbox_size[0]
            bbox[:, 4] = default_bbox_size[1]
            bbox[:, 5] = default_bbox_size[2]
        neuron_ids = np.arange(bbox.shape[0], dtype=np.int32)
        return bbox, neuron_ids

    vis.Plot3DResult(
        mip_fetcher,
        neuron_fetcher,
        split_number=1,
        bbox_thickness=1,
        trace_length=0,
    ).save_all(save_dir, total_frames, fps)

def extract_intensities_only(
    ex_vol_folders,
    ex_neuron_pt_tuple,
    output_dir,
    device='cuda',
    zrange=None
):
    """
    Extract intensities using existing coordinates (ex_neuron_pt_tuple).
    Skips alignment.
    """
    ex_folder_files = [load_datapath(folder) for folder in ex_vol_folders]
    ex_seg_counts = np.array([len(files) for files in ex_folder_files], dtype=np.int32)
    ex_files = []
    for files in ex_folder_files:
        ex_files.extend(files)
    
    num_ex_vols = len(ex_files)
    num_coords_frames, N_neurons, _ = ex_neuron_pt_tuple.shape
    
    if num_ex_vols != num_coords_frames:
        print_warning_message(f"Frame count mismatch: {num_ex_vols} volumes vs {num_coords_frames} coordinate frames. Using minimum.")
    
    num_frames = min(num_ex_vols, num_coords_frames)
    all_intensities_list = []
    
    print_log_message(f"Starting intensity extraction for {num_frames} volumes using existing coordinates...")

    for k in tqdm(range(num_frames), desc="Extracting intensities (only-extract mode)"):
        ex_file_path = ex_files[k]
        coords = ex_neuron_pt_tuple[k]
        
        ex_vol_data = np.load(ex_file_path)
        ex_vol_data = apply_zrange(ex_vol_data, zrange)
        
        final_valid_mask = ~np.isnan(coords[:, 0])
        valid_interp_tuple = coords[final_valid_mask]
        
        intensity_values, intensity_indices, _ = extract_neuron_intensities_torch(
            ex_vol_data,
            valid_interp_tuple,
            area_ratio=0.8,
            background_threshold=0,
            device=device,
        )
        
        if isinstance(intensity_values, torch.Tensor):
            intensity_values = intensity_values.detach().cpu().numpy()
        if isinstance(intensity_indices, torch.Tensor):
            intensity_indices = intensity_indices.detach().cpu().numpy()
            
        intensity_values = np.asarray(intensity_values)
        intensity_indices = np.asarray(intensity_indices)
        
        if intensity_indices.size == 0:
            all_intensities_list.append(pd.Series(np.nan, index=range(N_neurons)))
            continue

        if intensity_indices.dtype != np.int64 and intensity_indices.dtype != np.int32 and intensity_indices.dtype != np.bool_:
            intensity_indices = intensity_indices.astype(np.int64)

        valid_count = int(np.sum(final_valid_mask))
        if intensity_indices.dtype != np.bool_:
            in_range = (intensity_indices >= 0) & (intensity_indices < valid_count)
            intensity_indices = intensity_indices[in_range]
            intensity_values = intensity_values[in_range]

        vol_intensity = pd.Series(np.nan, index=range(N_neurons))
        original_indices = np.where(final_valid_mask)[0][intensity_indices]
        vol_intensity.iloc[original_indices] = intensity_values
        all_intensities_list.append(vol_intensity)

    if all_intensities_list:
        all_intensities_df = pd.concat(all_intensities_list, axis=1)
        all_intensities_df.columns = range(len(all_intensities_list))
    else:
        all_intensities_df = pd.DataFrame(index=range(N_neurons))

    output_csv_path = os.path.join(output_dir, "neuron_intensities_extracted.csv")
    all_intensities_df.to_csv(output_csv_path, index_label="neuron_index")
    
    match = re.search(r'w(\d+)', output_dir)
    if match:
        prefix = match.group(0)  
        file_name = prefix + '_trace.h5'
        file_path = os.path.join(output_dir, file_name)
    else:
        file_path = os.path.join(output_dir, 'trace.h5')
    
    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset('intensity', data=all_intensities_df.values)
        h5f.create_dataset('n_seg', data=ex_seg_counts.astype(np.int64))
    print_log_message(f"Saved extracted intensities to {output_csv_path} and {file_path}.")

    return all_intensities_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracts neuronal intensity from experimental volumes by interpolating coordinates from reference volumes.")
    
    # input arguments
    parser.add_argument('--ex-volumes-root', type=str, required=True, help="Root directory containing subfolders of experimental .npy volumes.")
    parser.add_argument('--ref-volumes-dir', type=str, required=False, help="Directory containing the sequence of reference .npy volumes.")
    parser.add_argument('--config', type=str, required=True, default = "/home/wenlab-user/JinghaoWang/new_new/code_v1.10/src/configs/inference/240623.json",help="Path to the model configuration JSON file.")
    
    # output arguments
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save all outputs, including reference inference and final intensity data.")
    parser.add_argument('--transfer-zephir-path', type=str, default=None,
                        help="If provided, skip intensity extraction and convert inference outputs into ZephIR format stored at this path.")
    parser.add_argument('--neuron-depth-limit', type=int, default=20, help="Maximum neuron depth limit")
    
    # inference arguments
    parser.add_argument("--pre-resize", type=int, default=1, choices=[0, 1], help="Whether to pre-resize the reference volumes (1: yes, 0: no).")
    parser.add_argument("--pre-resize-size", type=int, default=680, help="Size for pre-resizing the largest dimension of reference volumes.")
    parser.add_argument("--pre-rescale-pixels", type=int, default=1, choices=[0, 1], help="Whether to pre-rescale pixels of reference volumes (1: yes, 0: no).")
    parser.add_argument('--skip-inference', '--skip-ref-inference', dest='skip_inference', action='store_true',
                        help="Skip running inference when precomputed results already exist (applies to all modes, including mip).")
    
    # inference processing mode
    parser.add_argument('--processing-mode', type=str, choices=['interpolate', 'align', 'dual_propagate', 'mip'], 
                        default='interpolate', help="Strategy for processing ex_vols: 'interpolate' (linear), 'align' (shift to nearest ref_vol), or 'dual_propagate' (forward/backward adjacent align).")
    parser.add_argument('--align-shiftrange', type=str, default="21,21",
                        help="Local search range (Rows,Cols) or (Rows,Cols,Slices) for 'align' or 'dual_propagate' mode, e.g., '21,21' or '21,21,11'.")
    parser.add_argument('--align-preproc', type=str, choices=['dog', 'afp'], default='dog',
                        help="Preprocessing method for alignment: 'dog' (Difference of Gaussians) or 'afp' (Automatic Foreground Process).")
    
    parser.add_argument('--only-extract-intensity', action='store_true', help="Skip alignment and reuse existing ex_neuron_pt_tuple.npy for intensity extraction.")

    parser.set_defaults(skip_inference=False)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    # config path
    os.makedirs(args.output_dir, exist_ok=True)

    config_zrange = config.get('zrange')
    config_zratio = _compute_z_ratio(config)


    if torch.cuda.is_available():
        print_info_message(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        device = 'cuda'
    else:
        print_info_message('CUDA is not available. Using CPU')
        device = 'cpu'

    try:
        shiftrange = tuple(map(int, args.align_shiftrange.split(',')))
        if len(shiftrange) not in (2, 3): raise ValueError
    except ValueError:
        print_warning_message(f"Invalid shiftrange '{args.align_shiftrange}'. Using default (21,21).")
        shiftrange = (21, 21)

    ex_vol_folders = sorted(glob(os.path.join(args.ex_volumes_root, "ImgStk*")))
    ex_neuron_pt_tuple = None
    all_intensities_df = None
    
    if args.processing_mode == 'mip':
        ex_neuron_pt_tuple, all_intensities_df, ex_volume_path_list = run_mip_inference_and_extract(
            ex_vol_folders=ex_vol_folders,
            config_path=args.config,
            output_dir=args.output_dir,
            zrange=config_zrange,
            transfer_zephir_path=args.transfer_zephir_path,
            skip_inference=args.skip_inference,
            neuron_depth_limit=args.neuron_depth_limit,
            zephir_z_ratio=config_zratio,
            device=device,
            pre_resize=args.pre_resize,
            pre_resize_size=args.pre_resize_size,
            pre_rescale_pixels=args.pre_rescale_pixels,
            shiftrange=shiftrange,
            align_preproc=args.align_preproc
        )
        print_info_message("MIP mode processing finished.")
        
        if args.transfer_zephir_path:
            print_info_message("ZephIR export finished. Stopping pipeline as requested.")
            sys.exit(0)
    else:
        if args.only_extract_intensity:
            print_info_message("--- Only Extract Intensity Mode ---")
            ex_coords_path = os.path.join(args.output_dir, "ex_neuron_pt_tuple.npy")
            if not os.path.exists(ex_coords_path):
                raise FileNotFoundError(f"Could not find {ex_coords_path} required for --only-extract-intensity mode.")
            
            print_log_message(f"Loading coordinates from {ex_coords_path}")
            ex_neuron_pt_tuple = np.load(ex_coords_path)
            
            all_intensities_df = extract_intensities_only(
                ex_vol_folders,
                ex_neuron_pt_tuple,
                args.output_dir,
                device=device,
                zrange=config_zrange
            )
        else:
            if not args.ref_volumes_dir:
                raise ValueError("'--ref-volumes-dir' is required for non-mip processing modes.")

            print_info_message("--- Phase 1: Running sequence inference on reference volumes ---")
            ref_inference_output_dir = os.path.join(args.output_dir, "reference_inference_results")
            os.makedirs(ref_inference_output_dir, exist_ok=True)
            if args.skip_inference:
                print_info_message("Skipping reference inference; expecting existing ref_neuron_pt_tuple_filled.npy.")
            else:
                run_inference_on_volume_sequence(
                    volume_dir=args.ref_volumes_dir,
                    config_path=args.config,
                    output_dir=ref_inference_output_dir,
                    json_store_root=os.path.join(ref_inference_output_dir, "buffer_state"),
                    zrange=config_zrange,
                    pre_resize=args.pre_resize,
                    pre_resize_size=args.pre_resize_size,
                    pre_rescale_pixels=args.pre_rescale_pixels
                )
                print_log_message("Phase 1: Reference inference complete.")

            filled_path = os.path.join(ref_inference_output_dir, "ref_neuron_pt_tuple_filled.npy")
            raw_path = os.path.join(ref_inference_output_dir, "ref_neuron_pt_tuple.npy")
            for candidate in (filled_path, raw_path):
                clamp_neuron_depth_file(candidate, args.neuron_depth_limit)

            print_info_message(f"--- Phase 2: Starting {args.processing_mode} and intensity extraction ---")
            ref_coords_path = filled_path if os.path.exists(filled_path) else raw_path
            if not os.path.exists(ref_coords_path):
                raise FileNotFoundError("Could not find 'ref_neuron_pt_tuple_filled.npy' or 'ref_neuron_pt_tuple.npy'. Phase 1 may have failed.")
            elif ref_coords_path == raw_path:
                print_warning_message("Using non-interpolated reference coordinates ('ref_neuron_pt_tuple.npy').")

            ref_coords = np.load(ref_coords_path)
            clamp_neuron_depth(ref_coords, args.neuron_depth_limit)

        if args.transfer_zephir_path:
            export_volumes_to_zephir(
                args.ref_volumes_dir,
                ref_coords_path,
                args.transfer_zephir_path,
                z_ratio=config_zratio,
                zrange=config_zrange,
                max_depth=args.neuron_depth_limit,
            )
            print_info_message("ZephIR conversion complete; skipping experimental intensity extraction.")
            print_info_message("ZephIR export finished. Stopping pipeline as requested.")
            sys.exit(0)
        else:
            ref_vol_paths = sorted(glob(os.path.join(args.ref_volumes_dir, "*.npy")))
            ex_neuron_pt_tuple, all_intensities_df = interpolate_and_extract(
                ref_coords,
                ex_vol_folders,
                ref_vol_paths,
                args.output_dir,
                mode=args.processing_mode,
                device=device,
                zrange=config_zrange,
                z_ratio=config_zratio,
                shiftrange=shiftrange,
                align_preproc=args.align_preproc,
            )
            print_info_message("Processing finished.")

    print_info_message("--- Phase 3: Generating experimental volume video ---")

    if ex_neuron_pt_tuple is not None:
        ex_volume_path_list = []
        for folder in ex_vol_folders:
            ex_volume_path_list.extend(load_datapath(folder))
        if ex_volume_path_list:
            video_output_dir = os.path.join(args.output_dir, "experiment_volume_video")
            generate_experiment_volume_video(
                ex_volume_path_list,
                ex_neuron_pt_tuple,
                video_output_dir,
                fps=5,
                z_ratio=5.0,
                zrange=config_zrange,
            )
        else:
            print_warning_message("No experimental volumes found for video generation.")
    
    print_info_message("Video generation complete.")