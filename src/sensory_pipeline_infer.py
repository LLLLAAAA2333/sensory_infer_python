import os
import sys
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
import vis_trajectory as vis
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.infer_sequence import run_inference_on_volume_sequence
from src.infer_single import run_inference_on_single_volume
from src.inference.volume_alignment import compute_distance, volume_alignment
from src.inference.blur import get_image4processing, pixel_threshold
from src.inference.intensity_extract import extract_neuron_intensities_torch
from src.merge_resize_inference import Treeformer_End2End
from src.comm_utils.prints import print_info_message, print_log_message, print_warning_message

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

def translation_matching_bruteforce_with_dist(volume1_gpu, volume2_gpu, shiftrange=(21.21), device='cuda'):
    """
    Args:
        (Y, X, Z) format GPU Tensors.
        Return (row_shift, col_shift) and min_distance.
    """
    binary_volume1_mask = pixel_threshold(volume1_gpu)
    binary_volume2_mask = pixel_threshold(volume2_gpu)

    binary_image1 = get_image4processing(binary_volume1_mask).to(torch.int)
    binary_image2 = get_image4processing(binary_volume2_mask).to(torch.int)

    if binary_image1.shape != binary_image2.shape:
        print_warning_message(f"MIP shape mismatch {binary_image1.shape} vs {binary_image2.shape}. Skipping alignment.")
        return (0, 0), 99999999.0

    rows, cols = binary_image2.shape
    distance_matrix = compute_distance(binary_image1, binary_image2, rows, cols, shiftrange)
    min_distance, min_idx = torch.min(distance_matrix.view(-1), 0)
    min_distance_index = np.unravel_index(min_idx.cpu().numpy(), distance_matrix.shape)

    shift_yx = (min_distance_index[0] - shiftrange[0] // 2, min_distance_index[1] - shiftrange[1] // 2)
    return shift_yx, min_distance.item()

def load_ex_vol_gpu(path, cache, device='cuda'):
    if path not in cache:
        vol_data = np.load(path)
        if vol_data.dtype == np.uint16:
            vol_data = vol_data.astype(np.float32)
        cache[path] = torch.from_numpy(vol_data).to(device).float()
    return cache[path]

def interpolate_and_extract(ref_coords, ex_vol_folders, ref_vol_paths, output_dir,
                            mode='interpolate', device='cuda', shiftrange=(21, 21)):
    """
    Args:
        ref_coords: numpy array of shape (T_ref, N_neurons, F_features)
        ex_vol_folders: list of folders, each containing experimental .npy volumes
        output_dir: directory to save outputs
        mode: 'interpolate', 'align', or 'dual_propagate'
        shiftrange: tuple for bruteforce mode and align mode
    """
    T_ref, N_neurons, F_features = ref_coords.shape
    print_log_message(f"Loaded reference matrix: {T_ref} ref volumes, {N_neurons} unique neurons, {F_features} features.")
    print_info_message(f"Running in mode: '{mode}'")

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
                if vol_data.dtype == np.uint16:
                    vol_data = vol_data.astype(np.float32)
                ref_vols_gpu_cache[idx] = torch.from_numpy(vol_data).to(device).float()
            except Exception as e:
                print_warning_message(f"Error loading ref_vol {ref_vol_paths[idx]}: {e}. Cannot use 'phasecorr'.")
                return None
        return ref_vols_gpu_cache[idx]
    
    all_intensities_df = pd.DataFrame(index=range(N_neurons))
    global_frame_counter = 0
    ex_tuples = []

    for t in tqdm(range(min(T_ref - 1, len(ex_vol_folders))), desc="Extracting intensities"):
        ref_start_coords = ref_coords[t]
        ref_end_coords = ref_coords[t + 1]

        current_mode = mode

        ex_files = load_datapath(ex_vol_folders[t])
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

            print_log_message(f"  Aligning ref_A to ex_vol_0...")
            vol_k_minus_1_gpu = load_ex_vol_gpu(ex_files[0], {}, device)
            shift_yx, dist = translation_matching_bruteforce_with_dist(ref_A_vol_gpu, vol_k_minus_1_gpu, shiftrange, device)
            current_f_coords = ref_start_coords.copy()
            shift_vector = np.array([shift_yx[1], shift_yx[0], 0], dtype=current_f_coords.dtype)
            current_f_coords[:, :3] -= shift_vector

            forward_coords = np.full((num_ex_vols, N_neurons, F_features), np.nan, dtype=np.float32)
            forward_coords[0] = current_f_coords.copy()

            for k in range(1, num_ex_vols):
                vol_k_gpu = load_ex_vol_gpu(ex_files[k], {}, device)
                shift_yx, dist = translation_matching_bruteforce_with_dist(vol_k_minus_1_gpu, vol_k_gpu, shiftrange, device)
                shift_vector = np.array([shift_yx[1], shift_yx[0], 0], dtype=current_f_coords.dtype)
                current_f_coords[:, :3] -= shift_vector
                forward_coords[k] = current_f_coords.copy()
                vol_k_minus_1_gpu = vol_k_gpu

            del vol_k_gpu, vol_k_minus_1_gpu

            print_log_message(f"  Aligning ref_B to ex_vol_{num_ex_vols - 1}...")
            vol_k_plus_1_gpu = load_ex_vol_gpu(ex_files[-1], {}, device)
            shift_yx, dist = translation_matching_bruteforce_with_dist(ref_B_vol_gpu, vol_k_plus_1_gpu, shiftrange, device)

            current_b_coords = ref_end_coords.copy()
            shift_vector = np.array([shift_yx[1], shift_yx[0], 0], dtype=current_b_coords.dtype)
            current_b_coords[:, :3] -= shift_vector
            backward_coords = np.full((num_ex_vols, N_neurons, F_features), np.nan, dtype=np.float32)
            backward_coords[-1] = current_b_coords.copy()

            for k in range(num_ex_vols - 2, -1, -1):
                vol_k_gpu = load_ex_vol_gpu(ex_files[k], {}, device)
                shift_yx, dist = translation_matching_bruteforce_with_dist(vol_k_plus_1_gpu, vol_k_gpu, shiftrange, device)
                shift_vector = np.array([shift_yx[1], shift_yx[0], 0], dtype=current_b_coords.dtype)
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


        for k, ex_file_path in enumerate(tqdm(ex_files, desc=f"processing folder {t}", leave=False)):
            interp_pt_tuple = None

            if current_mode == 'interpolate':
                interp_ratio = (k + 1.0) / (num_ex_vols + 1.0)
                coord_features = [0,1,2]
                interp_coords = ref_start_coords[:, coord_features] + (ref_end_coords[:, coord_features] - ref_start_coords[:, coord_features]) * interp_ratio
                interp_pt_tuple = ref_start_coords.copy()
                interp_pt_tuple[:, coord_features] = interp_coords

            elif current_mode == 'align':
                ex_vol_k_data = np.load(ex_file_path)
                if ex_vol_k_data.dtype == np.uint16:
                    ex_vol_k_data = ex_vol_k_data.astype(np.float32)
                ex_vol_k_gpu = torch.from_numpy(ex_vol_k_data).to(device).float()
                
                # phase correlation to get shift
                shift_A, dist_A = translation_matching_bruteforce_with_dist(ref_A_vol_gpu, ex_vol_k_gpu, shiftrange, device)
                shift_B, dist_B = translation_matching_bruteforce_with_dist(ref_B_vol_gpu, ex_vol_k_gpu, shiftrange, device)

                if dist_A <= dist_B:
                    base_coords = ref_start_coords
                    shift = shift_A
                else:
                    base_coords = ref_end_coords
                    shift = shift_B

                interp_pt_tuple = base_coords.copy()
                shift_vector = np.array([shift[1], shift[0]], dtype=interp_pt_tuple.dtype)
                interp_pt_tuple[:, 0] -= shift_vector[0]  # X
                interp_pt_tuple[:, 1] -= shift_vector[1]  # Y

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
                all_intensities_df[global_frame_counter] = pd.Series(np.nan, index=range(N_neurons))
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
            all_intensities_df[global_frame_counter] = vol_intensity
            global_frame_counter += 1
        
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

def run_mip_inference_and_extract(ex_vol_folders, config_path, output_dir, **kwargs):
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
    ex_vol_paths = []
    for folder in ex_vol_folders:
        ex_vol_paths.append(load_datapath(folder))
    
    if not ex_vol_paths:
        print_warning_message("No .npy files found in any experimental volume folders. Stopping MIP mode.")
        return
    
    print_log_message(f"Found {len(ex_vol_paths)} total experimental volumes for MIP processing.")
    
    synthetic_volume_save_path = os.path.join(output_dir, "synthetic_mip_results")
    os.makedirs(synthetic_volume_save_path, exist_ok=True)

    shiftrange = kwargs.get('shiftrange', (101, 101))
    print_log_message("Running volume alignment to create artificial MIP...")
    aligned_volumes_mip, shift_list = volume_alignment(
        ex_vol_paths, 
        synthetic_volume_save_path, 
        shiftrange=shiftrange
    )
    np.save(os.path.join(synthetic_volume_save_path, 'aligned_volumes_mip.npy'), aligned_volumes_mip)
    np.save(os.path.join(synthetic_volume_save_path, 'shift_list.npy'), shift_list)
    print_log_message(f"Artificial MIP and shift list saved to {synthetic_volume_save_path}")

    print_log_message("Running inference on artificial MIP...")
    neuron_pt_tuple_np = run_inference_on_single_volume(aligned_volumes_mip, config_path, synthetic_volume_save_path, **kwargs)
    print_log_message("Inference on artificial MIP completed.")

    print_log_message("Extracting 3D intensities from all experimental volumes using shifts...")
    num_neurons = neuron_pt_tuple_np.shape[0]
    all_intensities_list = []
    ex_neuron_pt_tuple_list = []

    for index, file_path in enumerate(tqdm(ex_vol_paths, desc="Extracting 3D Intensities")):
        current_coords = neuron_pt_tuple_np.copy()
        if index > 0:
            # shift_list[index-1] corresponds to volume[index] vs volume[0]
            shift = shift_list[index - 1] 
            row_shift = -shift[1] # Y shift (inverse)
            col_shift = -shift[0] # X shift (inverse)
            current_coords[:, 0] += col_shift # X coord
            current_coords[:, 1] += row_shift # Y coord
        ex_neuron_pt_tuple_list.append(current_coords)

        ex_vol_data = np.load(file_path)
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
    fps=5,
    z_ratio=5.0,
    source_min=130,
    source_max=200,
    default_bbox_size=(6.0, 6.0, 6.0),
    red_pseudo_color=False,
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
            return np.load(ex_volume_paths[idx])

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extracts neuronal intensity from experimental volumes by interpolating coordinates from reference volumes.")
    
    # input arguments
    parser.add_argument('--ex-volumes-root', type=str, required=True, help="Root directory containing subfolders of experimental .npy volumes.")
    parser.add_argument('--ref-volumes-dir', type=str, required=True, help="Directory containing the sequence of reference .npy volumes.")
    parser.add_argument('--config', type=str, required=True, default = "/home/wenlab-user/JinghaoWang/new_new/code_v1.10/src/configs/inference/240623.json",help="Path to the model configuration JSON file.")
    
    # output arguments
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save all outputs, including reference inference and final intensity data.")
    
    # inference arguments
    parser.add_argument("--pre-resize", type=int, default=1, choices=[0, 1], help="Whether to pre-resize the reference volumes (1: yes, 0: no).")
    parser.add_argument("--pre-resize-size", type=int, default=680, help="Size for pre-resizing the largest dimension of reference volumes.")
    parser.add_argument("--pre-rescale-pixels", type=int, default=1, choices=[0, 1], help="Whether to pre-rescale pixels of reference volumes (1: yes, 0: no).")
    parser.add_argument('--processing-mode', type=str, choices=['interpolate', 'align', 'dual_propagate', 'mip'], 
                        default='interpolate', help="Strategy for processing ex_vols: 'interpolate' (linear), 'align' (shift to nearest ref_vol), or 'dual_propagate' (forward/backward adjacent align).")
    parser.add_argument('--align-shiftrange', type=str, default="21,21",
                        help="Local search range (Rows,Cols) for 'align' or 'dual_propagate' mode, e.g., '21,21' for +/- 10 pixels.")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    # config path
    ref_inference_output_dir = os.path.join(args.output_dir, "reference_inference_results")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(ref_inference_output_dir, exist_ok=True)

    if torch.cuda.is_available():
        print_info_message(f'Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        device = 'cuda'
    else:
        print_info_message('CUDA is not available. Using CPU')
        device = 'cpu'

    try:
        shiftrange = tuple(map(int, args.align_shiftrange.split(',')))
        if len(shiftrange) != 2: raise ValueError
    except ValueError:
        print_warning_message(f"Invalid shiftrange '{args.align_shiftrange}'. Using default (21,21).")
        shiftrange = (21, 21)

    ex_vol_folders = sorted(glob(os.path.join(args.ex_volumes_root, "ImgStk*")))
    
    if args.processing_mode == 'mip':
        ex_neuron_pt_tuple, all_intensities_df, ex_volume_path_list = run_mip_inference_and_extract(
            ex_vol_folders=ex_vol_folders,
            config_path=args.config,
            output_dir=args.output_dir,
            device=device, 
            pre_resize=args.pre_resize,
            pre_resize_size=args.pre_resize_size,
            pre_rescale_pixels=args.pre_rescale_pixels,
            shiftrange=shiftrange
        )
        print_info_message("MIP mode processing finished.")
        
    else:
        print_info_message("--- Phase 1: Running sequence inference on reference volumes ---")
        run_inference_on_volume_sequence(
            volume_dir=args.ref_volumes_dir,
            config_path=args.config,
            output_dir=ref_inference_output_dir,
            json_store_root=os.path.join(ref_inference_output_dir, "buffer_state"),
            pre_resize=args.pre_resize,
            pre_resize_size=args.pre_resize_size,
            pre_rescale_pixels=args.pre_rescale_pixels
        )
        print_log_message("Phase 1: Reference inference complete.")

        print_info_message(f"--- Phase 2: Starting {args.processing_mode} and intensity extraction ---")
        # load reference coordinates
        ref_coords_path = os.path.join(ref_inference_output_dir, "ref_neuron_pt_tuple_filled.npy")
        if not os.path.exists(ref_coords_path):
            ref_coords_path = os.path.join(ref_inference_output_dir, "ref_neuron_pt_tuple.npy")
            if not os.path.exists(ref_coords_path):
                raise FileNotFoundError("Could not find 'ref_neuron_pt_tuple_filled.npy' or 'ref_neuron_pt_tuple.npy'. Phase 1 may have failed.")
            else:
                print_warning_message("Using non-interpolated reference coordinates ('ref_neuron_pt_tuple.npy').")

        ref_coords = np.load(ref_coords_path)
        ref_vol_paths = sorted(glob(os.path.join(args.ref_volumes_dir, "*.npy")))

        ex_neuron_pt_tuple, all_intensities_df = interpolate_and_extract(
                                                        ref_coords, 
                                                        ex_vol_folders, 
                                                        ref_vol_paths,
                                                        args.output_dir, 
                                                        mode=args.processing_mode,
                                                        device=device,
                                                        shiftrange=shiftrange
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
                )
            else:
                print_warning_message("No experimental volumes found for video generation.")
        
        print_info_message("Video generation complete.")