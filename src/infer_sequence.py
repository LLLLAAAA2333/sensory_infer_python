import os
import sys
import json
import re
import torch
import glob
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.comm_utils.prints import print_log_message, print_warning_message, pad_num
from src.merge_resize_inference import Treeformer_End2End, VolumeMemoryBuffer, draw_volume_result


def _apply_zrange(volume_np, zrange=None):
    if zrange is None:
        return volume_np
    z_start, z_end = zrange
    z_start = max(z_start, 0)
    if z_end == -1 or z_end > volume_np.shape[2]:
        z_end = volume_np.shape[2]
    if z_start == 0 and z_end == volume_np.shape[2]:
        return volume_np
    return volume_np[:, :, z_start:z_end]

def _find_neighbor(neuron_pt_tuple, t_miss, n_miss, K=5, max_K=20, step=5):
    """Find stable neighbors for a missing neuron using incremental K search."""
    nan_mask = np.isnan(neuron_pt_tuple[:, :, [0, 1, 2]]).any(axis=2)

    valid_t_indices = np.where(~nan_mask[:, n_miss])[0]
    if valid_t_indices.size == 0:
        return np.array([], dtype=np.int64), None

    distances_to_t_miss = np.abs(valid_t_indices - t_miss)
    t_ref = valid_t_indices[np.argmin(distances_to_t_miss)]

    coords_ref = neuron_pt_tuple[t_ref, :, 0:3]
    target_coord_ref = coords_ref[n_miss]
    valid_mask_ref = ~np.isnan(coords_ref).any(axis=1)
    coords_ref_clean = coords_ref[valid_mask_ref]
    if coords_ref_clean.size == 0:
        return np.array([], dtype=np.int64), t_ref

    original_indices_mapping = np.where(valid_mask_ref)[0]
    kdtree = KDTree(coords_ref_clean)

    max_K = max(K, max_K)
    for cur_k in range(K, max_K + 1, step):
        query_k = min(cur_k + 1, coords_ref_clean.shape[0])
        distances, clean_indices = kdtree.query(target_coord_ref, k=query_k)
        clean_indices = np.atleast_1d(clean_indices)
        original_neighbor_indices = original_indices_mapping[clean_indices]

        if original_neighbor_indices.size == 0:
            continue

        # remove the neuron itself
        if original_neighbor_indices[0] == n_miss:
            potential_neighbor_indices = original_neighbor_indices[1:]
        else:
            potential_neighbor_indices = original_neighbor_indices

        if potential_neighbor_indices.size == 0:
            print_log_message(f"No stable neighbors found for neuron {n_miss} at time {t_miss} with K={cur_k}")
            continue

        is_present_at_t_miss = ~nan_mask[t_miss, potential_neighbor_indices]
        stable_neighbor_indices = potential_neighbor_indices[is_present_at_t_miss]
        if stable_neighbor_indices.size > 0:
            return stable_neighbor_indices.astype(np.int64), t_ref

        print_log_message(f"No stable neighbors found for neuron {n_miss} at time {t_miss} with K={cur_k}")

    return np.array([], dtype=np.int64), t_ref

def _interpolate_features(neuron_pt_tuple, t_miss, n_miss, t_ref, stable_neighbor_indices):

    target_coord_ref = neuron_pt_tuple[t_ref, n_miss, 0:3]
    coords_neighbors_ref = neuron_pt_tuple[t_ref, stable_neighbor_indices, 0:3]
    coords_neighbors_miss = neuron_pt_tuple[t_miss, stable_neighbor_indices, 0:3]
    # compute transformation vectors from ref to miss for neighbors
    displacement_vectors = coords_neighbors_miss - coords_neighbors_ref
    avg_displacement = np.nanmean(displacement_vectors, axis=0)
    estimated_coord = target_coord_ref + avg_displacement

    features_ref = neuron_pt_tuple[t_ref, n_miss, 3:]
    estamated_features = features_ref  # assuming features do not change significantly
    estimated_full_tuple = np.concatenate([estimated_coord, estamated_features], axis=0)
    return estimated_full_tuple

def interpolate_missing_neurons(neuron_pt_tuple, K=5):
    neuron_pt_tuple_filled = np.copy(neuron_pt_tuple)

    nan_mask_original = np.isnan(neuron_pt_tuple[:, :, [0,1,2]]).any(axis=2)
    nan_indices = np.argwhere(nan_mask_original)

    fill_count = 0
    fail_count = 0
    for t_miss, n_miss in nan_indices:
        stable_neighbor_indices, t_ref = _find_neighbor(neuron_pt_tuple, t_miss, n_miss, K=K)
        if stable_neighbor_indices.size > 0 and t_ref is not None:
            estimated_full_tuple = _interpolate_features(neuron_pt_tuple, t_miss, n_miss, t_ref, stable_neighbor_indices)
            neuron_pt_tuple_filled[t_miss, n_miss, :] = estimated_full_tuple
            fill_count += 1
        else:
            available_times = np.where(~np.isnan(neuron_pt_tuple[:, n_miss, 0]))[0]
            if available_times.size > 0:
                nearest_time = available_times[np.argmin(np.abs(available_times - t_miss))]
                neuron_pt_tuple_filled[t_miss, n_miss, :] = neuron_pt_tuple_filled[nearest_time, n_miss, :]
                fill_count += 1
            else:
                fail_count += 1
    
    print_log_message(f"Filled {fill_count} missing neurons, failed to fill {fail_count} missing neurons.")
    return neuron_pt_tuple_filled

def run_inference_on_volume_sequence(volume_dir, config_path, output_dir, **kwargs):
    print_log_message(f"Running inference on volume sequence in directory: {volume_dir}")
    os.makedirs(output_dir, exist_ok=True)
    results_subdir = os.path.join(output_dir, "tracked_neuron_results")
    os.makedirs(results_subdir, exist_ok=True)
    vis_subdir = os.path.join(output_dir, "visualizations_sequence")
    os.makedirs(vis_subdir, exist_ok=True)
    print_log_message(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    def _natural_key(path):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', os.path.basename(path))]
    volume_paths = sorted(glob.glob(os.path.join(volume_dir, "*.npy")),
                      key=_natural_key)
    if not volume_paths:
        raise FileNotFoundError(f"No .npy volumes found in {volume_dir}")

    zrange = kwargs.get('zrange') or config.get("zrange", [0, -1])
    zrange = list(zrange)
    z_start, z_end = zrange
    sample_volume = np.load(volume_paths[0])
    if z_end == -1 or z_end > sample_volume.shape[2]:
        z_end = sample_volume.shape[2]
    z_len = max(z_end - max(z_start, 0), 0)
    if z_len <= 0:
        raise ValueError(f"Invalid zrange {zrange} for volume depth {sample_volume.shape[2]}")
    del sample_volume

    print_log_message("Initializing model...")

    model = Treeformer_End2End(

        ext_path = config["ext_path"],
        det_path = config["det_path"],
        rec_path = config["rec_path"],

        ext_input_dim = [z_len] + config["ext_input_dim"],
        det_input_dim = [z_len] + config["det_input_dim"],
        rec_input_dim = config["rec_input_dim"],

        ext_conf = config["ext_conf"],

        det_conf = config["det_conf"], det_iou_t = config["det_iou_t"],
        region_shrink_pixel = config["region_shrink_pixel"],

        rec_pt_mode = config["rec_pt_mode"],
        xoy_unit = config["xoy_unit"], z_unit = config["z_unit"],  # um/pixel

        is_ext = config["is_ext"],
        magnification = config["magnification"],

        det_keep_ratio = config["det_keep_ratio"],

    ).eval().cuda().half()

    buffer = VolumeMemoryBuffer(
        save_path = kwargs.get('json_store_root', "./volume_memory_buffer"),
        deg_t = 45,
        weight_W = 0.6,
        warmup_num_vol = 10,
    )
    print_log_message("Model initialized.")
    all_results = {}
    temporal_results = []
    for idx, volume_path in enumerate(volume_paths):
        volume_basename = os.path.basename(volume_path)
        volume_identifier = os.path.splitext(volume_basename)[0]
        print_log_message(f"Processing volume {idx + 1}/{len(volume_paths)}: {volume_path}")
        current_neuron_pt_tuples = []
        current_neuron_pred_ids = []
        volume_np = np.load(volume_path)
        
        # Filter out high intensity noise
        if volume_np.max() >= 10000:
             print_warning_message(f"Found high intensity values (>= 10000) in {volume_basename}. Clamping to median.")
             volume_np[volume_np >= 10000] = np.median(volume_np).astype(volume_np.dtype)

        volume_np = _apply_zrange(volume_np, (z_start, z_end))
        volume_tensor = torch.HalfTensor(volume_np.transpose(2, 0, 1)[:, np.newaxis, :, :].astype(np.float32)).cuda()

        pre_resize = kwargs.get('pre_resize', 1)
        pre_resize_size = kwargs.get('pre_resize_size', 680)
        scale_factor = kwargs.get('scale_factor', 1.0)
        if pre_resize:
            scale_factor = pre_resize_size / max(volume_tensor.shape[2:])
            volume_tensor = F.interpolate(volume_tensor, scale_factor=scale_factor, mode="nearest")
            # print_log_message(f"Pre-resized volume to size: {volume_tensor.shape}")
        
        pre_rescale_pixels = kwargs.get('pre_rescale_pixels', 1)
        if pre_rescale_pixels:
            _std, _mean = volume_tensor.std(), volume_tensor.mean()
            print_log_message(f"std: {_std}, mean: {_mean}")
            volume_tensor[:] = -(_mean / _std * 13 - 103) + 13 / _std * volume_tensor
        
        print_log_message("Starting inference...")
        
        with torch.inference_mode():
            head_bboxes, regions, neuron_dict, neuron_emb, is_warnings, supp_r = model(volume_tensor)
            region_pt_tuple, region_ptrs, region_map, neuron_pt_tuple, (num_region, num_neuron), (std, max_min) = supp_r

            print_log_message("Alignment using buffer...")
            neuron_pred_ids, region_pred_ids, num_neuron_idv = buffer(neuron_emb, neuron_dict, num_region, any(is_warnings))
            print_log_message(f"Inference Done. Regions: {num_region}, Neurons: {num_neuron}, Assigned IDs: {num_neuron_idv}")

            if pre_resize and scale_factor != 1.0:
                neuron_pt_tuple[:, [0, 1, 3, 4]] = neuron_pt_tuple[:, [0, 1, 3, 4]] / scale_factor

            current_neuron_pt_tuple = neuron_pt_tuple.cpu().numpy()
            current_neuron_pred_ids = np.array(neuron_pred_ids)
            temporal_results.append((current_neuron_pt_tuple, current_neuron_pred_ids))
            current_results = {
                    'neuron_pt_tuple': neuron_pt_tuple.cpu().numpy(),
                    'neuron_pred_ids': np.array(neuron_pred_ids), # Save assigned IDs
                    'regions': regions, # Keep regions if needed for visualization or JSON later
                    'region_ptrs': region_ptrs,
                    'head_bboxes': head_bboxes.cpu().numpy()
                }
            all_results[volume_identifier] = current_results

            # save individual volume result
            output_npz_path = os.path.join(results_subdir, f"{volume_identifier}_results.npz")
            np.savez(output_npz_path,
                        neuron_pt_tuple=current_results['neuron_pt_tuple'],
                        neuron_pred_ids=current_results['neuron_pred_ids'])
            print_log_message(f"Saved results to {output_npz_path}")

            vis_output_name = f"{volume_identifier}_visualization"
            draw_volume_result(
                volume_tensor, head_bboxes, regions,
                region_pred_ids, # Use the IDs from buffer
                region_ptrs,
                vis_subdir, # Save in the visualization sub-directory
                name=vis_output_name,
                verbose=False
            )
            print_log_message(f"Saved visualization to {vis_subdir}/{vis_output_name}.pdf")
            
    # Save all results together
    print_log_message("Aggregating results into an neuron_pt_tuple matrix...")
    all_ids = set()
    for pt_tuple, pred_ids in temporal_results:
        all_ids.update(pred_ids)
        if pt_tuple.ndim == 2 and pt_tuple.shape[1] > 0: # 确保至少有一个有效的 pt_tuple 来确定特征数
                            num_features = pt_tuple.shape[1] # F=8
    if not all_ids:
        print_warning_message("No neurons detected in the entire sequence.")
        return
    else:
        valid_ids = {id_ for id_ in all_ids if id_ >= 0}
        if not valid_ids:
            print_warning_message("No valid neuron IDs detected in the entire sequence.")
            max_id = -1
        else:
            max_id = max(valid_ids)
        
    num_total_neurons = max_id + 1
    num_time_points = len(temporal_results)
    print_log_message(f"Creating aligned matrix of shape (T={num_time_points}, N={num_total_neurons}, F={num_features})")
    ref_neuron_pt_tuple = np.full((num_time_points, num_total_neurons, num_features), np.nan, dtype=np.float32)
    for t, (pt_tuple, pred_ids) in enumerate(temporal_results):
        for neuron_idx, neuron_id in enumerate(pred_ids):
            if neuron_id >= 0: # Only consider valid IDs
                ref_neuron_pt_tuple[t, neuron_id, :] = pt_tuple[neuron_idx,:]
        
    ref_output_path = os.path.join(output_dir, "ref_neuron_pt_tuple.npy")
    np.save(ref_output_path, ref_neuron_pt_tuple)
    print_log_message(f"Saved Ref neuron point tuple matrix to {ref_output_path}")

    print_log_message("Interpolate missing neurons in ref_neuron_pt_tuple")
    ref_neuron_pt_tuple_filled = interpolate_missing_neurons(ref_neuron_pt_tuple, K=5)
    filled_output_path = os.path.join(output_dir, "ref_neuron_pt_tuple_filled.npy")
    np.save(filled_output_path, ref_neuron_pt_tuple_filled)
    print_log_message(f"Saved Filled Ref neuron point tuple matrix to {filled_output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on sequence, align neurons, save aligned matrix.")
    parser.add_argument("--volume-dir", type=str, required=True, help="Path to the directory containing input .npy volume files.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the model configuration file (JSON).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output results.")
    parser.add_argument("--json-store-root", type=str, default=None, help="Base directory for buffer state (optional). Defaults to output_dir/buffer_state.")
    parser.add_argument("--pre-resize", type=int, default=1, choices=[0, 1], help="Whether to pre-resize the volume (1: yes, 0: no).")
    parser.add_argument("--pre-resize-size", type=int, default=680, help="Size for pre-resizing the largest dimension.")
    parser.add_argument("--pre-rescale-pixels", type=int, default=1, choices=[0, 1], help="Whether to pre-rescale pixel intensities (1: yes, 0: no).")

    args = parser.parse_args()

    if args.json_store_root is None:
        args.json_store_root = os.path.join(args.output_dir, "buffer_state")

    args_dict = vars(args)
    run_inference_on_volume_sequence(**args_dict)