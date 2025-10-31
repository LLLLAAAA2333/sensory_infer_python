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

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.infer_sequence import run_inference_on_volume_sequence
from src.inference.intensity_extract import extract_neuron_intensities_torch
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

def interpolate_and_extract(ref_coords, ex_vol_folders, output_dir, device='cuda'):
    T_ref, N_neurons, F_features = ref_coords.shape
    print_log_message(f"Loaded reference matrix: {T_ref} ref volumes, {N_neurons} unique neurons, {F_features} features.")

    if len(ex_vol_folders) != T_ref - 1:
        print_warning_message(f"Mismatch! Found {T_ref} ref volumes but {len(ex_vol_folders)} experimental volume folders. Expected {T_ref - 1} folders.")

    all_intensities_df = pd.DataFrame(index=range(N_neurons))
    global_frame_counter = 0
    ex_tuples = []
    for t in tqdm(range(min(T_ref - 1, len(ex_vol_folders))), desc="Interpolating and extracting intensities"):
        ref_start_coords = ref_coords[t]
        ref_end_coords = ref_coords[t + 1]

        # Get experimental volume number from 2 ref volumes
        ex_files = load_datapath(ex_vol_folders[t])
        num_ex_vols = len(ex_files)

        if num_ex_vols == 0:
            print_warning_message(f"No .npy files found in {ex_vol_folders[t]}. Skipping this folder.")
            continue

        for k, ex_file_path in enumerate(tqdm(ex_files, desc=f"processing folder {t}", leave=False)):
            
            interp_ratio = (k + 1.0) / (num_ex_vols + 1.0)

            coord_features = [0,1,2]
            interp_coords = ref_start_coords[:, coord_features] + (ref_end_coords[:, coord_features] - ref_start_coords[:, coord_features]) * interp_ratio

            interp_pt_tuple = ref_start_coords.copy()
            interp_pt_tuple[:, coord_features] = interp_coords

            # handle nan situations
            mask_nan_interp = np.isnan(interp_coords[:,0])
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
            intensity_values, intensity_indices, _ = extract_neuron_intensities_torch(ex_vol_data, valid_interp_tuple, intensity_threshold=120, background_threshold=102, device=device)
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

    print_info_message("--- Phase 2: Starting interpolation and intensity extraction ---")
    # load reference coordinates
    ref_coords_path = os.path.join(ref_inference_output_dir, "ref_neuron_pt_tuple_filled.npy")
    if not os.path.exists(ref_coords_path):
        ref_coords_path = os.path.join(ref_inference_output_dir, "ref_neuron_pt_tuple.npy")
        if not os.path.exists(ref_coords_path):
            raise FileNotFoundError("Could not find 'ref_neuron_pt_tuple_filled.npy' or 'ref_neuron_pt_tuple.npy'. Phase 1 may have failed.")
        else:
            print_warning_message("Using non-interpolated reference coordinates ('ref_neuron_pt_tuple.npy').")

    ref_coords = np.load(ref_coords_path)
    ex_vol_folders = sorted(glob(os.path.join(args.ex_volumes_root, "ImgStk*")))
    interpolate_and_extract(ref_coords, ex_vol_folders, args.output_dir, device=device)
    print_info_message("Processing finished.")