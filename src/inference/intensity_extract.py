import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
import pandas as pd
from tqdm import tqdm
import re
import h5py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from src.inference.blur import get_image4processing

# ---------------------------------old intensity extraction(use mip) ---------------------------------
def cxcywh2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] * .5  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] * .5  # top left y
    return y

def cxcywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    if len(x.shape) == 1:
        y[0] = x[0] - x[2] * .5  # top left x
        y[1] = x[1] - x[3] * .5  # top left y
        y[2] = x[0] + x[2] * .5  # bottom right x
        y[3] = x[1] + x[3] * .5  # bottom right y
        return y
    y[:, 0] = x[:, 0] - x[:, 2] * .5  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] * .5  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] * .5  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] * .5  # bottom right y
    return y

def region2misi_filter(regions: Dict, volume, area_ratio: float = .6):
    """
    :param regions: Dict, {z: [cx, cy, w, h]}
    :return: torch.Tensor, [[mean_i, std_i]]
    """
    area_reduction = torch.tensor([1, 1, area_ratio, area_ratio], dtype = torch.float32)
    output = list()
    for frame_idx, frame_region in regions.items():
        for b in cxcywh2xyxy(torch.tensor(frame_region) * area_reduction).to(dtype = torch.int32):
            pixel_box = torch.tensor(volume[b[1]:b[3], b[0]:b[2], int(frame_idx)]).to(dtype=torch.float32)
            std, mean = torch.std_mean(pixel_box[pixel_box > np.sort(pixel_box.flatten())[int(len(pixel_box.flatten())* 0.5)]])
            output.append(torch.FloatTensor([mean, std]))

    output = torch.stack(output)
    return output

def neuron2misi_filter(neuron_pt_tuple, mip_volume, area_ratio: float = .6):
    area_reduction = torch.tensor([1, 1, area_ratio, area_ratio], dtype = torch.float32)
    output = list()
    for neuron in neuron_pt_tuple:
        b = cxcywh2xyxy(torch.tensor(neuron[[0, 1, 3, 4]]) * area_reduction).to(dtype = torch.int32)
        pixel_box = torch.tensor(mip_volume[b[1]:b[3], b[0]:b[2]]).to(dtype=torch.float32)
        std, mean = torch.std_mean(pixel_box[pixel_box > np.sort(pixel_box.flatten())[int(len(pixel_box.flatten())* 0.5)]])
        # std, mean = torch.std_mean(pixel_box)
        output.append(torch.FloatTensor([mean, std]))

    output = torch.stack(output)
    return output

def translate_matrix(image, row_shift, col_shift):
    """
    Move the image according to the row shift and column shift using PyTorch
    """
    shifted_image = torch.roll(image, shifts=row_shift, dims=0)
    shifted_image = torch.roll(shifted_image, shifts=col_shift, dims=1)
    
    return shifted_image

def pixel_intensity_extraction(file_list, shift_list, neuron_pt_tuple, save_path):
    """
    extract pixel intensity from each volume using the inferenced neuron coordinates
    return the .xlsx file with n(neurons) x T(volumes)
    """
    intensity_dict = {}
    for index, file_path in enumerate(tqdm(file_list, desc="Extracting Pixel Intensities", leave=False)):
        volume = np.load(file_path).astype(np.int32)
        volume = torch.from_numpy(volume).cuda() 
        mip_volume = get_image4processing(volume).cpu().numpy()
        if index == 0:
            intensity_dict[index] = neuron2misi_filter(neuron_pt_tuple, mip_volume)
        else:
            row_shift = -shift_list[index-1][1]
            col_shift = -shift_list[index-1][0]
            neuron_pt_tuple_shift = neuron_pt_tuple.copy()
            neuron_pt_tuple_shift[:, 0] += row_shift
            neuron_pt_tuple_shift[:, 1] += col_shift
            intensity_dict[index] = neuron2misi_filter(neuron_pt_tuple_shift, mip_volume)
        
    intensity_df = pd.DataFrame({key: value[:, 0] for key, value in intensity_dict.items()})
    intensity_df_ID = np.array(range(len(intensity_df)))
    intensity_df_trace = intensity_df.iloc[:, :].values

    # row_std = intensity_df.std(axis=1)
    # sorted_intensity_df = intensity_df.iloc[row_std.argsort()[::-1]]
    # sorted_intensity_df.to_excel(os.path.join(save_path, 'pixel_intensity.xlsx'))
    intensity_df.to_excel(os.path.join(save_path, 'pixel_intensity.xlsx'))
    # match the worm name
    match = re.search(r'w(\d+)', save_path)  

    if match:
        prefix = match.group(0)  
        file_name = prefix + '_trace.h5'
        file_path = os.path.join(save_path, file_name)
    else:
        file_path = os.path.join(save_path, 'trace.h5')

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('intensity', data=intensity_df_trace)
        f.create_dataset('ID', data=intensity_df_ID)

    
    # return sorted_intensity_df
    return intensity_df

# ---------------------------------new intensity extraction(convolution) ---------------------------------
def extract_neuron_intensities_torch(volume, neuron_pt_tuple, area_ratio=0.8, background_threshold=0.0, device='cuda', median='none', depth_correction=1):
    """
    Extract per-neuron intensity using vectorized grid_sample and a sliding window to find the brightest contiguous Z segment.

    Args:
        volume (np.ndarray): 3D volume data (Y, X, Z).
        neuron_pt_tuple (np.ndarray): Neuron coords/sizes, shape (N, 6): [cx, cy, z*5, w, h, d*5].
        area_ratio (float): Central ROI size ratio (ellipse inside bbox). Default 0.8.
        background_threshold (float): Value to subtract from final average intensity. Default 0.0.
        device (str): 'cuda' or 'cpu'.
        median (str): 'none' to use all pixels, otherwise use median thresholding. Default 'none'.
        depth_correction (int): add a correction factor to depth calculation. Default 1.
    """
    # 1. Prepare Volume
    if isinstance(volume, np.ndarray):
        if volume.dtype == np.uint16:
            volume = volume.astype(np.float32)
        elif volume.dtype not in (np.float32, np.float64, np.int32, np.int16, np.int8, np.uint8, np.bool_):
            volume = volume.astype(np.float32)
        volume_3d = torch.from_numpy(volume).to(device)
    else:
        volume_3d = volume.to(device)

    vol_y, vol_x, vol_z = volume_3d.shape
    # Reshape to (1, 1, Z, Y, X) for grid_sample (treating Z as depth, Y as height, X as width)
    # grid_sample input: (N, C, D_in, H_in, W_in)
    vol_batch = volume_3d.permute(2, 0, 1).unsqueeze(0).unsqueeze(0) 
    
    # 2. Prepare Neuron Data
    if not isinstance(neuron_pt_tuple, torch.Tensor):
        neuron_pt_tuple = torch.tensor(neuron_pt_tuple, device=device, dtype=torch.float32)
    else:
        neuron_pt_tuple = neuron_pt_tuple.to(device=device, dtype=torch.float32)

    N = neuron_pt_tuple.shape[0]
    if N == 0:
        return np.array([]), np.array([]), {}

    # Extract columns: [cx, cy, z*5, w, h, d*5]
    cx = neuron_pt_tuple[:, 0]
    cy = neuron_pt_tuple[:, 1]
    z_raw = neuron_pt_tuple[:, 2]
    w = neuron_pt_tuple[:, 3]
    h = neuron_pt_tuple[:, 4]
    
    if neuron_pt_tuple.shape[1] > 5:
        d_raw = neuron_pt_tuple[:, 5]
    else:
        d_raw = torch.full((N,), 10.0, device=device) # default depth 2.0 * 5 = 10.0

    z_center = z_raw / 5.0
    depth = d_raw / 5.0
    
    # Calculate integer Z bounds
    z_min = torch.ceil(z_center - depth / 2.0).int()
    z_max = torch.ceil(z_center + depth / 2.0).int() + depth_correction
    
    d_int = z_max - z_min
    d_int = torch.clamp(d_int, min=1)
    
    max_d = int(d_int.max().item())
    
    # Determine ROI size for X, Y (use max size to avoid downsampling)
    max_w = int(w.max().item()) if w.numel() > 0 else 16
    max_h = int(h.max().item()) if h.numel() > 0 else 16
    
    # Clamp size to reasonable bounds
    S_x = max(16, min(max_w, 128))
    S_y = max(16, min(max_h, 128))
    
    # 3. Construct Grid (N, max_d, S_y, S_x, 3)
    # Z coordinates
    range_z = torch.arange(max_d, device=device, dtype=torch.float32)
    z_coords = z_min.unsqueeze(1) + range_z.unsqueeze(0) # (N, max_d)
    valid_z_mask = range_z.unsqueeze(0) < d_int.unsqueeze(1) # (N, max_d)
    
    # Normalize Z to [-1, 1]
    z_norm = (2.0 * z_coords / (vol_z - 1.0)) - 1.0
    
    # Y, X coordinates (normalized to [-1, 1] for the ROI)
    range_y = torch.linspace(-0.5, 0.5, S_y, device=device)
    range_x = torch.linspace(-0.5, 0.5, S_x, device=device)
    
    # Map ROI [-0.5, 0.5] to Volume Coordinates
    # y_vol = cy + y_roi * h
    y_grid = cy.view(N, 1, 1) + range_y.view(1, S_y, 1) * h.view(N, 1, 1)
    x_grid = cx.view(N, 1, 1) + range_x.view(1, 1, S_x) * w.view(N, 1, 1)
    
    # Normalize to [-1, 1]
    y_norm = (2.0 * y_grid / (vol_y - 1.0)) - 1.0
    x_norm = (2.0 * x_grid / (vol_x - 1.0)) - 1.0
    
    # Expand to (N, max_d, S_y, S_x)
    z_expanded = z_norm.view(N, max_d, 1, 1).expand(N, max_d, S_y, S_x)
    y_expanded = y_norm.view(N, 1, S_y, 1).expand(N, max_d, S_y, S_x)
    x_expanded = x_norm.view(N, 1, 1, S_x).expand(N, max_d, S_y, S_x)
    
    grid = torch.stack((x_expanded, y_expanded, z_expanded), dim=-1)
    
    # 4. Sample
    vol_batch_expanded = vol_batch.expand(N, -1, -1, -1, -1)
    samples = F.grid_sample(vol_batch_expanded, grid, align_corners=True, padding_mode='zeros')
    samples = samples.squeeze(1) # (N, max_d, S_y, S_x)
    
    # 5. Masking (Ellipse)
    y_rel = range_y.view(1, S_y, 1)
    x_rel = range_x.view(1, 1, S_x)
    dist_sq = y_rel**2 + x_rel**2
    threshold_sq = (0.5 * area_ratio)**2
    mask_2d = dist_sq <= threshold_sq # (1, S_y, S_x)
    
    samples_flat = samples.view(N, max_d, -1)
    mask_flat = mask_2d.view(-1)
    masked_samples = samples_flat[:, :, mask_flat] # (N, max_d, K)
    
    # 6. Compute Statistics per Slice
    sums_all = masked_samples.sum(dim=-1)
    counts_all = torch.tensor(masked_samples.shape[-1], device=device, dtype=torch.float32)
    sums_all = sums_all * valid_z_mask

    if median != 'none':
        # Median
        medians = torch.median(masked_samples, dim=-1).values # (N, max_d)
        mask_above = masked_samples >= medians.unsqueeze(-1)
        
        sums_above = (masked_samples * mask_above).sum(dim=-1)
        counts_above = mask_above.sum(dim=-1).float()
        
        # Zero out invalid Z slices
        sums_above = sums_above * valid_z_mask
        counts_above = counts_above * valid_z_mask
    
    # 7. Sliding Window (Kernel 3)
    sums_all_padded = F.pad(sums_all.unsqueeze(1), (0, 2))
    counts_all_padded = counts_all * F.pad(valid_z_mask.unsqueeze(1).float(), (0, 2))
    
    kernel = torch.ones((1, 1, 3), device=device)
    
    sums_all_win = F.conv1d(sums_all_padded, kernel).squeeze(1)
    counts_all_win = F.conv1d(counts_all_padded, kernel).squeeze(1)
    
    avg_all = sums_all_win / torch.clamp_min(counts_all_win, 1e-6)

    if median != 'none':
        sums_above_padded = F.pad(sums_above.unsqueeze(1), (0, 2))
        counts_above_padded = F.pad(counts_above.unsqueeze(1), (0, 2))
        sums_above_win = F.conv1d(sums_above_padded, kernel).squeeze(1)
        counts_above_win = F.conv1d(counts_above_padded, kernel).squeeze(1)
        
        avg_primary = sums_above_win / torch.clamp_min(counts_above_win, 1e-6)
        avg_win = torch.where(counts_above_win > 0, avg_primary, avg_all)
    else:
        avg_win = avg_all
    
    # Mask out invalid windows
    win_idx = torch.arange(max_d, device=device).unsqueeze(0)
    depth_tensor = d_int.unsqueeze(1)
    cond_ge_3 = (depth_tensor >= 3) & (win_idx <= depth_tensor - 3)
    cond_lt_3 = (depth_tensor < 3) & (win_idx == 0)
    valid_win_mask = cond_ge_3 | cond_lt_3
    
    avg_win[~valid_win_mask] = -float('inf')
    
    # 8. Find Max
    max_vals, max_indices = torch.max(avg_win, dim=1)
    is_valid_neuron = max_vals > -float('inf')
    
    # 9. Prepare Output
    neuron_intensity_list = (max_vals - background_threshold).cpu().numpy()
    neuron_indices_list = np.arange(N)
    neuron_intensity_list[~is_valid_neuron.cpu().numpy()] = np.nan
    
    neuron_slices = {}
    max_indices_np = max_indices.cpu().numpy()
    z_min_np = z_min.cpu().numpy()
    depth_np = d_int.cpu().numpy()
    
    for i in range(N):
        if not is_valid_neuron[i]:
            continue
        idx = max_indices_np[i]
        d = depth_np[i]
        w_size = min(3, d)
        start_z = z_min_np[i] + idx
        slices = np.arange(start_z, start_z + w_size)
        neuron_slices[i] = slices

    return neuron_intensity_list, neuron_indices_list, neuron_slices