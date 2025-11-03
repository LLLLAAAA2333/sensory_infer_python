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
def cxcywh2xyxy_torch(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] * 0.5  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] * 0.5  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] * 0.5  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] * 0.5  # bottom right y
    return y

def get_bounding_box_torch(x, y, z, width, height, depth, vol_dims_y_x_z, device='cuda'):
    y_dim, x_dim, z_dim = vol_dims_y_x_z
    
    bbox_cxcywh = torch.tensor([x, y, width, height], device=device)
    bbox_xyxy = cxcywh2xyxy_torch(bbox_cxcywh)
    
    x_min = max(0, int(bbox_xyxy[0]))
    y_min = max(0, int(bbox_xyxy[1]))
    x_max = min(x_dim, int(bbox_xyxy[2]))
    y_max = min(y_dim, int(bbox_xyxy[3]))
    
    z_min = max(0, int(np.ceil(z - depth / 2.0)))
    z_max = min(z_dim, int(np.ceil(z + depth / 2.0)))

    valid = (x_min <= x_max) and (y_min <= y_max) and (z_min <= z_max)
    return (x_min, x_max, y_min, y_max, z_min, z_max), valid

def create_ellipse_mask_torch(roi_shape_yx, device):
    h, w = roi_shape_yx # h = Y, w = X
    cy, cx = h / 2 - 0.5, w / 2 - 0.5
    y, x = torch.meshgrid(torch.arange(h, device=device), 
                          torch.arange(w, device=device), indexing='ij')
    
    w_radius = w / 2.0
    h_radius = h / 2.0
    
    if w_radius <= 0 or h_radius <= 0:
        return torch.zeros(roi_shape_yx, dtype=torch.bool, device=device)
        
    mask = ((x - cx)**2 / w_radius**2 + (y - cy)**2 / h_radius**2) <= 1
    return mask

def calculate_intensity_conv_torch(volume, bbox, threshold, background_threshold, area_ratio=0.8, device = 'cuda'):
    """
    Calculate neuron intensity within a 3D ROI using a median-based selection per Z slice
    and a sliding window along Z to find the brightest contiguous segment.

    Notes:
    - The 'threshold' argument is ignored (kept for backward compatibility).
    - We compute, per Z slice, the mean of pixels >= median inside an elliptical ROI.
      To avoid NaNs when no pixel is strictly above median (e.g., constant slices),
      we fall back to using all masked pixels in that window.
    - background_threshold is subtracted from the final average.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    # Use right-open slicing semantics consistently; depth equals number of slices
    vol_bbox = volume[y_min:y_max+1, x_min:x_max+1, z_min:z_max+1]

    if vol_bbox.numel() == 0:
        return np.nan, []

    y_range, x_range, z_range = vol_bbox.shape
    neuron_depth = z_range  # fix off-by-one (previously used +1)

    # Build central elliptical ROI with area_ratio
    eff_h = int(max(1, round(y_range * area_ratio)))
    eff_w = int(max(1, round(x_range * area_ratio)))
    h_start = (y_range - eff_h) // 2
    h_end = h_start + eff_h
    w_start = (x_range - eff_w) // 2
    w_end = w_start + eff_w

    if h_start >= h_end or w_start >= w_end:
        return np.nan, []

    mask_2d = torch.zeros((y_range, x_range), dtype=torch.bool, device=device)
    mask_2d[h_start:h_end, w_start:w_end] = create_ellipse_mask_torch((eff_h, eff_w), device=device)

    counts_masked = int(mask_2d.sum().item())
    if counts_masked == 0:
        return np.nan, []

    # Prepare per-Z sums/counts using median-based selection
    sums_above_med = torch.zeros((z_range,), dtype=vol_bbox.dtype, device=device)
    counts_above_med = torch.zeros((z_range,), dtype=torch.int32, device=device)
    sums_masked = torch.zeros((z_range,), dtype=vol_bbox.dtype, device=device)

    for zi in range(z_range):
        slice_2d = vol_bbox[:, :, zi]
        vals = slice_2d[mask_2d]
        # In rare empty cases (shouldn't happen due to counts_masked check), skip
        if vals.numel() == 0:
            continue
        med = torch.median(vals)
        sel = vals >= med  # use >= to reduce zero-count cases while matching robust behavior
        cnt_sel = int(sel.sum().item())
        if cnt_sel > 0:
            sums_above_med[zi] = vals[sel].sum()
            counts_above_med[zi] = cnt_sel
        else:
            # Keep zero here; we'll fall back to all masked in window aggregation
            sums_above_med[zi] = torch.tensor(0.0, dtype=vol_bbox.dtype, device=device)
            counts_above_med[zi] = 0
        sums_masked[zi] = vals.sum()

    # Sliding window along Z using conv1d on GPU
    window_size = min(3, int(neuron_depth))
    if z_range < window_size:
        window_size = z_range
    if window_size == 0:
        return np.nan, []

    kernel = torch.ones((1, 1, window_size), dtype=vol_bbox.dtype, device=device)

    sums_above_win = F.conv1d(sums_above_med.view(1, 1, -1), kernel, padding=0).view(-1)
    counts_above_win = F.conv1d(counts_above_med.to(dtype=vol_bbox.dtype).view(1, 1, -1), kernel, padding=0).view(-1)

    sums_masked_win = F.conv1d(sums_masked.view(1, 1, -1), kernel, padding=0).view(-1)
    counts_masked_win = counts_masked * window_size

    # Avoid division by zero: fallback to masked sums when no above-median pixels in a window
    use_fallback = counts_above_win <= 0
    avg_win = torch.empty_like(sums_above_win)
    # primary
    avg_primary = sums_above_win / torch.clamp_min(counts_above_win, 1e-6)
    # fallback
    avg_fallback = sums_masked_win / counts_masked_win
    avg_win = torch.where(use_fallback, avg_fallback, avg_primary)

    if avg_win.numel() == 0:
        return np.nan, []

    max_window_start_idx = int(torch.argmax(avg_win).item())
    max_average_intensity = float(avg_win[max_window_start_idx].item())

    best_slices = np.arange(
        z_min + max_window_start_idx,
        z_min + max_window_start_idx + window_size
    )

    return max_average_intensity - background_threshold, best_slices

def extract_neuron_intensities_torch(volume, neuron_pt_tuple, area_ratio=0.8, background_threshold=0.0, device='cuda'):
    """
    Extract per-neuron intensity using median-based selection per Z slice and
    a sliding window to find the brightest contiguous Z segment.

    Args:
        volume (np.ndarray): 3D volume data (Y, X, Z).
        neuron_pt_tuple (np.ndarray): Neuron coords/sizes, shape (N, 6): [cx, cy, z*5, w, h, d*5].
        area_ratio (float): Central ROI size ratio (ellipse inside bbox). Default 0.8.
        background_threshold (float): Value to subtract from final average intensity. Default 0.0.
        device (str): 'cuda' or 'cpu'.
    """
    if isinstance(volume, np.ndarray) and volume.dtype == np.uint16:
        volume = volume.astype(np.float32)
    elif isinstance(volume, np.ndarray) and volume.dtype not in (np.float32, np.float64, np.int32, np.int16, np.int8, np.uint8, np.bool_):
        volume = volume.astype(np.float32)
    volume_3d = torch.from_numpy(volume).to(device)
    vol_dims_y_x_z = volume_3d.shape
    
    neuron_slices = {}
    neuron_intensity_list = []
    neuron_indices_list = []

    for neuron_idx in range(neuron_pt_tuple.shape[0]):
        neuron_data = neuron_pt_tuple[neuron_idx]
        
        x, y, z = neuron_data[0], neuron_data[1], neuron_data[2]
        width, height = neuron_data[3], neuron_data[4]
        depth = neuron_data[5] if len(neuron_data) > 5 else 2.0 
        
        # z and depth scaling
        z /= 5.0 
        depth = depth / 5.0

        # 2. 获取边界框
        bbox, valid = get_bounding_box_torch(
            x, y, z, width, height, depth, 
            vol_dims_y_x_z, device=device
        )
        if not valid:
            continue
            
        # convolution to calculate intensity (threshold ignored in impl; pass 0.0)
        intensity, best_slices = calculate_intensity_conv_torch(
            volume_3d, bbox,
            0.0, background_threshold,
            area_ratio=area_ratio,
            device=device
        )
        
        if not np.isnan(intensity):
            neuron_intensity_list.append(intensity)
            neuron_indices_list.append(neuron_idx)
            neuron_slices[neuron_idx] = best_slices

    return np.array(neuron_intensity_list), np.array(neuron_indices_list), neuron_slices