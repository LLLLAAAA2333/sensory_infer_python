import torch
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

    valid = (x_min < x_max) and (y_min < y_max) and (z_min < z_max)
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
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    neuron_depth = z_max - z_min + 1
    vol_bbox = volume[y_min:y_max, x_min:x_max, z_min:z_max]

    if vol_bbox.numel() == 0:
        return np.nan, []
    
    y_range, x_range, z_range = vol_bbox.shape
    eff_h, eff_w = y_range * area_ratio, x_range * area_ratio
    roi_shape_yx = (y_range, x_range)

    h_start = (y_range - int(eff_h)) // 2
    h_end = h_start + int(eff_h)
    w_start = (x_range - int(eff_w)) // 2
    w_end = w_start + int(eff_w)

    if h_start >= h_end or w_start >= w_end:
        return np.nan, []
    
    mask_2d = torch.zeros(roi_shape_yx, dtype=torch.bool, device=device)
    mask_2d[h_start:h_end, w_start:w_end] = create_ellipse_mask_torch((int(eff_h), int(eff_w)), device=device)
    mask_3d = mask_2d.unsqueeze(-1).expand(-1, -1, z_range)
    above_thresh = (vol_bbox > threshold) & mask_3d

    valid_intensities = vol_bbox * above_thresh
    total_intensities_z = torch.sum(valid_intensities, dim=(0, 1)) # (z_range)
    pixel_counts_z = torch.sum(above_thresh, dim=(0, 1))      # (z_range)
    
    window_size = min(3, int(neuron_depth))
    if z_range < window_size:
        window_size = z_range
    
    if window_size == 0:
        return np.nan, []
    
    # convolution using numpy
    np_intensities = total_intensities_z.cpu().numpy()
    np_counts = pixel_counts_z.cpu().numpy()
    kernel_np = np.ones(window_size)
    
    intensity_sums_np = np.convolve(np_intensities, kernel_np, mode='valid')
    count_sums_np = np.convolve(np_counts, kernel_np, mode='valid')
    
    if count_sums_np.size == 0:
        if np.sum(np_counts) > 0:
            avg_intensity = np.sum(np_intensities) / np.sum(np_counts)
            return avg_intensity - background_threshold, np.arange(z_min, z_max)
        else:
            return np.nan, []

    avg_intensities_np = np.full_like(intensity_sums_np, -np.inf, dtype=float)
    valid_np = count_sums_np > 0
    
    if not np.any(valid_np):
         return np.nan, []
         
    avg_intensities_np[valid_np] = intensity_sums_np[valid_np] / count_sums_np[valid_np]
    
    max_window_start_idx = np.argmax(avg_intensities_np)
    max_average_intensity = avg_intensities_np[max_window_start_idx]
    
    best_slices = np.arange(z_min + max_window_start_idx, 
                            z_min + max_window_start_idx + window_size)
    
    return max_average_intensity - background_threshold, best_slices

def extract_neuron_intensities_torch(volume, neuron_pt_tuple, intensity_threshold=110, background_threshold=102, device='cuda'):
    """
    Args:
        volume (np.ndarray): 3D volume data (Y, X, Z).
        neuron_pt_tuple (np.ndarray): Neuron coordinates and sizes, shape (N, 6) with format [cx, cy, z*5, w, h, d*5].
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
            
       # convolution to calculate intensity
        intensity, best_slices = calculate_intensity_conv_torch(
            volume_3d, bbox,
            intensity_threshold, background_threshold, 
            device=device
        )
        
        if not np.isnan(intensity):
            neuron_intensity_list.append(intensity)
            neuron_indices_list.append(neuron_idx)
            neuron_slices[neuron_idx] = best_slices

    return np.array(neuron_intensity_list), np.array(neuron_indices_list), neuron_slices