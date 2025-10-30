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