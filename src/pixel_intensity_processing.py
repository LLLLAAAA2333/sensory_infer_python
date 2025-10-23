import os
import cv2
import sys
import json
import h5py
import random
import shutil
import colorsys
import time
import torch
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from src.comm_utils.prints import print_info_message
from src.comm_utils.dataset_building import extract_waiting_stacks, split_processing_streams
from src.merge_resize_inference import Treeformer_End2End, VolumeMemoryBuffer, draw_volume_result
from src.comm_utils.packages import *
from src.comm_utils.prints import print_log_message, print_warning_message, pad_num
from src.preproc.inference import auto_preprocess
from src.utils import store_result_as_json
from src.merge_resize_inference import *
from loop_base import load_matlab_volumes_dict
import vis_trajectory as vis



 # --------------------------------- Step 1. volume .mat to .npy ---------------------------------

def infer_one_batch(stream, args, volume_save_path):
    print_log_message(f"Processing stream: {stream}")
    results, preprc_results = dict(), dict()

    # load_stack
    # (preprc_results, head_bbox_dict, scale_factors), results = load_matlab_volumes_dict(args, stream), dict()
    preprc_results = auto_preprocess(mode = args.preprocessing_mode, paths = stream, args = args)
    
    # folder_path = os.path.dirname(stream[0]) + '/volume/' + stream[0].split('/')[-1].split('.')[0]
    # save volume result
    mat_save_path = os.path.join(volume_save_path, stream[0].split('/')[-1].split('.')[0]) 
    os.makedirs(mat_save_path, exist_ok=True)
    for key in preprc_results.keys():
        npy_save_path = os.path.join(mat_save_path, key.split('/')[-1]+'.npy')
        # print(preprc_results[key].cpu().numpy().shape)
        # np.save(npy_save_path, preprc_results[key].permute(2, 3, 0, 1).squeeze(dim=-1).cpu().numpy())
        np.save(npy_save_path, preprc_results[key])
        
        torch.cuda.empty_cache()

    return None, None


# ---------------------------------Step 2. Volume alignment ---------------------------------

def load_datapath(folder_list):
    """
    Load several .npy files from several folders
    """
    file_list = []
    for folder in folder_list:
        file_path = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        file_path.sort(key=lambda path: int(os.path.basename(path)[-10:-4]))
        file_list = file_list + file_path
    return file_list


def get_image4processing(volume, is_max=True):
    """
    Extracts an image from a volume by applying max or mean along the last dimension.
    """
    image = torch.max(volume, dim=-1).values if is_max else torch.mean(volume, dim=-1)
    return image


def calc_volume_bg_threshold(volume, kernel_and_stride: int = 256, t_ratio: float = 1.2, ):
    """Calculate threshold for volume based on background pixels.
    
    Args:
        volume (torch.Tensor): Input volume with shape [B, 1, H, W].
        kernel_and_stride (int): Kernel size and stride for max pooling.
        t_ratio (float): Ratio to multiply with the minimum pooled value to get the threshold.
    Returns:
        float: Calculated threshold value.
        torch.Tensor: Pooled MIP.
    """
    mip = torch.max(volume, dim = 0, keepdim = True)[0]
    pooled_mip = torch.nn.functional.max_pool2d(mip, kernel_size = kernel_and_stride, stride = kernel_and_stride)
    threshold = pooled_mip.min() * t_ratio
    return threshold


def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.1):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)), (kernel_size, kernel_size))
    return kernel[np.newaxis, np.newaxis]  # [1, 1, kernel_size, kernel_size]


def gaussian_blur(volume, kernel_size: int = 5):
    g_kernel = gaussian_kernel(kernel_size, 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8)

    return F.conv2d(volume, torch.tensor(g_kernel, dtype = volume.dtype, device = volume.device), padding = kernel_size // 2)


def pixel_threshold(volume, gaussian_k: int = 9, maxpool_k: int = 125, bg_t_r: float = 1.2, rescale_p: float = .97, only_scale: bool = False):
    volume_torch =  volume.to(torch.float32).permute(2, 0, 1).unsqueeze(1)
    if gaussian_k > 1:  
        volume_torch[:] = gaussian_blur(volume_torch, kernel_size = gaussian_k)
    threshold = calc_volume_bg_threshold(volume_torch, kernel_and_stride = maxpool_k, t_ratio = bg_t_r)
    mask = volume_torch > threshold
    return mask.squeeze(1).permute(1, 2, 0)


def pixel2binary(image, rate=0.98):
    index = round(torch.numel(image) * rate)
    threshold = torch.mean(torch.sort(image.view(-1))[0][index:].float())
    binary_image = image > threshold
    return binary_image.int()

def Euclidean_distance(image1, image2):
    return torch.sum((image1 - image2) ** 2)


def compute_distance(binary_image1, binary_image2, rows, cols, shiftrange=(61, 61)):
    """
    Each entry in distance_matrix represents the euclidean distance between image 1 and image 2 which moved (i-nrow/2, j-ncol/2)
    """
    distance_matrix = torch.zeros(shiftrange[0], shiftrange[1], device='cuda')

    for i_idx, i in enumerate(range(-shiftrange[0]//2+1,shiftrange[0]//2+1)):
        for j_idx, j in enumerate(range(-shiftrange[1]//2+1,shiftrange[1]//2+1)):
            aligned_image2 = torch.roll(binary_image2, shifts=i, dims=0)
            aligned_image2 = torch.roll(aligned_image2, shifts=j, dims=1)

            distance = torch.sum((binary_image1 - aligned_image2) ** 2)
            distance_matrix[i_idx, j_idx] = distance

    return distance_matrix


def translation_matching(volume1, volume2, shiftrange=(61, 61)):
    """
    Compute the intersection of image 1 and shifted image 2 
    return the proper row shift and column shift
    """
    if volume1.shape != volume2.shape:
        raise ValueError("Images must have the same shape")

    # binary_image1 = pixel2binary(image1)
    # binary_image2 = pixel2binary(image2)
    binary_volume1 = pixel_threshold(volume1)
    binary_volume2 = pixel_threshold(volume2)
    binary_image1 = get_image4processing(binary_volume1).to(torch.int)
    binary_image2 = get_image4processing(binary_volume2).to(torch.int)

    rows, cols = binary_image2.shape

    # if Euclidean_distance(binary_image1, binary_image2) < torch.sum(binary_image1):
    #     print(f"Minimum Distance: {Euclidean_distance(binary_image1, binary_image2).item()}, Translation Offset: {(0, 0)}")
    #     return (0, 0) 

    distance_matrix = compute_distance(binary_image1, binary_image2, rows, cols, shiftrange)
    min_distance, min_idx = torch.min(distance_matrix.view(-1), 0)
    min_distance_index = np.unravel_index(min_idx.cpu().numpy(), distance_matrix.shape)
    # print(f"Minimum Distance: {min_distance.item()}, Translation Offset: {(min_distance_index[0]-shiftrange[0]//2, min_distance_index[1]-shiftrange[1]//2)}")

    return (min_distance_index[0]-shiftrange[0]//2, min_distance_index[1]-shiftrange[1]//2)


def translate_matrix(image, row_shift, col_shift):
    """
    Move the image according to the row shift and column shift using PyTorch
    """
    
    shifted_image = torch.roll(image, shifts=row_shift, dims=0)
    shifted_image = torch.roll(shifted_image, shifts=col_shift, dims=1)
    
    return shifted_image


def volume_alignment(file_list, save_path, shiftrange=(51, 51)):
    """
    Align volumes, return the synthetic volume and shift pixels of each volume comparing to the initial volume
    """
    reference_volume = np.load(file_list[0]).astype(np.int32)
    reference_image = get_image4processing(torch.from_numpy(reference_volume).cuda())
    shift_list = []
    n = len(file_list) // 100 + 1  # Extract several volumes to combine as synthetic volume
    aligned_volumes = np.zeros((n, reference_volume.shape[0], reference_volume.shape[1], reference_volume.shape[2]))

    for index, file_path in enumerate(tqdm(file_list[1:], desc="Processing Volumes", leave=False)):
        volume = np.load(file_path).astype(np.int32)
        volume = torch.from_numpy(volume).cuda()
        # image = get_image4processing(volume)

        shift = translation_matching(torch.from_numpy(reference_volume).cuda(), volume, shiftrange)
        shift_list.append(shift)
        
        if index % 100 == 0:
            if shift != (0, 0):
                aligned_volume = volume.clone()
                for z in range(volume.shape[-1]):
                    aligned_volume[:, :, z] = translate_matrix(volume[:, :, z], shift[0], shift[1])
                aligned_volumes[index // 100] = aligned_volume.cpu().numpy()
            else:
                aligned_volume = volume.clone()
                aligned_volumes[index // 100] = aligned_volume.cpu().numpy()
        
        # print(f"Processed Group {os.path.basename(file_path)[13:15]}, volume {os.path.basename(file_path)[-8:-4]}, time {end_time - start_time}")

    aligned_volumes_mip = np.max(aligned_volumes, axis=0)
    shift_list = np.array(shift_list)
    np.save(os.path.join(save_path, 'aligned_volumes_mip.npy'), aligned_volumes_mip)
    np.save(os.path.join(save_path, 'shift_list.npy'), shift_list)
    
    return aligned_volumes_mip, shift_list

# ---------------------------------Step 3. Inference on synthetic volume---------------------------------

def benchmark(model, volume, dtype = 'fp32', nwarmup = 50, nruns = 1000):
    """Benchmark the model's inference performance, measuring the average batch time (supports FP32/FP16 precision).
    
    Args:
        model (torch.nn.Module): The PyTorch model to be tested
        volume (torch.Tensor): The input tensor (must match the model's input dimensions)
        dtype (str, optional): The precision for computation, can be either 'fp32' or 'fp16', default is 'fp32'
        nwarmup (int, optional): The number of warmup iterations, default is 50
        nruns (int, optional): The number of official timing iterations, default is 1000

    Returns:
        None: The results are directly printed and not returned as values

    """
    if dtype == 'fp16':
        model.half()
        volume = volume.half()

    print("Warm up ...")
    with torch.inference_mode():
        for _ in range(nwarmup):
            features = model(volume)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.inference_mode():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(volume)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 100 == 0:
                print('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", volume.size())
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))


# ---------------------------------Step 4. Pixel intensity extraction---------------------------------


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

# ---------------------------------Step 5. Plot pixel intensity---------------------------------

def compute_delta_F_over_F(pixel_intensity, stim_intervals):
    """
    Get ΔF/F 
    """
    frame_rate = 1  
    baseline_intervals = [(start - 10, start) for start, _ in stim_intervals]
    
    F0 = np.zeros(pixel_intensity.shape[0])
    for start, end in baseline_intervals:
        F0 += pixel_intensity.iloc[:, start:end].mean(axis=1)
    F0 /= len(baseline_intervals)

    delta_F = pixel_intensity.sub(F0, axis=0)
    delta_F_over_F = delta_F.div(F0, axis=0)

    return delta_F_over_F


def curve_filter(pixel_intensity, stim_interval, spont_interval):
    """
    Smoothing 
    """
    exceeds = pixel_intensity > 500
    for location in exceeds.stack()[exceeds.stack()].index:
        row, col = location
        if col > 0 and col < pixel_intensity.shape[1] - 1:
            pixel_intensity.at[row, col] = (pixel_intensity.iat[row, col-1] + pixel_intensity.iat[row, col+1]) / 2
        elif col == 0:  
            pixel_intensity.at[row, col] = pixel_intensity.iat[row, col+1]
        elif col == pixel_intensity.shape[1] - 1:  
            pixel_intensity.at[row, col] = pixel_intensity.iat[row, col-1]
            
    rows_to_drop = []
     
    if stim_interval is not None:
        for row_index, row in pixel_intensity.iterrows():
            stim_avg = np.mean([np.mean(row.iloc[interval[0]:interval[1]]) for interval in stim_interval])
            spont_avg = np.mean([np.mean(row.iloc[interval[0]:interval[1]]) for interval in spont_interval])
            if np.abs(stim_avg - spont_avg) < 0.3 or np.abs(row.iloc[-1]-row.iloc[0]) > 50:
                rows_to_drop.append(row_index)

        pixel_intensity.drop(rows_to_drop, inplace=True)
        
    smoothed_pixel_intensity = pixel_intensity.rolling(window=5, axis=1, min_periods=1, center=True).mean()
    # smoothed_pixel_intensity = compute_delta_F_over_F(smoothed_pixel_intensity, stim_interval)

    return smoothed_pixel_intensity


def lineplot(plot_data, stim_interval, save_path, label=None):
    plt.figure(figsize=(12, 8))
    colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
    "#17becf", "#9edae5", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#1f77b4", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
    ]
    if label is not None:
        for index, (_, row) in enumerate(plot_data.iterrows()):
            plt.plot(row, label=label.iloc[index, 1], color = colors[index])
    else:
        for index, row in plot_data.iterrows():
            plt.plot(row, label=f'Neuron {index + 1}')
    
    if stim_interval is not None:
        for start, end in stim_interval:
            plt.axvspan(start, end, color='grey', alpha=0.3)

    plt.xlabel('Volume')
    plt.ylabel('Pixel Intensity')
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    
    plt.savefig(os.path.join(save_path, 'pixel_intensity.png')) if save_path else plt.show()
    return None

def load_xlsx_data(data_path, label_path):
    data = pd.read_excel(data_path)
    if label_path is not None:
        biological_id = pd.read_excel(label_path)
        biological_id = biological_id.dropna()
        biological_id.reset_index(drop=True, inplace=True)
        pixel_intensity = data.loc[data.iloc[:, 0].isin(biological_id.iloc[:, 0])]
        pixel_intensity = pixel_intensity.sort_values(by = pixel_intensity.columns[0])
        pixel_intensity = pixel_intensity.drop(pixel_intensity.columns[0], axis=1)
        pixel_intensity.reset_index(drop=True, inplace=True)
    else:
        # biological_id = data.columns[0]
        biological_id = pd.DataFrame(np.arange(0, len(data)))
        biological_id['id'] = np.arange(0, len(data))
        pixel_intensity = data.drop(data.columns[0], axis=1)
        pixel_intensity = pixel_intensity.sort_values(by=pixel_intensity.columns[0])
        pixel_intensity.reset_index(drop=True, inplace=True)

    return pixel_intensity, biological_id


# ---------------------------------step. 6 npy to h5 and zephir visualization ---------------------------------

def create_zephir_support(source_folder_path, target_folder_path):
    """
    Paste args.json and getters.py to destinated folder
    """
    shutil.copytree(source_folder_path, target_folder_path)
    
    return None


def write_metadata_json(folder_path, metadata):
    """
    Based on input data, write metadata.json
    """
    file_path = os.path.join(folder_path, 'metadata.json')
    with open(file_path, 'w') as file:
        json.dump(metadata, file, indent=4)
    
    print('Finished creating metadata.json')
    
    return None
    


def rescale_image(image, target_min, target_max, source_min = None, source_max = None):
    """
    Rescale the values in an image to a new specified range.This function includes
    checks to ensure that the target and source ranges are valid.
    Parameters:
    image (numpy.ndarray): The input image array with pixel values.
    target_min : The minimum value of the target range.
    target_max : The maximum value of the target range.
    source_min (optional): The minimum value of the image's original range.
                           If None, it is automatically computed from the image.
    source_max (optional): The maximum value of the image's original range.
                           If None, it is automatically computed from the image.
    Returns:
    numpy.ndarray: The rescaled image array where the original image values have been
                   scaled to fit within the new target range, while ensuring that all
                   values lie within this range using clipping.
    Raises:
    ValueError: If the target or source ranges are invalid (i.e., min is not less than max).
    """
    # Check that target_min is less than target_max
    if target_min >= target_max:
        raise ValueError("target_min must be less than target_max")
    # If source_min or source_max are not provided, compute them from the image
    if source_min is None:
        source_min = np.min(image)
    if source_max is None:
        source_max = np.max(image)
    image_float64 = image.astype(np.float64)
    # Check that source_min is less than source_max
    if source_min >= source_max:
        raise ValueError("source_min must be less than source_max")
    # Compute the rescaled image with values adjusted to the new range and clip to ensure
    # values stay within target_min and target_max
    rescaled_image = np.clip((image_float64 - source_min) / (source_max - source_min) * (target_max - target_min) + target_min,
                             target_min, target_max).astype(image.dtype)
    return rescaled_image



def volume_stack_and_create_h5_0(gcamp_file_list, ref_file_list, zephir_folder_path):
    """
    Input: list of gcamp_volume.npy file paths and ref_volume.npy file paths
    Output: HDF5 file containing stacked volumes, this function remains the original volumes
    """
    h5file_path = os.path.join(zephir_folder_path, 'data.h5')
    if os.path.exists(h5file_path):
        os.remove(h5file_path)
        print('Deleting the existing data.h5')

    # Initialize the HDF5 file
    with h5py.File(h5file_path, 'a') as f:
        for index, (gcamp_path, ref_path) in enumerate(tqdm(zip(gcamp_file_list, ref_file_list), desc="Creating data.h5", leave=False, total=len(gcamp_file_list))):
            gcamp_volume = np.load(gcamp_path)
            ref_volume = np.load(ref_path)
            volume_channel = np.stack((ref_volume, gcamp_volume), axis=0)
            # print(volume_channel.shape)
            rescaled_volume = rescale_image(volume_channel, 0, 255).astype(np.uint8).transpose(0, 3, 2, 1)
            
            if index == 0:
                T, C, Z, Y, X = len(gcamp_file_list), rescaled_volume.shape[0], rescaled_volume.shape[1], rescaled_volume.shape[2], rescaled_volume.shape[3]
                dset = f.create_dataset('/data', (T, C, Z, Y, X), dtype='uint8')
            
            dset[index] = rescaled_volume

    print('Finished creating data.h5')
    return None


def volume_stack_and_create_h5(gcamp_file_list, ref_file_list, zephir_folder_path):
    """
    Input: list of gcamp_volume.npy file paths and ref_volume.npy file paths
    Output: HDF5 file containing stacked volumes, cut off the vols to 30
    """

    # cut off gcamp_file_list and ref_file_list to the same length 30
    gcamp_file_list = gcamp_file_list[:30]
    ref_file_list = ref_file_list[:30]
    
    h5file_path = os.path.join(zephir_folder_path, 'data.h5')
    if os.path.exists(h5file_path):
        os.remove(h5file_path)
        print('Deleting the existing data.h5')

    # Initialize the HDF5 file
    with h5py.File(h5file_path, 'a') as f:
        for index, (gcamp_path, ref_path) in enumerate(tqdm(zip(gcamp_file_list, ref_file_list), desc="Creating data.h5", leave=False, total=len(gcamp_file_list))):
            gcamp_volume = np.load(gcamp_path)
            ref_volume = np.load(ref_path)
            volume_channel = np.stack((ref_volume, gcamp_volume), axis=0)
            # print(volume_channel.shape)
            rescaled_volume = rescale_image(volume_channel, 0, 255).astype(np.uint8).transpose(0, 3, 2, 1)
            
            if index == 0:
                T, C, Z, Y, X = len(gcamp_file_list), rescaled_volume.shape[0], rescaled_volume.shape[1], rescaled_volume.shape[2], rescaled_volume.shape[3]
                dset = f.create_dataset('/data', (T, C, Z, Y, X), dtype='uint8')
            
            dset[index] = rescaled_volume

    print('Finished creating data.h5')
    return None


def generate_uniform_colors(n):
    """
    Generate n uniformly distributed colors in hexadecimal format.
    Colors are sampled by evenly spacing the hue value in the HSL color space.
    Saturation and lightness are fixed to ensure vibrant, consistent colors.
    """
    colors = []
    for i in range(n):
        # Calculate the hue spaced evenly around the color wheel
        hue = i / n
        # Fixed saturation and lightness
        saturation, lightness = 0.9, 0.6
        # Convert HSL to RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert RGB from 0-1 range to 0-255 range
        rgb = tuple(int(x * 255) for x in rgb)
        # Format as hex
        color = "#{:02x}{:02x}{:02x}".format(*rgb)
        colors.append(color)
    return colors

def create_worldlines(worldlines_id, zephir_folder_path):
    '''
    input: neurons numeric id
    output: worldlines.h5
    '''
    h5file_path = os.path.join(zephir_folder_path, 'worldlines.h5')
    l = np.unique(worldlines_id).shape[0]
    # l = len(worldlines_id)
    random_colors = generate_uniform_colors(l)
    color_map = dict(zip(np.unique(worldlines_id), random_colors))
    colors = [color_map[id] for id in worldlines_id]
    
    try:
        with h5py.File(h5file_path, 'a') as f:
            f.create_dataset('/name', data=np.array(['null']*l, dtype='S'))
            f.create_dataset('/id', data=np.array(worldlines_id, dtype='uint8'), dtype='uint8')
            f.create_dataset('/color', data=np.array(random_colors, dtype='S'))
    except Exception as e:
        print("An error occurred:", e)
        return None


def neuron_alignment(neuron_pt_tuple, shift_list):
    '''
    align neuron_pt_tuple to each volume
    '''
    neuron_all = []

    for index, shift in enumerate(shift_list):
        row_shift = -shift[1]
        col_shift = -shift[0]
        neuron_pt_tuple_shift = neuron_pt_tuple.copy()
        neuron_pt_tuple_shift[:, 0] += row_shift
        neuron_pt_tuple_shift[:, 1] += col_shift

        neuron_all.append(neuron_pt_tuple_shift)

    return neuron_all

def create_raw_annotations(neuron_all, zephir_folder_path):
    '''
    input: all neuron_pt_tuple (time x neuron_pt_tuple)
    output: annotations.h5
    '''
    h5file_path = os.path.join(zephir_folder_path, 'annotations.h5')
    l = len(neuron_all)
    if os.path.exists(h5file_path):
        os.remove(h5file_path)
        print('deleting the existing annotations.h5')
    try:
        with h5py.File(h5file_path, 'a') as f:
            if '/x' not in f:
                max_shape = (None,)
                f.create_dataset('/x', shape=(0,), maxshape=max_shape, dtype='float32')
                f.create_dataset('/y', shape=(0,), maxshape=max_shape, dtype='float32')
                f.create_dataset('/z', shape=(0,), maxshape=max_shape, dtype='float32')
                f.create_dataset('/id', shape=(0,), maxshape=max_shape, dtype='uint32')
                f.create_dataset('/parent_id', shape=(0,), maxshape=max_shape, dtype='uint16')
                f.create_dataset('/worldline_id', shape=(0,), maxshape=max_shape, dtype='uint8')
                f.create_dataset('/provenance', shape=(0,), maxshape=max_shape, dtype='S4')
                f.create_dataset('/t_idx', shape=(0,), maxshape=max_shape, dtype='uint16')

            for volume_index, neuron_pt_tuple in enumerate(neuron_all):
                n = neuron_pt_tuple.shape[0]
                new_size = f['/x'].shape[0] + n
                f['/x'].resize(new_size, axis=0)
                f['/x'][-n:] = neuron_pt_tuple[:, 1] / 1024
                f['/y'].resize(new_size, axis=0)
                f['/y'][-n:] = neuron_pt_tuple[:, 0] / 1024
                f['/z'].resize(new_size, axis=0)
                f['/z'][-n:] = neuron_pt_tuple[:, 2] / (1.5/0.3*18) # z-axis normalization factor: frame_num * (z_unit / XOY_unit)
                # f['/id'].resize(new_size, axis=0)
                # f['/id'][-n:] = np.zeros(n)
                f['/worldline_id'].resize(new_size, axis=0)
                f['/worldline_id'][-n:] = np.arange(0, n)
                f['/provenance'].resize(new_size, axis=0)
                f['/provenance'][-n:] = np.array(['ANTT']*n, dtype='S4')
                f['/t_idx'].resize(new_size, axis=0)
                f['/t_idx'][-n:] = np.full(n, volume_index, dtype='uint16')

                # 计算当前全局起始索引（即之前已写入的数据总量）
                current_start = new_size - n  # 等同于 f['/id'].shape[0] - n
                # 生成从 current_start 开始的连续整数序列
                id_values = np.arange(current_start, current_start + n, dtype='uint32')
                # 追加到 id 和 parent_id 数据集
                f['/id'].resize(new_size, axis=0)
                f['/id'][-n:] = id_values
                f['/parent_id'].resize(new_size, axis=0)
                f['/parent_id'][-n:] = id_values  # 假设 parent_id 和 id 相同
                
            print('finish creating new annotations.h5')

    except Exception as e:
        print("An error occurred:", e)
        return None

    return None


# ---------------------------------step. 7 video generation ---------------------------------


def load_volume_neuron_info_from_nparray(neuron_pt_tuple, shift_list, volume_number):
    
    neuron_3d_bbox = neuron_pt_tuple[:, :6].copy() # Read the 3D neuron boxes [cx, cy, cz, w, h, d]

    if volume_number>0:
        neuron_3d_bbox[:,0] -= shift_list[volume_number-1,1]
        neuron_3d_bbox[:,1] -= shift_list[volume_number-1,0]

    
    neuron_pred_id = np.arange(neuron_pt_tuple.shape[0])

    return neuron_3d_bbox, neuron_pred_id


def load_volume_image_from_npy(image_path, x_ratio = 1., y_ratio = 1., z_ratio = 1., source_min = None, source_max = None, red_pseudo_color = False):
    volume = np.load(image_path)
    volume =  np.transpose(volume, (2, 0, 1))
    mips = vis.get_mip_from_uint8_gray_volume(volume, x_ratio, y_ratio, z_ratio, source_min, source_max,red_pseudo_color)
    return mips



def list_files_by_pattern(folder_path_list, pattern= r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[w|fly]\d+_?Dt\d{6}_(\d{6})"):
    files_dict = {}
    compiled_pattern = re.compile(pattern)
    for index, folder_path in enumerate(folder_path_list):
        for file_name in os.listdir(folder_path):
            match = compiled_pattern.match(file_name)
            if match:
                last_number = 200*index+int(match.group(1))
                files_dict[last_number] = os.path.join(folder_path, file_name)
    
    return files_dict


def generate_volume_dict(sorted_volume_list):
    file_dict = {}
    for index, file_name in enumerate(sorted_volume_list):
        file_dict[index] = file_name
    
    return file_dict

def generate_volume_video(sorted_volume_list, synthetic_volume_save_path, neuron_pt_tuple, shift_list):
    
    # image_name_dict = list_files_by_pattern(sorted_volume_folder)
    image_name_dict = generate_volume_dict(sorted_volume_list)
    vis.Plot3DResult(lambda ptr: load_volume_image_from_npy(image_name_dict[ptr], z_ratio = 15.0/3.0,source_min = 130, source_max = 200,red_pseudo_color = False),
                lambda ptr: load_volume_neuron_info_from_nparray(neuron_pt_tuple, shift_list, ptr),
                split_number = 1,bbox_thickness=1,trace_length=0,
                ).save_all(synthetic_volume_save_path, len(image_name_dict), 5)
    
    return None

def zprojection_to_frame(volume):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(volume.shape[2]), np.max(volume, axis=(0, 1)))
    ax.set_ylim(120, 500)
    ax.set_ylabel('Maximum Pixel Intensity')
    ax.set_title(f"Volume Z-projection")

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img

def zprojection_to_video(volume_list, video_path):
    
    # Generate the first frame to get the dimensions
    first_frame = zprojection_to_frame(np.load(volume_list[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, 1.0, (width, height))

    for i, volume_path in enumerate(tqdm(volume_list, desc='Processing Z-projection')):
        volume = np.load(volume_path)
        frame = zprojection_to_frame(volume)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()
    
    return None


def extract_subset(input_file, datasets, output_file, num_rows=None):
    with h5py.File(input_file, 'r') as original_file:
        with h5py.File(output_file, 'w') as new_file:
            for dataset_name in datasets:
                if num_rows is not None:
                    data = original_file[dataset_name][:num_rows]
                else:
                    data = original_file[dataset_name][:]
                new_file.create_dataset(dataset_name, data=data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "CBMI 1 pipeline")
    parser.add_argument('--process-stack-root', type = str, default = "/home/data4/WJH/olfactory_data/20241116_wen0065/w1/green", help = '')
    parser.add_argument('--save-preprocess-result-root', type = str, default = '')
    parser.add_argument('--json-store-root', type = str, default = "/home/data4/WJH/result/20241116_wen0065/w1/2")
    parser.add_argument('--zephir-support-root', type = str, default="/home/wenlab-user/JinghaoWang/new_new/code_v1.10/src/zephir")
    parser.add_argument('--config', type = str, default = "/home/wenlab-user/JinghaoWang/new_new/code_v1.10/src/configs/inference/240623.json", help = "")
    parser.add_argument('--error-filename', type = str, default = "AErrorStacks.txt", help = "")
    parser.add_argument('--re-infer-error-stacks', action = "store_true", help = "")
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[w|fly]\d+_?Dt\d{6}")
    parser.add_argument('--preprocessing-mode', type = int, default = 0)
    parser.add_argument('--only-preprocessing', action = "store_true")
    parser.add_argument('--volume-window', type = int, default = 1, help = "number of processing volumes once")
    parser.add_argument('--volume-start-idx', type = int, default = 0, help = "")
    parser.add_argument('--pre-resize', type = int, default = 1)
    parser.add_argument('--pre-rescale-pixels', type = int, default = 1)
    parser.add_argument('--pre-resize-size', type = int, default = 680)
    parser.add_argument('--pre-scale-factor', type= int, default = 1)
    parser.add_argument('--mp', type= int, default = 0)
    parser.add_argument('--shift-range', type=str, default="51,51", help="Shift range as a comma-separated pair of integers (e.g., 51,51)")
    parser.add_argument('--run-processing', type=str, choices=['True', 'False'], default='True', help="Set to 'True' to run processing, 'False' to skip.")
    parser.add_argument('--zephir', action = "store_true", help = "save zephir result")
    parser.add_argument('--save-neuron-pt-tuple', action = "store_true", help = "save neuron points")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    args.zrange = config["zrange"]
    save_fig_root = os.path.join(args.json_store_root, "figs")
    args.save_preprocess_result_root = os.path.join(args.json_store_root, "mip") if args.preprocessing_mode >= 3 else args.save_preprocess_result_root
    error_file_path = os.path.join(args.json_store_root, args.error_filename)
    volume_save_path =  os.path.join(args.json_store_root, "volume")
    synthetic_volume_save_path =  os.path.join(args.json_store_root, "synthetic_volume")

    
    os.makedirs(args.json_store_root, exist_ok = True)
    os.makedirs(save_fig_root, exist_ok = True)
    os.makedirs(volume_save_path, exist_ok = True)
    os.makedirs(synthetic_volume_save_path, exist_ok = True)
    # os.makedirs(zephir_save_path, exist_ok = True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f'Using GPU:{current_device} - {device_name}')
    else:
        print('CUDA is not available, Using CPU')
    
    print_info_message(args)
    
    if args.run_processing == 'True':
        processing_stacks = extract_waiting_stacks(
            args.json_store_root, error_file_path, args.name_reg,
            re_infer_error=args.re_infer_error_stacks,
            paths=glob(os.path.join(args.process_stack_root, '*.mat'))
        )[args.volume_start_idx:]

        for stream in tqdm(split_processing_streams(processing_stacks, max_mats_one_stream=args.volume_window), colour="#FF1493"):
            results, preprc_results = infer_one_batch(stream, args, volume_save_path)
    else:
        print("Processing skipped as 'run-processing' is set to 'False'.")
    
        
    volume_folder = glob(os.path.join(volume_save_path, '*'))
    # volume_folder = glob(os.path.join(os.path.join(args.process_stack_root, 'volume'), '*'))
    sorted_volume_folder = sorted(volume_folder, key=lambda x: int(os.path.basename(x).split('_')[0][6:]))
    sorted_volume_list = load_datapath(sorted_volume_folder)
    
    zprojection_to_video(sorted_volume_list, os.path.join(synthetic_volume_save_path, 'z_projection.avi'))
    # Convert shift_range string to a tuple of integers
    shift_range = tuple(map(int, args.shift_range.split(',')))
    aligned_volumes_mip, shift_list = volume_alignment(sorted_volume_list, synthetic_volume_save_path, shiftrange=shift_range)
   
    model = Treeformer_End2End(

        ext_path = config["ext_path"],
        det_path = config["det_path"],
        rec_path = config["rec_path"],

        ext_input_dim = [config["zrange"][1] - config["zrange"][0]] + config["ext_input_dim"],
        det_input_dim = [config["zrange"][1] - config["zrange"][0]] + config["det_input_dim"],
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
        save_path = args.json_store_root,
        deg_t = 100,
        weight_W = 0.2,
        warmup_num_vol = 3,
    )
    

            
    volume = torch.HalfTensor(aligned_volumes_mip.transpose(2, 0, 1)[:, np.newaxis, :, :].astype(np.float32)).cuda()
    if args.pre_resize:
        scale_factor = args.pre_resize_size / max(volume.shape[2:])
        volume = F.interpolate(volume, scale_factor = scale_factor, mode = "nearest")
    if args.pre_rescale_pixels:
        _std, _mean = volume.std(), volume.mean()
        print_log_message(f"std: {_std}, mean: {_mean}")
        volume[:] = -(_mean / _std * 13 - 103) + 13 / _std * volume

    with torch.inference_mode():
        head_bboxes, regions, neuron_dict, neuron_emb, is_warnings, supp_r = model(volume)
        region_pt_tuple, region_ptrs, region_map, neuron_pt_tuple, (num_region, num_neuron), (std, max_min) = supp_r
        if args.pre_resize:
            neuron_pt_tuple[:, [0, 1, 3, 4]] = neuron_pt_tuple[:, [0, 1, 3, 4]] / scale_factor
        np.save(os.path.join(synthetic_volume_save_path, 'neuron_pt_tuple.npy'), neuron_pt_tuple.cpu().numpy())
        buffer.pickup.vol_infos = {os.path.join(synthetic_volume_save_path, 'aligned_volumes_mip.npy'): [std, max_min]}
        
        neuron_pred_ids, region_pred_ids, num_neuron_idv = buffer(neuron_emb, neuron_dict, num_region, any(is_warnings))
        # pt_tuple_data = torch.concat([neuron_pt_tuple.cpu(), torch.FloatTensor(neuron_pred_ids).unsqueeze(1)], dim = 1).cpu().numpy()
        print(f" the number of regions: {num_region}, \t the number of neurons: {num_neuron}, \t  the number of neurons: {num_neuron_idv}")
        draw_volume_result(volume, head_bboxes, regions, region_pred_ids, region_ptrs, synthetic_volume_save_path, name = "aligned_volumes_mip", verbose = False)

    pixel_intensity_df = pixel_intensity_extraction(sorted_volume_list,
                                                    shift_list, 
                                                    neuron_pt_tuple.cpu().numpy(), 
                                                    synthetic_volume_save_path)
    # pixel_intensity_df, labels = load_xlsx_data(os.path.join(synthetic_volume_save_path, 'pixel_intensity.xlsx'), None)

    
    # stim_interval = None
    # spont_interval = None
    
    # smoothed_pixel_intensity = curve_filter(pixel_intensity_df, stim_interval, spont_interval)
    # lineplot(smoothed_pixel_intensity, stim_interval, synthetic_volume_save_path)


    generate_volume_video(sorted_volume_list, synthetic_volume_save_path, neuron_pt_tuple.cpu().numpy(), shift_list)
    
    # get neuron_pt_tuple in all volumes
    aligned_neuron_pt_tuple = neuron_alignment(neuron_pt_tuple.cpu().numpy(), shift_list)

    if args.save_neuron_pt_tuple:
        neuron_pt_tuple_save_path = os.path.join(synthetic_volume_save_path, 'all_neuron_pt_tuple.npy')
        all_neuron_pt_tuple = np.empty((len(shift_list),neuron_pt_tuple.shape[0], neuron_pt_tuple.shape[1]), dtype=np.float32)
        for index, neuron in enumerate(aligned_neuron_pt_tuple):
            all_neuron_pt_tuple[index] = neuron
        np.save(neuron_pt_tuple_save_path, all_neuron_pt_tuple)
        print(f"Neuron points cloud have been saved to {neuron_pt_tuple_save_path}")
            

    if args.zephir:
        zephir_save_path = os.path.join(args.json_store_root, 'zephir')
        create_zephir_support(args.zephir_support_root, zephir_save_path)
        # metadata = {
        # "shape_t": len(sorted_volume_list),
        # "shape_c": 2,
        # "shape_z": 18,
        # "shape_y": 1024,
        # "shape_x": 1024,
        # "dtype": "uint8"
        # }

        metadata = {
        "shape_t": 10,
        "shape_c": 2,
        "shape_z": 18,
        "shape_y": 1024,
        "shape_x": 1024,
        "dtype": "uint8"
        }
        
        write_metadata_json(zephir_save_path, metadata)
        volume_stack_and_create_h5(sorted_volume_list, sorted_volume_list, zephir_save_path)
        create_worldlines(np.arange(0, len(neuron_pt_tuple)), zephir_save_path)
        create_raw_annotations(aligned_neuron_pt_tuple, zephir_save_path)
        

        # process data.h5 file, take 10 volumes
        data_datasets = ['data']
        extract_subset(zephir_save_path + '/data.h5', data_datasets, zephir_save_path + '/data_sub.h5', num_rows=10)

        # process annotations.h5 file, take 10 neurons
        annotations_datasets = ['id','parent_id','provenance', 't_idx', 'worldline_id', 'x', 'y', 'z']
        extract_subset(zephir_save_path + '/annotations.h5', annotations_datasets, zephir_save_path + '/subset_annotations.h5', num_rows=num_neuron*10)
        print("nwe HDF5 file 'data_sub.h5'、'subset_annotations.h5' have been created")   
    
    
    
    