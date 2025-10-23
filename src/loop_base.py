# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com
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

# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
# from src.infers.utils.prints import print_log_message, print_warning_message
# from src.infers.preproc.inference import auto_preprocess
# from src.infers.utils.packages import *
# from src.infers.inference import save_h5file
# from src.infers.utils.utils import load_aravi_lab_RFP_volumes, load_h5_results_to_capture_calcium, capture_calcium_dynamics_in_a_volume, organize_dynamics, export_to_excel



def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.1):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)), (kernel_size, kernel_size))
    return kernel[np.newaxis, np.newaxis]  # [1, 1, kernel_size, kernel_size]


def gaussian_blur(volume, kernel_size: int = 5):
    g_kernel = gaussian_kernel(kernel_size, 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8)

    return F.conv2d(volume, torch.tensor(g_kernel, dtype = volume.dtype, device = volume.device), padding = kernel_size // 2)


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


def estimate_A_est(pixels, noise_mean, threshold, percentile: float = 0.97):
    A_est = (torch.quantile((pixels - noise_mean).float(), percentile) / (threshold - noise_mean) ** (1 - percentile)) ** (1 / percentile)
    print_log_message(f"A_est: {A_est}")
    return A_est


def normalize_pixels(pixels, noise_mean, A_est, bias = 1.):
    pixels = (pixels - noise_mean) / A_est + bias
    return pixels


def geometric_normalization(pixels, noise_mean, threshold, percentile: float = 0.97):
    A_est = estimate_A_est(pixels, noise_mean, threshold, percentile)
    pixels = pixels / A_est + 1
    return pixels


def print_gpu_memory():
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
    
def automatic_foreground_process(volume, gaussian_k: int = 9, maxpool_k: int = 128, bg_t_r: float = 1.2, rescale_p: float = .97, only_scale: bool = False):
    """TODO: try 3D gaussian blur"""
    if only_scale:
        raw = volume.clone()
    # Gaussian blur
    if gaussian_k > 1:
        volume[:] = gaussian_blur(volume, kernel_size = gaussian_k)
    # Calculate background/foreground threshold
    threshold = calc_volume_bg_threshold(volume, kernel_and_stride = maxpool_k, t_ratio = bg_t_r)
    # Thresholding
    mask = volume > threshold
    # Background mean
    try:
        # Move the data to CPU for the mean calculation
        volume_cpu = volume.cpu()
        mask_cpu = mask.cpu()
        noise_mean = volume_cpu[~mask_cpu].mean()
        
        noise_mean = noise_mean.cuda()
        
        print(f"Noise mean: {noise_mean}")
        
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print("CUDA out of memory. Try reducing batch size or freeing up memory.")
            torch.cuda.empty_cache()
            print_gpu_memory()
        else:
            raise e
    # noise_mean = volume[~mask].mean()
    volume[~mask] = 0.
    # TODO: delete debug info
    # print_log_message(f" \n Threshold: {threshold}， FG / whole: {mask.float().sum() / volume.numel() * 100 :.3f} %， "
    #                   f"Background mean: {noise_mean}, Foreground mean: {volume[mask].mean()}")

    # Geometric normalization
    A_est = estimate_A_est(volume[mask], noise_mean, threshold, percentile = rescale_p)

    if only_scale:
        return normalize_pixels(raw, noise_mean, A_est, bias = 1.)
    else:
        volume[mask] = normalize_pixels(volume[mask], noise_mean, A_est, bias = 1.)
        return volume


def load_matlab_volumes_dict(args, stream):
    # Stack loading
    preprc_results = auto_preprocess(mode = args.preprocessing_mode, paths = stream, args = args)

    head_bbox_dict, scale_factors = dict(), dict()

    # In-place operation
    for key, volume in preprc_results.items():
        head_bbox_dict[key] = volume[2][3] if args.preprocessing_mode > 0 else None
        preprc_results[key] = torch.HalfTensor((volume[0] if args.preprocessing_mode > 0 else volume).transpose((2, 0, 1))[:, np.newaxis].astype(np.float32)).cuda()
        scale_factors[key] = args.pre_scale_factor
        if args.mp > 0:
            preprc_results[key] = torch.max_pool2d(preprc_results[key], kernel_size = args.mp, stride = args.mp)
            scale_factors[key] /= args.mp
            print_warning_message(f"mp: {scale_factors[key]}")

        if args.pre_resize:
            scale_factor = args.pre_resize_size / max(preprc_results[key].shape[2:])
            scale_factors[key] *= scale_factor
            # print("pre_resize:", scale_factor)
            preprc_results[key] = F.interpolate(preprc_results[key], scale_factor = scale_factor, mode = "nearest")

        if args.pre_rescale_pixels:
            _std, _mean = preprc_results[key].std(), preprc_results[key].mean()
            print_log_message(f"std: {_std}, mean: {_mean}")
            # print("pre_rescale_pixels")
            # preprc_results[key][:] = -(_mean / _std * 22 - 119) + 22 / _std * preprc_results[key]
            preprc_results[key][:] = -(_mean / _std * 13 - 103) + 13 / _std * preprc_results[key]

        if head_bbox_dict[key] is not None:
            head_bbox_dict[key] = [i * scale_factors[key] for i in head_bbox_dict[key]]

        # ------------ New preprocessing ------------
        # preprc_results[key] = automatic_foreground_process(preprc_results[key], gaussian_k = 13, maxpool_k = 128, bg_t_r = 1.2, rescale_p = .97, only_scale = False)

    return preprc_results, head_bbox_dict, scale_factors


def load_aravi_RFP_stream_dict(args, stream, model):
    return {name: value for path in stream for name, value in load_aravi_lab_RFP_volumes(args, path, model.det_input_dim, name = "Helena")}  # TODO: name


def load_matlab_calcium(args, stream):
    args.preprocessing_mode = 0
    preprc_results = auto_preprocess(mode = args.preprocessing_mode, paths = stream, args = args)

    for key, volume in preprc_results.items():
        preprc_results[key] = torch.tensor(volume.transpose((2, 0, 1))[:, np.newaxis].astype(np.float32), dtype = torch.float32).cuda()

    return preprc_results


def infer_one_batch(model, buffer, stream, args):
    print_log_message(f"Processing stream: {stream}")

    if args.loading_mode == "matlab":
        (preprc_results, head_bbox_dict, scale_factors), results = load_matlab_volumes_dict(args, stream), dict()
    elif args.loading_mode == "hdf5_aravi":
        preprc_results, results = load_aravi_RFP_stream_dict(args, stream, model), dict()
    else:
        raise ValueError(f"Unsupported loading mode: {args.loading_mode}!")

    with torch.inference_mode():
        for name, volume in tqdm(preprc_results.items(), colour = "#9370DB"):
            if args.store_track_tri_view_result or args.store_ptcloud_result or args.store_number_result or args.store_pdf_result or args.store_json_result:
                # np.save(f"/home/cbmi/CBMI_python/data/temp/for_zephir/numpy_vol/{name}.npy", volume.cpu().numpy())
                # try:
                head_bboxes, regions, neuron_dict, neuron_emb, is_warnings, supp_r = model(volume, head_bbox_dict[name] if args.preprocessing_mode > 0 else None)
                region_pt_tuple, region_ptrs, region_map, neuron_pt_tuple, _filtered_region_mask, (num_region, num_neuron), (std, max_min, _mean, rot) = supp_r
                # buffer.pickup.vol_infos = {name: [std, max_min]}

                is_warnings = (False, False, False, False)

                neuron_pred_ids, region_pred_ids, num_neuron_idv, (is_warmup, is_count4sorting) = buffer(neuron_emb, neuron_pt_tuple, neuron_dict, num_region, any(is_warnings))
                # except Exception as e:
                #     print_warning_message(f"Warning: {name} can't be processed, because {e}!")
                #     results[name] = [None,] * 5
                #     continue
                tqdm.write(f"name: {name}, \t shape: {volume.shape} \t the number of regions: {num_region}, \t the number of neurons: {num_neuron}, \t  the number of W: {num_neuron_idv}")
                if any(is_warnings):
                    print_warning_message(f"Warning: {name} has been processed with warning {is_warnings} (ext, det, merge, rec)!")

                # ---------------- Saving for neuron traces ----------------
                if args.store_track_tri_view_result:
                    save_h5file(path = args.h5_file_path, volume_name = name, volume = volume,
                                pre_scale_factor = scale_factors[name],
                                is_warnings = is_warnings,
                                is_warmup = is_warmup,
                                is_count4sorting = is_count4sorting,
                                head_bboxes = head_bboxes,

                                neuron_pred_ids = neuron_pred_ids,
                                region_pred_ids = region_pred_ids,

                                region_ptrs = region_ptrs,
                                regions = regions,
                                region_pt_dim2 = region_pt_tuple[_filtered_region_mask][:, :2],  # for point cloud result
                                neuron_pt_tuple = recover_cxcy_on_pt_tuple(pt_tuple = neuron_pt_tuple, rot = rot, mean = _mean),
                                aligned_neuron_pt_tuple = neuron_pt_tuple,

                                neuron_emb = neuron_emb,
                                num_region = num_region,
                                neuron_dict = neuron_dict,
                                num_neuron_idv = num_neuron_idv,
                                mean = _mean,
                                rot = rot,
                                num_info = (num_region, num_neuron) + (num_neuron_idv, int(any(is_warnings))),
                                )


            elif args.store_raw_tri_view_result:
                save_h5file(path = args.h5_file_path, volume_name = name, volume = volume)


def capture_calcium_one_batch(stream, args):
    for vol_name, vol_calcium in tqdm(load_matlab_calcium(args, stream).items(), colour = "#9370DB"):
        # np.save(f"/home/cbmi/CBMI_python/data/temp/for_zephir/gcamp/{vol_name}.npy", vol_calcium.cpu().numpy())
        vol_bools, regions, region_pred_ids, region_ptrs = load_h5_results_to_capture_calcium(args.h5_file_path, vol_name)
        vol_neuronal_dynamics = capture_calcium_dynamics_in_a_volume(vol_calcium, regions, region_pred_ids, region_ptrs, calc_type = "Mean")
        yield vol_name, vol_neuronal_dynamics, vol_bools