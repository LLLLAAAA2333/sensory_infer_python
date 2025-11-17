import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

from src.loop_base import automatic_foreground_process

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

def pixel_threshold_old(volume, gaussian_k: int = 9, maxpool_k: int = 125, bg_t_r: float = 1.2, rescale_p: float = .97, only_scale: bool = False):
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)
    volume_torch = volume.to(torch.float32)

    if volume_torch.dim() == 3:
        volume_torch =  volume.to(torch.float32).permute(2, 0, 1).unsqueeze(1)
    elif volume_torch.dim() == 4:
        pass
    if gaussian_k > 1:  
        volume_torch[:] = gaussian_blur(volume_torch, kernel_size = gaussian_k)
    threshold = calc_volume_bg_threshold(volume_torch, kernel_and_stride = maxpool_k, t_ratio = bg_t_r)
    mask = volume_torch > threshold
    return mask.squeeze(1).permute(1, 2, 0)

def pixel_threshold(volume, gaussian_k: int = 9, maxpool_k: int = 125, bg_t_r: float = 1.2, rescale_p: float = .97, only_scale: bool = False):
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)

    volume_torch = volume.to(torch.float32)

    if volume_torch.dim() == 3:
        volume_4d = volume_torch.permute(2, 0, 1).unsqueeze(1).contiguous()
        transpose_back = lambda tensor: tensor.squeeze(1).permute(1, 2, 0)
    elif volume_torch.dim() == 4:
        volume_4d = volume_torch.contiguous()
        transpose_back = lambda tensor: tensor.squeeze(1)
    else:
        raise ValueError(f"Unsupported volume dimension {volume_torch.dim()} for pixel_threshold")

    processed = automatic_foreground_process(
        volume_4d.clone(),
        gaussian_k = gaussian_k,
        maxpool_k = maxpool_k,
        bg_t_r = bg_t_r,
        rescale_p = rescale_p,
        only_scale = only_scale,
    )

    if only_scale:
        return transpose_back(processed)

    mask = processed > 0
    return transpose_back(mask)





