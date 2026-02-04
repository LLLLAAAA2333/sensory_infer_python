import os
import sys
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import math
from typing import Tuple, Optional
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from src.inference.blur import pixel_threshold, get_image4processing
from src.comm_utils.prints import print_info_message, print_log_message, print_warning_message


def get_gaussian_kernel_2d(sigma: float, device: str = 'cuda') -> torch.Tensor:
    """
    Dynamically generates a 2D Gaussian convolution kernel.
    """
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    # Create 1D Gaussian
    k1d = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
    k1d = torch.exp(-k1d**2 / (2 * sigma**2))
    k1d = k1d / k1d.sum()
    
    # Outer product for 2D
    kernel = k1d.view(-1, 1) @ k1d.view(1, -1)
    return kernel.view(1, 1, kernel_size, kernel_size)


def apply_dog_filter(img_2d: torch.Tensor, sigma_fg: float = 1.0, sigma_bg: float = 6.0) -> torch.Tensor:
    """
    Applies Difference of Gaussians (DoG) filter to 2D image.
    Formula: dog = blur(sigma_fg) - blur(sigma_bg)
    Threshold: dog = ReLU(dog), then values < 0.05 * max are set to 0.
    
    Uses reflection padding to avoid boundary artifacts.
    """
    device = img_2d.device
    # Ensure input is 4D [B, C, H, W]
    if img_2d.dim() == 2:
        input_tensor = img_2d.unsqueeze(0).unsqueeze(0).float()
    else:
        input_tensor = img_2d.float()

    k_fg = get_gaussian_kernel_2d(sigma_fg, device)
    k_bg = get_gaussian_kernel_2d(sigma_bg, device)

    # Calculate padding sizes
    p_fg = k_fg.shape[-1] // 2
    p_bg = k_bg.shape[-1] // 2

    # Apply blurs with reflection padding
    # Format for F.pad: (left, right, top, bottom)
    blur_fg = F.conv2d(F.pad(input_tensor, (p_fg, p_fg, p_fg, p_fg), mode='reflect'), k_fg, padding=0)
    blur_bg = F.conv2d(F.pad(input_tensor, (p_bg, p_bg, p_bg, p_bg), mode='reflect'), k_bg, padding=0)

    dog = blur_fg - blur_bg
    dog = F.relu(dog)
    
    # Adaptive soft threshold
    dog_max = dog.max()
    if dog_max > 0:
        mask = dog >= (0.05 * dog_max)
        dog = dog * mask.float()
    
    return dog.squeeze()


# ----------------------------- Old methods for volume alignment(Exhaustive) ----------------------------- #
# ----------------------------- Old methods for volume alignment(Exhaustive) ----------------------------- #
def translate_matrix(image, row_shift, col_shift):
    """
    Move the image according to the row shift and column shift using PyTorch
    """
    
    shifted_image = torch.roll(image, shifts=row_shift, dims=0)
    shifted_image = torch.roll(shifted_image, shifts=col_shift, dims=1)
    
    return shifted_image

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


def volume_alignment(file_list, save_path, shiftrange=(51, 51), zrange=None, method='dog'):
    """
    Align volumes, return the synthetic volume and shift pixels of each volume comparing to the initial volume
    """
    reference_volume = np.load(file_list[0])
    reference_volume = _apply_zrange(reference_volume, zrange).astype(np.int32)
    reference_image = get_image4processing(torch.from_numpy(reference_volume).cuda())
    shift_list = []
    n = len(file_list) // 20 + 1  # Extract several volumes to combine as synthetic volume
    aligned_volumes = np.zeros((n, reference_volume.shape[0], reference_volume.shape[1], reference_volume.shape[2]))

    for index, file_path in enumerate(tqdm(file_list[1:], desc="Processing Volumes", leave=False)):
        volume = np.load(file_path)
        volume = _apply_zrange(volume, zrange).astype(np.int32)
        volume = torch.from_numpy(volume).cuda()
        # image = get_image4processing(volume)

        shift = translation_matching_fft(torch.from_numpy(reference_volume).cuda(), volume, shiftrange, method=method)
        shift_list.append(shift)
        
        if index % 20 == 0:
            if shift != (0, 0):
                aligned_volume = volume.clone()
                for z in range(volume.shape[-1]):
                    aligned_volume[:, :, z] = translate_matrix(volume[:, :, z], shift[0], shift[1])
                aligned_volumes[index // 20] = aligned_volume.cpu().numpy()
            else:
                aligned_volume = volume.clone()
                aligned_volumes[index // 20] = aligned_volume.cpu().numpy()
        
        # print(f"Processed Group {os.path.basename(file_path)[13:15]}, volume {os.path.basename(file_path)[-8:-4]}, time {end_time - start_time}")

    aligned_volumes_mip = np.max(aligned_volumes, axis=0)
    shift_list = np.array(shift_list)
    np.save(os.path.join(save_path, 'aligned_volumes_mip.npy'), aligned_volumes_mip)
    np.save(os.path.join(save_path, 'shift_list.npy'), shift_list)
    
    return aligned_volumes_mip, shift_list

# ----------------------------- volume alignment(Phase Correlation) ----------------------------- #
def translation_matching_phase_corr(volume1_gpu, volume2_gpu, device='cuda'):
    """
    Compute the translation offset between volume1 and volume2 using phase correlation
    return the proper row shift and column shift
    """
    binary_volume1 = pixel_threshold(volume1_gpu.permute(2, 0, 1).unsqueeze(1)).squeeze(1).permute(1, 2, 0)
    binary_volume2 = pixel_threshold(volume2_gpu.permute(2, 0, 1).unsqueeze(1)).squeeze(1).permute(1, 2, 0)

    binary_image1 = get_image4processing(binary_volume1).to(torch.int) # (Y, X)
    binary_image2 = get_image4processing(binary_volume2).to(torch.int) # (Y, X)

    img1_np = binary_image1.cpu().numpy().astype(np.float32)
    img2_np = binary_image2.cpu().numpy().astype(np.float32)

    if img1_np.shape != img2_np.shape:
        print_warning_message(f"MIP shape mismatch {img1_np.shape} vs {img2_np.shape}. Skipping alignment.")
        return (0, 0), -1.0
    
    shift_xy, response = cv2.phaseCorrelate(img1_np, img2_np)
    shift_x, shift_y = shift_xy
    shift_yx = (shift_y, shift_x)
    return shift_yx, -response

def compute_distance_fft(binary_image1, binary_image2, rows, cols, shiftrange=(61, 61)):
    """
    Optimized distance computation using FFT.
    Calculates sum((A - B_shift)^2) = sum(A^2) + sum(B^2) - 2*convolution(A, B)
    """
    # Ensure inputs are float for FFT
    img1 = binary_image1.float()
    img2 = binary_image2.float()

    # 1. Compute constant terms (Sum of squares)
    # Note: sum(rolled_img^2) is constant for circular shifts
    sum_sq1 = torch.sum(img1 ** 2)
    sum_sq2 = torch.sum(img2 ** 2)

    # 2. Compute Cross-Correlation using FFT
    # CrossCorr(A, B) = IFFT( FFT(A) * conj(FFT(B)) )
    f1 = torch.fft.rfft2(img1)
    f2 = torch.fft.rfft2(img2)
    cross_corr = torch.fft.irfft2(f1 * torch.conj(f2), s=img1.shape)

    # 3. Calculate Euclidean Distance Map
    # distance^2 = A^2 + B^2 - 2AB
    dist_map = sum_sq1 + sum_sq2 - 2 * cross_corr

    # 4. Handle Quadrant Shift
    # FFT output has shift 0 at index [0,0]. We need to center it.
    dist_map = torch.fft.fftshift(dist_map)

    # 5. Crop the center region corresponding to shiftrange
    H, W = dist_map.shape
    cy, cx = H // 2, W // 2
    rh, rw = shiftrange[0] // 2, shiftrange[1] // 2
    
    # Note: Adjust logic to match exact range of original loop (-h/2+1 to h/2+1)
    # Original loop creates matrix of size shiftrange[0] x shiftrange[1]
    
    # Safe cropping
    y1 = cy - rh + 1 
    y2 = y1 + shiftrange[0]
    x1 = cx - rw + 1
    x2 = x1 + shiftrange[1]
    
    distance_matrix = dist_map[y1:y2, x1:x2]

    return distance_matrix

def translation_matching_fft(volume1, volume2, shiftrange=(61, 61), method='dog'):
    """
    Compute the intersection of image 1 and shifted image 2 using FFT
    return the proper row shift and column shift
    """
    if volume1.shape != volume2.shape:
        raise ValueError("Images must have the same shape")

    # Use DoG-based or traditional thresholding preprocessing
    mask1, _ = preprocess_for_alignment(volume1, method=method)
    mask2, _ = preprocess_for_alignment(volume2, method=method)

    rows, cols = mask2.shape

    distance_matrix = compute_distance_fft(mask1, mask2, rows, cols, shiftrange)
    min_distance, min_idx = torch.min(distance_matrix.reshape(-1), 0)
    min_distance_index = np.unravel_index(min_idx.cpu().numpy(), distance_matrix.shape)

    return (min_distance_index[0]-shiftrange[0]//2, min_distance_index[1]-shiftrange[1]//2)


def preprocess_for_alignment(vol_gpu, z_ratio=1.0, sigma_fg=1.0, sigma_bg=6.0, get_xz=False, method='dog', percentile=0.98):
    """
    Fast preprocessing for alignment using MIP + Filter (DoG or PixelThreshold).
    Returns processed float32 tensors.
    """
    if method == 'dog':
        # 1. XY MIP
        img_xy = torch.max(vol_gpu, dim=-1).values
        mask_xy = apply_dog_filter(img_xy, sigma_fg, sigma_bg)
        
        # 2. XZ MIP (if requested)
        mask_xz = None
        if get_xz:
            # MIP along Y (dim 0) -> (X, Z) -> (Z, X)
            mip_xz = torch.max(vol_gpu, dim=0).values.permute(1, 0)
            
            # Resize Z
            input_tensor = mip_xz.unsqueeze(0).unsqueeze(0).float()
            new_h = int(input_tensor.shape[2] * z_ratio)
            resized = F.interpolate(input_tensor, size=(new_h, int(input_tensor.shape[3])), 
                                mode='bilinear', align_corners=False)
            img_xz = resized.squeeze()
            mask_xz = apply_dog_filter(img_xz, sigma_fg, sigma_bg)
    else:
        # Traditional PixelThreshold logic (Binary mask converted to float)
        binary_mask = pixel_threshold(vol_gpu, rescale_p=percentile)
        mask_xy = get_image4processing(binary_mask).float()
        
        mask_xz = None
        if get_xz:
            # MIP along Y (dim 0) -> (X, Z) -> (Z, X)
            mask_xz_raw = torch.max(binary_mask, dim=0).values.permute(1, 0)
            input_tensor = mask_xz_raw.unsqueeze(0).unsqueeze(0).float()
            new_h = int(input_tensor.shape[2] * z_ratio)
            resized = F.interpolate(input_tensor, size=(new_h, int(input_tensor.shape[3])), 
                                mode='bilinear', align_corners=False)
            mask_xz = resized.squeeze()

    return mask_xy, mask_xz

def compute_3d_shift(ref_vol_gpu, ex_vol_gpu, matching_func, shiftrange=(21, 21), z_ratio=1.0, device='cuda', method='dog'):
    """
    Compute 3D shift (dx, dy, dz) using optimized dual XY and XZ projections.
    Args:
        shiftrange: (dy, dx) or (dy, dx, dz)
        matching_func: function with signature (img1, img2, shiftrange, device) -> ((dy, dx), dist)
    Returns:
        shift_xyz: np.array([dx, dy, dz])
        min_dist: float
    """
    use_z = len(shiftrange) == 3
    
    # 1. Preprocess Projections
    ref_xy, ref_xz = preprocess_for_alignment(ref_vol_gpu, z_ratio, get_xz=use_z, method=method)
    ex_xy, ex_xz = preprocess_for_alignment(ex_vol_gpu, z_ratio, get_xz=use_z, method=method)
    
    # 2. XY Alignment
    # Use only first two dims of shiftrange
    sr_xy = (shiftrange[0], shiftrange[1])
    shift_yx, dist_xy = matching_func(ref_xy, ex_xy, sr_xy, device)
    
    dy = shift_yx[0]
    dx = shift_yx[1]
    dz_scaled = 0.0
    
    # 3. XZ Alignment (Optional)
    if use_z:
        # Scale Z-radius (shiftrange[2]) by z_ratio
        # shiftrange for XZ image (H=Z_new, W=X) is (dz_scaled, dx)
        sr_xz = (int(shiftrange[2] * z_ratio), shiftrange[1])
        shift_zx, dist_zx = matching_func(ref_xz, ex_xz, sr_xz, device)
        dz_scaled = shift_zx[0]
    
    # Combine
    shift_xyz = np.array([dx, dy, dz_scaled])
    
    return shift_xyz, dist_xy
