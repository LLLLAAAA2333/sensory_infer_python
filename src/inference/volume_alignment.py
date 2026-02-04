import os
import sys
import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from typing import Tuple
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from src.inference.blur import pixel_threshold, get_image4processing
from src.comm_utils.prints import print_info_message, print_log_message, print_warning_message


# ----------------------------- Old methods for volume alignment(Exhaustive) ----------------------------- #
def pixel2binary(image, rate=0.98):
    index = round(torch.numel(image) * rate)
    threshold = torch.mean(torch.sort(image.view(-1))[0][index:].float())
    binary_image = image > threshold
    return binary_image.int()

def Euclidean_distance(image1, image2):
    return torch.sum((image1 - image2) ** 2)

@torch.jit.script
def compute_distance(binary_image1: torch.Tensor, binary_image2: torch.Tensor, rows: int, cols: int, shiftrange: Tuple[int, int]=(61, 61)):
    """
    Each entry in distance_matrix represents the euclidean distance between image 1 and image 2 which moved (i-nrow/2, j-ncol/2)
    """
    sr0 = int(shiftrange[0])
    sr1 = int(shiftrange[1])
    distance_matrix = torch.zeros(sr0, sr1, device='cuda')

    for i_idx, i in enumerate(range(-sr0//2+1, sr0//2+1)):
        for j_idx, j in enumerate(range(-sr1//2+1, sr1//2+1)):
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


def volume_alignment(file_list, save_path, shiftrange=(51, 51), zrange=None, align_method='bruteforce'):
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

        if align_method == 'fft':
            shift = translation_matching_fft(torch.from_numpy(reference_volume).cuda(), volume, shiftrange)
        else:
            shift = translation_matching(torch.from_numpy(reference_volume).cuda(), volume, shiftrange)
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

def translation_matching_fft(volume1, volume2, shiftrange=(61, 61)):
    """
    Compute the intersection of image 1 and shifted image 2 using FFT
    return the proper row shift and column shift
    """
    if volume1.shape != volume2.shape:
        raise ValueError("Images must have the same shape")

    binary_volume1 = pixel_threshold(volume1)
    binary_volume2 = pixel_threshold(volume2)
    binary_image1 = get_image4processing(binary_volume1).to(torch.int)
    binary_image2 = get_image4processing(binary_volume2).to(torch.int)

    rows, cols = binary_image2.shape

    distance_matrix = compute_distance_fft(binary_image1, binary_image2, rows, cols, shiftrange)
    min_distance, min_idx = torch.min(distance_matrix.reshape(-1), 0)
    min_distance_index = np.unravel_index(min_idx.cpu().numpy(), distance_matrix.shape)

    return (min_distance_index[0]-shiftrange[0]//2, min_distance_index[1]-shiftrange[1]//2)


def preprocess_for_alignment_optimized(vol_gpu, z_ratio=1.0, percentile=0.98, get_xz=False):
    """
    Fast preprocessing for alignment using MIP + Top Percentile.
    Returns:
        mask_xy: (Y, X) int tensor
        mask_xz: (Z_new, X) int tensor or None
    """
    # 1. XY MIP (standard)
    # vol_gpu: (Y, X, Z)
    img_xy = torch.max(vol_gpu, dim=-1).values
    
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
        
        val_xz = torch.quantile(img_xz.float(), percentile)
        mask_xz = (img_xz > val_xz).to(torch.int)

    # 3. Simple Percentile Thresholding for XY
    val_xy = torch.quantile(img_xy.float(), percentile)
    mask_xy = (img_xy > val_xy).to(torch.int)

    return mask_xy, mask_xz

def compute_3d_shift(ref_vol_gpu, ex_vol_gpu, matching_func, shiftrange=(21, 21), z_ratio=1.0, device='cuda'):
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
    ref_xy, ref_xz = preprocess_for_alignment_optimized(ref_vol_gpu, z_ratio, get_xz=use_z)
    ex_xy, ex_xz = preprocess_for_alignment_optimized(ex_vol_gpu, z_ratio, get_xz=use_z)
    
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
