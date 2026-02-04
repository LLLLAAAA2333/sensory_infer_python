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


# ----------------------------- 3D Alignment (XY + Z) ----------------------------- #

def get_xz_projection(volume_gpu, z_ratio=1.0):
    """
    Compute XZ MIP projection and resize Z axis to match pixel isotropy.
    Args:
        volume_gpu: (Y, X, Z) tensor
        z_ratio: float, ratio of Z unit / XY unit (typically ~5.0)
    Returns:
        xz_img: (Z_new, X) int tensor suitable for alignment
    """
    # 1. MIP along Y axis -> (X, Z)
    # volume is (Y, X, Z). dim=0 is Y.
    # values shape: (X, Z)
    mip_xz = torch.max(volume_gpu, dim=0).values 
    
    # 2. Transpose to (Z, X) -> treating Z as "Height" (rows), X as "Width" (cols)
    mip_xz = mip_xz.permute(1, 0) # (Z, X)
    
    # 3. Resize Z axis
    # interpolate expects (N, C, H, W)
    input_tensor = mip_xz.unsqueeze(0).unsqueeze(0).float() # (1, 1, Z, X)
    
    new_h = int(input_tensor.shape[2] * z_ratio)
    new_w = int(input_tensor.shape[3])
    
    # Bilinear interpolation for smoothness
    resized = F.interpolate(input_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Back to (Z', X)
    resized_img = resized.squeeze(0).squeeze(0) # (Z', X)
    
    # 4. Thresholding
    # Treat as volume (Z', X, 1) for pixel_threshold
    vol_fake = resized_img.unsqueeze(-1)
    
    # Dynamic maxpool_k to prevent "Output size is too small" error
    h, w = resized_img.shape
    safe_maxpool_k = max(1, min(125, h // 2, w // 2))

    # pixel_threshold handles normalization and background removal
    mask = pixel_threshold(vol_fake, maxpool_k=safe_maxpool_k) # (Z', X, 1)
    
    # Convert to standard image format for matching
    img = get_image4processing(mask).to(torch.int) # (Z', X)
    
    return img

def compute_3d_shift(ref_vol_gpu, ex_vol_gpu, matching_func, shiftrange=(21, 21), z_ratio=1.0, device='cuda'):
    """
    Compute 3D shift (dx, dy, dz) using dual XY and XZ projections.
    Args:
        matching_func: function with signature (img1, img2, shiftrange, device) -> ((dy, dx), dist)
    Returns:
        shift_xyz: np.array([dx, dy, dz])
        min_dist: float (combined or avg distance)
    """
    # 1. XY Alignment
    ref_mask = pixel_threshold(ref_vol_gpu)
    ex_mask = pixel_threshold(ex_vol_gpu)
    
    ref_xy = get_image4processing(ref_mask).to(torch.int)
    ex_xy = get_image4processing(ex_mask).to(torch.int)
    
    shift_yx, dist_xy = matching_func(ref_xy, ex_xy, shiftrange, device)
    
    # 2. XZ Alignment
    ref_xz = get_xz_projection(ref_vol_gpu, z_ratio)
    ex_xz = get_xz_projection(ex_vol_gpu, z_ratio)
    
    # Note: Z is scaled by z_ratio, so pixel shift in Z is magnified.
    # We use the same shiftrange for simplicity, assuming Z shift isn't massive in *pixels* after scaling?
    # Actually if Z shift is massive, we might need larger range. 
    # But usually sample drift is small.
    shift_zx, dist_zx = matching_func(ref_xz, ex_xz, shiftrange, device)
    
    # shift_yx = (row_shift, col_shift) -> (y, x)
    dy = shift_yx[0]
    dx = shift_yx[1]
    
    # shift_zx = (row_shift, col_shift) -> (z_scaled, x)
    # We take Z from here.
    dz_scaled = shift_zx[0]
    
    # Return scaled Z shift to match coordinate system (Z * z_ratio)
    # Rescale Z back to original units -> dz = dz_scaled / z_ratio
    # But pipeline expects scaled shift for coords.
    
    # Combine
    shift_xyz = np.array([dx, dy, dz_scaled])
    
    return shift_xyz, dist_xy # Return XY dist as primary metric, or average?
