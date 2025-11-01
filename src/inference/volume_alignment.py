import os
import sys
import torch
import numpy as np
import tqdm
import cv2
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