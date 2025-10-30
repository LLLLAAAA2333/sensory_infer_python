import shutil
import os
import json
import numpy as np
import h5py
from tqdm import tqdm
import colorsys

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