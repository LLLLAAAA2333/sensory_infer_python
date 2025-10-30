import os
import cv2
import sys
import json
import h5py
import shutil
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
from src.preproc.mat2npy import infer_one_batch
from src.inference.blur import get_image4processing
from src.inference.volume_alignment import volume_alignment
from src.inference.intensity_extract import pixel_intensity_extraction
from src.zephir.zephir_support import create_zephir_support, write_metadata_json, volume_stack_and_create_h5, create_worldlines, create_raw_annotations, neuron_alignment
from src.merge_resize_inference import *
import vis_trajectory as vis

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