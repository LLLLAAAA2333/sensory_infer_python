import os
import sys
import json
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.comm_utils.prints import print_log_message, print_warning_message, pad_num
from src.merge_resize_inference import Treeformer_End2End, VolumeMemoryBuffer, draw_volume_result

def run_inference_on_single_volume(volume_path, config_path, output_dir, **args):
    print_log_message(f"Running inference on volume: {volume_path}")
    os.makedirs(output_dir, exist_ok=True)
    neuron_output_path = os.path.join(output_dir, "neuron_pt_tuple.npy")
    vis_output_path_base = os.path.join(output_dir, "inference_result")
    os.makedirs(vis_output_path_base, exist_ok=True)
    print_log_message(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print_log_message("Initializing model...")
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
        save_path = args.get('json_store_root', "./volume_memory_buffer"),
        deg_t = 100,
        weight_W = 0.2,
        warmup_num_vol = 3,
    )
    print_log_message("Model initialized.")

    print_log_message("Loading volume data...")
    volume_np = np.load(volume_path)
    volume_tensor = torch.HalfTensor(volume_np.transpose(2, 0, 1)[:, np.newaxis, :, :].astype(np.float32)).cuda()
    print_log_message("Volume data loaded, shape: " + str(volume_tensor.shape))

    pre_resize = args.get('pre_resize', 1)
    pre_resize_size = args.get('pre_resize_size', 680)
    scale_factor = args.get('scale_factor', 1.0)
    if pre_resize:
        scale_factor = pre_resize_size / max(volume_tensor.shape[2:])
        volume_tensor = F.interpolate(volume_tensor, scale_factor=scale_factor, mode="nearest")
        print_log_message(f"Pre-resized volume to size: {volume_tensor.shape}")
    
    pre_rescale_pixels = args.get('pre_rescale_pixels', 1)
    if pre_rescale_pixels:
        _std, _mean = volume_tensor.std(), volume_tensor.mean()
        print_log_message(f"std: {_std}, mean: {_mean}")
        volume_tensor[:] = -(_mean / _std * 13 - 103) + 13 / _std * volume_tensor
    
    print_log_message("Starting inference...")
    neuron_pt_tuple_to_save = None

    with torch.inference_mode():
        head_bboxes, regions, neuron_dict, neuron_emb, is_warnings, supp_r = model(volume_tensor)
        region_pt_tuple, region_ptrs, region_map, neuron_pt_tuple, (num_region, num_neuron), (std, max_min) = supp_r
        if pre_resize:
            neuron_pt_tuple[:, [0, 1, 3, 4]] = neuron_pt_tuple[:, [0, 1, 3, 4]] / scale_factor

        neuron_pt_tuple_to_save = neuron_pt_tuple.cpu().numpy()
        if any(is_warnings):
            print_warning_message("Warnings during inference:")
            for w in is_warnings:
                if w:
                    print_warning_message(w)

        if neuron_pt_tuple_to_save is not None:
            np.save(neuron_output_path, neuron_pt_tuple_to_save)
            print_log_message(f"Saved neuron point tuple to {neuron_output_path}")
        
        
        neuron_pred_ids, region_pred_ids, num_neuron_idv = buffer(neuron_emb, neuron_dict, num_region, any(is_warnings))
        print_log_message("Generating visualization...")
        draw_volume_result(
            volume_tensor, head_bboxes, regions,
            region_pred_ids,region_ptrs,
            vis_output_path_base,
            name="inference_result", verbose=False
        )

    print_log_message("Inference completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a single volume.")
    parser.add_argument("--volume-path", type=str, required=True, help="Path to the input volume (.npy file).")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the model configuration file (JSON).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output results.")
    parser.add_argument("--json-store-root", type=str, default="./volume_memory_buffer", help="Directory for volume memory buffer storage.")
    parser.add_argument("--pre-resize", type=int, default=1, help="Whether to pre-resize the volume (1: yes, 0: no).")
    parser.add_argument("--pre-resize-size", type=int, default=680, help="Size to pre-resize the largest dimension of the volume.")
    parser.add_argument("--pre-rescale-pixels", type=int, default=1, help="Whether to pre-rescale pixel intensities (1: yes, 0: no).")

    args = parser.parse_args()
    args_dict = vars(args)
    run_inference_on_single_volume(
        **args_dict
    )