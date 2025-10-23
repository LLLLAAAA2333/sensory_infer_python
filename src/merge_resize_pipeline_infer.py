# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import sys
import json
import time
import argparse
from glob import glob
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from src.comm_utils.prints import print_info_message
from src.comm_utils.dataset_building import extract_waiting_stacks, split_processing_streams
from src.merge_resize_iter import infer_one_batch
from src.merge_resize_inference import Treeformer_End2End, VolumeMemoryBuffer, draw_volume_result

if __name__ == '__main__':
    # Initialize parameters
    parser = argparse.ArgumentParser(description = "CBMI 1 pipeline")
    # --------------------------------- Stack loading and Stage 1: Pre-processing ---------------------------------
    parser.add_argument('--process-stack-root', type = str, default = '/home/data4/WJH/WEN0065/0617/w1', help = '')
    parser.add_argument('--save-preprocess-result-root', type = str, default = '')
    parser.add_argument('--name-reg', type = str, default = r"[iI]ma?ge?_?[sS]t(?:ac)?k_?\d+_dk?\d+.*[w|fly]\d+_?Dt\d{6}")
    parser.add_argument('--preprocessing-mode', type = int, default = 0)
    parser.add_argument('--only-preprocessing', action = "store_true")
    parser.add_argument('--volume-window', type = int, default = 1, help = "number of processing volumes once")
    parser.add_argument('--volume-start-idx', type = int, default = 0, help = "")
    parser.add_argument('--error-filename', type = str, default = "AErrorStacks.txt", help = "")
    parser.add_argument('--re-infer-error-stacks', action = "store_true", help = "")
    parser.add_argument('--config', type = str, default = "/home/wenlab-user/RongWei/olfactory/code_v1.10/src/configs/inference/240623.json", help = "")

    # --------------------------------- Result saving ---------------------------------
    parser.add_argument('--store-vis-result', action = "store_true", help = "")
    parser.add_argument('--json-store-root', type = str, default = "/home/wenlab-user/RongWei/olfactory/WEN0065/0617/w1")
    parser.add_argument('--json-store-sub-root', type = str, default = "id")
    parser.add_argument('--json-mnr-store-sub-root', type = str, default = "id_mnr")  # Multi-Neuron Recognition 0-indexed

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    args.zrange = config["zrange"]
    save_fig_root = os.path.join(args.json_store_root, "figs")
    args.save_preprocess_result_root = os.path.join(args.json_store_root, "mip") if args.preprocessing_mode >= 3 else args.save_preprocess_result_root
    args.json_mnr_store_root = os.path.join(args.json_store_root, args.json_mnr_store_sub_root if args.json_mnr_store_sub_root else f"{time.strftime('%m-%d')}_mnr")
    args.json_store_root = os.path.join(args.json_store_root, args.json_store_sub_root if args.json_store_sub_root else time.strftime('%m-%d'))
    error_file_path = os.path.join(args.json_store_root, args.error_filename)

    os.makedirs(args.json_store_root, exist_ok = True)
    os.makedirs(args.json_mnr_store_root, exist_ok = True)
    os.makedirs(save_fig_root, exist_ok = True)

    print_info_message(args)

    processing_stacks = extract_waiting_stacks(
        args.json_store_root, error_file_path, args.name_reg,
        re_infer_error = args.re_infer_error_stacks,
        paths = glob(os.path.join(args.process_stack_root, '*.mat')))[args.volume_start_idx:]
        # paths = glob(os.path.join(args.process_stack_root, 'ImgStk003*')))[args.volume_start_idx:]

    treeformer = Treeformer_End2End(

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

        mp = config.get("mp", 0),

    ).eval().cuda().half()

    treeformer_he = VolumeMemoryBuffer(
        save_path = "/home/wenlab-user/RongWei/olfactory/WEN0065/0617/w1",
        deg_t = 100,
        weight_W = 0.2,
        warmup_num_vol = 0,
    )
    import pandas as pd

    num_regions = list()
    num_neurons = list()
    ab_num_neurons = list()
    is_warnings = list()

    for stream in tqdm(split_processing_streams(processing_stacks, max_mats_one_stream = args.volume_window), colour = "#FF1493"):
        results, preprc_results = infer_one_batch(treeformer, treeformer_he, stream, args, error_file_path, is_saved = True)
        if not args.only_preprocessing:
            num_regions.extend([results[name][-1][0] for name in results.keys() if results[name][-1] is not None])
            num_neurons.extend([results[name][-1][1] for name in results.keys() if results[name][-1] is not None])
            ab_num_neurons.extend([results[name][-1][2] for name in results.keys() if results[name][-1] is not None])
            is_warnings.extend([results[name][-1][3] for name in results.keys() if results[name][-1] is not None])
            df = pd.DataFrame({"regions": num_regions, "neurons": num_neurons, "ab_num_neurons": ab_num_neurons, "is_warnings": is_warnings})
            df.to_excel(save_fig_root.replace("figs", "num_regions.xlsx"), index = False)

            if args.store_vis_result:
                for name in results.keys():
                    draw_volume_result(preprc_results[name], results[name][0], results[name][1], results[name][2], results[name][3], save_fig_root, name, verbose = False)
