import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from src.comm_utils.prints import print_log_message, print_warning_message, pad_num
from src.preproc.inference import auto_preprocess
import numpy as np
import torch

def infer_one_batch(stream, args, volume_save_path):
    print_log_message(f"Processing stream: {stream}")
    results, preprc_results = dict(), dict()

    # load_stack
    # (preprc_results, head_bbox_dict, scale_factors), results = load_matlab_volumes_dict(args, stream), dict()
    preprc_results = auto_preprocess(mode = args.preprocessing_mode, paths = stream, args = args)
    
    # folder_path = os.path.dirname(stream[0]) + '/volume/' + stream[0].split('/')[-1].split('.')[0]
    # save volume result
    mat_save_path = os.path.join(volume_save_path, stream[0].split('/')[-1].split('.')[0]) 
    os.makedirs(mat_save_path, exist_ok=True)
    for key in preprc_results.keys():
        npy_save_path = os.path.join(mat_save_path, key.split('/')[-1]+'.npy')
        # print(preprc_results[key].cpu().numpy().shape)
        # np.save(npy_save_path, preprc_results[key].permute(2, 3, 0, 1).squeeze(dim=-1).cpu().numpy())
        np.save(npy_save_path, preprc_results[key])
        
        torch.cuda.empty_cache()

    return None, None