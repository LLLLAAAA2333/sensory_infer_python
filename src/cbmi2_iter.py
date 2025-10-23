# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from src.comm_utils.packages import *
from src.comm_utils.prints import print_log_message, print_warning_message, pad_num
from src.preproc.inference import auto_preprocess
from src.utils import store_result_as_json


def infer_one_batch(model, buffer, stream, args, error_file_path: str, is_saved: bool = True):
    print_log_message(f"Processing stream: {stream}")
    results, preprc_results = dict(), dict()

    # Stack loading
    preprc_results = auto_preprocess(mode = args.preprocessing_mode, paths = stream, args = args)

    if args.only_preprocessing:
        return None, None
    if args.preprocessing_mode > 0:
        preprc_results = {key: torch.HalfTensor(volume[0].transpose((2, 0, 1))[:, np.newaxis].astype(np.float32)).cuda() for key, volume in preprc_results.items()}
    else:
        preprc_results = {key: torch.HalfTensor(volume.transpose((2, 0, 1))[:, np.newaxis].astype(np.float32)).cuda() for key, volume in preprc_results.items()}
    # preprc_results = {key: torch.max(volume, dim = 0, keepdim = True)[0] for key, volume in preprc_results.items()}
    if model.mp > 0:
        preprc_results = {key: torch.max_pool2d(volume, kernel_size = model.mp, stride = model.mp) for key, volume in preprc_results.items()}
        print_log_message(f"Max pooling with kernel size {model.mp}.")

    with torch.inference_mode():
        for name, volume in tqdm(preprc_results.items()):
            tqdm.write(f"shape: {volume.shape}")
            # try:
            #     head_bboxes, regions, pts_emb, pt_tuple, region_ptrs, neuron_emb, neuron_info, num_region, is_warnings = model(volume * 5.)
            #     neuron_pred_ids, region_pred_ids, num_neuron_idv = buffer(neuron_emb, neuron_info[0], pts_emb.shape[0], any(is_warnings))
            # except Exception as e:
            #     print_warning_message(f"Warning: {name} can't be processed, because {e}!")
            #     results[name] = [None,] * 5
            #     continue
            head_bboxes, regions, pts_emb, pt_tuple, region_ptrs, neuron_emb, neuron_info, num_region, is_warnings = model(volume)
            neuron_pred_ids, region_pred_ids, num_neuron_idv = buffer(neuron_emb, neuron_info[0], pts_emb.shape[0], any(is_warnings))

            print_log_message(f"name: {name}, \t the number of regions: {num_region[0]}, \t the number of neurons: {num_region[1]}, \t threshold: {num_region[2]} \t the number of neurons: {num_neuron_idv}")
            if any(is_warnings):
                print_warning_message(f"Warning: {name} has been processed with warning {is_warnings}!")
            else:
                _r = f"/home/cbmi/CBMI_python/data/gene/rec/cleaned_neuron/CBMI_unsup/w10/{name}"
                # os.makedirs(_r, exist_ok = True)
                # np.save(f"{_r}/{name}.npy", torch.concat([neuron_info[2].cpu(), torch.FloatTensor(neuron_pred_ids).unsqueeze(1)], dim = 1).cpu().numpy())

                # _r = f"/home/cbmi/CBMI_python/data/gene/rec/merge_regions/w8"
                # os.makedirs(_r, exist_ok = True)
                # np.save(f"{_r}/{name}.npy", neuron_info[3].cpu().numpy())
            pt_tuple_data = torch.concat([neuron_info[2].cpu(), torch.FloatTensor(neuron_pred_ids).unsqueeze(1)], dim = 1).cpu().numpy()
            root_path = args.json_store_root.replace("/id", "/raw")
            os.makedirs(root_path, exist_ok = True)
            np.save(os.path.join(root_path, f"{name}.npy"), pt_tuple_data)

            results[name] = [head_bboxes, regions, region_pred_ids, region_ptrs, num_region + (num_neuron_idv, int(any(is_warnings)))]

    if is_saved:
        store_result_as_json(root = args.json_store_root, results = results, is_1_indexed = True)  # 1-indexed 3d dict for MATLAB loading
        store_result_as_json(root = args.json_mnr_store_root, results = results, is_1_indexed = False)  # 0-indexed 3d dict for python loading to evaluate Multi-Neuron Recognition

    return results, preprc_results
