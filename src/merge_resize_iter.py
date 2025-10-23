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
    
    # folder_path = os.path.dirname(stream[0]) + '/volume/' + stream[0].split('/')[-1].split('.')[0]
    save_path = '/home/wenlab-user/RongWei/olfactory/WEN0065/0617/w1/volume/' + stream[0].split('/')[-1].split('.')[0]
    # print(preprc_results.keys())
    for key in preprc_results.keys():
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path + '/'+ key.split('/')[-1]+'.npy', preprc_results[key][0])


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
            head_bboxes, regions, neuron_dict, neuron_emb, is_warnings, supp_r = model(volume)
            region_pt_tuple, region_ptrs, region_map, neuron_pt_tuple, (num_region, num_neuron), (std, max_min) = supp_r
            buffer.pickup.vol_infos = {name: [std, max_min]}
            neuron_pred_ids, region_pred_ids, num_neuron_idv = buffer(neuron_emb, neuron_dict, num_region, any(is_warnings))
            # except Exception as e:
            #     print_warning_message(f"Warning: {name} can't be processed, because {e}!")
            #     results[name] = [None,] * 5
            #     continue
            print_log_message(f"name: {name}, \t the number of regions: {num_region}, \t the number of neurons: {num_neuron}, \t  the number of W: {num_neuron_idv}")
            if any(is_warnings):
                print_warning_message(f"Warning: {name} has been processed with warning {is_warnings} (ext, det, merge, rec)!")

            # _r = f"/home/cbmi/CBMI_python/data/zone/data_20230525_w10_1222_/region_pts"
            # os.makedirs(_r, exist_ok = True)
            # np.save(f"{_r}/{name}.npy", region_pt_tuple.cpu().numpy())
            #
            # _r = f"/home/cbmi/CBMI_python/data/zone/data_20230525_w10_1222_/neuron_pts"
            # os.makedirs(_r, exist_ok = True)
            # np.save(f"{_r}/{name}.npy", neuron_pt_tuple.cpu().numpy())

            # pt_tuple_data = torch.concat([neuron_pt_tuple.cpu(), torch.FloatTensor(neuron_pred_ids).unsqueeze(1)], dim = 1).cpu().numpy()
            # root_path = args.json_store_root.replace("/id", "/raw")
            # os.makedirs(root_path, exist_ok = True)
            # np.save(os.path.join(root_path, f"{name}.npy"), pt_tuple_data)

            results[name] = [head_bboxes, regions, region_pred_ids, region_ptrs, (num_region, num_neuron) + (num_neuron_idv, int(any(is_warnings)))]

    if is_saved:
        store_result_as_json(root = args.json_store_root, results = results, is_1_indexed = True)  # 1-indexed 3d dict for MATLAB loading
        store_result_as_json(root = args.json_mnr_store_root, results = results, is_1_indexed = False)  # 0-indexed 3d dict for python loading to evaluate Multi-Neuron Recognition

    return results, preprc_results
