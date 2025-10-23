# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from src.comm_utils.packages import *
from src.comm_utils.prints import print_log_message, print_warning_message, print_info_message, pad_num


def cxcywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] * .5  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] * .5  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] * .5  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] * .5  # bottom right y
    return y


# --------------------------------- Result saving ---------------------------------
def store_result_as_json(root, results, is_1_indexed = True):
    [save_dict_as_json(root, name, result, is_1_indexed) for name, result in results.items() if result[0] is not None]
    print_info_message(f"Results have been saved in {root} !")


def save_dict_as_json(root, name, result, is_1_indexed = True):
    """{neuron_id: [xyxyz, ...], ...}"""
    # assert isinstance(output, dict), f'{name} must be dict type!'
    _, regions, pred_ids, region_ptrs = result[:4]
    pred_ids = [0, ] * len(region_ptrs) if pred_ids is None else pred_ids
    mat_name = "_".join(name.split('_')[:-1])
    num_volume = pad_num(int(name.split('_')[-1]) + is_1_indexed, 6)

    # head_bboxes = cxcywh2xyxy(head_bboxes)

    new_dict = dict()
    for _id, (frame_idx, ptr) in zip(pred_ids, region_ptrs):
        new_id = str(_id + is_1_indexed)
        # ox, oy, oz = [int(x) + 1 for x in head_bboxes[frame_idx][:2]] + [1]
        ox, oy, oz = [1, 1, 1] if is_1_indexed else [0, 0, 0]
        i = cxcywh2xyxy(regions[frame_idx])[ptr]
        _r = new_dict.get(new_id, list())
        new_dict[new_id] = _r + [[i[0].item() + ox, i[1].item() + oy, i[2].item() + ox, i[3].item() + oy, frame_idx + oz]]

    # store
    root = os.path.join(root, mat_name)
    os.makedirs(root, exist_ok = True)
    with open(os.path.join(root, num_volume + '.json'), 'w') as f:
        json.dump(new_dict, f)
