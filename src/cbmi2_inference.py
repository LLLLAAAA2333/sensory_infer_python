# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import os
import sys
import torch
import torchvision
import numpy as np
from torch import nn
from typing import Dict
import matplotlib.pyplot as plt
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))


# --------------- general funcs ---------------
def cxcywh2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] * .5  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] * .5  # top left y
    return y


def cxcywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] * .5  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] * .5  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] * .5  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] * .5  # bottom right y
    return y


def _xyxy2cxcywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) * .5  # cx
    y[:, 1] = (x[:, 1] + x[:, 3]) * .5  # cy
    y[:, 2] = x[:, 2] - x[:, 0]  # w
    y[:, 3] = x[:, 3] - x[:, 1]  # h
    return y


def bboxes_fp2int(bboxes: torch.FloatTensor or torch.HalfTensor):
    """cxcywh type"""
    bboxes[:, :2] = torch.floor(bboxes[:, :2])
    bboxes[:, 2:4] = torch.ceil(bboxes[:, 2:4])
    return bboxes.to(dtype = torch.int32)


def index_region(regions: Dict):
    return [[key, i] for key, value in regions.items() for i in range(len(value))]


def region2cxcyz(regions: Dict, z_ratio):
    """
    :param regions: Dict, {z: [cx, cy, w, h]}
    :return: torch.Tensor, [[cx, cy, z]]
    """

    output = torch.concat([torch.concat([frame_region[:, :2], z_ratio * frame_idx * torch.ones_like(frame_region)[:, :1]], dim = 1) for frame_idx, frame_region in regions.items()], dim = 0)

    return output


def region2cxcyz_shape(regions: Dict, z_ratio):
    """
    :param regions: Dict, {z: [cx, cy, w, h]}
    :return: torch.Tensor, [[cx, cy, z, w, h]]
    """
    output = torch.concat(
        [torch.concat([frame_region[:, :2], z_ratio * frame_idx * torch.ones_like(frame_region)[:, :1], frame_region[:, 2:]], dim = 1) for frame_idx, frame_region in regions.items()], dim = 0
    )

    return output


def region2misi(regions: Dict, volume, roi_origin):
    """
    :param regions: Dict, {z: [cx, cy, w, h]}
    :return: torch.Tensor, [[mean_i, std_i]]
    """

    output = list()
    for frame_idx, frame_region in regions.items():
        for b in cxcywh2xyxy(frame_region).to(dtype = torch.int32):
            std, mean = torch.std_mean(volume[frame_idx, 0, b[1]:b[3], b[0]:b[2]].float())
            output.append(torch.FloatTensor([mean, std]))

    output = torch.stack(output)
    return output


# --------------- head extraction ---------------
# @profile
def post_processing_ext(preds, conf_thresh = .4, ratio: int = 8):
    """

    :param preds:
    :param conf_thresh:
    :param ratio:
    :return: torch.int32, [batch_size, 5] (cx,cy,w,h,flag), where flag represents whether the frame has a head.
    """

    idxes = torch.argmax(preds[:, :, 4], dim = 1)
    bboxes = torch.zeros(len(preds), 5, dtype = preds.dtype, device = preds.device)
    for i, p in enumerate(idxes):
        if preds[i, p, -1] >= conf_thresh:
            bboxes[i, :4] = preds[i, p, :4]
            bboxes[i, -1] = 1

    bboxes = bboxes_fp2int(bboxes)
    bboxes[:, :4] = torch.mul(bboxes[:, :4], ratio)

    return bboxes


def _ext_relative2absolute(head_bboxes):
    """delete the first frame by setting its score to 0"""
    new_head_bboxes = torch.ones_like(head_bboxes)
    head_bboxes = cxcywh2xyxy(head_bboxes[:, :4].clone())
    roi_vol_idxes = torch.nonzero(head_bboxes[:, -1]).squeeze(1)
    m = torch.concat([torch.min(head_bboxes[roi_vol_idxes][:, :2], dim = 0)[0], torch.max(head_bboxes[roi_vol_idxes][:, 2:4], dim = 0)[0]])
    new_head_bboxes[:, :4] = _xyxy2cxcywh(torch.IntTensor([int(m[0]), int(m[1]), int(m[2]), int(m[3])]).unsqueeze(0).repeat(new_head_bboxes.shape[0], 1))
    # new_head_bboxes[0, :] = 0
    return new_head_bboxes


# --------------- neuronal region detection ---------------
def build_det_batch(volume, bboxes, side_t: int = 384, batch_size_t: int = 0, magnification: int = 40):
    """
    Extract foreground and set dynamic batch_size and static shape in a batch.
    Resize frames to the given shape.
    :param volume: [17, 1, 1024, 1024]
    :param bboxes: torch.int32, [batch_size, 5] (cx,cy,w,h,flag)
    :param side_t: static shape (side_t, side_t)
    :return:
    """

    is_warning = False

    dtype, device = volume.dtype, volume.device

    p_batch_idxes = torch.argwhere(bboxes[:, -1]).squeeze(1)
    batch_size = len(p_batch_idxes) if batch_size_t == 0 else batch_size_t
    print(f"Processed batch size {batch_size}")

    batch = torch.ones(batch_size, 1, side_t, side_t, dtype = dtype, device = device)
    scales = list()

    for rel_idx, vol_idx in enumerate(p_batch_idxes):
        bb = bboxes[vol_idx, :4]
        image = volume[vol_idx, 0, int(bb[1] - bb[3] * .5):int(bb[1] + bb[3] * .5), int(bb[0] - bb[2] * .5):int(bb[0] + bb[2] * .5)]

        scale = min(side_t / max(*image.shape[:2]), 1.0)
        # scale = 40 / magnification

        temp_img = image

        if scale < 1.0:
            temp_img = F.interpolate(image.unsqueeze(0).unsqueeze(0), scale_factor = scale).squeeze(0).squeeze(0)
            is_warning = True
            print(f"----------------   Resized image from {image.shape} to {temp_img.shape}")

        batch[rel_idx, 0, :temp_img.shape[0], :temp_img.shape[1]] = temp_img
        scales.append(scale)

    return batch, p_batch_idxes, scales, batch_size, is_warning


def post_processing_det(preds, num_sampels, conf_thresh = .6, iou_thresh = .6, max_nms = 30000,
                        det_num_region_t_frame: int = 300,
                        det_num_region_t_vol: int = 0,
                        shrink_pixel: int = 2,
                        smooth_factor: float = 0.90,
                        det_num_region_t_vol_factor: float = .10
                        ):
    """
    :return: torch.int32, [batch_size, 5] (x,y,w,h,flag), where flag represents whether the frame has a head.
    """
    is_warning = False
    # candidates = preds[..., 4] >= (conf_thresh * .70)
    candidates = preds[..., 4] >= conf_thresh

    raw_output = [torch.zeros((0, preds.shape[2]), device = preds.device, dtype = preds.dtype)] * num_sampels
    scores = torch.zeros((0,), device = preds.device, dtype = preds.dtype)

    # Clean up by the score threshold, number threshold and IOU threshold (NMS)
    for pi in range(num_sampels):
        p = preds[pi][candidates[pi]]

        if p.shape[0] == 0:
            continue
        elif p.shape[0] > max_nms:
            p = p[p[:, 4].argsort(descending = True)[:max_nms]]

        # shrink bbox for better performance, but it will produce non-positive value in width and height columns
        p[:, 2:4] = p[:, 2:4] - shrink_pixel
        p = p[(p[:, 2] > 0) & (p[:, 3] > 0)]

        idx = torchvision.ops.nms(cxcywh2xyxy(p[:, :4]), p[:, 4], iou_thresh)[:det_num_region_t_frame]
        raw_output[pi] = p[idx]
        scores = torch.cat([scores, p[idx, 4]])

    # data = scores.detach().cpu().numpy()
    # plt.hist(data, bins = 50, color = 'blue', alpha = 0.7)
    # plt.title('Histogram of Tensor Values')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    cur_num_region = torch.sum(scores >= conf_thresh).item()

    if det_num_region_t_vol == 0:
        det_num_region_t_vol = cur_num_region
    elif (cur_num_region >= det_num_region_t_vol * (1 + det_num_region_t_vol_factor)) or (cur_num_region <= det_num_region_t_vol * (1 - det_num_region_t_vol_factor)):
        is_warning = True  # This volume will be ignored to update global variables, but still have a result.
        # print(f"Warning: {det_num_region_t_vol} is too different from {len(scores)}")
    else:
        det_num_region_t_vol = int(det_num_region_t_vol * smooth_factor + cur_num_region * (1 - smooth_factor))
        cur_num_region = min(len(scores), det_num_region_t_vol)

    return raw_output, is_warning, det_num_region_t_vol, cur_num_region


# --------------- region merge by position and shape ---------------
def recover_region_pos_in_roi(raw_regions, roi_frame_idxes, scales, head_bboxes):
    """input: [cx, cy, w, h, score, upper_score, cx_upper, cy_upper, lower_score, cx_lower, cy_lower]
    return [cx, cy, w, h, ......]"""
    _device, _dtype = raw_regions[0].device, raw_regions[0].dtype

    xys = cxcywh2xyxy(head_bboxes[roi_frame_idxes])[:, :2]
    roi_origin = torch.min(xys, dim = 0)[0]  # [xmin, ymin] for whole volume

    if raw_regions[0].shape[1] == 5:
        scales = torch.tensor([[s, s, s, s, 1.] for s in scales], dtype = _dtype, device = _device)
        xys = torch.tensor([[x, y, 0, 0, 0] for x, y in xys], dtype = _dtype, device = _device)
    elif raw_regions[0].shape[1] == 11:
        scales = torch.tensor([[s, s, s, s, 1., 1., s, s, 1., s, s] for s in scales], dtype = _dtype, device = _device)
        xys = torch.tensor([[x, y, 0, 0, 0, 0, x, y, 0, x, y] for x, y in xys], dtype = _dtype, device = _device)

    # transition and scale
    regions = {idx.item(): r / scale + xy for r, idx, scale, xy in zip(raw_regions, roi_frame_idxes, scales, xys) if len(r)}

    # scores = torch.hstack([r[:, 4] for r, idx in zip(raw_regions, roi_frame_idxes) if len(r)])

    return regions, roi_origin


def depict_region_by_tuple(volume, regions, roi_origin, region_ptrs, z_ratio, rec_len_pt: int = 8, rec_pt_mode: str = "p") -> torch.Tensor:
    """region: [cx, cy, w, h]"""
    pt_tuple = torch.zeros((len(region_ptrs), rec_len_pt), device = volume.device, dtype = volume.dtype)

    if "ps" in rec_pt_mode:
        pt_tuple[:, :5] = region2cxcyz_shape(regions, z_ratio).to(volume.dtype)
    elif "p" in rec_pt_mode:
        pt_tuple[:, :3] = region2cxcyz(regions, z_ratio).to(volume.dtype)
    if "i" in rec_pt_mode:
        pt_tuple[:, -2:] = region2misi(regions, volume, roi_origin).to(volume.dtype)

    # data = pt_tuple[:, -2].detach().cpu().numpy()
    # plt.hist(data, bins = 50, color = 'blue', alpha = 0.7)
    # plt.title('Histogram of Tensor Values')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    return pt_tuple


def filter_regions_by_scores(scores, cur_num_region, ts: float = 0.):
    cur_intensity_t = torch.sort(scores, descending = True)[0][cur_num_region - 1] if ts == 0. else ts
    cur_region_idxes = torch.where(scores >= cur_intensity_t)[0]
    deleted_idxes = torch.where(scores < cur_intensity_t)[0]
    return cur_region_idxes, deleted_idxes, cur_intensity_t


def refresh_variables(regions, region_pt_tuple, region_ptrs, cur_region_idxes):
    new_regions = dict()
    cur_region_ptrs = torch.LongTensor(region_ptrs)[cur_region_idxes]
    for frame_idx in regions.keys():
        frame_ptrs = cur_region_ptrs[cur_region_ptrs[:, 0] == frame_idx, :][:, 1]
        if len(frame_ptrs) > 0:
            new_regions[frame_idx] = regions[frame_idx][frame_ptrs]
    return new_regions, index_region(new_regions), region_pt_tuple[cur_region_idxes]


def filter_regions_by_neuron_depth(pt_tuple: torch.tensor, neuron_dict: Dict, num_region_one_neuron_t: int = 1):
    cur_region_idxes, deleted_idxes = list(), list()
    save_neuron_ids = list()
    for nid, neuron_idxes in neuron_dict.items():
        if len(neuron_idxes) > num_region_one_neuron_t:
            cur_region_idxes += neuron_idxes
            save_neuron_ids.append(nid)
        else:
            deleted_idxes += neuron_idxes
    idx_map = {idx: i for i, idx in enumerate(sorted(cur_region_idxes))}
    new_neuron_dict = {n: [idx_map[idx] for idx in neuron_dict[nid]] for n, nid in enumerate(save_neuron_ids)}
    new_region_map = {v: key for key, value in new_neuron_dict.items() for v in value}
    return torch.LongTensor(cur_region_idxes).to(pt_tuple.device), torch.LongTensor(deleted_idxes).to(pt_tuple.device), new_neuron_dict, new_region_map


def calc_shape_t(a_wh, b_wh):
    t = 0.5 * (torch.norm(a_wh / 2) + torch.norm(b_wh / 2))
    return t


def merge_region_by_pos_shape(X, ratio: float = 0.5):
    """

    Args:
        X: 8 tuples, (cx, cy, z, w, h, d, intensity_mean, intensity_std). (cx, cy, z, w, h, intensity_mean) are used for merging.

    Returns:

    """

    # cosine_threshold = torch.cos(torch.deg2rad(torch.tensor(deg_threshold, device = X.device, dtype = X.dtype)))
    zidexes = [torch.argwhere(X[:, 2] == z_value).squeeze(1) for z_value in torch.unique(X[:, 2])]
    bi_mats, bi_graphs = list(), list()  # the distance matrix and matched connection of a bipartite graph between two adjacent frames

    neuron_dict, region_map = dict(), dict()  # digital id -> index , index -> digital id

    # initialize ids from the first frame
    for did, idx in enumerate(zidexes[0]):
        neuron_dict[did] = [idx.item()]
        region_map[idx.item()] = did

    # allocate digital ids for the rest frames
    for prev_i, cur_i in zip(zidexes[:-1], zidexes[1:]):
        prev_X, cur_X = X[prev_i], X[cur_i]
        bi_mat = torch.norm(prev_X[:, :2].unsqueeze(1) - cur_X[:, :2].unsqueeze(0), dim = 2).detach().cpu().numpy()
        shape_dis = ratio * (torch.norm(prev_X[:, 3:5] * .5, dim = 1).unsqueeze(1) + torch.norm(cur_X[:, 3:5] * .5, dim = 1).unsqueeze(0)).detach().cpu().numpy()
        bi_graph = [[_y, _x] for _y, _x in zip(*linear_sum_assignment(bi_mat)) if bi_mat[_y, _x] <= shape_dis[_y, _x]]

        # merge regions with neurons allocated
        for _y, _x in bi_graph:
            neuron_dict[region_map[prev_i[_y].item()]].append(cur_i[_x].item())
            region_map[cur_i[_x].item()] = region_map[prev_i[_y].item()]

        # create new neurons
        for _cur_i in cur_i:
            if _cur_i.item() not in region_map:
                neuron_dict[len(neuron_dict)] = [_cur_i.item()]
                region_map[_cur_i.item()] = len(neuron_dict) - 1

        bi_mats.append(bi_mat)
        bi_graphs.append(bi_graph)

    return neuron_dict, region_map


def find_local_minima(tensor, grad_int_t: float = 10.):
    return torch.nonzero((tensor + grad_int_t < torch.cat([tensor[1:], tensor[-1:]])) & (tensor + grad_int_t < torch.cat([tensor[:1], tensor[:-1]]))).squeeze(1) + 1


def merge_single_length_sublists(lists, min_len: int = 1):
    result = []
    for i, sub_lst in enumerate(lists):
        if len(sub_lst) == min_len and i > 0:
            result[-1].extend(sub_lst)
        else:
            result.append(sub_lst)
    return result


def split_neuron_by_intensity_mean(neuron_dict: Dict, region_map: Dict, region_pt_tuple: torch.Tensor, len_neuron_t4split: int = 3):
    """region_pt_tuple: [cx, cy, z, w, h, d, intensity_mean, intensity_std]"""
    new_neuron_dict = neuron_dict.copy()
    for neuron_id, neuron_idxes in neuron_dict.items():
        if len(neuron_idxes) > len_neuron_t4split:
            mins = find_local_minima(region_pt_tuple[neuron_idxes, -2])
            if len(mins) > 0:
                _split_neuron_idxes = [neuron_idxes[:mins[0]]] + [neuron_idxes[mins[i]:mins[i + 1]] for i in range(len(mins) - 1)] + [neuron_idxes[mins[-1]:]]
                split_neuron_idxes = merge_single_length_sublists(_split_neuron_idxes)
                # if len(split_neuron_idxes) != len(_split_neuron_idxes):
                #     print(neuron_id, region_pt_tuple[neuron_idxes, -2], split_neuron_idxes)

                for i, cur_neuron_idxes in enumerate(split_neuron_idxes):
                    if i == 0:
                        new_neuron_dict[neuron_id] = cur_neuron_idxes
                    else:
                        cur_neuron_id = len(new_neuron_dict)
                        new_neuron_dict[cur_neuron_id] = cur_neuron_idxes
                        for idx in new_neuron_dict[cur_neuron_id]:
                            region_map[idx] = cur_neuron_id

    return new_neuron_dict, region_map


def filter_neurons(neuron_dict: Dict, region_map: Dict, span_t4filter: int = 2):
    new_neuron_dict, new_region_map = dict(), dict()
    cur_region_idxes, deleted_region_idxes = list(), list()
    for _, neuron_idxes in neuron_dict.items():
        if len(neuron_idxes) >= span_t4filter:
            cur_neuron_id = len(new_neuron_dict)
            new_neuron_dict[cur_neuron_id] = neuron_idxes
            cur_region_idxes += neuron_idxes
            new_region_map.update({idx: cur_neuron_id for idx in neuron_idxes})
        else:
            deleted_region_idxes += neuron_idxes

    maps = dict()
    for i in range(len(region_map)):
        if i not in deleted_region_idxes:
            maps[i] = len(maps)
    assert len(maps) == len(cur_region_idxes)

    new_neuron_dict = {nid: [maps[idx] for idx in neuron_idxes] for nid, neuron_idxes in new_neuron_dict.items()}
    new_region_map = {maps[idx]: nid for idx, nid in new_region_map.items()}

    return new_neuron_dict, new_region_map, cur_region_idxes, deleted_region_idxes


# --------------- neuron recognition ---------------
def depict_neuron_by_tuple(region_pt_tuple: torch.Tensor, neuron_dict: Dict, rec_len_pt: int = 8, rec_pt_mode: str = "p"):
    """region_pt_tuple: [cx, cy, z, w, h, d, intensity_mean, intensity_std]"""
    neuron_pt_tuple = torch.zeros((len(neuron_dict), rec_len_pt), device = region_pt_tuple.device, dtype = region_pt_tuple.dtype)

    for idx, (nid, idxes) in enumerate(neuron_dict.items()):
        neurons = region_pt_tuple[idxes]
        mean_infos = torch.sum(1 / len(neurons) * neurons[:, :5], dim = 0)  # cx, cy, z, w, h
        if "p" in rec_pt_mode:
            neuron_pt_tuple[idx, :3] = mean_infos[:3]
        if "s" in rec_pt_mode:
            depth = torch.max(neurons[:, 2]) - torch.min(neurons[:, 2])
            neuron_pt_tuple[idx, 3:5] = mean_infos[3:5]
            neuron_pt_tuple[idx, 5] = depth
        if "i" in rec_pt_mode:
            areas = neurons[:, 3] * neurons[:, 4]
            neuron_pt_tuple[idx, 6:8] = torch.sum(1 / torch.sum(areas) * neurons[:, 6:8] * areas.unsqueeze(1), dim = 0)

    if torch.isinf(neuron_pt_tuple).any():
        raise ValueError("inf value in neuron_pt_tuple")

    return neuron_pt_tuple


def normalize_pt_tuple(pt_tuple, mode = "p"):
    max_extent_xyz = torch.max(pt_tuple[:, :3]) - torch.min(pt_tuple[:, :3])
    if "p" in mode:
        pt_tuple[:, :3] = (pt_tuple[:, :3] - pt_tuple[:, :3].mean(axis = 0)) / (max_extent_xyz + 1e-8)
    if "s" in mode:
        pt_tuple[:, 3:6] = pt_tuple[:, 3:6] / (max_extent_xyz + 1e-8)
    if "i" in mode:
        pt_tuple[:, 6:8] = (pt_tuple[:, 6:8] - pt_tuple[:, 6:8].mean(axis = 0)) / (pt_tuple[:, 6:8].std(axis = 0) + 1e-8)

    return pt_tuple


def build_rec_input(tensor: torch.Tensor, target_num, pt_mode: str):
    tensor = tensor.unsqueeze(0) if tensor.dim() == 2 else tensor
    other_info_mask = torch.zeros((tensor.shape[0], 1, 1), device = tensor.device, dtype = tensor.dtype) if pt_mode == "p" else torch.ones((tensor.shape[0], 1, 1), device = tensor.device, dtype = tensor.dtype)
    mask = torch.zeros((tensor.shape[0], target_num), device = tensor.device, dtype = torch.bool)
    is_warning = False

    if tensor.shape[1] > target_num:
        resize_t = tensor[:, :target_num]
        idx = target_num
        is_warning = True
    elif tensor.shape[1] < target_num:
        resize_t = torch.cat([tensor, torch.zeros((tensor.shape[0], target_num - tensor.shape[1], tensor.shape[2]), device = tensor.device, dtype = tensor.dtype)], dim = 1)
        idx = tensor.shape[1]
        mask[:, idx:] = True
    else:
        resize_t = tensor
        idx = tensor.shape[1]

    return resize_t, idx, other_info_mask, mask, is_warning


def cos2deg(cos_sim, eps = 1e-7):
    return torch.rad2deg_(torch.acos(torch.clamp(cos_sim, min = -1 + eps, max = 1 - eps)))


# --------------- Inference Time Augmentation(ITA) for neuron recognition ---------------

def get_random_rotation_matrix(ita_num, device, dtype):
    random_angles = torch.rand(ita_num, device = device, dtype = dtype) * 2 * torch.pi
    random_rotation_matrix = torch.stack([torch.cos(random_angles), -torch.sin(random_angles), torch.sin(random_angles), torch.cos(random_angles)], dim = 1).view(ita_num, 2, 2)
    return random_rotation_matrix


def get_ita_random_rotation(rec_input, r_matrix, ita_num: int = 1):
    """rec_input: [1, num_neuron, 8]"""

    rotated_rec_input = rec_input.repeat(ita_num, 1, 1)
    torch.matmul(rotated_rec_input[:, :, :2], r_matrix, out = rotated_rec_input[:, :, :2])

    return rec_input


# --------------- New Merge ---------------
def _find_matching_pairs(base, base_pred, base_pred_mask, base_init,
                         tgt, tgt_pred, tgt_pred_mask, tgt_init,
                         topk: int = 3, dis_t: float = 10., delta_dis_t: float = 1.0):
    """pred: [num_pred, 2], target: [num_target, 2]"""
    _dtype, _device = torch.long, base.device
    # -1 represents no match
    sorted_indices_base_tgt, sorted_indices_tgt_base = -1 + torch.zeros((base.shape[0], topk), dtype = _dtype, device = _device), -1 + torch.zeros((tgt.shape[0], topk), dtype = _dtype, device = _device)
    topk_t = min(topk, tgt.shape[0])
    mat1 = torch.norm(base_pred.unsqueeze(1) - tgt.unsqueeze(0), dim = 2)
    mat2 = torch.norm(tgt_pred.unsqueeze(1) - base.unsqueeze(0), dim = 2)
    mask = base_pred_mask * tgt_pred_mask.squeeze(1).unsqueeze(0)  # predicted mask
    mask *= ((mat1 < dis_t) * (mat2.T < dis_t))  # distance mask # TODO: evaluate
    # TODO:     mask *= (mat1 < dis_t) * (mat2 < dis_t) # distance mask
    mask *= (torch.abs(mat1 - mat2.T) < delta_dis_t)  # matching mask

    # the pair, whose item is True in mask and whose item in base_pred, is one of the prediction result.
    # symmetric results
    for i, (mat1_raw, l_mat1) in enumerate(zip(torch.argsort(mat1, dim = 1), torch.sum(mask, dim = 1))):
        if l_mat1:
            _r_mat1 = mat1_raw[:topk_t][:l_mat1]
            sorted_indices_base_tgt[i, :len(_r_mat1)] = _r_mat1.to(_dtype) + tgt_init  # relative index to absolute index
    for i, (mat2_raw, l_mat2) in enumerate(zip(torch.argsort(mat2, dim = 1), torch.sum(mask.T, dim = 1))):
        if l_mat2:
            _r_mat2 = mat2_raw[:topk_t][:l_mat2]
            sorted_indices_tgt_base[i, :len(_r_mat2)] = _r_mat2.to(_dtype) + base_init  # relative index to absolute index

    return sorted_indices_base_tgt, sorted_indices_tgt_base


def _find_matching_pairs_volume(regions, pointers,
                                score_t: float = .8,
                                topk: int = 3, dis_t: float = 10., delta_dis_t: float = 1.0):
    cur, upper_mask, upper, lower_mask, lower = pointers
    frame_idxes = list(regions.keys())
    _dtype, _device = torch.long, regions[frame_idxes[0]].device
    # -1 represents no match
    results = {idx: -1 + torch.zeros((regions[idx].shape[0], topk * 2), dtype = _dtype, device = _device) for idx in frame_idxes}  # {i: [upper, lower]}
    # TODO: if volume is not continuous, the following code will be wrong.
    inits = {key: value for key, value in zip(frame_idxes, torch.cumsum(torch.tensor([0] + [regions[idx].shape[0] for idx in frame_idxes[:-1]]), dim = 0))}
    for i, idx in enumerate(frame_idxes):
        if i != len(frame_idxes) - 1:
            # i -> i+1; cur_upper, upper_lower
            c_p, u_p = frame_idxes[i], frame_idxes[i + 1]
            results[c_p][:, :topk], results[u_p][:, topk:] = _find_matching_pairs(base = regions[c_p][:, cur], base_pred = regions[c_p][:, upper], base_pred_mask = regions[c_p][:, upper_mask] >= score_t, base_init = inits[c_p],
                                                                                  tgt = regions[u_p][:, cur], tgt_pred = regions[u_p][:, lower], tgt_pred_mask = regions[u_p][:, lower_mask] >= score_t, tgt_init = inits[u_p],
                                                                                  topk = topk, dis_t = dis_t, delta_dis_t = delta_dis_t)

        if i != 0:
            # i -> i-1; cur_lower, lower_cur
            c_p, l_p = frame_idxes[i], frame_idxes[i - 1]
            results[c_p][:, topk:], results[l_p][:, :topk] = _find_matching_pairs(base = regions[c_p][:, cur], base_pred = regions[c_p][:, lower], base_pred_mask = regions[c_p][:, lower_mask] >= score_t, base_init = inits[c_p],
                                                                                  tgt = regions[l_p][:, cur], tgt_pred = regions[l_p][:, upper], tgt_pred_mask = regions[l_p][:, upper_mask] >= score_t, tgt_init = inits[l_p],
                                                                                  topk = topk, dis_t = dis_t, delta_dis_t = delta_dis_t)
    return results


def _reformat_regions(regions, results):
    """10-tuple for every region:
            (cx, cy, w, h, upper_top1, upper_top2, upper_top3, lower_top1, lower_top2, lower_top3), where "top" is the absolute idx of region in the volume"""
    return {key: torch.cat([regions[key][:, :4].long(), results[key]], dim = 1) for key in list(regions.keys())}


def _calc_product_brightenss_score(regions, regions_w_info, volume, max_intensity: float):
    """weighted score = score * normalized brightness; approximate brightness"""

    frame_idxes = list(regions.keys())

    scores = torch.hstack([regions[key][:, 4] for key in frame_idxes])
    # import matplotlib.pyplot as plt
    # a = scores[:, None].detach().cpu().numpy()
    # plt.hist(a, bins = 10, alpha = 0.7, color = 'red')
    # plt.title('Histogram of Scores')
    # plt.xlabel('Scores')
    # plt.ylabel('Frequency')
    # plt.show()

    brightness = torch.hstack([volume[key, 0, regions_w_info[key][:, 1], regions_w_info[key][:, 0]] for key in frame_idxes])  # IMPORTANT: volume: bchw, regions_w_info: x,y,...
    # import matplotlib.pyplot as plt
    # a = brightness[:, None].detach().cpu().numpy()
    # plt.hist(a, bins = 10, alpha = 0.7, color = 'red')
    # plt.title('Histogram of Brightness')
    # plt.xlabel('Brightness')
    # plt.ylabel('Frequency')
    # plt.show()

    max_intensity = max_intensity if max_intensity else torch.max(brightness)

    weighted_score = scores * brightness / max_intensity

    return weighted_score, max_intensity


def _tree_search(topk_indices, region_list: list, regions: dict, region_ptrs: list, is_processed: torch.Tensor, upper: bool = True, topk: int = 3):
    """recursion algorithm"""
    # inplace operation
    cur_idx = -1
    for idx in topk_indices:
        if (idx + 1) and (not is_processed[idx]):
            is_processed[idx] = True
            cur_idx = idx
            region_list.append(idx)  # absolute idx
            break
    if cur_idx + 1:  # when cur_idx = -1, that will be false
        info = regions[region_ptrs[cur_idx][0]][region_ptrs[cur_idx][1]]  # absolute idx -> relative idx -> absolute idx
        next_topk_indices = info[-topk * 2:-topk] if upper else info[-topk:]
        next_topk_indices = next_topk_indices[next_topk_indices != -1]
        _tree_search(next_topk_indices, region_list, regions, region_ptrs, is_processed, upper = upper, topk = topk)


def _dp_merge(regions: dict, scores: torch.Tensor, region_ptrs: list, topk: int = 3):
    """dynamic programming"""
    is_processed = torch.zeros(scores.shape, dtype = torch.bool, device = scores.device)
    neuron_dict = dict()
    # a = scores[:, None].detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.hist(a, bins = 10, alpha = 0.7, color = 'red')
    # plt.title('Histogram of Weighted scores')
    # plt.xlabel('Weighted scores')
    # plt.ylabel('Frequency')
    # plt.show()
    indices = torch.argsort(scores, descending = True)
    for cur_idx in indices:
        if not is_processed[cur_idx]:
            is_processed[cur_idx] = True
            info = regions[region_ptrs[cur_idx][0]][region_ptrs[cur_idx][1]]  # absolute idx -> relative idx -> absolute idx
            upper_topk_indices, lower_topk_indices = info[-topk * 2:-topk], info[-topk:]
            upper_topk_indices, lower_topk_indices = upper_topk_indices[upper_topk_indices != -1], lower_topk_indices[lower_topk_indices != -1]
            # symmetric operations
            region_list = [cur_idx]  # store absolute idx
            # upper
            _tree_search(upper_topk_indices, region_list, regions, region_ptrs, is_processed, upper = True, topk = topk)  # inplace operation
            # lower
            _tree_search(lower_topk_indices, region_list, regions, region_ptrs, is_processed, upper = False, topk = topk)  # inplace operation

            neuron_dict[len(neuron_dict)] = sorted(region_list)

    return neuron_dict


def post_processing_merge(regions, merge_pointer, max_intensity, topk: int = 3):
    """regions: {frame_idx: torch.tensor([[]])}"""
    region_ptrs = index_region(regions)  # [frame_idx, region_idx_in_frame]
    results = _find_matching_pairs_volume(regions, merge_pointer, score_t = .8, topk = topk, dis_t = 10., delta_dis_t = 1.)
    regions_w_info = _reformat_regions(regions, results)  # Long dtype
    weighted_score, max_intensity = _calc_product_brightenss_score(regions, regions_w_info, volume, max_intensity)
    neuron_dict = _dp_merge(regions = regions_w_info, scores = weighted_score, region_ptrs = region_ptrs, topk = topk)  # dynamic programming
    return neuron_dict, region_ptrs, max_intensity, weighted_score


# ------------------- Models -------------------
class Treeformer_End2End(nn.Module):
    def __init__(self, ext_path, det_path, merge_path, rec_path,
                 ext_input_dim: tuple or list, det_input_dim: tuple or list, rec_input_dim: tuple or list,
                 ext_conf: float = .4, ext_ratio: int = 8,
                 det_conf: float = .6, det_iou_t: float = .6, det_num_region_t_frame: int = 300, region_shrink_pixel: int = 2,
                 rec_pt_mode: str = "p",
                 xoy_unit = 0.3, z_unit = 1.5,  # xoy: 1 pixel = 0.3 um, z: 1 pixel = 1.5 um
                 rec_ita_num: int = 1,

                 is_ext: bool = True,
                 magnification: int = 40,
                 mp: int = 0,
                 verbose: bool = True,
                 ) -> None:
        super().__init__()

        self.ext = torch.jit.load(ext_path)
        self.ext_input_dim = ext_input_dim
        self.ext_conf = ext_conf
        self.ext_ratio = ext_ratio
        self.ext_batch_size_t = ext_input_dim[0]

        self.is_ext = is_ext
        self.magnification = magnification

        self.det = torch.jit.load(det_path)
        self.det_input_dim = det_input_dim
        self.det_side_t = det_input_dim[-1]
        self.det_batch_size_t = det_input_dim[0]
        self.det_conf = det_conf
        self.det_iou_t = det_iou_t
        self.det_num_region_t_frame = det_num_region_t_frame
        self.det_num_region_t_vol = 0
        self.region_shrink_pixel = region_shrink_pixel

        self.merge = torch.jit.load(merge_path)
        self.MERGE_POINTER = [0, 1], [5, ], [6, 7], [8, ], [9, 10]

        self.rec = torch.jit.load(rec_path)
        self.rec_input_dim = rec_input_dim
        self.z_ratio = z_unit / xoy_unit
        self.rec_len_pt = rec_input_dim[-1]
        self.rec_num_neuron = rec_input_dim[1]
        self.rec_pt_mode = rec_pt_mode
        self.rec_ita_num = rec_ita_num
        self.PT_MODE = {"p": [0, 1, 2], "ps": [0, 1, 2, 3, 4, 5], "pi": [0, 1, 2, 6, 7], "psi": [0, 1, 2, 3, 4, 5, 6, 7]}

        self.rec_ita_rotated_matrix = None
        self.ts = 0.

        self.mp = mp
        self.max_intensity = 0.
        self._verbose = verbose

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose
        print("verbose has been set to {}".format(self._verbose))

    def forward(self, volume):
        """a unit of the reigion: [cx, cy, w, h]"""
        basic_r, warning_r, supp_r = self._forward_once(volume)
        return (*basic_r, warning_r) if not self.verbose else (*basic_r, warning_r, supp_r)

    def _forward_once(self, volume):
        # ------------------- head extraction -------------------
        if self.is_ext:
            ext_input = self.ext(volume)
            head_bboxes = post_processing_ext(ext_input, self.ext_conf, self.ext_ratio)
            head_bboxes = _ext_relative2absolute(head_bboxes)
        else:
            head_bboxes = torch.ones((volume.shape[0], 5), device = volume.device, dtype = volume.dtype)
            head_bboxes[:, :4] = torch.IntTensor([[volume.shape[-1] // 2, volume.shape[-2] // 2, self.det_input_dim[-1], self.det_input_dim[-2]]]).repeat(volume.shape[0], 1)  # [cx, cy, w, h]

        # ------------------- neuronal region detection -------------------
        batch, roi_frame_idxes, scales, batch_size, ext_is_warning = build_det_batch(volume, head_bboxes, self.det_side_t, batch_size_t = self.det_batch_size_t)
        coarse_preds, coarse_feas = self.det(batch)
        region_preds = self.merge(batch, coarse_feas)

        raw_regions, det_is_warning, self.det_num_region_t_vol, cur_num_region = post_processing_det(region_preds, batch_size, self.det_conf, self.det_iou_t,
                                                                                                     det_num_region_t_frame = self.det_num_region_t_frame,
                                                                                                     det_num_region_t_vol = self.det_num_region_t_vol,
                                                                                                     shrink_pixel = self.region_shrink_pixel)
        regions, roi_origin = recover_region_pos_in_roi(raw_regions, roi_frame_idxes, scales, head_bboxes)

        # ------------------- neuronal region merge -------------------
        neuron_dict, region_ptrs, self.max_intensity, weighted_score = post_processing_merge(regions, self.MERGE_POINTER, self.max_intensity, topk = 3)

        # setup return values
        basic_r = regions, neuron_dict
        warning_r = ext_is_warning, det_is_warning
        supp_r = head_bboxes, region_ptrs, weighted_score

        return basic_r, warning_r, supp_r
    #
    # # @profile
    # def _forward_backup(self, volume):
    #     # ------------------- head extraction -------------------
    #     if self.is_ext:
    #         ext_input = self.ext(volume)
    #         head_bboxes = post_processing_ext(ext_input, self.ext_conf, self.ext_ratio)
    #         head_bboxes = _ext_relative2absolute(head_bboxes)
    #     else:
    #         head_bboxes = torch.ones((volume.shape[0], 5), device = volume.device, dtype = volume.dtype)
    #         head_bboxes[:, :4] = torch.IntTensor([[volume.shape[-1] // 2, volume.shape[-2] // 2, self.det_input_dim[-1], self.det_input_dim[-2]]]).repeat(volume.shape[0], 1)  # [cx, cy, w, h]
    #
    #     # ------------------- neuronal region detection -------------------
    #     batch, roi_frame_idxes, scales, batch_size, ext_is_warning = build_det_batch(volume, head_bboxes, self.det_side_t, batch_size_t = self.det_batch_size_t)
    #     coarse_preds, coarse_feas = self.det(batch)
    #     region_preds = self.merge(batch, coarse_feas)
    #
    #     raw_regions, det_is_warning, self.det_num_region_t_vol, cur_num_region = post_processing_det(region_preds, batch_size, self.det_conf, self.det_iou_t,
    #                                                                                                  det_num_region_t_frame = self.det_num_region_t_frame,
    #                                                                                                  det_num_region_t_vol = self.det_num_region_t_vol,
    #                                                                                                  shrink_pixel = self.region_shrink_pixel)
    #     regions, roi_origin = recover_region_pos_in_roi(raw_regions, roi_frame_idxes, scales, head_bboxes)
    #
    #     # ------------------- neuronal region merge -------------------
    #     neuron_dict, region_ptrs, self.max_intensity, weighted_score = post_processing_merge(regions, self.MERGE_POINTER, self.max_intensity, topk = 3)
    #
    #     region_pt_tuple = depict_region_by_tuple(volume, regions, roi_origin, region_ptrs, self.z_ratio, rec_len_pt = 8, rec_pt_mode = "psi")
    #     ppp = torch.concat([region_pt_tuple, weighted_score.unsqueeze(1)], dim = 1)
    #
    #     # cur_region_idxes, deleted_region_idxes = filter_regions_by_scores(region_pt_tuple[:, -2], 60)  # Sparse mode
    #     cur_region_idxes, deleted_region_idxes, self.ts = filter_regions_by_scores(ppp[:, -3] / ppp[:, -3].max() * ppp[:, -1], int(cur_num_region * .8))  # Sparse mode
    #     # cur_region_idxes, deleted_region_idxes = filter_regions_by_intensity(region_pt_tuple, cur_num_region)  # Whole mode
    #
    #     regions, region_ptrs, region_pt_tuple = refresh_variables(regions, region_pt_tuple, region_ptrs, cur_region_idxes)
    #     neuron_dict, region_map = merge_region_by_pos_shape(region_pt_tuple, ratio = 0.9)
    #     new_neuron_dict, region_map = split_neuron_by_intensity_mean(neuron_dict, region_map, region_pt_tuple, len_neuron_t4split = 4)
    #
    #     # filter the neuron only live in one frame
    #     cur_region_idxes, deleted_region_idxes, new_new_neuron_dict, region_map = filter_regions_by_neuron_depth(region_pt_tuple, new_neuron_dict, num_region_one_neuron_t = 1)
    #     regions, region_ptrs, region_pt_tuple = refresh_variables(regions, region_pt_tuple, region_ptrs, cur_region_idxes)
    #
    #     print(
    #         f"\n==================== neuron dict ====================\t{len(neuron_dict)}"
    #         # f"\n==================== new neuron dict ====================\t {len(new_neuron_dict)}"
    #         # f"\n==================== new new neuron dict ====================\t {len(new_new_neuron_dict)}"
    #     )
    #
    #     # ------------------- neuron recognition -------------------
    #     _neuron_pt_tuple = depict_neuron_by_tuple(region_pt_tuple, neuron_dict, rec_len_pt = self.rec_len_pt, rec_pt_mode = "psi")
    #     neuron_pt_tuple = normalize_pt_tuple(_neuron_pt_tuple.clone(), mode = self.rec_pt_mode)
    #     if torch.isinf(neuron_pt_tuple).any():
    #         raise ValueError("inf value in neuron_pt_tuple")
    #     rec_input, idx, other_info_mask, mask, rec_is_warning = build_rec_input(neuron_pt_tuple, self.rec_num_neuron, pt_mode = self.rec_pt_mode)
    #     if self.rec_ita_num == 1:
    #         neuron_emb = self.rec(rec_input, other_info_mask, mask).squeeze(0)[:idx]
    #     else:
    #         self.rec_ita_rotated_matrix = get_random_rotation_matrix(self.rec_ita_num, rec_input.device, rec_input.dtype) if self.rec_ita_rotated_matrix is None else self.rec_ita_rotated_matrix
    #         rec_input = get_ita_random_rotation(rec_input, self.rec_ita_rotated_matrix, ita_num = self.rec_ita_num)
    #         rec_output = self.rec(rec_input, other_info_mask.repeat(self.rec_ita_num, 1, 1), mask.repeat(self.rec_ita_num, 1))
    #         neuron_emb = F.normalize(torch.mean(rec_output[:, :idx], dim = 0), dim = -1)
    #
    #     is_warnings = (ext_is_warning, det_is_warning, rec_is_warning)
    #     return head_bboxes, regions, region_pt_tuple, region_pt_tuple, region_ptrs, neuron_emb, (neuron_dict, region_map, _neuron_pt_tuple, ppp), (len(region_map), len(neuron_dict), None), is_warnings
    #


class VolumeMemoryBuffer:
    def __init__(self, save_path = None, deg_t: int = 45, weight_W: float = 1.0, warmup_num_vol: int = 3):
        self.deg_t = deg_t
        self.save_path = save_path
        self.warmup_num_vol = warmup_num_vol
        self.weight_W = weight_W if weight_W != -1 else (1.0 / warmup_num_vol)

        self.init_buffers()

    def init_buffers(self):
        self.W = None  # [N, dim]
        self.ids = list()  # Every item is an allocated digital id corresponding to W
        self.pairs = list()  # Every pair is a list that records idx tuples of a neuron corresponding to W. e.g. [(0, 0), (1, 1)]
        self.W_pointer = list()  # every item is the newest idx tuple of a neuron corresponding to W.
        self.num_vol_counter = -1  # the number of volumes that have been recognized

        self.W_map = dict()  # key: i for W, value: neuron id

    @property
    def size(self):
        return len(self.ids)

    def save(self):
        torch.save(self.W, self.save_path + "_W.pt")

    def allocate_id_neuron2regions(self, neuron_pred_ids, neuron_dict, num_regs):
        region_pred_ids = [-1, ] * num_regs
        for pred_id, (_, r_ids) in zip(neuron_pred_ids, neuron_dict.items()):
            for r in r_ids:
                region_pred_ids[r] = pred_id
        return region_pred_ids

    def _recognize_neuron_volume_first(self, he):
        self.W = he
        self.ids = list(range(he.shape[0]))
        self.pairs = [[(self.num_vol_counter, _id)] for _id in self.ids]
        self.W_pointer = [(self.num_vol_counter, _id) for _id in self.ids]
        self.W_map = {i: _id for i, _id in enumerate(self.ids)}
        return self.ids

    def _recognize_neuron_volume_rest(self, he, deg_t: int = 30, is_updated: bool = True):
        bi_mat = cos2deg(F.linear(he, self.W)).detach().cpu().numpy()  # TODO: calculate degree into cosine
        bi_graph = [[_y, _x] for _y, _x in zip(*linear_sum_assignment(bi_mat)) if bi_mat[_y, _x] <= deg_t]
        preds = [-1, ] * he.shape[0]

        # if is_updated and self.num_vol_counter < 3:
        #     for i, (cur_i, W_i) in enumerate(bi_graph):
        #         preds[cur_i] = self.W_map[W_i]
        #
        #         # replace buffer by the last one
        #         self.W[W_i] = F.normalize((he[cur_i] * self.weight_W + self.W[W_i] * (1 - self.weight_W)).unsqueeze(0)).squeeze(0)
        #         self.pairs[W_i].append((self.num_vol_counter, cur_i))
        #         self.W_pointer[W_i] = (self.num_vol_counter, cur_i)
        #
        #     # allocate new ids for -1
        #     for i, pred in enumerate(preds):
        #         if pred == -1:
        #             self.W_map[self.size] = max(self.W_map.values()) + 1
        #             preds[i] = self.W_map[self.size]
        #             self.ids.append(self.W_map[self.size])
        #
        #             self.W = torch.cat([self.W, he[i].unsqueeze(0)], dim = 0)
        #             self.pairs.append([(self.num_vol_counter, i)])
        #             self.W_pointer.append((self.num_vol_counter, i))
        # else:
        #     for i, (cur_i, W_i) in enumerate(bi_graph):
        #         preds[cur_i] = self.W_map[W_i]

        for i, (cur_i, W_i) in enumerate(bi_graph):
            preds[cur_i] = self.W_map[W_i]

            # replace buffer by the last one
            self.W[W_i] = F.normalize((he[cur_i] * self.weight_W + self.W[W_i] * (1 - self.weight_W)).unsqueeze(0)).squeeze(0)
            self.pairs[W_i].append((self.num_vol_counter, cur_i))
            self.W_pointer[W_i] = (self.num_vol_counter, cur_i)

            # allocate new ids for -1
        for i, pred in enumerate(preds):
            if pred == -1:
                self.W_map[self.size] = max(self.W_map.values()) + 1
                preds[i] = self.W_map[self.size]
                self.ids.append(self.W_map[self.size])

                self.W = torch.cat([self.W, he[i].unsqueeze(0)], dim = 0)
                self.pairs.append([(self.num_vol_counter, i)])
                self.W_pointer.append((self.num_vol_counter, i))

        return preds

    def __call__(self, he, neuron_dict, num_regions, is_warning: bool = False):
        """
        Recognize neurons in a volume and update buffers.
        Args:
            he: hypersphere embedding matrix for neurons in a volume

        Returns:
            a list of predicted digital ids corresponding to the input he.
        """
        self.num_vol_counter += 1
        neuron_pred_ids = self._recognize_neuron_volume_first(he) if self.W is None else self._recognize_neuron_volume_rest(he, deg_t = self.deg_t, is_updated = not is_warning)
        region_pred_ids = self.allocate_id_neuron2regions(neuron_pred_ids, neuron_dict, num_regions)
        return neuron_pred_ids, region_pred_ids, len(self.W)


# ------------------- Utils -------------------
def draw_volume_result(volume, head_bboxes, regions, pred_ids, region_ptrs, save_fig_root, name, verbose: bool = False, others_class_start_id: int = 2000, shift: int = 1):
    vis_id = True if pred_ids is not None else False
    only_show_ig = True if regions is None else False
    if not only_show_ig:
        head_bboxes = cxcywh2xyxy(head_bboxes)
        roi_vol_idxes = torch.nonzero(head_bboxes[:, -1]).squeeze(1)
        m = torch.concat([torch.min(head_bboxes[roi_vol_idxes][:, :2], dim = 0)[0], torch.max(head_bboxes[roi_vol_idxes][:, 2:4], dim = 0)[0]])
        max_scope_xyxy = torch.IntTensor([int(m[0]), int(m[1]), int(m[2]), int(m[3])])

    # _min, _max = volume.min(), volume.max()

    reg_i = 0
    num_rows = int(np.ceil(np.sqrt(volume.shape[0])))
    num_cols = int(np.ceil(volume.shape[0] / num_rows))
    plt.figure(figsize = (num_cols * 5, num_rows * 5))
    for vol_idx in range(volume.shape[0]):
        plt.subplot(num_rows, num_cols, vol_idx + 1)
        if not only_show_ig:
            img = volume[vol_idx, 0, max_scope_xyxy[1]: max_scope_xyxy[3], max_scope_xyxy[0]: max_scope_xyxy[2]]
        else:
            img = volume[vol_idx, 0]
        _min, _max = img.min(), img.max()
        img = np.array(img.cpu())
        plt.imshow(img, cmap = "afmhot", vmin = _min, vmax = _max)
        plt.colorbar()
        if not only_show_ig:
            if vol_idx in regions.keys():
                if vis_id:
                    if vol_idx != region_ptrs[reg_i][0]:
                        pass
                    assert vol_idx == region_ptrs[reg_i][0], f"vol_idx: {vol_idx}, region_ptrs[reg_i][0]: {region_ptrs[reg_i][0]}"
                bb = np.array(head_bboxes[vol_idx].cpu() - torch.IntTensor([max_scope_xyxy[0], max_scope_xyxy[1], max_scope_xyxy[0], max_scope_xyxy[1], 0]))
                # head bbox
                plt.gca().add_patch(plt.Rectangle((int(bb[0]), int(bb[1])), bb[2] - bb[0], bb[3] - bb[1], linewidth = 1, edgecolor = '#7B68EE', facecolor = "None"))
                # neuronal regions
                for r in np.array(cxcywh2xywh(regions[vol_idx]).cpu() - torch.IntTensor([max_scope_xyxy[0], max_scope_xyxy[1], 0, 0])):
                    plt.gca().add_patch(plt.Rectangle((int(r[0]), int(r[1])), r[2], r[3], linewidth = 1, edgecolor = '#FF1493', facecolor = "None"))
                    if vis_id:
                        plt.text(int(r[0] + r[2] * .5), int(r[1] + r[3] * .5), s = str(pred_ids[reg_i] + shift) if pred_ids[reg_i] < others_class_start_id else "-1",
                                 verticalalignment = 'center', horizontalalignment = 'center',
                                 fontdict = {"color": "blue", "fontsize": 2, "weight": "bold"}
                                 )
                        reg_i += 1

        plt.title(f"frame {vol_idx + shift}" if not only_show_ig else f"W frame {vol_idx + shift}")

    if save_fig_root != "":
        plt.savefig(f"{save_fig_root}/{name}.pdf")
    if verbose:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import time
    import json


    def benchmark(model, volume, dtype = 'fp32', nwarmup = 50, nruns = 1000):
        if dtype == 'fp16':
            model.half()
            volume = volume.half()

        print("Warm up ...")
        with torch.inference_mode():
            for _ in range(nwarmup):
                features = model(volume)
        torch.cuda.synchronize()
        print("Start timing ...")
        timings = []
        with torch.inference_mode():
            for i in range(1, nruns + 1):
                start_time = time.time()
                features = model(volume)
                torch.cuda.synchronize()
                end_time = time.time()
                timings.append(end_time - start_time)
                if i % 100 == 0:
                    print('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

        print("Input shape:", volume.size())
        # print("Output features size:", features.size())

        print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))


    config = "/home/cbmi/CBMI_python/src/configs/inference/cbmi2/zm9644_m_c.json"
    with open(config, "r") as f:
        config = json.load(f)

    treeformer = Treeformer_End2End(

        ext_path = config["ext_path"],
        det_path = config["det_path"],
        merge_path = config["merge_path"],  # its input is taken care of by the output of detection model
        rec_path = config["rec_path"],

        ext_input_dim = [config["zrange"][1] - config["zrange"][0]] + config["ext_input_dim"],
        det_input_dim = [config["zrange"][1] - config["zrange"][0]] + config["det_input_dim"],
        rec_input_dim = config["rec_input_dim"],

        ext_conf = config["ext_conf"],

        det_conf = config["det_conf"], det_iou_t = config["det_iou_t"],
        region_shrink_pixel = config["region_shrink_pixel"],

        rec_pt_mode = config["rec_pt_mode"],
        xoy_unit = config["xoy_unit"], z_unit = config["z_unit"],  # um/pixel
        rec_ita_num = config["rec_ita_num"],

        is_ext = config["is_ext"],
        magnification = config["magnification"],

        verbose = True,

    ).eval().cuda().half()

    for path in ["/home/cbmi/CBMI_python/data/zone/cache2/ImgStk001_dk001_w10_Dt230525_000030.pt",
                 "/home/cbmi/CBMI_python/data/zone/cache2/ImgStk001_dk001_w10_Dt230525_000000.pt",
                 "/home/cbmi/CBMI_python/data/zone/cache2/ImgStk001_dk001_w10_Dt230525_000010.pt",
                 "/home/cbmi/CBMI_python/data/zone/cache2/ImgStk001_dk001_w10_Dt230525_000032.pt"]:
        volume = torch.load(path)[config["zrange"][0]: config["zrange"][1]]

        with torch.inference_mode():
            regions, neuron_dict, is_warnings, supp_r = treeformer(volume)
            head_bboxes, region_ptrs, weighted_score = supp_r
            print(f" the number of regions: {weighted_score.shape[0]}, \t the number of neurons: {len(neuron_dict)}")
