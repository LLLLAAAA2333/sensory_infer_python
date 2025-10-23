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


def region2ci(regions: Dict, volume):
    """ extract the center intensity only
    :param regions: Dict, {z: [cx, cy, w, h]}
    :return: torch.Tensor, [center intensity]
    """
    return torch.stack([volume[frame_idx, 0, b[1], b[0]] for frame_idx, frame_region in regions.items() for b in frame_region.to(dtype = torch.int32)])


def region2misi(regions: Dict, volume, area_ratio: float = .6):
    """
    :param regions: Dict, {z: [cx, cy, w, h]}
    :return: torch.Tensor, [[mean_i, std_i]]
    """
    area_reduction = torch.tensor([1, 1, area_ratio, area_ratio], device = volume.device, dtype = torch.float32)
    output = list()
    for frame_idx, frame_region in regions.items():
        for b in cxcywh2xyxy(frame_region * area_reduction).to(dtype = torch.int32):
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
                        det_num_region_t_vol_factor: float = .10,
                        det_keep_ratio: float = .8,
                        ):
    """
    :return: torch.int32, [batch_size, 5] (x,y,w,h,flag), where flag represents whether the frame has a head.
    """
    is_warning = False
    candidates = preds[..., -1] >= (conf_thresh * .70)
    raw_output = [torch.zeros((0, 5), device = preds.device, dtype = preds.dtype)] * num_sampels
    scores = torch.zeros((0,), device = preds.device, dtype = preds.dtype)

    # Clean up by the score threshold, number threshold and IOU threshold (NMS)
    for pi in range(num_sampels):
        p = preds[pi][candidates[pi]]

        if p.shape[0]:
            p = p[p[:, 4].argsort(descending = True)[:max_nms]] if p.shape[0] > max_nms else p

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
        cur_num_region = int(cur_num_region * det_keep_ratio)
    elif (cur_num_region >= det_num_region_t_vol * (1 + det_num_region_t_vol_factor)) or (cur_num_region <= det_num_region_t_vol * (1 - det_num_region_t_vol_factor)):
        is_warning = True  # This volume will be ignored to update global variables, but still have a result.
        # print(f"Warning: {det_num_region_t_vol} is too different from {len(scores)}")
    else:
        det_num_region_t_vol = int(det_num_region_t_vol * smooth_factor + cur_num_region * (1 - smooth_factor))
        cur_num_region = min(len(scores), int(det_num_region_t_vol * det_keep_ratio))

    return raw_output, is_warning, det_num_region_t_vol, cur_num_region


def recover_region_pos_in_roi(raw_regions, roi_frame_idxes, scales, head_bboxes):
    """return [cx, cy, w, h]"""
    xys = cxcywh2xyxy(head_bboxes[roi_frame_idxes])[:, :2]
    roi_origin = torch.min(xys, dim = 0)[0]  # [xmin, ymin] for whole volume
    xys = torch.cat([xys, torch.zeros_like(xys)], dim = 1)

    regions = {idx.item(): bboxes_fp2int(r[:, :4] / scale + xy) for r, idx, scale, xy in zip(raw_regions, roi_frame_idxes, scales, xys) if len(r)}
    scores = torch.hstack([r[:, 4] for r, idx in zip(raw_regions, roi_frame_idxes) if len(r)])

    return regions, roi_origin, scores


# --------------- region merge by position and shape ---------------
def depict_region_by_tuple(volume, regions, region_ptrs, z_ratio, rec_len_pt: int = 8, rec_pt_mode: str = "p") -> torch.Tensor:
    """region: [cx, cy, w, h]"""
    pt_tuple = torch.zeros((len(region_ptrs), rec_len_pt), device = volume.device, dtype = volume.dtype)

    if "ps" in rec_pt_mode:
        pt_tuple[:, :5] = region2cxcyz_shape(regions, z_ratio).to(volume.dtype)
    elif "p" in rec_pt_mode:
        pt_tuple[:, :3] = region2cxcyz(regions, z_ratio).to(volume.dtype)
    if "i" in rec_pt_mode:
        # pt_tuple[:, -2:] = region2misi(regions, volume).to(volume.dtype)
        pt_tuple[:, -2] = region2ci(regions, volume).to(volume.dtype)
    pt_tuple[torch.isnan(pt_tuple)] = 0  # to avoid nans

    # bb = pt_tuple[:, -2].cpu().detach().numpy()
    # aa = region2ci(regions, volume).to(volume.dtype).cpu().detach().numpy()
    # print(np.corrcoef(aa, bb)[0, 1])
    #
    # plt.figure(figsize = (10, 6))
    # plt.plot(bb, label = 'Average Brightness', marker = 'o')
    # plt.plot(aa, label = 'Center Brightness', marker = 'x')
    #
    # plt.title('Average vs Center Brightness')
    # plt.xlabel('Image Index')
    # plt.ylabel('Brightness')
    # plt.legend()
    # plt.show()

    # data = pt_tuple[:, -2].detach().cpu().numpy()
    # plt.hist(data, bins = 50, color = 'blue', alpha = 0.7)
    # plt.title('Histogram of Tensor Values')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    return pt_tuple


def calc_weighted_score(brightness, score):
    return brightness / brightness.max() * score


def filter_regions_by_scores(scores, cur_num_region):
    cur_intensity_t = torch.sort(scores, descending = True)[0][cur_num_region - 1]
    mask = scores >= cur_intensity_t
    return mask


def refresh_variables_region(regions, region_ptrs, mask):
    new_regions = dict()
    cur_region_ptrs = torch.LongTensor(region_ptrs)[mask]
    for frame_idx in regions.keys():
        frame_ptrs = cur_region_ptrs[cur_region_ptrs[:, 0] == frame_idx, :][:, 1]
        if len(frame_ptrs) > 0:
            new_regions[frame_idx] = regions[frame_idx][frame_ptrs]
    return new_regions, index_region(new_regions)


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


def merge_region_by_pos_shape(X, iou_t: float = 0.2):
    """X: 8 tuples, (cx, cy, z, w, h, d, intensity_mean, intensity_std). (cx, cy, z, w, h, intensity_mean) are used for merging."""

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
        bi_mat = torchvision.ops.box_iou(cxcywh2xyxy(prev_X[:, [0, 1, 3, 4]]), cxcywh2xyxy(cur_X[:, [0, 1, 3, 4]])).detach().cpu().numpy()
        bi_graph = [[_y, _x] for _y, _x in zip(*linear_sum_assignment(-bi_mat)) if bi_mat[_y, _x] >= iou_t]  # Calculate the maximum Sum of IoU

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
            # Pay attention to data overflow
            area_ratio = (neurons[:, 3] * neurons[:, 4]) / torch.sum(neurons[:, 3] * neurons[:, 4])
            neuron_pt_tuple[idx, 6:8] = torch.sum(neurons[:, 6:8] * area_ratio.unsqueeze(1), dim = 0)

    # TODO: verbose mode, delete it in the future
    if torch.isinf(neuron_pt_tuple).any():
        raise ValueError("inf value in neuron_pt_tuple, because data overflow (HalfTensor)")

    return neuron_pt_tuple


def calc_cur_num_neuron(cur_num_neuron,
                        rec_num_region_t_vol: int = 0,
                        smooth_factor: float = .90,
                        rec_num_region_t_vol_factor: float = .20,
                        rec_keep_ratio: float = .90, ):
    is_warning = False
    if rec_num_region_t_vol == 0:
        rec_num_region_t_vol = cur_num_neuron
        cur_num_neuron = int(cur_num_neuron * rec_keep_ratio)
    elif (cur_num_neuron >= rec_num_region_t_vol * (1 + rec_num_region_t_vol_factor)) or (cur_num_neuron <= rec_num_region_t_vol * (1 - rec_num_region_t_vol_factor)):
        is_warning = True
    else:
        rec_num_region_t_vol = int(rec_num_region_t_vol * smooth_factor + cur_num_neuron * (1 - smooth_factor))
        cur_num_neuron = min(cur_num_neuron, int(rec_num_region_t_vol * rec_keep_ratio))

    return is_warning, rec_num_region_t_vol, cur_num_neuron


def refresh_variables_neuron(neuron_dict, region_map, neuron_mask: torch.tensor, region_mask):
    new_neuron_dict, new_region_map = dict(), dict()
    # cur_region_idxes, deleted_region_idxes = list(), list()
    deleted_region_idxes = list()
    for nid, neuron_idxes in neuron_dict.items():
        if neuron_mask[nid]:
            cur_neuron_id = len(new_neuron_dict)
            new_neuron_dict[cur_neuron_id] = neuron_idxes
            # cur_region_idxes += neuron_idxes
            new_region_map.update({idx: cur_neuron_id for idx in neuron_idxes})
        else:
            region_mask[neuron_idxes] = False
            deleted_region_idxes += neuron_idxes

    maps = dict()
    for i in range(len(region_map)):
        if i not in deleted_region_idxes:
            maps[i] = len(maps)
    # assert len(maps) == len(cur_region_idxes)

    new_neuron_dict = {nid: [maps[idx] for idx in neuron_idxes] for nid, neuron_idxes in new_neuron_dict.items()}
    new_region_map = {maps[idx]: nid for idx, nid in new_region_map.items()}

    return new_neuron_dict, new_region_map, region_mask


def normalize_pt_tuple(pt_tuple, mode = "p"):
    max_extent_xyz = torch.max(pt_tuple[:, :3]) - torch.min(pt_tuple[:, :3])
    if "p" in mode:
        pt_tuple[:, :3] = (pt_tuple[:, :3] - pt_tuple[:, :3].mean(axis = 0)) / (max_extent_xyz + 1e-8)
    if "s" in mode:
        pt_tuple[:, 3:6] = pt_tuple[:, 3:6] / (max_extent_xyz + 1e-8)
    if "i" in mode:
        pt_tuple[:, 6:8] = (pt_tuple[:, 6:8] - pt_tuple[:, 6:8].mean(axis = 0)) / (pt_tuple[:, 6:8].std(axis = 0) + 1e-8)

    # TODO: verbose mode, delete it in the future
    if torch.isinf(pt_tuple).any():
        raise ValueError("inf value in neuron_pt_tuple")

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


class Aligner_PCA(nn.Module):
    """Class to use PCA for aligning 2D point clouds on a torch tensor."""

    def __init__(self, dtype = torch.float32, device = torch.device("cpu")):
        """
        Initializes the Aligner_PCA class.
        :param dtype: Data type for tensors.
        :param device: The device on which tensors will be allocated.
        """
        super().__init__()
        self.last_pc = nn.Parameter(torch.tensor([0, 1], dtype = dtype, device = device), requires_grad = False)

    def forward(self, X):
        """
        Aligns the given point cloud.
        :param X: torch.FloatTensor of shape (n, 2) representing the point cloud.
        :return: Tuple containing rotated point cloud, standard deviation, range, and last principal component.
        """
        X_rot, std, max_min, self.last_pc[:] = self.align(X, self.last_pc)
        return X_rot, std, max_min

    @staticmethod
    def align(X, last_pc):
        """
        Static method to align the point cloud.
        :param X: Input tensor of the point cloud.
        :param last_pc: Last principal component for alignment reference.
        :return: Rotated point cloud, standard deviation, range, and new principal component.
        """
        # assert X.shape[1] == 2  # Ensure the input tensor is 2D
        is_half = X.dtype == torch.half

        # Center the data
        X -= X.mean(0)

        # Perform Singular Value Decomposition (SVD)
        X = X.float() if is_half else X
        U = torch.svd(torch.mm(X.t(), X))[0]
        U = U.half() if is_half else U

        # Find the first principal component
        pc = U[:, 0]

        # Determine the anterior/posterior and adjust the principal component
        pc = pc if torch.dot(last_pc, pc) > 0 else -pc

        # Build the rotation matrix
        rot = torch.tensor([[pc[1], pc[0]], [-pc[0], pc[1]]], device = X.device, dtype = X.dtype)

        # Rotate the point cloud
        X_rot = torch.mm(X, rot)

        # Calculate standard deviation and range (max-min) for template selection
        std = torch.std(X_rot[:, 0])
        max_min = X_rot[:, 0].max() - X_rot[:, 0].min()

        return X_rot, std, max_min, pc


class Warmup_Pickup:
    """Class for selecting a template key based on warmup volumes with caching mechanism."""

    def __init__(self, warmup_num: int = 10):
        """
        Initializes the Warmup_Pickup class.
        :param warmup_num: Number of volumes to use for warmup.
        """
        self._vol_infos = dict()  # Dictionary to store volume information
        self._template_key = None  # Cached template key
        self.WARMUP_NUM = warmup_num  # Number of volumes for warmup

    @property
    def vol_infos(self):
        """
        Getter for volume information.
        :return: Volume information dictionary.
        """
        return self._vol_infos

    @vol_infos.setter
    def vol_infos(self, value: dict):
        """
        Setter for volume information. Updates the volume info until the warmup number is reached.
        :param value: Dictionary containing new volume information.
        """
        if len(self._vol_infos) < self.WARMUP_NUM:
            self._vol_infos.update(value)

    @property
    def template_key(self):
        """
        Getter for the template key. The key is calculated once and then cached.
        :return: Calculated template key.
        """
        if self._template_key is None and len(self._vol_infos) == self.WARMUP_NUM:
            keys, values = list(self._vol_infos.keys()), torch.tensor(list(self._vol_infos.values()))
            self._template_key = keys[torch.argmin(values[:self.WARMUP_NUM, 0])]
        return self._template_key

    def __len__(self):
        """Return the number of volumes in the volume information."""
        return len(self._vol_infos)


# ------------------- Models -------------------
class Treeformer_End2End(nn.Module):
    def __init__(self, ext_path, det_path, rec_path,
                 ext_input_dim: tuple or list, det_input_dim: tuple or list, rec_input_dim: tuple or list,
                 ext_conf: float = .4, ext_ratio: int = 8,
                 det_conf: float = .6, det_iou_t: float = .6, det_num_region_t_frame: int = 300, region_shrink_pixel: int = 2,
                 rec_pt_mode: str = "p",
                 xoy_unit = 0.3, z_unit = 1.5,  # xoy: 1 pixel = 0.3 um, z: 1 pixel = 1.5 um

                 det_keep_ratio: float = .8, rec_keep_ratio: float = .95,

                 is_ext: bool = True,
                 magnification: int = 40,
                 mp: int = 0,
                 verbose: bool = True,

                 warmup_num_vol: int = 10,
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

        self.rec_num_region_t_vol = 0
        self.rec = torch.jit.load(rec_path)
        self.rec_input_dim = rec_input_dim
        self.z_ratio = z_unit / xoy_unit
        self.rec_len_pt = rec_input_dim[-1]
        self.rec_num_neuron = rec_input_dim[1]
        self.rec_pt_mode = rec_pt_mode
        self.PT_MODE = {"p": [0, 1, 2], "ps": [0, 1, 2, 3, 4, 5], "pi": [0, 1, 2, 6, 7], "psi": [0, 1, 2, 3, 4, 5, 6, 7]}

        self.det_keep_ratio = det_keep_ratio
        self.rec_keep_ratio = rec_keep_ratio

        self.rec_ita_rotated_matrix = None

        self.mp = mp
        self.verbose: bool = verbose

        self.aligner = Aligner_PCA()

    def forward(self, volume):
        main_r, warning_r, supp_r = self._forward_once(volume)
        return (*main_r, warning_r) if not self.verbose else (*main_r, warning_r, supp_r)

    # @profile
    def _forward_once(self, volume):
        # ------------------- head extraction -------------------
        head_bboxes = self.ext_opt(volume)
        # ------------------- neuronal region detection -------------------
        ext_is_warning, det_is_warning, cur_num_region, regions, scores = self.det_opt(volume, head_bboxes)
        # ------------------- merge -------------------
        regions, region_ptrs, region_map, region_pt_tuple, neuron_dict = self.merge_opt(volume, regions, scores, cur_num_region)
        # ------------------- neuron recognition -------------------
        merge_is_warning, _neuron_pt_tuple, neuron_dict, region_map, regions, region_ptrs, rec_is_warning, neuron_emb, (std, max_min) = self.rec_opt(region_pt_tuple, neuron_dict, region_map, regions, region_ptrs)

        print(f"\n==================== neuronal region ====================\t{len(region_ptrs)}"
              f"\n==================== neuron dict ====================\t{len(neuron_dict)}"
              )

        # setup return values
        main_r = head_bboxes, regions, neuron_dict, neuron_emb
        warning_r = ext_is_warning, det_is_warning, merge_is_warning, rec_is_warning
        supp_r = region_pt_tuple, region_ptrs, region_map, _neuron_pt_tuple, (len(region_map), len(neuron_dict)), (std, max_min)
        return main_r, warning_r, supp_r

    def ext_opt(self, volume):
        if self.is_ext:
            ext_input = self.ext(volume)
            head_bboxes = post_processing_ext(ext_input, self.ext_conf, self.ext_ratio)
            head_bboxes = _ext_relative2absolute(head_bboxes)
        else:
            head_bboxes = torch.ones((volume.shape[0], 5), device = volume.device, dtype = volume.dtype)
            head_bboxes[:, :4] = torch.IntTensor([[volume.shape[-1] // 2, volume.shape[-2] // 2, self.det_input_dim[-1], self.det_input_dim[-2]]]).repeat(volume.shape[0], 1)  # [cx, cy, w, h]
        return head_bboxes

    def det_opt(self, volume, head_bboxes: torch.tensor):
        batch, roi_frame_idxes, scales, batch_size, ext_is_warning = build_det_batch(volume, head_bboxes, self.det_side_t, batch_size_t = self.det_batch_size_t)
        raw_regions, det_is_warning, self.det_num_region_t_vol, cur_num_region = post_processing_det(self.det(batch), batch_size, self.det_conf, self.det_iou_t,
                                                                                                     det_num_region_t_frame = self.det_num_region_t_frame,
                                                                                                     det_num_region_t_vol = self.det_num_region_t_vol,
                                                                                                     shrink_pixel = self.region_shrink_pixel,
                                                                                                     det_keep_ratio = self.det_keep_ratio,
                                                                                                     )
        regions, roi_origin, scores = recover_region_pos_in_roi(raw_regions, roi_frame_idxes, scales, head_bboxes)
        return ext_is_warning, det_is_warning, cur_num_region, regions, scores

    def merge_opt(self, volume, regions, scores, cur_num_region):
        region_ptrs = index_region(regions)  # [frame_idx, region_idx_in_frame]
        region_pt_tuple = depict_region_by_tuple(volume, regions, region_ptrs, self.z_ratio, rec_len_pt = 8, rec_pt_mode = "psi")
        region_pt_tuple[:, -1] = calc_weighted_score(region_pt_tuple[:, -2], scores)

        _filtered_region_mask = filter_regions_by_scores(region_pt_tuple[:, -1], cur_num_region)  # Sparse mode
        region_pt_tuple = region_pt_tuple[_filtered_region_mask]
        regions, region_ptrs = refresh_variables_region(regions, region_ptrs, mask = _filtered_region_mask)

        neuron_dict, region_map = merge_region_by_pos_shape(region_pt_tuple, iou_t = 0.2)
        # neuron_dict, region_map = split_neuron_by_intensity_mean(neuron_dict, region_map, region_pt_tuple, len_neuron_t4split = 4)

        return regions, region_ptrs, region_map, region_pt_tuple, neuron_dict

    def rec_opt(self, region_pt_tuple, neuron_dict, region_map, regions, region_ptrs):
        _neuron_pt_tuple = depict_neuron_by_tuple(region_pt_tuple, neuron_dict, rec_len_pt = self.rec_len_pt, rec_pt_mode = "psi")
        merge_is_warning, self.rec_num_region_t_vol, cur_num_neuron = calc_cur_num_neuron(len(_neuron_pt_tuple), self.rec_num_region_t_vol, rec_keep_ratio = self.rec_keep_ratio)
        _filtered_neuron_mask: torch.tensor = filter_regions_by_scores(_neuron_pt_tuple[:, -1], cur_num_neuron)  # Sparse mode: filter neurons by area * brightness
        _neuron_pt_tuple = _neuron_pt_tuple[_filtered_neuron_mask]
        # neuron_coordinate = _neuron_pt_tuple.cpu().numpy()[:,0:6]/(680/1024)
        # np.save('/home/wenlab-user/RongWei/olfactory/0411/ImgStk002_dk001_w10_Dt20240411/synthetic_volume/w10_neuron_pt_tuple_1.npy', neuron_coordinate)
        
        neuron_dict, region_map, _filtered_region_mask = refresh_variables_neuron(neuron_dict, region_map, _filtered_neuron_mask, torch.ones_like(region_pt_tuple[:, 0], dtype = torch.bool))
        regions, region_ptrs = refresh_variables_region(regions, region_ptrs, mask = _filtered_region_mask)
        raw_neuron_pt_tuple = _neuron_pt_tuple.clone()
        
        # point cloud alignment
        _neuron_pt_tuple[:, :2], std, max_min = self.aligner(_neuron_pt_tuple[:, :2])
        
        neuron_pt_tuple = normalize_pt_tuple(_neuron_pt_tuple.clone(), mode = self.rec_pt_mode)
        neuron_pt_tuple[:, -1] = 0.

        rec_input, idx, other_info_mask, mask, rec_is_warning = build_rec_input(neuron_pt_tuple, self.rec_num_neuron, pt_mode = self.rec_pt_mode)
        neuron_emb = self.rec(rec_input, other_info_mask, mask).squeeze(0)[:idx]
        return merge_is_warning, raw_neuron_pt_tuple, neuron_dict, region_map, regions, region_ptrs, rec_is_warning, neuron_emb, (std, max_min)


class VolumeMemoryBuffer:
    def __init__(self, save_path = None, deg_t: int = 45, weight_W: float = 1.0, warmup_num_vol: int = 3):
        self.deg_t = deg_t
        self.save_path = save_path
        self.warmup_num_vol = warmup_num_vol
        self.weight_W = weight_W if weight_W != -1 else (1.0 / warmup_num_vol)

        self.pickup = Warmup_Pickup(warmup_num_vol)

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

        if is_updated and self.num_vol_counter < self.warmup_num_vol:
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
        else:
            for i, (cur_i, W_i) in enumerate(bi_graph):
                preds[cur_i] = self.W_map[W_i]

        # for i, (cur_i, W_i) in enumerate(bi_graph):
        #     preds[cur_i] = self.W_map[W_i]
        #
        #     # replace buffer by the last one
        #     self.W[W_i] = F.normalize((he[cur_i] * self.weight_W + self.W[W_i] * (1 - self.weight_W)).unsqueeze(0)).squeeze(0)
        #     self.pairs[W_i].append((self.num_vol_counter, cur_i))
        #     self.W_pointer[W_i] = (self.num_vol_counter, cur_i)
        #
        # # allocate new ids for -1
        # for i, pred in enumerate(preds):
        #     if pred == -1:
        #         self.W_map[self.size] = max(self.W_map.values()) + 1
        #         preds[i] = self.W_map[self.size]
        #         self.ids.append(self.W_map[self.size])
        #
        #         self.W = torch.cat([self.W, he[i].unsqueeze(0)], dim = 0)
        #         self.pairs.append([(self.num_vol_counter, i)])
        #         self.W_pointer.append((self.num_vol_counter, i))

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


# --------------- Utils for verbose, which will be deleted in someday ---------------
def plot_things(neuron_dict, region_pt_tuple):
    # -------------------------- DEBUG --------------------------
    brightss, areas, poses = dict(), dict(), dict()
    for key, n_idxes in neuron_dict.items():
        brightss[key] = region_pt_tuple[n_idxes, -2].cpu().detach().numpy()
        areas[key] = (region_pt_tuple[n_idxes, 3] * region_pt_tuple[n_idxes, 4]).cpu().detach().numpy()
        poses[key] = np.zeros_like(region_pt_tuple[n_idxes, 0].cpu().detach().numpy())
        matrix = region_pt_tuple[n_idxes, :2].cpu().detach().numpy()
        # distance between two pts
        poses[key][1:] = np.sqrt(np.sum(np.square(matrix[1:] - matrix[:-1]), axis = 1))

    depths = [len(value) for value in brightss.values()]
    from collections import Counter
    counts = Counter(depths)
    numbers, frequencies = zip(*counts.items())

    plt.bar(numbers, frequencies)
    plt.title(f'Depth Histogram, {len(depths)} neurons ')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(list(numbers))
    plt.show()

    idxes = np.argsort(-np.array(depths))
    for i in range(3):
        idx = idxes[i]
        x = list(range(len(brightss[idx])))
        fig, ax1 = plt.subplots()
        ax1.plot(x, brightss[idx], 'g-', marker = "x", label = 'intensity')
        ax1.plot(x, areas[idx], 'g-', marker = 'o', label = 'area')
        ax1.set_xlabel('frame')
        ax1.set_ylabel('brightness', color = 'g')
        ax1.tick_params('y', colors = 'g')
        ax1.legend(loc = 'upper left')

        ax2 = ax1.twinx()
        ax2.plot(x, poses[idx], 'b-', marker = '^', label = "delta")
        ax2.set_ylabel('delta', color = 'b')
        ax2.tick_params('y', colors = 'b')
        ax2.legend(loc = 'upper right')

        plt.title(f"neuron {idx + 1}")

        plt.show()
    # -------------------------- DEBUG --------------------------


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


def plot_volume_pt(pt, title: str = "", path: str = ""):
    """
    Plot the 2D point cloud.
    :param pt: 2D points in shape (n, 2).
    :param title: Title for the plot.
    """
    x, y = pt[:, 0], pt[:, 1]

    # Define the point color and background color
    point_color = '#89CFF0'  # Light blue color

    # Plot the point cloud
    plt.figure(figsize = (8, 6))
    plt.scatter(x, y, color = point_color, edgecolor = 'black')  # Black edge for points

    plt.axis('equal')  # Ensure equal unit length for x and y axes

    # Set labels and title
    plt.xlabel('X Axis', color = 'black')
    plt.ylabel('Y Axis', color = 'black')
    plt.title(title, color = 'black')

    # Save the plot image if a path is provided
    if path:
        plt.savefig(path)
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


    # config = "/home/wenlab-user/RongWei/olfactory/code_v1.10/src/configs/inference/wenlab_olfactory_si_womp.json"
    config = "/home/wenlab-user/RongWei/olfactory/code_v1.10/src/configs/inference/240411_new.json"
    # config = "/home/cbmi/CBMI_python/src/configs/inference/wenlab_olfactory.json"
    with open(config, "r") as f:
        config = json.load(f)

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

    # benchmark(model, volume, 'fp16')
    buffer = VolumeMemoryBuffer(
        save_path = "/home/wenlab-user/RongWei/olfactory/wnemos/0402",
        deg_t = 100,
        weight_W = 0.2,
        warmup_num_vol = 3,
    )

    for path in [
        # "/home/cbmi/CBMI_python/data/zone_end20231230/cache2/ImgStk001_dk001_w10_Dt230525_000050.pt",
        # "/home/cbmi/CBMI_python/data/zone_end20231230/cache2/ImgStk001_dk001_w10_Dt230525_000003.pt",
        # "/home/cbmi/CBMI_python/data/zone_end20231230/cache2/ImgStk001_dk001_w10_Dt230525_000025.pt",
        # "/home/cbmi/CBMI_python/data/zone_end20231230/cache2/ImgStk001_dk001_w10_Dt230525_000032.pt",
        # "/home/cbmi/CBMI_python/data/temp/tttt/ImgStk001_dk001_w1_Dt231102_volume.npy"
         "/home/wenlab-user/RongWei/olfactory/0411/ImgStk002_dk001_w10_Dt20240411/synthetic_volume/w10_aligned_volumes_all_mip.npy"
        # '/home/wenlab-user/RongWei/olfactory/20231118/w2/synthetic_volume/w2_aligned_volumes_part_mip.npy'
        # '/home/wenlab-user/RongWei/olfactory/20231118/w2/volume/ImgStk003_dk001_w2_Dt20231118_{AF}_{c2E-02_repeat1_green-1-to-930}/ImgStk003_dk001_w2_Dt202311_000007.npy'
        # '/home/wenlab-user/RongWei/olfactory/20231118/w2/volume/ImgStk002_dk001_w2_Dt20231118_{AF}_{c1E-01_repeat1_green-1-to-930}/ImgStk002_dk001_w2_Dt202311_000009.npy'
    ]:
        # volume = torch.load(path)[config["zrange"][0]: config["zrange"][1]]
        volume = torch.HalfTensor(np.load(path).transpose(2, 0, 1)[:, np.newaxis, :, :].astype(np.float32)).cuda()
        # volume = F.interpolate(volume, scale_factor=680/1024, mode='nearest')
        # _std, _mean = volume.std(), volume.mean()
        # volume[:] = -(_mean / _std * 22 - 119) + 22 / _std * volume

        with torch.inference_mode():
            head_bboxes, regions, neuron_dict, neuron_emb, is_warnings, supp_r = model(volume)
            region_pt_tuple, region_ptrs, region_map, neuron_pt_tuple, (num_region, num_neuron), (std, max_min) = supp_r
            buffer.pickup.vol_infos = {path: [std, max_min]}
            
            neuron_pred_ids, region_pred_ids, num_neuron_idv = buffer(neuron_emb, neuron_dict, num_region, any(is_warnings))
            # pt_tuple_data = torch.concat([neuron_pt_tuple.cpu(), torch.FloatTensor(neuron_pred_ids).unsqueeze(1)], dim = 1).cpu().numpy()
            # print(neuron_pred_ids)
            print(f" the number of regions: {num_region}, \t the number of neurons: {num_neuron}, \t  the number of neurons: {num_neuron_idv}")
            draw_volume_result(volume, head_bboxes, regions, region_pred_ids, region_ptrs, '/home/wenlab-user/RongWei/olfactory/0411/ImgStk002_dk001_w10_Dt20240411/synthetic_volume', name = "w10_aligned_volumes_mip_1", verbose = False)
