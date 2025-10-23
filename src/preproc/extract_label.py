# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import re
import os
import cv2
import mat73
import numpy as np
from tqdm import tqdm
from typing import List
from scipy.io import loadmat
from collections import OrderedDict

from src.comm_utils.prints import pad_num, print_warning_message


def is_include(bbox: list, pts: List[np.ndarray]):
    """

    :param bbox: [z, xmin, ymin, xmax, ymax, iskey]
    :param pts: [ys, xs]
    :return:
    """
    xmin, ymin, xmax, ymax = bbox[1:5]
    ys_xmin = pts[0][pts[1] == xmin]
    ys_xmax = pts[0][pts[1] == xmax]

    if (len(ys_xmin) != 0) & (len(ys_xmax) != 0):
        return (np.min(ys_xmin) <= ymin) & (np.min(ys_xmax) <= ymin) & (np.max(ys_xmin) >= ymax) & (np.max(ys_xmax) >= ymax)
    else:
        return False


def extract_stack_label_mat(anno_path: str, vol_idxes: List, label_reg: str, frame_scope: tuple[int, int] = (0, 18)):
    raw_labels = loadmat(anno_path)['neuron_boxes']
    stack_name = re.search(label_reg, anno_path).group()
    labels = OrderedDict()
    # assert max(vol_idxes) <= (raw_labels.shape[0] - 1)
    for vol_idx in vol_idxes:
        vol_name = f"{stack_name}_{pad_num(vol_idx, 6)}"
        vol_labels = dict()
        for slice_idx in range(*frame_scope):
            raw_label_slice = raw_labels[vol_idx, slice_idx]
            if raw_label_slice.size:
                raw_label_slice = raw_label_slice[0]
                for neuron in raw_label_slice:

                    neuron_id = neuron[1].squeeze() - 1
                    if neuron_id.size > 1:
                        neuron_id = neuron_id[0]
                    elif neuron_id.size == 0:
                        print_warning_message(f"{stack_name} vol_{vol_idx + 1} (1-based indexing) {neuron} doesn't annotate ID, and default ID is -1")
                        neuron_id = -1
                    neuron_name = int(neuron_id)

                    neuron_bbox = neuron[4].squeeze()
                    neuron_bbox[0:2] -= 1
                    neuron_bbox[2:4] += (neuron_bbox[0:2] + 1)
                    # neuron_bbox[2:4] += (neuron_bbox[0:2] - 1)

                    neuron_label = [slice_idx - frame_scope[0], *neuron_bbox.tolist(), neuron[0].squeeze().tolist()]
                    neuron_label = [int(item) for item in neuron_label]
                    if neuron_name not in vol_labels.keys():
                        vol_labels[neuron_name] = [neuron_label]
                    else:
                        vol_labels[neuron_name].append(neuron_label)

        labels[vol_name] = vol_labels

    return labels


def extract_stack_label_mat73(anno_path: str, vol_idxes: List, label_reg: str, frame_scope: tuple[int, int] = (0, 18)):
    raw_labels = mat73.loadmat(anno_path)['neuron_boxes']
    stack_name = re.search(label_reg, anno_path).group()
    labels = OrderedDict()
    # assert max(vol_idxes) <= (raw_labels.shape[0] - 1)
    for vol_idx in vol_idxes:
        vol_name = f"{stack_name}_{pad_num(vol_idx, 6)}"
        vol_labels = dict()
        for slice_idx in range(*frame_scope):
            raw_label_slice = raw_labels[vol_idx][slice_idx]
            if len(raw_label_slice):
                raw_label_slice = {key: np.array([value]) for key, value in raw_label_slice.items()} if isinstance(raw_label_slice['idx'], np.ndarray) else raw_label_slice
                for neuron in zip(*list(raw_label_slice.values())):
                    neuron_id = neuron[2].squeeze() - 1
                    if neuron_id.size > 1:
                        neuron_id = neuron_id[0]
                    elif neuron_id.size == 0:
                        print_warning_message(f"{stack_name} vol_{vol_idx + 1} (1-based indexing) {neuron} doesn't annotate ID, and default ID is -1")
                        neuron_id = -1
                    neuron_name = int(neuron_id)

                    neuron_bbox = neuron[0].squeeze()
                    neuron_bbox[0:2] -= 1
                    neuron_bbox[2:4] += (neuron_bbox[0:2] + 1)
                    # neuron_bbox[2:4] += (neuron_bbox[0:2] - 1)
                    # TODO: check in det
                    neuron_label = [slice_idx - frame_scope[0], *neuron_bbox.tolist(), neuron[3].squeeze().tolist()]
                    neuron_label = [int(item) for item in neuron_label]
                    if neuron_name not in vol_labels.keys():
                        vol_labels[neuron_name] = [neuron_label]
                    else:
                        vol_labels[neuron_name].append(neuron_label)

        labels[vol_name] = vol_labels
    return labels


def extract_stack_label(anno_path: str, vol_idxes: List, label_reg: str, frame_scope: tuple[int, int] = (0, 18)):
    """[z, xmin, ymin, xmax, ymax, _]"""
    try:
        labels = extract_stack_label_mat(anno_path, vol_idxes, label_reg, frame_scope)
    except NotImplementedError:
        labels = extract_stack_label_mat73(anno_path, vol_idxes, label_reg, frame_scope)

    return labels


def transfer_raw2region(raw_label, p = [0, 0, 0], is_sorted: bool = False):
    """(xmin, ymin, xmax, ymax, z)"""
    regions = [[x1 - p[0], y1 - p[1], x2 - p[0], y2 - p[1], z - p[2]] for neuron in raw_label.values() for z, x1, y1, x2, y2, _ in neuron]
    regions = sorted(regions, key = lambda r: r[-1]) if is_sorted else regions
    return regions


def transfer_raw2region_wid(raw_label, p = [0, 0, 0], is_sorted: bool = False):
    """(xmin, ymin, xmax, ymax, z, id)"""
    regions = [[x1 - p[0], y1 - p[1], x2 - p[0], y2 - p[1], z - p[2], int(neuron_id)] for neuron_id, neuron in raw_label.items() for z, x1, y1, x2, y2, _ in neuron]
    regions = sorted(regions, key = lambda r: r[-1]) if is_sorted else regions
    return regions


def split_vol2frame_regions(vol_regions):
    nd_regs = np.array(vol_regions)
    regions = {z: nd_regs[nd_regs[:, 4] == z].tolist() for z in np.unique(nd_regs[:, 4])}
    return regions


def load_region_label(label_root, infos, name_reg, frame_scope: tuple[int, int] = (0, 18), include_id: bool = False):
    infos = [stack_info for _, idv_infos in infos.items() for stack_info in idv_infos]
    regions = OrderedDict()
    for file_name, scope in tqdm(infos, desc = "loading region label \t"):
        path = os.path.join(label_root, file_name)
        raw_stack_label = extract_stack_label(path, scope, name_reg, frame_scope)
        if include_id:
            stack_regions = {vol_name: split_vol2frame_regions(transfer_raw2region_wid(vol_raw_label, is_sorted = True)) for vol_name, vol_raw_label in raw_stack_label.items()}
        else:
            stack_regions = {vol_name: split_vol2frame_regions(transfer_raw2region(vol_raw_label)) for vol_name, vol_raw_label in raw_stack_label.items()}
        regions.update(stack_regions)
    return regions


def dilate_heatmap(heatmaps):
    """ inplace operation """
    for stack_idx in range(heatmaps.shape[0]):
        for frame_idx in range(heatmaps.shape[1]):
            img = heatmaps[stack_idx, frame_idx].copy().astype(np.uint8)
            img = cv2.dilate(img, np.ones((13, 13), np.uint8), iterations = 5)
            img = cv2.erode(img, np.ones((13, 13), np.uint8), iterations = 3)
            heatmaps[stack_idx, frame_idx] = img.astype(np.bool)
