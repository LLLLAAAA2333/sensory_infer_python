# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

"""
    3D semantic segmentation metric (IoU, SEG, MUCov):
    https://www.nature.com/articles/s41540-020-00152-8.pdf
    https://github.com/funalab/QCANet/blob/master/src/tools/evaluation_seg.py
"""

from src.comm_utils.packages import *
from src.comm_utils.prints import print_error_message
from src.comm_utils.metrics.MNR_metric.utils import calc_intersection_union, calc_pixel_intersection_union_by_brightness


def calc_merge_score_wlabel(preds: Dict, labels: Dict, threshold: float, method: str = "SEG",
                            iou_type: str = "box", **kwargs):
    """Calculate SEG, MUCov, and F1 score like det.

    :param preds:
    :param labels:
    :param threshold:
    :param method:
    :return:
    """

    if method.upper() == "SEG":
        method = calc_SEG_score
    elif method.upper() == "MUCOV":
        method = calc_MUCov_score
    elif method.upper() == "F1SCORE":
        method = calc_3d_neuron_score
    else:
        print_error_message(f" Merge Metric {method} doesn't exist! ")

    score, results = method(preds, labels, threshold, iou_type, **kwargs)

    return score, results


def calc_neuron_iou(a: List, b: List,
                    iou_type: str = "box", **kwargs):
    """IoU of two list of regions

    :param a: [R1, R2, ...]. R: [xmin, ymin, xmax, ymax, z]
    :param b: the same as a
    :return:
    """

    a, b = a.copy(), b.copy()
    inter_areas = 0
    union_areas = 0

    for r_a in a:
        flag = False
        for i_b, r_b in enumerate(b):
            if r_a[-1] == r_b[-1]:
                flag = True
                inter_area, union_area = calc_intersection_union(r_a[:4], r_b[:4]) if iou_type == "box" else calc_pixel_intersection_union_by_brightness(r_a, r_b, kwargs['vol'], kwargs['ratio'])
                inter_areas += inter_area
                union_areas += union_area
                b.pop(i_b)
                break
        if not flag:
            union_areas += (r_a[2] - r_a[0]) * (r_a[3] - r_a[1]) if iou_type == "box" else int((r_a[2] - r_a[0]) * (r_a[3] - r_a[1]) * kwargs['ratio'])
    union_areas += ((sum([(r[2] - r[0]) * (r[3] - r[1]) for r in b])) if iou_type == "box" else sum([int((r[2] - r[0]) * (r[3] - r[1]) * kwargs['ratio']) for r in b]))
    iou = inter_areas / (union_areas + 1e-5)

    return iou


def calc_3d_neuron_score(preds: Dict, labels: Dict, threshold: float,
                         iou_type: str = "box", **kwargs):
    """Merge Metric like IoU.

    :param preds: {vol_name: {id: [R1, ...] ...}}. R: [xmin, ymin, xmax, ymax, z]
    :param labels: the same as results
    :param threshold:
    :return: [precision, recall, f1_score, results]
    """

    preds, labels = preds.copy(), labels.copy()
    results = dict()
    tps, fps, fns = 0, 0, 0
    for vol_name in preds.keys():
        tp, fp = 0, 0
        pred, label = preds[vol_name], labels[vol_name]
        pred_list = sorted(list(pred.values()), key = lambda x: len(x), reverse = True)
        label_list = sorted(list(label.values()), key = lambda x: len(x), reverse = True)
        if iou_type != "box":
            vol = torch.FloatTensor(kwargs['volumes'][vol_name].astype(np.int32))

        for n in pred_list:
            max_iou = sys.float_info.min
            max_idx = -1
            for gt_i, gt in enumerate(label_list):
                iou = calc_neuron_iou(n, gt, iou_type = "box")
                if iou >= max_iou:
                    max_iou = iou if iou_type == "box" else calc_neuron_iou(n, gt, "pixel", vol = vol, **kwargs)
                    max_idx = gt_i
            if max_iou >= threshold:
                tp += 1
                label_list.pop(max_idx)
            else:
                fp += 1
        fn = len(label_list)
        tps, fps, fns = tps + tp, fps + fp, fns + fn
        p, r = tp / (tp + fp), tp / (tp + fn)
        results[vol_name] = [p, r, (2 * p * r) / (p + r + 1e-5)]

    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-5)

    return [precision, recall, f1_score], results


def calc_SEG_score(preds: Dict, labels: Dict, threshold: float = 0.5,
                   iou_type: str = "box", **kwargs):
    """SEG Metric. Like recall metric

    :param preds: {vol_name: {id: [R1, ...] ...}}. R: [xmin, ymin, xmax, ymax, z]
    :param labels: the same as results
    :param threshold:
    :return:
    """

    preds, labels = preds.copy(), labels.copy()
    results = dict()
    ious = 0.0
    num_labels = 0

    for vol_name in preds.keys():
        pred, label = preds[vol_name], labels[vol_name]
        pred_list = sorted(list(pred.values()), key = lambda x: len(x), reverse = True)
        label_list = sorted(list(label.values()), key = lambda x: len(x), reverse = True)
        # recording variable
        vol_ious = 0.0
        num_labels += len(label)
        if iou_type != "box":
            vol = torch.FloatTensor(kwargs['volumes'][vol_name].astype(np.int32))

        for n in pred_list:
            max_iou = sys.float_info.min
            max_idx = -1
            for gt_i, gt in enumerate(label_list):
                iou = calc_neuron_iou(n, gt, iou_type = "box")
                if iou >= max_iou:
                    max_iou = iou if iou_type == "box" else calc_neuron_iou(n, gt, "pixel", vol = vol, **kwargs)
                    max_idx = gt_i
            if max_iou >= threshold:
                vol_ious += max_iou
                label_list.pop(max_idx)
        ious += vol_ious
        results[vol_name] = vol_ious / len(label)
    SEG_score = ious / num_labels

    return SEG_score, results


def calc_MUCov_score(preds: Dict, labels: Dict, threshold: float = 0.5, iou_type: str = "box", **kwargs):
    """MUCov Metric. Like precision metric

    :param preds: {vol_name: {id: [R1, ...] ...}}. R: [xmin, ymin, xmax, ymax, z]
    :param labels: the same as results
    :param threshold:
    :return:
    """

    preds, labels = preds.copy(), labels.copy()
    results = dict()
    ious = 0.0
    num_preds = 0

    for vol_name in preds.keys():
        pred, label = preds[vol_name], labels[vol_name]
        pred_list = sorted(list(pred.values()), key = lambda x: len(x), reverse = True)
        label_list = sorted(list(label.values()), key = lambda x: len(x), reverse = True)
        # recording variable
        vol_ious = 0.0
        num_preds += len(pred)
        if iou_type != "box":
            vol = torch.FloatTensor(kwargs['volumes'][vol_name].astype(np.int32))

        for gt in label_list:
            max_iou = sys.float_info.min
            max_idx = -1
            for p_i, p in enumerate(pred_list):
                iou = calc_neuron_iou(gt, p, iou_type = "box")
                if iou >= max_iou:
                    max_iou = iou if iou_type == "box" else calc_neuron_iou(gt, p, "pixel", vol = vol, **kwargs)
                    max_idx = p_i
            if max_iou >= threshold:
                vol_ious += max_iou
                pred_list.pop(max_idx)
        ious += vol_ious
        results[vol_name] = vol_ious / len(pred)
    MUCov_score = ious / num_preds

    return MUCov_score, results
