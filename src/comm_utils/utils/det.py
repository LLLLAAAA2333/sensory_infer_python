# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from src.comm_utils.packages import *


def calc_iou(a: List, b: List):
    """

    :param a: [xmin, ymin, xmax, ymax, ...]
    :param b: [xmin, ymin, xmax, ymax, ...]
    :return: intersection of union between two 2d bboxes (float scalar)
    """

    inter_area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union_area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter_area
    iou = inter_area / (union_area + 1e-5)

    return iou


def calc_ious4score_reg(score, reg, pred_score, pred_reg, pred_threshold: float, threshold: float = 0.5):
    """

    :param score: [N, 1]
    :param reg: [N, 4]
    :param pred_score: [N, 1]
    :param pred_reg: [N, 4]
    :param threshold: float type
    :return: [precision, mean_ious]
    """

    tps = 0
    ious = list()
    # inplace operation !!!
    reg, pred_reg = reg.clone(), pred_reg.clone()
    reg[:, 2:] += reg[:, :2]
    pred_reg[:, 2:] += pred_reg[:, :2]
    for s, r, s_h, r_h in zip(score, reg, pred_score, pred_reg):
        if (s_h < pred_threshold) and (s < threshold):
            tps += 1

        elif (s_h >= pred_threshold) and (s >= threshold):
            tps += 1
            iou = calc_iou(r, r_h)
            ious.append(iou)

    precision = tps / (len(score) + 1e-5)
    mean_ious = torch.mean(torch.Tensor(ious)) if len(ious) > 0 else 0.

    return precision, mean_ious
