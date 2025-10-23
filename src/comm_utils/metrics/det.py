# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from src.comm_utils.packages import *
from src.utils import cxcywh2xyxy


def metric_at(iou_matrix, iou_threshold):
    """
        Compute the metric at a specific IoU threshold.
    :param iou_matrix: [N, M], IoU matrix. N = len(labels), M = len(preds)
    :param iou_threshold: float, IoU threshold
    :return:
    """

    matches = iou_matrix >= iou_threshold
    tps = torch.sum(torch.sum(matches, dim = 1) > 0)  # Correct objects
    return tps


def calc_det_metric(preds, targets, scores, iou_thresholds = []):
    """

    :param preds:  List of predicted boxes, each element is a list of boxes for a single image
    :param targets: List of target boxes, each element is a list of boxes
    :param scores: List of scores, each element is a list of scores
    :param iou_thresholds: List or int, IoU thresholds
    :return: Dict: [[t, num_tps, num_fps, num_fns, precision, recall, f1_score]]
    """

    iou_matrics = [torchvision.ops.box_iou(cxcywh2xyxy(t), cxcywh2xyxy(p)) for t, p in zip(targets, preds) if len(t) and len(p)]

    num_preds, num_labels = sum([len(p) for p in preds]), sum([len(t) for t in targets])

    metrics = list()
    for thd in iou_thresholds:
        tps = sum([metric_at(m, thd) for m in iou_matrics])
        precision = tps / num_preds if num_preds > 0 else 0.0
        recall = tps / num_labels if num_labels > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        metrics.append({"thd": thd.item(), "tp": tps, "num_preds": num_preds, "num_labels": num_labels, "p": precision, "r": recall, "f": f1_score})

    return metrics
