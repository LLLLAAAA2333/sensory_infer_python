# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

"""
    Reference: https://pypi.org/project/object-detection-metrics/

"""

from src.comm_utils.packages import *
from src.comm_utils.metrics.MNR_metric.utils import calc_intersection_union, calc_iou
from src.comm_utils.metrics.MNR_metric.utils import calc_pixel_iou_by_brightness


def calc_det_score_wlabel(preds: Dict, labels: Dict, threshold: float, iou_type: str = "box", **kwargs):
    """Calculate recall, precision and F1 score for detection with raw label"""

    preds = {name: [r for s in vol_res.values() for r in s] for name, vol_res in preds.items()}
    labels = {name: [r for s in vol_res.values() for r in s] for name, vol_res in labels.items()}
    precision, recall, f1_score, results = calc_det_score(preds, labels, threshold, iou_type = iou_type, **kwargs)

    return precision, recall, f1_score, results


def calc_det_score(preds: Dict, labels: Dict, threshold: float, iou_type: str = "box", **kwargs):
    """Calculate recall, precision and F1 score for detection

    :param preds: {vol_name: [R1, R2, ...]} value is a list of slice regions. R: [xmin, ymin, xmax, ymax, z]
    :param labels: {vol_name: [R1, R2, ... ]}.
    :param threshold: float type
    :return: [precision, recall, f1_score, results]
    """

    results = dict()
    tps, fps, fns = 0, 0, 0
    for vol_name in preds.keys():
        tp = 0
        fp = 0
        pred, label = preds[vol_name], labels[vol_name]
        if iou_type != "box":
            vol = torch.FloatTensor(kwargs['volumes'][vol_name].astype(np.int32))
        for r in pred:
            max_iou = sys.float_info.min
            max_idx = -1
            for gt_i, gt in enumerate(label):
                if r[-1] == gt[-1]:
                    iou = calc_iou(gt[:4], r[:4])
                    if iou >= max_iou:
                        max_iou = iou if iou_type == "box" else calc_pixel_iou_by_brightness(gt, r, vol, kwargs['ratio'])
                        max_idx = gt_i
            if max_iou >= threshold:
                tp += 1
                label.pop(max_idx)
            else:
                fp += 1
        fn = len(label)
        tps, fps, fns = tps + tp, fps + fp, fns + fn
        p, r = tp / (tp + fp), tp / (tp + fn)
        results[vol_name] = [p, r, (2 * p * r) / (p + r + 1e-5)]

    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall + 1e-5)

    return precision, recall, f1_score, results
