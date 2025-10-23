# -*- coding: utf-8 -*-
#
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

from src.comm_utils.packages import *
from src.utils import cxcywh2xyxy


def calc_intersection_union(a: List, b: List):
    """
    :param a: [xmin, ymin, xmax, ymax, ...]
    :param b: [xmin, ymin, xmax, ymax, ...]
    :return:
    """

    inter_area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union_area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter_area

    return inter_area, union_area


def calc_iou(a: list, b: list):
    """"""
    inter_area, union_area = calc_intersection_union(a, b)
    iou = inter_area / (union_area + 1e-5)

    return iou


def calc_top_brightest_points(bbox: torch.FloatTensor,
                              volume: torch.HalfTensor,
                              ratio: float) -> torch.FloatTensor:
    """
        Calculate the top brightest points in a box.
        box: xyxyz
        volume: [C, H, W]
        ratio: float, ratio of the brightest points to be selected
    """
    bbox = torch.LongTensor(bbox)
    region = volume[bbox[1]: bbox[3], bbox[0]: bbox[2], bbox[4]]
    flatten_region = region.flatten()
    _, indexes = torch.topk(flatten_region, k = int(len(flatten_region) * ratio))
    relative_ccords = torch.stack([indexes % region.shape[1], torch.div(indexes, region.shape[1], rounding_mode = "floor")], dim = 1)  # [x, y]
    absolute_ccords = relative_ccords + bbox[:2]
    return absolute_ccords


def calc_euclidean_distance_matrix(p: torch.FloatTensor,
                                   gt: torch.FloatTensor):
    return torch.sum((p.unsqueeze(1) - gt.unsqueeze(0)) ** 2, dim = -1)


def calc_pixel_intersection_union_by_brightness(p_bbox: torch.FloatTensor,
                                                gt_bbox: torch.FloatTensor,
                                                volume: torch.HalfTensor,
                                                ratio: float):
    _p = calc_top_brightest_points(p_bbox, volume, ratio)
    _gt = calc_top_brightest_points(gt_bbox, volume, ratio)
    matrix = calc_euclidean_distance_matrix(_p, _gt)
    inter_area = torch.sum(torch.sum(matrix == 0, dim = 0) > 0)
    union_area = _p.shape[0] + _gt.shape[0] - inter_area
    return inter_area, union_area


def calc_pixel_iou_by_brightness(p_bbox: torch.FloatTensor,
                                 gt_bbox: torch.FloatTensor,
                                 volume: torch.HalfTensor,
                                 ratio: float):
    """
        Calculate the pixel IoU between a predicted box and a ground truth box by their brightness.
    """

    inter_area, union_area = calc_pixel_intersection_union_by_brightness(p_bbox, gt_bbox, volume, ratio)
    iou = inter_area / (union_area + 1e-6)
    return iou
