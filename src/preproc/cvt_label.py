# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com

import numpy as np
from typing import Dict, List


def shift_volume_label(label: Dict, shift: List[int]) -> Dict:
    """
    Shift a label corresponding to crop operation on a volume.
    :param label: (xmin, ymin, xmax, ymax, z) type
    :param shifts: (x, y, w, h)
    :return:
    """
    (sx, sy, sw, sh) = shift
    shifted_label = {neuron_id: [[x1 - sx, y1 - sy, x2 - sx, y2 - sy, z] for (z, x1, y1, x2, y2) in neuron] for neuron_id, neuron in label.items()}
    return shifted_label


def shift_stack_label(label: Dict[str, Dict], shifts: Dict):
    """Shift stack labels volume by volume."""
    shifted_stack = {volume_name: shift_volume_label(volume_label, shifts[volume_name]) for volume_name, volume_label in label.items()}
    return shifted_stack


def cvt_volume_label_to_lattice(volume, label):
    """

    :param label:
    :param volume_shape:
    :return: [F_RFP, probability, delta_center_x, delta_center_y, w/2, h/2]
    """
    F_rfp, prb, delta_x, delta_y, delta_w, delta_h = np.zeros()
    return
