# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
from src.comm_utils.packages import *
from src.utils import cxcywh2xyxy


def plot_volume():
    return


def cvt_tensorboard_image_torch(image, y_score, y_reg, y_score_pred, y_reg_pred):
    """
        bbox type [x, y, w, h].
        All are tensor type.
    :param image: shape [1, height, width], defalut 1024 x 1024
    :param y_score: [x, y, w, h], tensor value
    :param y_reg: bbox, tensor type
    :param y_score_pred: [x, y, w, h], tensor value
    :param y_reg_pred: bbox, tensor type
    :return: tensor type
    """

    # inplace operation !!!
    y_reg, y_reg_pred = y_reg.clone(), y_reg_pred.clone()
    vis_img = image.to(torch.uint8)
    # label
    y_reg[2] += y_reg[0]
    y_reg[3] += y_reg[1]
    vis_img = torchvision.utils.draw_bounding_boxes(vis_img, boxes = y_reg.unsqueeze(0), labels = [f"label {float(y_score):.2f}"])

    y_reg_pred[2] += y_reg_pred[0]
    y_reg_pred[3] += y_reg_pred[1]
    vis_img = torchvision.utils.draw_bounding_boxes(vis_img, boxes = y_reg_pred.unsqueeze(0), labels = [f"pred {float(y_score_pred):.2f}"])

    return vis_img


def draw_neuronal_regions_in_tensorboard_image_torch(image, preds, targets, vis_score = False):
    # inplace operation !!!
    vis_img = image.to(torch.uint8)

    if len(preds):
        bboxes, scores = cxcywh2xyxy(preds[:, :4]), preds[:, 4]
        if vis_score:
            vis_img = torchvision.utils.draw_bounding_boxes(vis_img, boxes = bboxes, colors = "red", labels = [f"{int(s * 100)}" for s in scores])
        else:
            vis_img = torchvision.utils.draw_bounding_boxes(vis_img, boxes = bboxes, colors = "red")

    if len(targets):
        tgt_bboxes = cxcywh2xyxy(targets[:, :4])
        vis_img = torchvision.utils.draw_bounding_boxes(vis_img, boxes = tgt_bboxes, colors = "blue")

    if vis_img.shape[0] == 1:
        vis_img = vis_img.repeat(3, 1, 1)

    return vis_img


def draw_batch_image_with_bbox(preds, tgts, imgs, vis_score = False):
    images = [draw_neuronal_regions_in_tensorboard_image_torch(img, preds[i] if len(preds) else [], tgts[i] if len(tgts) else [], vis_score) for i, img in enumerate(imgs)]
    images = torchvision.utils.make_grid(images, nrow = int(math.sqrt(len(images))))
    return images

# def draw_vector_field(delta_x, delta_y, score):
#     X, Y = torch.meshgrid(torch.arange(delta_x.shape[0]), torch.arange(delta_x.shape[1])
# def draw_batch_image_with_vector_field(preds, tgts, imgs, vis_score = False):
#     images = [draw_neuronal_regions_in_tensorboard_image_torch(img, preds[i] if len(preds) else [], tgts[i] if len(tgts) else [], vis_score) for i, img in enumerate(imgs)]
#     images = torchvision.utils.make_grid(images, nrow = int(math.sqrt(len(images))))
#     return images