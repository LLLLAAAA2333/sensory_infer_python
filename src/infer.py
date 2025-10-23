import os
import re
import sys
import cv2
import onnx
import thop
import time
import math
import yaml
import json
import h5py
import mat73
import torch
import random
import onnxsim
import warnings
import argparse
import numpy as np
import torchvision
import pandas as pd
from torch import nn
import seaborn as sns
from glob import glob
from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from copy import deepcopy
from sklearn import metrics
import albumentations as DA  # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
from scipy.io import loadmat
from threading import Thread  # https://docs.python.org/3/library/threading.html#threading.Thread
import bbox_visualizer as bbv  # https://github.com/shoumikchow/bbox-visualizer based on cv2
from lion_pytorch import Lion
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.manifold import TSNE
from dataclasses import dataclass
from torch.nn import functional as F
from scipy.spatial.distance import pdist
import torchvision.transforms as transforms
from collections import Counter, OrderedDict
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms.functional as TF
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from finetuning_scheduler import FinetuningScheduler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Union, List, Optional
from pytorch_lightning.loggers import TensorBoardLogger
from itertools import product, permutations, combinations
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from torchmetrics.detection.mean_ap import Metric, MeanAveragePrecision


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from src.merge_resize_inference import *
from src.comm_utils.packages import *
from src.comm_utils.prints import pad_num


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


if __name__ == '__main__':
    config = "/home/cbmi/CBMI_python/src/configs/inference/vlad.json"
    with open(config, "r") as f:
        config = json.load(f)

    treeformer = Treeformer_End2End(

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

    ).eval().cuda().half()

    # benchmark(treeformer, volume, 'fp16')
    treeformer_he = VolumeMemoryBuffer(
        save_path = "/home/cbmi/CBMI_python/data/zone/cache2",
        deg_t = 100,
    )

    path = "/home/cbmi/CBMI_python/data/fresh_raw/others2evaluate/CorePark/data/184.h5"
    save_fig_root = "/home/cbmi/CBMI_python/data/zone/aravi/vlad/184_0/figs"
    os.makedirs(save_fig_root, exist_ok = True)

    save_mip_root = "/home/cbmi/CBMI_python/data/zone/aravi/vlad/184_0/mip"
    os.makedirs(save_mip_root, exist_ok = True)


    vols = list()
    with h5py.File(path, 'r') as f:
        keys = sorted([int(k) for k in f.keys() if k.isdigit()])
        for i in tqdm(keys):
            data = np.array(f[f'{i}/frame'])[0]
            vol = torch.FloatTensor(data[None]).cuda().half().permute(3, 0, 1, 2)

            mip = torch.max(vol, dim = 0, keepdim = True)[0].squeeze()
            plt.imsave(os.path.join(save_mip_root, f"Volume_{pad_num(i, 5)}.png"), mip.detach().cpu().numpy(), cmap = 'afmhot', vmin = 0, vmax = 255)
            plt.close()

            with torch.inference_mode():
                if vol.shape[2] < config["det_input_dim"][-2] or vol.shape[3] < config["det_input_dim"][-1]:
                    print(f"volume shape {vol.shape} is smaller than ext_input_dim {config['det_input_dim']}")
                    vv = torch.zeros([vol.shape[0]] + config["det_input_dim"], device = vol.device, dtype = vol.dtype)
                    vv[:, :, :vol.shape[2], :vol.shape[3]] = vol * 4
                else:
                    vv = vol

                head_bboxes, regions, pts_emb, pt_tuple, region_ptrs, neuron_emb, neuron_info, num_region, is_warnings = treeformer(vv)
                neuron_pred_ids, region_pred_ids, num_neuron_idv = treeformer_he(neuron_emb, neuron_info[0], pts_emb.shape[0], any(is_warnings))
                bb = torch.concat([neuron_info[2].cpu(), torch.FloatTensor(neuron_pred_ids).unsqueeze(1)], dim = 1).cpu().numpy()
                print(f" the number of regions: {num_region[0]}, \t the number of neurons: {num_region[1]}, \t  the number of neurons: {num_neuron_idv}")

                draw_volume_result(vol, head_bboxes, regions, region_pred_ids, region_ptrs, save_fig_root = save_fig_root, name = f"Volume_{pad_num(i, 5)}", verbose = False)




