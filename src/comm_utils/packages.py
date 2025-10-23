# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


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
