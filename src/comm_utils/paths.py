# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com
import os


def create_engine_path(root: str, name: str
                       ) -> str:
    """ create TensorRT serialized model path"""
    path = os.path.join(root, name + ".trt")
    return path


def create_tensorrt_ts_path(root: str, name: str
                            ) -> str:
    """ create TorchScript model path after being compiled by Torch-TensorRT JIT """
    path = os.path.join(root, name + ".ts")
    return path


def get_dataset_files_root(root, pile_name, mode):
    assert mode in ["label", "preprocessing_proofreading", "raw", "result"], "Wrong mode name!"
    return os.path.join(root, mode, pile_name)
