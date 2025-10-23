# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


import time
import torch
import numpy as np


# Helper function to benchmark the model
def benchmark_torch_model_det(model, input_shape = (1, 1, 32, 32), dtype = 'fp32', nwarmup = 50, nruns = 1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.inference_mode():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.inference_mode():
        for i in range(1, nruns + 1):
            start_time = time.time()
            pred_loc, pred_label = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())
    print("Output location prediction size:", pred_loc.size())
    print("Output label prediction size:", pred_label.size())
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))


def benchmark_torch_model_cls(model, input_shape = (1024, 1, 32, 32), dtype = 'fp32', nwarmup = 50, nruns = 1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.inference_mode():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.inference_mode():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 100 == 0:
                print('Iteration %d/%d, ave batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())

