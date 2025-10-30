import torch
import time
import numpy as np


def benchmark(model, volume, dtype = 'fp32', nwarmup = 50, nruns = 1000):
    """Benchmark the model's inference performance, measuring the average batch time (supports FP32/FP16 precision).
    
    Args:
        model (torch.nn.Module): The PyTorch model to be tested
        volume (torch.Tensor): The input tensor (must match the model's input dimensions)
        dtype (str, optional): The precision for computation, can be either 'fp32' or 'fp16', default is 'fp32'
        nwarmup (int, optional): The number of warmup iterations, default is 50
        nruns (int, optional): The number of official timing iterations, default is 1000

    Returns:
        None: The results are directly printed and not returned as values

    """
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
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))