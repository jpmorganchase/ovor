# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co

from collections import deque
import os
from pathlib import Path
import time

import torch
from torch.cuda import max_memory_allocated
import tqdm

class Metric:
    def __init__(self, fmt="{avg:.4f}", window=1000):
        self.fmt = fmt
        self.data = deque(maxlen=window)
        self.reset()

    def reset(self):
        self.data.clear()
        self.sum = 0.0
        self.count = 0

    def add(self, val, n=1):
        self.data.extend([val] * n)
        self.sum += val * n
        self.count += n
    
    @property
    def avg(self):
        return self.sum / self.count

    @property
    def max(self):
        return max(self.data)

    @property
    def last(self):
        return self.data[-1]

    def __str__(self):
        return self.fmt.format(avg=self.avg, max=self.max, last=self.last)

    def __repr__(self):
        return f"({self.sum}, {self.count})"

class DummyLogger:
    def __getattr__(self, attr):
        return self._dummy

    def _dummy(self, *args, **kwargs):
        pass

class Logger:
    def __init__(self, config, seed=1):
        self.log_dir = config.log_dir + f"/run-{seed}"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.logger = LclLogger(config.log_dir + '/output.log')
        self.metrics = self.setup_metrics(config.metrics)
        self.counters = config.counters
        for k in self.counters:
            setattr(self, k, 0)
        self.timer = Timer()

    def setup_metrics(self, config):
        d = dict()
        if config is None:
            return d
        for k, v in config.items():
            d[k] = Metric(v)
        return d

    def __getattr__(self, attr):
        if attr in self.metrics:
            return self.metrics[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def log(self, msg):
        self.logger.write(msg)

    def get_metrics(self, metrics):
        return {m: self.metrics[m].avg for m in metrics}

    def reset(self):
        for m in self.metrics.values():
            m.reset()
        for c in self.counters:
            setattr(self, c, 0)

    def __str__(self):
        return ', '.join(
            [f"{c}: {getattr(self, c)}" for c in self.counters] +
            [f"{name}: {str(metric)}" for name, metric in self.metrics.items() if metric.count > 0]
        )
    
    def get_memory_usage(self):
        return max_memory_allocated() / 1048576

    def log_model(self, model_state_dict):
        for k, v in model_state_dict.items():
            model_state_dict[k] = v.cpu()
        checkpoint_path = f"{self.log_dir}/models/task{self.task}.pth"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state_dict, checkpoint_path)

class LclLogger:
    def __init__(self, path):
        self.log = open(path, 'a')

    def write(self, message):
        tqdm.tqdm.write(message)
        self.log.write(message + '\n')  

    def flush(self):
        self.log.flush()

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time

    def tic(self):
        self.time = time.time()

    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval