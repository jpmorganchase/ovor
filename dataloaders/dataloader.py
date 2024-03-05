# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

import numpy as np
import random
import torch
from torch.utils.data import Subset, DataLoader

from .dataset import *

def get_dataloaders(args, seed):
    if args.dataset == 'ImageNet_R':
        loader = ImagenetRLoader(args.dataroot)
    elif args.dataset.startswith('CIFAR100'):
        loader = CIFAR100Loader(args.dataroot)
    elif args.dataset.startswith('CUB'):
        loader = CUBLoader(args.dataroot)
    elif args.dataset == 'ImageNet_A':
        loader = ImagenetALoader(args.dataroot)
    print(f"Train set: {len(loader.data['train'][0])}")
    print(f"Test set: {len(loader.data['test'][0])}")

    classes = np.arange(loader.num_cls).tolist()
    if seed > 0:
        random.seed(seed)
        random.shuffle(classes)
    mapping = {c: i for i, c in enumerate(classes)}
    tasks = [classes[i*args.split_size:(i+1)*args.split_size] for i in range(args.tasks)]

    train_ds = get_dataset(args.dataset, loader, mapping, 'train')
    train_dls = split_dataset(train_ds, tasks, 'train', args.batch_size, args.workers, seed)
    test_ds = get_dataset(args.dataset, loader, mapping, 'test')
    test_dls = split_dataset(test_ds, tasks, 'test', args.batch_size, args.workers, seed)
    return {'train': train_dls, 'test': test_dls}

def get_dataset(dataset, loader, mapping, mode):
    data, targets = loader.data[mode]
    params = {'data': data, 'targets': targets, 'cls_mapping': mapping, 'transform': loader.get_transform(mode)}
    if dataset == 'ImageNet_R':
        return ImagenetR(**params)
    elif dataset.startswith('CIFAR100'):
        return CIFAR100(**params)
    elif dataset.startswith('CUB'):
        return CUB200(**params)
    elif dataset == 'ImageNet_A':
        return ImagenetA(**params)

def split_dataset(dataset, tasks, mode, bsz, workers, seed):
    dataloaders = []
    for t in tasks:
        indices = np.isin(dataset.targets, t).nonzero()[0]
        g = torch.Generator()
        g.manual_seed(seed)
        dataloaders.append(DataLoader(
            dataset=Subset(dataset, indices), batch_size=bsz,
            shuffle=(mode=='train'), num_workers=workers,
            worker_init_fn=seed_worker, generator=g
        ))
    return dataloaders

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)