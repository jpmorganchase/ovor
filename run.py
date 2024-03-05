# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

import argparse
import numpy as np
import torch
import random
from utils.config import Config
from utils.logger import Logger
from trainer import Trainer

def get_args():
    cfg = Config(yaml_path='configs/common.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='yaml experiment config input')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat the experiment N times')
    parsed, unparsed = parser.parse_known_args()
    if parsed.config is not None:
        cfg.merge(parsed.config)
    cfg.update(gpus=parsed.gpus, repeat=parsed.repeat)

    ood = cfg.ood
    parser = argparse.ArgumentParser()
    parser.add_argument('--cov', type=float, default=ood.cov)
    parser.add_argument('--thres_id', type=float, default=ood.thres_id)
    parser.add_argument('--thres_ood', type=float, default=ood.thres_ood)
    parser.add_argument('--num_per_class', type=int, default=40)
    parser.add_argument('--sample_from', type=int, default=600)
    parser.add_argument('--select', type=int, default=50)
    parser.add_argument('--pick_nums', type=int, default=30)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--lmda', type=float, default=ood.lmda)
    parser.add_argument('--huber', action='store_false')
    parsed, unparsed = parser.parse_known_args(unparsed)
    ood.merge(vars(parsed))

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Dir to save experiment results, named with hps if not set')
    parser.add_argument('--learner_type', type=str, default='prompt', help='The type (filename) of learner')
    parser.add_argument('--learner_name', type=str, default='OnePrompt', help='The class name of learner')
    parser.add_argument('--prompt_param', nargs='+', type=float, default=[10, 40, 10],
                        help='e prompt pool size, e prompt length, g prompt length')
    parser.add_argument('--epochs', nargs='+', type=int, default=cfg.epochs,
                        help='warmup epochs, regular training epochs, ood epochs')
    args = parser.parse_args(unparsed)
    cfg.update(learner_type=args.learner_type, learner_name=args.learner_name,
               prompt_param=args.prompt_param, epochs=args.epochs)
    cfg.logging.log_dir = args.log_dir or generate_log_dir(cfg)
    for i in range(cfg.tasks):
        cfg.logging.metrics[f"eval_acc_cil_{i+1}"] = "{last:.4f}"
        cfg.logging.metrics[f"eval_acc_til_{i+1}"] = "{last:.4f}"
    return cfg

def generate_log_dir(cfg):
    learner_dict = {'CODAPrompt': 'coda', 'DualPrompt': 'dualprompt', 'L2P': 'l2p', 'OnePrompt': 'oneprompt'}
    prefix = '/'.join([
        'outputs', cfg.dataset, str(cfg.tasks),
        learner_dict.get(cfg.learner_name, 'default')
    ])
    hps = [prefix]
    epochs = cfg.epochs
    if epochs[0] > 0 or epochs[2] > 0:
        hps.append(f"e{epochs[0]}{epochs[1]}{epochs[2]}")
    if epochs[2] > 0:
        ood = cfg.ood
        hps.append(f"ood-{int(abs(ood.thres_id * 10))}-{int(abs(ood.thres_ood * 10))}-cov{int(10 * ood.cov):02d}-{ood.num_per_class}pc")
        hps.append(f"lmda{int(10 * ood.lmda):02d}")
    return '-'.join(hps)

if __name__ == '__main__':
    torch.backends.cudnn.deterministic=True

    args = get_args()

    results = {'Avg_acc': [], 'Forgetting': []}
    for r in range(args.repeat):
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        logger = Logger(args.logging, seed+1)
        args.dump(f"{logger.log_dir}/args.yaml")

        logger.log('************************************')
        logger.log(f"* TRIAL {r+1}")
        logger.log('************************************')
        trainer = Trainer(args, seed, logger)
        trainer.run()

        results['Avg_acc'].append(logger.avg_acc.last)
        results['Forgetting'].append(logger.forgetting.last)
    avg_acc = np.array(results['Avg_acc'])
    forget = np.array(results['Forgetting'])
    logger.log(f"===Summary of all {args.repeat} runs===")
    logger.log(f"Average accuracy: {avg_acc.mean():.2f}, std {avg_acc.std():.2f}")
    logger.log(f"Forgetting: {forget.mean():.2f}, std {forget.std():.2f}")