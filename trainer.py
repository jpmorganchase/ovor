# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co

import datetime
import numpy as np
from tqdm import tqdm, trange

from dataloaders import get_dataloaders
import learners

class Trainer:
    def __init__(self, args, seed, logger):
        self.logger = logger
        self.data = get_dataloaders(args, seed)
        self.split_size = args.split_size
        self.num_task = args.tasks
        self.learner = learners.__dict__[args.learner_type].__dict__[args.learner_name](
            self.num_task * self.split_size, args, logger
        )

        self.logger.log(f"Number of parameters: {self.learner.count_params()}")
        self.logger.log(f"Number of learnable parameters: {self.learner.count_learnable_params()}")

    def run(self):
        for task in trange(self.num_task, desc='Task'):
            self.logger.task = task
            self.learner.subset_start = task * self.split_size
            self.learner.subset_end = (task + 1) * self.split_size
            if task > 0:
                self.learner.model.module.prompt.process_task_count()
            self.learner.train(dataloader=self.data['train'][task], task=task)
            self.learner.log_model()

            forgetting = []
            accs = []
            forgetting_til = []
            for task2 in range(task + 1):
                if task - task2 > 0:
                    max_acc = self.logger.metrics[f"eval_acc_cil_{task2+1}"].max
                    max_acc_til = self.logger.metrics[f"eval_acc_til_{task2+1}"].max
                cur_acc, preds, targets = self.learner.eval_cil(
                    dataloader=self.data['test'][task2], model_task=task, data_task=task2
                )
                stats = np.zeros(self.num_task)
                for p, t in zip(preds, targets):
                    stats[p // self.split_size] += 1
                self.logger.log(f"num of test samples: {stats.sum()}")
                cur_acc_til = self.learner.eval_til(
                    dataloader=self.data['test'][task2], model_task=task, data_task=task2,
                    start=task2*self.split_size, end=(task2+1)*self.split_size
                )
                accs.append(cur_acc)
                if task - task2 > 0:
                    forgetting.append(max_acc - cur_acc)
                    forgetting_til.append(max_acc_til - cur_acc_til)
            avg_acc = np.array(accs).mean()
            self.logger.avg_acc.add(avg_acc)
            if task > 0:
                avg_forget = np.array(forgetting).mean()
                avg_forget_til = np.array(forgetting_til).mean()
                self.logger.forgetting.add(avg_forget)
                self.logger.forgetting_til.add(avg_forget_til)
            self.logger.log(
                f"Task {task+1}: Average acc {avg_acc:.4f}" + (
                    f", Forgetting {avg_forget:.4f}, Forgetting-TIL {avg_forget_til:.4f}"
                    if task > 0 else ''
                )
            )

            train_time = self.logger.train_time.sum
            eval_time = self.logger.eval_time.sum
            self.logger.log(f"Train time till now: {datetime.timedelta(seconds=train_time)}")
            self.logger.log(f"Eval time till now: {datetime.timedelta(seconds=eval_time)}")