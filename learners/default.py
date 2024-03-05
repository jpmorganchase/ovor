# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co

import torch
from torch import nn
from tqdm import tqdm

import models
from .ood import NPOS
from utils.schedulers import CosineSchedule

class NormalNN(nn.Module):
    def __init__(self, out_dim, args, logger):
        super().__init__()
        self.logger = logger
        self.out_dim = out_dim
        self.bsz = args.batch_size
        self.opt_cfg = {k: args[k] for k in ('optimizer', 'lr', 'momentum', 'weight_decay', 'scheduler')}
        self.subset_start = 0
        self.subset_end = out_dim
        self.epochs = args.epochs
        self.model_type = args.model_type
        self.model_name = args.model_name

        torch.cuda.set_device(args.gpus[0])
        self.model = self._create_model().cuda()
        self.ood = NPOS(args.ood)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.model = nn.DataParallel(self.model, device_ids=args.gpus, output_device=args.gpus[0])

    def _create_model(self):
        return models.__dict__[self.model_type].__dict__[self.model_name](out_dim=self.out_dim)

    def init_optimizer(self):
        optim_args = {
            'params': self._get_learnable_params(),
            'lr': self.opt_cfg['lr'],
            'weight_decay': self.opt_cfg['weight_decay']
        }
        optim = self.opt_cfg['optimizer']
        if optim in ['SGD', 'RMSprop']:
            optim_args['momentum'] = self.opt_cfg['momentum']
        elif optim == 'Rprop':
            optim_args.pop('weight_decay')
        elif optim == 'amsgrad':
            optim_args['amsgrad'] = True
            optim = 'Adam'
        elif optim == 'Adam':
            optim_args['betas'] = (self.opt_cfg['momentum'], 0.999)
        self.optimizer = torch.optim.__dict__[optim](**optim_args)

        if self.opt_cfg['scheduler'] == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=sum(self.epochs))
        else: # decay
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=sum(self.epochs), gamma=0.1
            )

    def _get_learnable_params(self):
        return self.model.parameters()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def count_learnable_params(self):
        return sum(p.numel() for p in self._get_learnable_params())

    def train(self, dataloader, task):
        self.init_optimizer()
        self.model.train()
        pbar = tqdm(desc='Epoch', total=sum(self.epochs), leave=False)
        self.logger.timer.tic()

        start_epoch = 0
        if task > 0:
            for e in range(self.epochs[0]):
                stats_str = self._train_one_epoch(dataloader, e, warmup=True)
                pbar.postfix = stats_str
                pbar.update()
            start_epoch = self.epochs[0]
        
        for e in range(start_epoch, start_epoch + self.epochs[1]):
            stats_str = self._train_one_epoch(dataloader, e)
            pbar.postfix = stats_str
            pbar.update()

        start_epoch += self.epochs[1]
        for e in range(start_epoch, start_epoch + self.epochs[2]):
            stats_str = self._train_one_epoch_ood(dataloader, e)
            pbar.postfix = stats_str
            pbar.update()
        
        train_time = self.logger.timer.toc()
        self.logger.train_time.add(train_time)
        pbar.close()

    def _get_ood_samples(self, dataloader):
        subset_size = self.subset_end - self.subset_start
        id_feats = [torch.empty(0, 768) for _ in range(subset_size)]
        with torch.no_grad():
            for imgs, targets, _ in dataloader:
                imgs = imgs.cuda()
                targets = targets.cuda()
                feats = self.model(x=imgs, feat=True).detach().cpu()
                for i, idx in enumerate(targets):
                    key = (idx % subset_size).item()
                    id_feats[key] = torch.cat((id_feats[key], feats[i].view(1, -1)), 0)
        return self.ood.generate(id_feats, self.subset_start)

    def _train_one_epoch(self, dataloader, epoch, warmup=False):
        if epoch > 0:
            self.scheduler.step()
        self.logger.train_loss.reset()
        self.logger.train_acc.reset()
        self.logger.memory.reset()

        for imgs, targets, _ in dataloader:
            imgs = imgs.cuda()
            targets = targets.cuda()
            logits, prompt_loss = self.model(x=imgs, train=True, warmup=warmup)
            logits = logits[:, :self.subset_end]
            logits[:, :self.subset_start] = -float('inf')
            loss = self.criterion(logits, targets)
            loss += prompt_loss.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.train_loss.add(loss.item(), n=targets.size(0))
            self.logger.train_acc.add(accuracy(logits, targets), n=targets.size(0))
            self.logger.memory.add(self.logger.get_memory_usage())
            self.logger.step += 1

        return f"Loss: {str(self.logger.train_loss)}, Acc: {str(self.logger.train_acc)}"

    def _train_one_epoch_ood(self, dataloader, epoch):
        self.scheduler.step()
        self.logger.train_loss.reset()
        self.logger.train_acc.reset()
        self.logger.memory.reset()
        self.logger.id_score.reset()
        self.logger.ood_score.reset()
        id_loader, ood_loader = self._get_ood_samples(dataloader)
        for ((ids, targets), oods) in zip(id_loader, ood_loader):
            ids = ids.cuda()
            targets = targets.cuda()
            oods = oods[0].cuda()
            id_logits = self.model(x=ids, last=True)[:, :self.subset_end]
            ood_logits = self.model(x=oods, last=True)[:, self.subset_start:self.subset_end]
            id_logits[:, :self.subset_start] = -float('inf')
            loss = self.criterion(id_logits, targets)
            ood_loss, id_score, ood_score = self.ood.loss(id_logits[:, self.subset_start:], ood_logits)
            loss += ood_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.train_loss.add(loss.item(), n=targets.size(0))
            self.logger.train_acc.add(accuracy(id_logits, targets), n=targets.size(0))
            self.logger.memory.add(self.logger.get_memory_usage())
            self.logger.id_score.add(id_score.item(), n=targets.size(0))
            self.logger.ood_score.add(ood_score.item(), n=targets.size(0))
            self.logger.step += 1

        return f"Loss: {str(self.logger.train_loss)}, Acc: {str(self.logger.train_acc)}"

    def log_model(self):
        self.logger.log_model(self.model.module.state_dict())

    def eval_cil(self, dataloader, model_task, data_task):
        acc, time, preds, targets = self._eval(dataloader=dataloader, start=0, end=self.subset_end, return_predictions=True)
        metric = f"eval_acc_cil_{data_task+1}"
        self.logger.metrics[metric].add(acc)
        self.logger.eval_time.add(time)
        self.logger.log(f"Eval-CIL M{model_task+1}D{data_task+1}: Acc {acc:.4f}, Time {time:.2f}")
        return acc, preds, targets

    def eval_til(self, dataloader, model_task, data_task, start, end):
        acc, time = self._eval(dataloader=dataloader, start=start, end=end)
        metric = f"eval_acc_til_{data_task+1}"
        self.logger.metrics[metric].add(acc)
        self.logger.eval_time.add(time)
        self.logger.log(f"Eval-TIL M{model_task+1}D{data_task+1}: Acc {acc:.4f}, Time {time:.2f}")
        return acc

    def _eval(self, dataloader, start, end, return_predictions=False):
        self.model.eval()
        self.logger.timer.tic()
        correct, total = 0., 0.
        preds = []
        gts = []
        with torch.no_grad():
            for imgs, targets, _ in dataloader:
                imgs = imgs.cuda()
                targets = targets.cuda()
                logits = self.model(x=imgs)[:, :end]
                logits[:, :start] = -float('inf')
                correct += accuracy(logits, targets) * targets.size(0)
                total += targets.size(0)
                preds.extend(logits.argmax(dim=1).tolist())
                gts.extend(targets.tolist())
        if return_predictions:
            return correct / total, self.logger.timer.toc(), preds, gts
        return correct / total, self.logger.timer.toc()

class SL(NormalNN):
    def init_optimizer(self):
        self.optim_slow = self._get_optim(self.model.module.feat.parameters(), 0.0001)
        self.optim_fast = self._get_optim(self.model.module.last.parameters(), 0.01)
        if self.opt_cfg['scheduler'] == 'cosine':
            self.sched_slow = CosineSchedule(self.optim_slow, K=self.epochs[1])
            self.sched_fast = CosineSchedule(self.optim_fast, K=sum(self.epochs))
        else: # decay
            self.sched_slow = torch.optim.lr_scheduler.MultiStepLR(
                self.optim_slow, milestones=sum(self.epochs), gamma=0.1
            )
            self.sched_fast = torch.optim.lr_scheduler.MultiStepLR(
                self.optim_fast, milestones=sum(self.epochs), gamma=0.1
            )

    def _get_optim(self, params, lr):
        optim_args = {
            'params': params,
            'lr': lr,
            'weight_decay': self.opt_cfg['weight_decay']
        }
        optim = self.opt_cfg['optimizer']
        if optim in ['SGD', 'RMSprop']:
            optim_args['momentum'] = self.opt_cfg['momentum']
        elif optim == 'Rprop':
            optim_args.pop('weight_decay')
        elif optim == 'amsgrad':
            optim_args['amsgrad'] = True
            optim = 'Adam'
        elif optim == 'Adam':
            optim_args['betas'] = (self.opt_cfg['momentum'], 0.999)
        return torch.optim.__dict__[optim](**optim_args)

    def _train_one_epoch(self, dataloader, epoch, warmup=False):
        if epoch > 0:
            self.sched_slow.step()
            self.sched_fast.step()
        self.logger.train_loss.reset()
        self.logger.train_acc.reset()
        self.logger.memory.reset()

        for imgs, targets, _ in dataloader:
            imgs = imgs.cuda()
            targets = targets.cuda()
            logits = self.model(x=imgs, train=True, warmup=warmup)
            logits = logits[:, :self.subset_end]
            logits[:, :self.subset_start] = -float('inf')
            loss = self.criterion(logits, targets)

            self.optim_slow.zero_grad()
            self.optim_fast.zero_grad()
            loss.backward()
            self.optim_slow.step()
            self.optim_fast.step()

            self.logger.train_loss.add(loss.item(), n=targets.size(0))
            self.logger.train_acc.add(accuracy(logits, targets), n=targets.size(0))
            self.logger.memory.add(self.logger.get_memory_usage())
            self.logger.step += 1

        return f"Loss: {str(self.logger.train_loss)}, Acc: {str(self.logger.train_acc)}"

    def _train_one_epoch_ood(self, dataloader, epoch):
        self.sched_fast.step()
        self.logger.train_loss.reset()
        self.logger.train_acc.reset()
        self.logger.memory.reset()
        self.logger.id_score.reset()
        self.logger.ood_score.reset()
        id_loader, ood_loader = self._get_ood_samples(dataloader)
        for ((ids, targets), oods) in zip(id_loader, ood_loader):
            ids = ids.cuda()
            targets = targets.cuda()
            oods = oods[0].cuda()
            id_logits = self.model(x=ids, last=True)[:, :self.subset_end]
            ood_logits = self.model(x=oods, last=True)[:, self.subset_start:self.subset_end]
            id_logits[:, :self.subset_start] = -float('inf')
            loss = self.criterion(id_logits, targets)
            ood_loss, id_score, ood_score = self.ood.loss(id_logits[:, self.subset_start:], ood_logits)
            loss += ood_loss

            self.optim_fast.zero_grad()
            loss.backward()
            self.optim_fast.step()

            self.logger.train_loss.add(loss.item(), n=targets.size(0))
            self.logger.train_acc.add(accuracy(id_logits, targets), n=targets.size(0))
            self.logger.memory.add(self.logger.get_memory_usage())
            self.logger.id_score.add(id_score.item(), n=targets.size(0))
            self.logger.ood_score.add(ood_score.item(), n=targets.size(0))
            self.logger.step += 1

        return f"Loss: {str(self.logger.train_loss)}, Acc: {str(self.logger.train_acc)}"

def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[0].view(-1).float().sum().item() * 100.0 / batch_size