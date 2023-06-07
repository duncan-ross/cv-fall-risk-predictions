import math
import logging

from tqdm import tqdm
from functools import partialmethod

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from typing import List

logger = logging.getLogger(__name__)


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 2e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.01  # Only applied to weight matrices
    lr_decay = False
    # Toying with warmup--didn't end up using LR decay
    warmup_tokens = 1e5 
    final_tokens = 1e10
    ckpt_path = None
    num_workers = 0  # for DataLoaderx
    writer = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        config,
        val_dataloader=None,
        test_dataloader=None,
        median_freq_weights: bool = False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.val_dataloader = val_dataloader
        self.losses = []
        self.optimizer = self.create_optimizer()

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if median_freq_weights:
            self.median_freq_weights = calculate_weights(
                self.train_dataloader, self.device
            )
        else:
            self.median_freq_weights = None

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def create_optimizer(self):
        model, config = self.model, self.config

        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups, lr=config.learning_rate, betas=config.betas
        )
        return optimizer

    def train(self, split, step):
        model, config = self.model, self.config
        is_train = split == "train"
        model.train(is_train)
        loader = self.train_dataloader if is_train else self.val_dataloader

        pbar = (
            tqdm(enumerate(loader), total=len(loader))
            if is_train
            else enumerate(loader)
        )
        losses = []
        for it, (id, x, y) in pbar:
            if type(x) == tuple: 
                x = tuple(xx.to(self.device) for xx in x)
            else:
                x = x.to(self.device)
            if type(y) == list or type(y) == tuple:
                y = [yy.to(self.device) for yy in y]
            else:
                y = y.to(torch.float32).to(self.device)

            if is_train:
                model.train()
            else:
                model.eval()
            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y, median_freq_weights=self.median_freq_weights)
                losses.append(loss.item())
            if is_train:
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_norm_clip
                )
                self.optimizer.step()

                lr = config.learning_rate
                if config.lr_decay:
                    if type(y) == list:
                        self.tokens += (y[0] >= 0).sum()
                    else:
                        self.tokens += (
                            y >= 0
                        ).sum()
                    # Still warmup for LR
                    if self.tokens < config.warmup_tokens:
                        lr_mult = float(self.tokens) / float(
                            max(1, config.warmup_tokens)
                        )
                    # Cosine LR decay
                    else:
                        progress = float(self.tokens - config.warmup_tokens) / float(
                            max(1, config.final_tokens - config.warmup_tokens)
                        )
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                # No LR Decay: Don't change LR
                else:
                    lr = config.learning_rate
                pbar.set_description(
                    f"epoch {step+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                )

                # Write results to file
                if config.writer is not None:
                    config.writer.add_scalar("train/loss", loss.item(), step)
                    config.writer.add_scalar("train/lr", lr, step)

        return np.mean(losses)


def calculate_weights(dataloader, device, cumulative: bool = False):
    """
    Calculate weights for each class based on median frequency balancing
    """
    unique, counts = np.unique(dataloader.dataset.labels[:, 0], return_counts=True)
    # cumulative sum of counts but in reverse order
    if cumulative:
        counts = np.flip(np.cumsum(np.flip(counts)))
    median = np.median(counts)
    class_weights = median / counts
    class_weights = torch.tensor(class_weights).to(device)
    return class_weights.float()
