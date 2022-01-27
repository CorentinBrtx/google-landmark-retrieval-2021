from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.logger import SummaryWriter
from torch.utils.data import DataLoader

from .utils import calculate_accuracy


def pass_epoch(
    loader: DataLoader,
    backbone: nn.Module,
    angular_margin: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    summary_writer: SummaryWriter,
    log_interval: int,
    device: str,
) -> Tuple[float]:

    loss = 0
    acc = 0
    tot_iter = 0
    with torch.set_grad_enabled(backbone.training):
        for i_batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            embeddings = backbone(x)
            logits = angular_margin(embeddings, y)

            loss_batch = loss_fn(logits, y)
            acc_batch = calculate_accuracy(logits, y)

            if backbone.training:
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

            loss_batch = loss_batch.item()
            if i_batch % log_interval == 0:
                mode = "train" if backbone.training else "validation"
                summary_writer(mode, i_batch, {"loss": loss_batch, "acc": acc_batch})

            loss += loss_batch
            acc += acc_batch
            tot_iter += 1

    loss /= tot_iter
    acc /= tot_iter
    return loss, acc
