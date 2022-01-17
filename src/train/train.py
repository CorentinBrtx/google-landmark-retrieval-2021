import math

import torch.nn as nn
import torch.optim as optim
from src.models.backbone import EfficientNetBackbone
from src.utils.logger import SummaryWriter, logger
from torch.utils.data import DataLoader

from .epoch import pass_epoch


def train(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    efficientNet: nn.Module,
    angular_margin: nn.Module,
    loss_fn: nn.Module,
    feature_size: int,
    lr: float,
    nb_epochs: int,
    log_interval: int,
    early_stop_after: int,
    device: str,
) -> nn.Module:

    backbone = EfficientNetBackbone(feature_size, efficientNet)
    backbone.to(device)
    angular_margin.to(device)

    backbone_state_dict = backbone.state_dict()
    head_state_dict = angular_margin.state_dict()

    optimizer = optim.Adam(list(backbone.parameters()) + list(angular_margin.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    summary_writer = SummaryWriter(nb_epochs, len(train_loader))

    min_loss = math.inf
    for epoch in range(nb_epochs):
        summary_writer.set_epoch(epoch + 1)

        backbone.train()
        angular_margin.train()

        loss, acc = pass_epoch(
            train_loader,
            backbone,
            angular_margin,
            optimizer,
            loss_fn,
            summary_writer,
            log_interval,
            device,
        )
        logger.info(f"Train Epoch Loss: {loss:.2f} | Accuracy: {acc:.2f}\n")

        backbone.eval()
        angular_margin.eval()
        loss, acc = pass_epoch(
            validation_loader,
            backbone,
            angular_margin,
            optimizer,
            loss_fn,
            summary_writer,
            log_interval,
            device,
        )
        acc = round(acc, 2)
        logger.info(f"Validation Epoch Loss: {loss:.2f} | Accuracy: {acc}\n")

        if loss > min_loss:
            min_loss = loss
            early_stop_counter = 0
            backbone_state_dict = backbone.state_dict()
            head_state_dict = angular_margin.state_dict()
        else:
            early_stop_counter += 1

        if early_stop_counter == early_stop_after:
            break

        scheduler.step(loss)

    backbone.load_state_dict(backbone_state_dict)
    angular_margin.load_state_dict(head_state_dict)
    return backbone, angular_margin, min_loss
