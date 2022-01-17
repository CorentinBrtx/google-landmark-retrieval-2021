import math
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
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
    checkpoint_interval: int,
    early_stop_after: int,
    device: str,
    data_dir: str,
    resume_training: bool = False,
) -> nn.Module:

    backbone = EfficientNetBackbone(feature_size, efficientNet)
    backbone.to(device)
    angular_margin.to(device)

    backbone_state_dict = deepcopy(backbone.state_dict())
    head_state_dict = deepcopy(angular_margin.state_dict())

    optimizer = optim.Adam(list(backbone.parameters()) + list(angular_margin.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    summary_writer = SummaryWriter(nb_epochs, len(train_loader))

    os.makedirs(os.path.join(data_dir, "models", "checkpoints"), exist_ok=True)

    train_losses = []
    val_losses = []

    min_loss = math.inf
    epoch = 0

    early_stop_counter = 0

    if resume_training:
        checkpoint = torch.load(
            os.path.join(data_dir, "models", "checkpoints", "latest_checkpoint.pth")
        )
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        angular_margin.load_state_dict(checkpoint["head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        backbone_state_dict = checkpoint["best_backbone_state_dict"]
        head_state_dict = checkpoint["best_head_state_dict"]
        epoch = checkpoint["epoch"]
        min_loss = checkpoint["min_loss"]
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        early_stop_counter = checkpoint["early_stop_counter"]

    while epoch < nb_epochs:
        epoch += 1

        summary_writer.set_epoch(epoch)

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
        train_losses.append(loss)

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
        val_losses.append(loss)

        if loss < min_loss:
            min_loss = loss
            early_stop_counter = 0
            backbone_state_dict = deepcopy(backbone.state_dict())
            head_state_dict = deepcopy(angular_margin.state_dict())
        else:
            early_stop_counter += 1

        if early_stop_counter == early_stop_after:
            break

        scheduler.step(loss)

        if epoch % checkpoint_interval == 0:
            os.remove(os.path.join(data_dir, "models", "checkpoints", "latest_checkpoint.pth"))
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state_dict": backbone.state_dict(),
                    "head_state_dict": angular_margin.state_dict(),
                    "best_backbone_state_dict": backbone_state_dict,
                    "best_head_state_dict": head_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "min_loss": min_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "early_stop_counter": early_stop_counter,
                },
                os.path.join(data_dir, "models", "checkpoints", "latest_checkpoint.pth"),
            )

        os.remove(os.path.join(data_dir, "models", "progress.png"))

        fig = plt.figure(figsize=(12, 7), dpi=200, facecolor="w")
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss evolution")
        fig.savefig(os.path.join(data_dir, "models", "progress.png"))
        plt.close(fig)

    backbone.load_state_dict(backbone_state_dict)
    angular_margin.load_state_dict(head_state_dict)

    os.remove(os.path.join(data_dir, "models", "checkpoints", "latest_checkpoint.pth"))
    torch.save(
        {
            "backbone_state_dict": backbone_state_dict,
            "head_state_dict": head_state_dict,
        },
        os.path.join(data_dir, "models", "final_model.pth"),
    )

    return backbone, angular_margin, min_loss
