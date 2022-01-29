import math
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.train.epoch import pass_epoch
from src.utils.file_manipulation import silentremove
from src.utils.logger import SummaryWriter
from torch.utils.data import DataLoader


def train(
    model_name: str,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    backbone: nn.Module,
    angular_margin: nn.Module,
    loss_fn: nn.Module,
    lr: float,
    nb_epochs: int,
    log_interval: int,
    checkpoint_interval: int,
    early_stop_after: int,
    device: str,
    data_dir: str,
    logger,
    resume_training: bool = False,
) -> nn.Module:

    backbone.to(device)
    angular_margin.to(device)

    backbone_state_dict = deepcopy(backbone.state_dict())
    head_state_dict = deepcopy(angular_margin.state_dict())

    optimizer = optim.Adam(list(backbone.parameters()) + list(angular_margin.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    summary_writer = SummaryWriter(nb_epochs, len(train_loader), len(validation_loader), logger)

    path_to_model = os.path.join(data_dir, "models", model_name)

    os.makedirs(path_to_model, exist_ok=True)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    min_loss = math.inf
    epoch = 0

    early_stop_counter = 0

    if resume_training:
        logger.info(f"Loading checkpoint for model {model_name}")
        checkpoint = torch.load(os.path.join(path_to_model, "latest_checkpoint.pth"))
        backbone.load_state_dict(checkpoint["backbone_state_dict"])
        angular_margin.load_state_dict(checkpoint["head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        backbone_state_dict = checkpoint["best_backbone_state_dict"]
        head_state_dict = checkpoint["best_head_state_dict"]
        epoch = checkpoint["epoch"]
        min_loss = checkpoint["min_loss"]
        train_losses = checkpoint["train_losses"]
        train_accuracies = checkpoint["train_accuracies"]
        val_losses = checkpoint["val_losses"]
        val_accuracies = checkpoint["val_accuracies"]
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
        logger.info(f"Train Epoch Loss: {loss:.4f} | Accuracy: {acc:.4f}\n")
        train_accuracies.append(acc)
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
        logger.info(f"Validation Epoch Loss: {loss:.4f} | Accuracy: {acc:.4f}\n")
        val_accuracies.append(acc)
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
            silentremove(os.path.join(path_to_model, "latest_checkpoint.pth"))
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
                    "train_accuracies": train_accuracies,
                    "val_losses": val_losses,
                    "val_accuracies": val_accuracies,
                    "early_stop_counter": early_stop_counter,
                },
                os.path.join(path_to_model, "latest_checkpoint.pth"),
            )

        silentremove(os.path.join(path_to_model, "progress.png"))

        fig, ax1 = plt.subplots(figsize=(12, 7), dpi=200, facecolor="w")
        ax2 = ax1.twinx()
        ax1.plot(train_losses, label="loss train", color="blue")
        ax1.plot(val_losses, label="loss validation", color="red")
        ax2.plot(train_accuracies, label="accuracy train", color="orange", linestyle="--")
        ax2.plot(val_accuracies, label="accuracy validation", color="green", linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax1.legend()
        ax2.legend(loc="upper center")
        fig.savefig(os.path.join(path_to_model, "progress.png"))
        plt.close(fig)

    backbone.load_state_dict(backbone_state_dict)
    angular_margin.load_state_dict(head_state_dict)

    silentremove(os.path.join(path_to_model, "latest_checkpoint.pth"))
    torch.save(
        {
            "backbone_state_dict": backbone_state_dict,
            "head_state_dict": head_state_dict,
        },
        os.path.join(path_to_model, "final_model.pth"),
    )

    return backbone, angular_margin, min_loss
