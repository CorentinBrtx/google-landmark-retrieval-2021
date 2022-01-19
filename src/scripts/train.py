import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from argparse import ArgumentParser

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from src.data.dataloader import load_dataset
from src.models.angular_margin import ArcFace
from src.models.backbone import EfficientNetBackbone
from src.train.train import train
from src.utils.logger import logger

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a model on the training set available in the data folder."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        dest="data_dir",
        help="Directory containing the data.",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--feature_size",
        dest="feature_size",
        type=int,
        help="Size of the embeddings vector.",
        default=512,
    )
    parser.add_argument(
        "-n",
        "--nb_epochs",
        dest="epochs",
        type=int,
        help="Number of epochs to train.",
        default=10,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        type=int,
        help="Batch size for the data loaders.",
        default=64,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        dest="lr",
        type=float,
        help="Initial learning rate for the Adam optimizer.",
        default=1e-3,
    )
    parser.add_argument(
        "-m",
        dest="m",
        type=float,
        help="m parameter for the Angular Margin Loss.",
        default=0.5,
    )
    parser.add_argument(
        "-s",
        dest="s",
        type=int,
        help="s parameter for the Angular Margin Loss.",
        default=64,
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        help="alpha parameter for the Curricular Face Loss.",
        default=0.99,
    )
    parser.add_argument(
        "-t",
        dest="t",
        type=float,
        help="t parameter for the MV-Arc-Softmax Loss.",
        default=1.2,
    )
    parser.add_argument(
        "--image_size",
        dest="image_size",
        type=int,
        help="Size to resize the images to",
        default=224,
    )
    parser.add_argument(
        "--log_interval",
        dest="log_interval",
        type=int,
        help="Number of batches between each logging.",
        default=50,
    )
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        type=int,
        help="Number of epochs between each model save.",
        default=1,
    )
    parser.add_argument(
        "--early_stop",
        dest="early_stop_after",
        type=int,
        help="Number of epochs to wait before early stopping.",
        default=11,
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="num_workers argument for the data loaders.",
        default=0,
    )
    parser.add_argument(
        "--resume_training",
        dest="resume_training",
        type=bool,
        help="Whether to resume training from a checkpoint.",
        default=False,
    )
    parser.add_argument(
        "--load_all",
        dest="load_all",
        type=bool,
        help="Whether to load all the images in memory.",
        default=False,
    )

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {DEVICE}")

    train_loader, validation_loader, nb_classes = load_dataset(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, load_all=args.load_all, image_size=args.image_size
    )

    logger.info("Dataset loaded")

    efficient_net = EfficientNet.from_pretrained("efficientnet-b0", num_classes=args.feature_size)

    backbone, head, acc = train(
        train_loader,
        validation_loader,
        EfficientNetBackbone(
            args.feature_size,
            efficient_net,
        ),
        ArcFace(args.feature_size, nb_classes, args.s, args.m),
        nn.CrossEntropyLoss(),
        args.feature_size,
        args.lr,
        args.epochs,
        args.log_interval,
        args.checkpoint_interval,
        args.early_stop_after,
        DEVICE,
        args.data_dir,
        args.resume_training,
    )
