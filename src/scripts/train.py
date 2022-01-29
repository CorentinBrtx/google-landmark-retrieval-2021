import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from numpy import random
from src.data.dataloader import load_dataset
from src.models.angular_margin import ArcFace, CurricularFace, MVArcSoftmax
from src.models.backbone import EfficientNetBackbone
from src.models.saving import load_model_config, save_model_config
from src.train.train import train
from src.utils.logger import setup_logger

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a model on the training set available in the data folder."
    )
    parser.add_argument(
        dest="model_name",
        help="Name of the model.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir",
        help="Directory containing the data.",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--feature-size",
        dest="feature_size",
        type=int,
        help="Size of the embeddings vector.",
        default=512,
    )
    parser.add_argument(
        "-n",
        "--nb-epochs",
        dest="epochs",
        type=int,
        help="Number of epochs to train.",
        default=10,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Batch size for the data loaders.",
        default=64,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        dest="lr",
        type=float,
        help="Initial learning rate for the Adam optimizer.",
        default=1e-3,
    )
    parser.add_argument(
        "--angular-margin",
        dest="angular_margin",
        type=str,
        help="Angular margin to use. Can be one of 'arcface', 'mvarcsoftmax' or 'curricularface'.",
        default="arcface",
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
        "--image-size",
        dest="image_size",
        type=int,
        help="Size to resize the images to",
        default=224,
    )
    parser.add_argument(
        "--efficientnet",
        dest="efficient_net",
        type=str,
        help="EfficientNet model to use",
        default="efficientnet-b0",
    )
    parser.add_argument(
        "--log-interval",
        dest="log_interval",
        type=int,
        help="Number of batches between each logging.",
        default=50,
    )
    parser.add_argument(
        "--checkpoint-interval",
        dest="checkpoint_interval",
        type=int,
        help="Number of epochs between each model save.",
        default=1,
    )
    parser.add_argument(
        "--early-stop",
        dest="early_stop_after",
        type=int,
        help="Number of epochs to wait before early stopping.",
        default=11,
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="num_workers argument for the data loaders.",
        default=0,
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        help="Seed for the train-val split.",
    )
    parser.add_argument(
        "--resume-training",
        dest="resume_training",
        action="store_true",
        help="Resume training from a checkpoint. Note that in that case, all other parameters are ignored.",
    )
    parser.add_argument(
        "--load-all",
        dest="load_all",
        action="store_true",
        help="Load all the images in memory.",
    )

    args = parser.parse_args()

    os.environ["LANDMARK_RETRIEVAL_DATA_DIR"] = args.data_dir

    model_dir = os.path.join(args.data_dir, "models", args.model_name)

    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(os.path.join(model_dir, "train.log"))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {DEVICE}")

    if args.resume_training:
        args = Namespace(**load_model_config(args.model_name))
        args.resume_training = True

    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 2 ** 32)

    train_loader, validation_loader, nb_classes = load_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        load_all=args.load_all,
        image_size=args.image_size,
        seed=seed,
    )

    logger.info("Dataset loaded")

    efficient_net = EfficientNet.from_pretrained(args.efficient_net, num_classes=args.feature_size)

    if args.angular_margin == "arcface":
        angular_margin = ArcFace(
            feature_size=args.feature_size, nb_classes=nb_classes, m=args.m, s=args.s
        )
    elif args.angular_margin == "mvarcsoftmax":
        angular_margin = MVArcSoftmax(
            feature_size=args.feature_size, nb_classes=nb_classes, m=args.m, s=args.s, t=args.t
        )
    elif args.angular_margin == "curricularface":
        angular_margin = CurricularFace(
            feature_size=args.feature_size,
            nb_classes=nb_classes,
            m=args.m,
            s=args.s,
            alpha=args.alpha,
        )
    else:
        raise ValueError("Unknown angular margin")

    save_model_config(
        {
            **vars(args),
            "train_batches": len(train_loader),
            "val_batches": len(validation_loader),
            "nb_classes": nb_classes,
            "seed": seed,
        },
        args.model_name,
    )

    backbone, head, acc = train(
        args.model_name,
        train_loader,
        validation_loader,
        EfficientNetBackbone(
            args.feature_size,
            efficient_net,
        ),
        angular_margin,
        nn.CrossEntropyLoss(),
        args.lr,
        args.epochs,
        args.log_interval,
        args.checkpoint_interval,
        args.early_stop_after,
        DEVICE,
        args.data_dir,
        logger,
        args.resume_training,
    )
