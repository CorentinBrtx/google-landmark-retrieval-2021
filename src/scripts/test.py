import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from argparse import ArgumentParser

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from src.data.dataloader import load_test_dataset
from src.inference.extract_embeddings import extract_embeddings
from src.models.backbone import EfficientNetBackbone
from src.utils.logger import setup_logger

if __name__ == "__main__":
    parser = ArgumentParser(description="Use a trained model to compute embeddings on a test set.")
    parser.add_argument(
        dest="model_path",
        help="Path to the saved model.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir",
        help="Directory containing images to embed (respecting the google landmarks dataset structure).",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to save embeddings to.",
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
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Batch size for the data loaders.",
        default=64,
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
        "--num-workers",
        dest="num_workers",
        type=int,
        help="num_workers argument for the data loaders.",
        default=0,
    )
    parser.add_argument(
        "--incomplete-model",
        dest="incomplete_model",
        action="store_true",
        help="Use an incomplete model (i.e. whose training was not complete).",
    )

    args = parser.parse_args()

    os.environ["LANDMARK_RETRIEVAL_DATA_DIR"] = args.data_dir

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = os.path.join(args.data_dir, "models", args.model_name)

    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(os.path.join(model_dir, "test.log"))

    logger.info(f"Using device: {DEVICE}")

    test_loader = load_test_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    logger.info("Dataset loaded")

    efficientnet = EfficientNet.from_name(args.efficient_net, num_classes = args.feature_size)

    backbone = EfficientNetBackbone(feature_size=args.feature_size, efficientNet=efficientnet)

    loaded_model = torch.load(args.model_path)

    if args.incomplete_model:
        backbone.load_state_dict(loaded_model["best_backbone_state_dict"])
    else:
        backbone.load_state_dict(loaded_model["backbone_state_dict"])

    test_embeddings, test_ids = extract_embeddings(test_loader, backbone, DEVICE, logger)
    np.save(os.path.join(args.output_dir, "test_embeddings.npy"), test_embeddings)
    np.save(os.path.join(args.output_dir, "test_ids.npy"), test_ids)
