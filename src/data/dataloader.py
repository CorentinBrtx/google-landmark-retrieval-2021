import os
from typing import Tuple

import torch
from src.data.dataset import GoogleLandmarkDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def load_dataset(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    load_all: bool = False,
    image_size: int = 224,
    seed: int = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create the train and validation dataloaders from the training set.

    Parameters
    ----------
    data_dir : str
        directory containing the data
    batch_size : int
        batch_size to use
    num_workers : int, optional
        num_workers for the data loaders, by default 0
    load_all : bool, optional
        whether to load all the images in memory, by default False

    Returns
    -------
    train_loader, val_loader, nb_classes : Tuple[DataLoader, DataLoader, int]
        the two loaders, and the total number of classes
    """
    transformations = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            ),
        ]
    )

    dataset = GoogleLandmarkDataset(
        img_dir=os.path.join(data_dir, "train"),
        annotations_file=os.path.join(data_dir, "train.csv"),
        transform=transformations,
        load_all=load_all,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()

    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size], generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    nb_classes = dataset.num_classes

    return train_loader, validation_loader, nb_classes
