from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(
    test_loader: DataLoader, backbone: nn.Module, device: str, logger=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from the given test set.

    Parameters
    ----------
    test_loader : DataLoader
        DataLoader containing the test set.
    backbone : nn.Module
        Backbone model to use for embedding.
    device : str
        Device to use for pytorch operations.

    Returns
    -------
    test_embbedings, test_ids: Tuple[np.ndarray, np.ndarray]
        Embeddings and corresponding image ids of the test set.
    """
    log_interval = len(test_loader) // 10
    test_embeddings = []
    test_ids = []

    backbone.to(device)

    backbone.eval()
    for i_batch, (x, y) in enumerate(test_loader):
        x = x.to(device)

        test_embeddings.append(backbone(x))
        test_ids += y

        if logger is not None and i_batch % log_interval == 0:
            logger.info(f"Extracting embeddings Batch {i_batch}/{len(test_loader)}")

    return torch.cat(test_embeddings).cpu().numpy(), np.array(test_ids)
