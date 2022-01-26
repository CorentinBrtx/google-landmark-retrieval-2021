from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(
    test_loader: DataLoader, backbone: nn.Module, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    log_interval = len(test_loader) // 10
    test_embeddings = []
    test_ids = []

    backbone.eval()
    for i_batch, (x, y) in enumerate(test_loader):
        x = x.to(device)

        test_embeddings.append(backbone(x))
        test_ids += y

        if i_batch % log_interval == 0:
            print(f"Extracting embedings Batch {i_batch}/{len(test_loader)}")

    return torch.cat(test_embeddings).cpu().numpy(), np.array(test_ids)
