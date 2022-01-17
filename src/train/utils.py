import torch


@torch.no_grad()
def calculate_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    y_pred = torch.argmax(logits, dim=1)
    return torch.mean((y_pred == y).float()).item()
