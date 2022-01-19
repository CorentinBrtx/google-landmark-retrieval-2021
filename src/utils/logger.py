import logging
from logging.handlers import WatchedFileHandler
import sys
from typing import Dict


def setup_logger():
    _logger = logging.getLogger("train")
    _logger.setLevel(logging.INFO)
    if len(_logger.handlers) == 0:
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        _logger.addHandler(stream_handler)
        file_handler = WatchedFileHandler("train.log")
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    return _logger


logger = setup_logger()


class SummaryWriter:
    def __init__(self, nb_epochs: int, nb_batchs: int) -> None:
        self.nb_epochs = nb_epochs
        self.nb_batchs = nb_batchs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, mode: str, i_batch: int, metrics: Dict[str, float]) -> None:
        summary = f"{mode.title()} Epoch {self.epoch}/{self.nb_epochs} | Batch {i_batch}/{self.nb_batchs} | "
        for metric_name, metric_value in metrics.items():
            summary += f"{metric_name.title()} {metric_value:.2f} | "
        logger.info(summary[:-2])
