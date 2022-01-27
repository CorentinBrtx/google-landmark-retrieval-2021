import math
from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class AngularMarginHead(nn.Module):
    def __init__(
        self, feature_size: int, nb_classes: int, s: int, m: float, clip: Optional[bool] = True
    ) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.min_allowed = math.cos(math.pi - m)
        self.clip = clip

        self._cosine = None
        self.sine = None

        self.weight = nn.parameter.Parameter(torch.Tensor(nb_classes, feature_size))
        nn.init.xavier_uniform_(self.weight)

    @property
    def cosine(self):
        return self._cosine

    @cosine.setter
    def cosine(self, value):
        self._cosine = value
        self.sine = torch.sqrt(1 - self.cosine ** 2)

    @abstractmethod
    def positive_cosine_similarity_modulator(self) -> torch.Tensor:
        """
        Positive cosine similarity modulator
        """

    @abstractmethod
    def negative_cosine_similarity_modulator(
        self, cosine_after_positive_modulator: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative cosine similarity modulator
        """

    def forward(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if self.clip:
            self.cosine = torch.clip(self.cosine, self.min_allowed, 0.99)

        one_hot = torch.zeros_like(self.cosine).to(y.device)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        cosine_after_positive_modulator = self.positive_cosine_similarity_modulator()
        cosine_after_negative_modulator = self.negative_cosine_similarity_modulator(
            cosine_after_positive_modulator
        )

        output = torch.where(
            one_hot == 1, cosine_after_positive_modulator, cosine_after_negative_modulator
        )
        return self.s * output


class ArcFace(AngularMarginHead):
    def __init__(
        self,
        feature_size: int,
        nb_classes: int,
        s: Optional[int] = 64,
        m: Optional[float] = 0.5,
        clip: Optional[bool] = True,
    ) -> None:
        super().__init__(feature_size, nb_classes, s, m, clip)

    def positive_cosine_similarity_modulator(self) -> torch.Tensor:
        """
        Positive cosine modulator for ArcFace is: T(cos(theta)) = cos(theta + m),
        if we expand this expression it becomes: cos(theta)*cos(m) - sin(theta)*sin(m)
        """
        return self.cosine * self.cos_m - self.sine * self.sin_m

    def negative_cosine_similarity_modulator(
        self, cosine_after_positive_modulator: torch.Tensor
    ) -> torch.Tensor:
        """
        There is no modulation for negative cosine similarity in ArcFace
        """
        return self.cosine


class MVArcSoftmax(AngularMarginHead):
    def __init__(
        self,
        feature_size: int,
        nb_classes: int,
        s: Optional[int] = 64,
        m: Optional[float] = 0.5,
        t: Optional[float] = 1.2,
        clip: Optional[bool] = True,
    ) -> None:
        super().__init__(feature_size, nb_classes, s, m, clip)
        self.t = t

    def positive_cosine_similarity_modulator(self) -> torch.Tensor:
        """
        Positive cosine modulator for MVArcSoftmax is: T(cos(theta_{y_i})) = cos(theta_{y_i} + m),
        if we expand this expression it becomes: cos(theta_{y_i})*cos(m) - sin(theta_{y_i})*sin(m)
        """
        return self.cosine * self.cos_m - self.sine * self.sin_m

    def negative_cosine_similarity_modulator(
        self, cosine_after_positive_modulator: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative cosine similarity modulator for MVArcSoftmax:

        if T(cos(theta_y_i)) - cos(theta_j) >= 0:
            return cos(theta_j)
        else:
            return t*cos(theta_j) + t - 1

        """
        return torch.where(
            cosine_after_positive_modulator - self.cosine >= 0,
            self.cosine,
            self.t * self.cosine + self.t - 1,
        )


class CurricularFace(AngularMarginHead):
    def __init__(
        self,
        feature_size: int,
        nb_classes: int,
        s: Optional[int] = 64,
        m: Optional[float] = 0.5,
        alpha: Optional[float] = 0.99,
        clip: Optional[bool] = True,
    ) -> None:
        super().__init__(feature_size, nb_classes, s, m, clip)
        self.t = 0
        self.ts = []
        self.alpha = alpha

    def positive_cosine_similarity_modulator(self) -> torch.Tensor:
        """
        Positive cosine modulator for CurricularFace is: T(cos(theta)) = cos(theta + m),
        if we expand this expression it becomes: cos(theta)*cos(m) - sin(theta)*sin(m)
        """
        return self.cosine * self.cos_m - self.sine * self.sin_m

    def negative_cosine_similarity_modulator(
        self, cosine_after_positive_modulator: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative cosine similarity modulator for CurricularFace:

        if T(cos(theta_y_i)) - cos(theta_j) >= 0:
            return cos(theta_j)
        else:
            return cos(theta_j) * (t + cos(theta_j))

        """
        return torch.where(
            cosine_after_positive_modulator - self.cosine >= 0,
            self.cosine,
            self.cosine ** 2 + self.t * self.cosine,
        )

    @torch.no_grad()
    def adjust_t(self, y: torch.Tensor) -> None:
        """
        We adjust t based on Exponential Moving Avarage
        """
        r = self.cosine[torch.arange(len(self.cosine)), y].mean()
        self.t = self.alpha * r + (1 - self.alpha) * self.t

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ts.append(self.t)
        output = super().forward(x, y)
        self.adjust_t(y)
        return output
