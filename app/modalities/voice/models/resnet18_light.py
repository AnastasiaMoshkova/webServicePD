from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .base import BaseModel


class _ResNet18LightBackbone(nn.Module):
    OUT_DIM: int = 512
    STAGES: tuple[str, ...] = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")

    def __init__(
        self,
        unfreeze_from: str = "layer4",
        gradual_unfreeze: bool = False,
        pretrained_backbone: bool = False,
    ):
        super().__init__()

        weights = "DEFAULT" if pretrained_backbone else None
        rn = torchvision.models.resnet18(weights=weights)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.stem = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        self.avgpool = rn.avgpool

        if unfreeze_from not in self.STAGES:
            raise ValueError(f"Stage '{unfreeze_from}' not found. Available: {list(self.STAGES)}")

        for param in self.parameters():
            param.requires_grad = False

        self._unfreeze_queue = self._build_unfreeze_queue(unfreeze_from)
        self._queue_pos = 0
        if not gradual_unfreeze:
            self._unfreeze_to_target()

    def _build_unfreeze_queue(self, unfreeze_from: str) -> list[nn.Module]:
        start_idx = self.STAGES.index(unfreeze_from)
        return [getattr(self, name) for name in reversed(self.STAGES[start_idx:])]

    def _unfreeze_to_target(self) -> None:
        while self.unfreeze_next():
            pass

    def unfreeze_next(self) -> bool:
        if self._queue_pos >= len(self._unfreeze_queue):
            return False
        module = self._unfreeze_queue[self._queue_pos]
        for p in module.parameters():
            p.requires_grad = True
        self._queue_pos += 1
        return True

    @property
    def fully_unfrozen(self) -> bool:
        return self._queue_pos >= len(self._unfreeze_queue)

    @property
    def unfrozen_depth(self) -> int:
        return self._queue_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)


class ResNet18LightModel(BaseModel):
    def __init__(
        self,
        num_classes: int = 1,
        dropout: float = 0.3,
        unfreeze_from: str = "layer4",
        gradual_unfreeze: bool = False,
        init_mode: str = "custom",
        pretrained_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = _ResNet18LightBackbone(
            unfreeze_from=unfreeze_from,
            gradual_unfreeze=gradual_unfreeze,
            pretrained_backbone=pretrained_backbone,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(_ResNet18LightBackbone.OUT_DIM, num_classes),
        )
        self._init_weights(init_mode)