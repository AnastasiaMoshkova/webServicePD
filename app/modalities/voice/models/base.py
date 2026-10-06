import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _init_weights(self, init_mode: str = "custom") -> None:
        valid = {"custom", "xavier", "kaiming", "default"}
        if init_mode not in valid:
            raise ValueError(
                f"Unknown init_mode '{init_mode}'. Choose from: {sorted(valid)}."
            )
        if init_mode == "default":
            return

        def _conv(weight: torch.Tensor) -> None:
            if init_mode == "xavier":
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")

        def _linear(weight: torch.Tensor) -> None:
            if init_mode == "kaiming":
                nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _conv(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                _linear(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)