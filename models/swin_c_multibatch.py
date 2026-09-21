import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t

from .transformer_cosine_multibatch import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


__all__ = ["swin_t_trans"]


class SwinTransMultiBatch(nn.Module):
    """Swin-T backbone followed by the original multibatch MAN head."""

    def __init__(self, pretrained=True):
        super().__init__()

        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = swin_t(weights=weights)
        self.backbone_features = backbone.features
        self.backbone_norm = backbone.norm

        # Swin-T stage 4 is stride 32 with 768 channels.  MAN expects 512.
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1),
            nn.GELU(),
        )

        d_model = 512
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=None)

        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        density_size = (input_h // 16, input_w // 16)

        # torchvision Swin features and norm use channels-last tensors.
        x = self.backbone_features(x)
        x = self.backbone_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.feature_adapter(x)

        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).permute(2, 0, 1)
        tokens, consistency_features = self.encoder(tokens, (height, width))
        x = tokens.permute(1, 2, 0).reshape(
            batch_size, channels, height, width
        )

        x = F.interpolate(
            x, size=density_size, mode="bilinear", align_corners=False
        )
        density = self.reg_layer_0(x)
        return torch.relu(density), consistency_features


def swin_t_trans(pretrained=True):
    return SwinTransMultiBatch(pretrained=pretrained)
