"""ACT policy for vision + proprioception imitation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """ResNet-18 image backbone (ImageNet-pretrained by default, as in the original ACT)."""

    def __init__(self, in_channels: int = 3, output_dim: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(backbone.fc.in_features, output_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        feat = self.features(x).flatten(1)
        return torch.relu(self.fc(feat))


class ACTPolicy(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        chunk_size: int = 8,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dropout: float = 0.1,
        encoder_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.image_encoder = ImageEncoder(output_dim=hidden_dim, pretrained=encoder_pretrained)
        self.proprio_encoder = nn.Linear(proprio_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.action_queries = nn.Embedding(chunk_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, images: torch.Tensor | list[torch.Tensor], proprio: torch.Tensor) -> torch.Tensor:
        # ``images`` is a single (B,H,W,C) tensor or a list of them (one per camera);
        # the shared image_encoder produces one memory token per camera view.
        if isinstance(images, torch.Tensor):
            images = [images]
        batch_size = images[0].size(0)
        img_tokens = [self.image_encoder(img).unsqueeze(1) for img in images]
        prop_feat = self.proprio_encoder(proprio).unsqueeze(1)
        memory = self.encoder(torch.cat([*img_tokens, prop_feat], dim=1))
        queries = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        out = self.decoder(queries, memory)
        return torch.tanh(self.action_head(out))
