"""Small Transformer that consumes the fixed Unity PCM input row."""

import torch
from torch import nn


class LipSyncTransformer(nn.Module):
    """Map one `[batch, sample_count]` PCM tensor to viseme score logits."""

    def __init__(self, output_size: int, hidden_size: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=128, stride=64, padding=32),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size))

    def forward(self, audio_pcm: torch.Tensor) -> torch.Tensor:
        """Input is float32 `[batch, 1024]`; output is unnormalized logits `[batch, 5]`."""
        features = self.frontend(audio_pcm.unsqueeze(1)).transpose(1, 2)
        encoded = self.encoder(features)
        return self.classifier(encoded.mean(dim=1))
