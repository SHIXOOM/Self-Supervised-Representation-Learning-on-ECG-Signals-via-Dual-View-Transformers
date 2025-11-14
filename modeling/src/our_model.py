from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ECGModelConfig:
    sequence_length: int = 1000
    positional_embedding_length: int = 1000
    num_channels: int = 12
    d_model: int = 512
    channel_token_length: int = 512
    time_heads: int = 8
    channel_heads: int = 4
    time_layers: int = 12
    channel_layers: int = 12
    ff_multiplier: int = 6
    dropout: float = 0.1
    temperature: float = 0.5
    projection_dim: int = 512
    dtype: torch.dtype = torch.bfloat16


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Parameter(
            torch.empty(1, max_len, d_model, dtype=dtype)
        )
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding[:, : x.size(1)]


class TimeTransformer(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        positional_length = config.positional_embedding_length
        self.input_proj = nn.Linear(
            config.num_channels, config.d_model, dtype=config.dtype
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.time_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            dtype=config.dtype,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.time_layers
        )
        self.positional_encoding = LearnablePositionalEncoding(
            config.d_model, positional_length, config.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.input_proj.weight.dtype:
            x = x.to(self.input_proj.weight.dtype)
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        return self.encoder(x)


class ChannelTransformer(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        target_length = config.channel_token_length
        # Lightweight depthwise separable stack learns downsampling before projection
        self.downsample = nn.Sequential(
            nn.Conv1d(
                config.num_channels,
                config.num_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                groups=config.num_channels,
                bias=False,
                dtype=config.dtype,
            ),
            nn.GELU(),
            nn.Conv1d(
                config.num_channels,
                config.num_channels,
                kernel_size=1,
                dtype=config.dtype,
            ),
            nn.GELU(),
            nn.Conv1d(
                config.num_channels,
                config.num_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                groups=config.num_channels,
                bias=False,
                dtype=config.dtype,
            ),
            nn.GELU(),
            nn.Conv1d(
                config.num_channels,
                config.num_channels,
                kernel_size=1,
                dtype=config.dtype,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(target_length),
        )
        self.channel_proj = nn.Linear(target_length, config.d_model, dtype=config.dtype)
        self.pre_encoder_norm = nn.LayerNorm(config.d_model, dtype=config.dtype)
        self.pre_encoder_dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.channel_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            dtype=config.dtype,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.channel_layers
        )
        self.num_channels = config.num_channels
        self.d_model = config.d_model
        self.dtype = config.dtype
        self.to(dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch, channels, time) - e.g., (B, 12, 5000)
        x = self.downsample(x)  # (batch, channels, target_length)
        x = self.channel_proj(x.permute(0, 1, 2))  # (batch, channels, d_model)
        x = self.pre_encoder_norm(x)
        x = self.pre_encoder_dropout(x)
        return self.encoder(x)


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.time_to_channel = nn.MultiheadAttention(
            config.d_model, config.time_heads, batch_first=True, dtype=config.dtype
        )
        self.channel_to_time = nn.MultiheadAttention(
            config.d_model, config.channel_heads, batch_first=True, dtype=config.dtype
        )
        self.time_norm = nn.LayerNorm(config.d_model, dtype=config.dtype)
        self.channel_norm = nn.LayerNorm(config.d_model, dtype=config.dtype)

    def forward(
        self, time_tokens: torch.Tensor, channel_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        time_cross, _ = self.time_to_channel(
            time_tokens, channel_tokens, channel_tokens
        )
        fused_time = self.time_norm(time_tokens + time_cross)
        channel_cross, _ = self.channel_to_time(
            channel_tokens, time_tokens, time_tokens
        )
        fused_channel = self.channel_norm(channel_tokens + channel_cross)
        return fused_time, fused_channel


class FusionHead(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.d_model * 2, config.d_model, dtype=config.dtype)
        self.norm = nn.LayerNorm(config.d_model, dtype=config.dtype)
        self.activation = nn.ReLU()

    def forward(
        self, time_repr: torch.Tensor, channel_repr: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat([time_repr, channel_repr], dim=-1)
        mapped = self.linear(fused)
        return self.activation(self.norm(mapped))


class ProjectionHead(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model, dtype=config.dtype),
            nn.LayerNorm(config.d_model, dtype=config.dtype),
            nn.ReLU(),
            nn.Linear(config.d_model, config.projection_dim, dtype=config.dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ECGEncoder(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.time_encoder = TimeTransformer(config)
        self.channel_encoder = ChannelTransformer(config)
        self.cross_attention = BidirectionalCrossAttention(config)
        self.fusion = FusionHead(config)
        self.projection = ProjectionHead(config)
        self.dtype = config.dtype
        self.to(dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        time_tokens = self.time_encoder(x)
        channel_tokens = self.channel_encoder(x)
        fused_time, fused_channel = self.cross_attention(time_tokens, channel_tokens)
        time_repr = fused_time.mean(dim=1)
        channel_repr = fused_channel.mean(dim=1)
        representation = self.fusion(time_repr, channel_repr)
        projection = self.projection(representation)
        return representation, projection