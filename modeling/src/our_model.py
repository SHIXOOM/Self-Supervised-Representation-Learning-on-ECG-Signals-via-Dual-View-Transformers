import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ECGModelConfig:
    sequence_length: int = 1000
    num_channels: int = 12
    d_model: int = 512
    time_heads: int = 8
    channel_heads: int = 4
    time_layers: int = 12
    channel_layers: int = 12
    ff_multiplier: int = 6
    dropout: float = 0.1
    temperature: float = 0.5
    projection_dim: int = 512
    dtype: torch.dtype = torch.float32


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dtype: torch.dtype) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=dtype) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeTransformer(nn.Module):
    def __init__(self, config: ECGModelConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(
            config.num_channels, config.d_model, dtype=config.dtype
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.time_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="relu",
            dtype=config.dtype,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.time_layers
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.sequence_length, config.dtype
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
        # Pool across time dimension first, then project to d_model
        self.pool = nn.AdaptiveAvgPool1d(512)
        self.channel_proj = nn.Linear(512, config.d_model, dtype=config.dtype)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.channel_heads,
            dim_feedforward=config.d_model * config.ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="relu",
            dtype=config.dtype,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.channel_layers
        )
        self.num_channels = config.num_channels
        self.d_model = config.d_model
        self.dtype = config.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (batch, channels, time) - e.g., (B, 12, 5000)
        x = self.pool(x)  # (batch, channels, 512) - aggregate time for each channel
        x = self.channel_proj(x.permute(0, 1, 2))  # (batch, channels, d_model)
        
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