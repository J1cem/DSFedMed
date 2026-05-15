"""Lightweight ControlNet-style generator for DSFedMed synthetic data.

This module intentionally avoids a hard dependency on Stable Diffusion so the
repository can run in restricted medical-imaging environments.  It preserves the
DSFedMed protocol boundary: each client trains and uploads only a small
condition branch (`ControlNetBranch`) while the base generator is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ControlNetConfig:
    in_channels: int = 3
    mask_channels: int = 1
    hidden_channels: int = 32
    noise_channels: int = 8
    num_blocks: int = 4


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _valid_group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FrozenMedicalGenerator(nn.Module):
    """Small frozen image prior used as a Stable-Diffusion stand-in.

    The weights are frozen during client training; only residual control features
    are learned.  In deployments with `diffusers`, this class can be replaced by
    a frozen SD UNet/VAE while keeping the same ControlNet branch API.
    """

    def __init__(self, config: ControlNetConfig):
        super().__init__()
        channels = config.hidden_channels
        self.stem = ConvBlock(config.mask_channels + config.noise_channels, channels)
        self.blocks = nn.ModuleList([ConvBlock(channels, channels) for _ in range(config.num_blocks)])
        self.head = nn.Conv2d(channels, config.in_channels, kernel_size=1)
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        mask: torch.Tensor,
        noise: torch.Tensor,
        controls: Optional[Iterable[torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.stem(torch.cat([mask, noise], dim=1))
        controls = list(controls or [])
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx < len(controls):
                x = x + controls[idx]
        return torch.sigmoid(self.head(x))


class ControlNetBranch(nn.Module):
    """Trainable client-specific residual branch conditioned on masks."""

    def __init__(self, config: ControlNetConfig):
        super().__init__()
        channels = config.hidden_channels
        self.encoder = ConvBlock(config.mask_channels, channels)
        self.blocks = nn.ModuleList([ConvBlock(channels, channels) for _ in range(config.num_blocks)])
        self.zero_convs = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for _ in range(config.num_blocks)])
        for conv in self.zero_convs:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, mask: torch.Tensor) -> List[torch.Tensor]:
        x = self.encoder(mask)
        controls = []
        for block, zero_conv in zip(self.blocks, self.zero_convs):
            x = block(x)
            controls.append(zero_conv(x))
        return controls


class ControlNetGenerator(nn.Module):
    """Frozen generator plus trainable ControlNet branch."""

    def __init__(self, config: Optional[ControlNetConfig] = None):
        super().__init__()
        self.config = config or ControlNetConfig()
        self.base = FrozenMedicalGenerator(self.config)
        self.control = ControlNetBranch(self.config)

    def forward(self, mask: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            b, _, h, w = mask.shape
            noise = torch.randn(b, self.config.noise_channels, h, w, device=mask.device, dtype=mask.dtype)
        return self.base(mask, noise, self.control(mask))

    def control_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.control.state_dict()

    def load_control_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        return self.control.load_state_dict(state_dict, strict=strict)


def average_control_states(
    states: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Server-side FedAvg/mixture aggregation for uploaded ControlNet branches."""

    if not states:
        raise ValueError("states must contain at least one client ControlNet state_dict")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    total = float(sum(weights))
    weights = [w / total for w in weights]
    averaged = {}
    for key in states[0].keys():
        averaged[key] = sum(state[key].detach().float() * weights[idx] for idx, state in enumerate(states))
    return averaged


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Ensure masks are single-channel float tensors in [0, 1]."""

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) > 1:
        mask = mask.mean(dim=1, keepdim=True)
    mask = mask.float()
    max_value = mask.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    return (mask / max_value).clamp(0.0, 1.0)


def reconstruction_loss(pred: torch.Tensor, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Image reconstruction loss used for private client ControlNet training."""

    image = image.float().clamp(0.0, 1.0)
    l1 = F.l1_loss(pred, image)
    masked_l1 = F.l1_loss(pred * mask, image * mask)
    tv = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean() + (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
    total = l1 + 0.5 * masked_l1 + 0.01 * tv
    return total, {"l1": l1.detach(), "masked_l1": masked_l1.detach(), "tv": tv.detach()}
