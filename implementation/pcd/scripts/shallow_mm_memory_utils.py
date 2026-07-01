from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


def parse_layer_list(spec: str | None) -> tuple[int, ...]:
    if spec is None:
        return ()

    text = spec.strip()
    if not text:
        return ()

    parsed: list[int] = []
    seen: set[int] = set()
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        value = int(item)
        if value not in seen:
            parsed.append(value)
            seen.add(value)
    return tuple(parsed)


def resolve_layer_list(spec: str | None, total_layers: int, env_name: str) -> tuple[int, ...]:
    indices = parse_layer_list(spec)
    if not indices:
        raise ValueError(f"{env_name} must specify at least one decoder layer.")

    for index in indices:
        if index < 0 or index >= total_layers:
            raise ValueError(
                f"{env_name} contains out-of-range layer index {index}; valid range is [0, {total_layers - 1}]."
            )
    return indices


def resolve_decoder_layers(model: Any) -> nn.ModuleList:
    candidates = [
        getattr(getattr(getattr(model, "model", None), "language_model", None), "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
    raise AttributeError("Could not locate decoder layers for SMMA-Pool.")


def pool_hidden_states(
    hidden_states: Sequence[torch.Tensor | None],
    attention_mask: torch.Tensor | None,
    pool_mode: str,
) -> torch.Tensor | None:
    valid_hiddens = [hidden for hidden in hidden_states if hidden is not None]
    if not valid_hiddens:
        return None

    if attention_mask is None:
        token_mask = torch.ones(
            valid_hiddens[0].shape[:2],
            device=valid_hiddens[0].device,
            dtype=torch.bool,
        )
    else:
        token_mask = attention_mask.to(device=valid_hiddens[0].device).bool()

    pooled = [_pool_single_hidden(hidden, token_mask, pool_mode) for hidden in valid_hiddens]
    if len(pooled) == 1:
        return pooled[0]
    return torch.stack(pooled, dim=0).mean(dim=0)


def _pool_single_hidden(hidden: torch.Tensor, token_mask: torch.Tensor, pool_mode: str) -> torch.Tensor:
    if pool_mode == "mean":
        mask = token_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    if pool_mode == "last":
        positions = token_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_indices, positions]

    raise ValueError(f"Unsupported SMMA_POOL_MODE={pool_mode!r}. Expected one of: mean, last.")


def replace_primary_tensor(output: Any, new_hidden: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return new_hidden
    if isinstance(output, tuple):
        return (new_hidden, *output[1:])
    if isinstance(output, list):
        return [new_hidden, *output[1:]]
    raise TypeError(f"Unsupported decoder layer output type for SMMA-Pool: {type(output)!r}")


class ShallowMemoryAdapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_hidden_dim: int, init_gate: float) -> None:
        super().__init__()
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.down_proj = nn.Linear(hidden_size, adapter_hidden_dim)
        self.act = nn.SiLU()
        self.up_proj = nn.Linear(adapter_hidden_dim, hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        memory_update = self.up_proj(self.act(self.down_proj(self.memory_norm(memory))))
        memory_update = memory_update.to(dtype=hidden_states.dtype).unsqueeze(1)
        gate = torch.sigmoid(self.gate).to(dtype=hidden_states.dtype)
        return hidden_states + gate * memory_update
