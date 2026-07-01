#!/usr/bin/env python3
"""Utilities for prompt-aware mRoPE remapping."""

from __future__ import annotations

import math
from typing import Any

import torch


def _normalize_int_list(value: Any, expected_len: int) -> list[int] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def estimate_prompt_rows(
    image_size: list[int] | tuple[int, int],
    prompt_box: list[int] | tuple[int, int, int, int],
    llm_grid_h: int,
) -> int:
    image_size = _normalize_int_list(image_size, 2)
    prompt_box = _normalize_int_list(prompt_box, 4)
    if image_size is None or prompt_box is None or llm_grid_h <= 0:
        return 0

    _, image_h = image_size
    _, top, _, bottom = prompt_box
    if image_h <= 0:
        return 0

    prompt_h = max(0, min(image_h, bottom) - max(0, top))
    if prompt_h <= 0:
        return 0

    prompt_rows = math.ceil(llm_grid_h * (prompt_h / image_h))
    return max(1, min(llm_grid_h, prompt_rows))


def remap_prompt_band_positions(
    position_ids: torch.Tensor,
    token_indices: torch.Tensor,
    llm_grid_h: int,
    llm_grid_w: int,
    prompt_rows: int,
    *,
    mode: str,
    batch_index: int = 0,
    offset: int = 256,
) -> torch.Tensor:
    if mode not in {"row_major", "box_offset"}:
        return position_ids
    if llm_grid_h <= 0 or llm_grid_w <= 0 or prompt_rows <= 0 or token_indices.numel() == 0:
        return position_ids

    prompt_rows = min(prompt_rows, llm_grid_h)
    prompt_token_count = min(int(token_indices.numel()), prompt_rows * llm_grid_w)
    if prompt_token_count <= 0:
        return position_ids

    prompt_token_indices = token_indices[:prompt_token_count]
    remapped = position_ids.clone()
    base_position = remapped[0, batch_index, prompt_token_indices[0]].to(remapped.dtype)

    if mode == "row_major":
        row_major = torch.arange(prompt_token_count, device=remapped.device, dtype=remapped.dtype) + base_position
        remapped[:, batch_index, prompt_token_indices] = row_major.unsqueeze(0).expand(3, -1)
        return remapped

    remapped[:, batch_index, prompt_token_indices] = remapped[:, batch_index, prompt_token_indices] + offset
    return remapped


def apply_prompt_box_mrope(
    position_ids: torch.Tensor,
    input_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    prompt_boxes: list[Any] | None,
    prompt_image_sizes: list[Any] | None,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
    mode: str,
    offset: int = 256,
) -> torch.Tensor:
    if mode not in {"row_major", "box_offset"}:
        return position_ids
    if input_ids is None or image_grid_thw is None:
        return position_ids
    if prompt_boxes is None or prompt_image_sizes is None:
        return position_ids

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    remapped = position_ids.clone()
    image_index = 0

    for batch_index in range(input_ids.size(0)):
        active_positions = torch.nonzero(attention_mask[batch_index] == 1, as_tuple=False).flatten()
        input_ids_i = input_ids[batch_index, active_positions]
        if input_ids_i.numel() == 0:
            continue

        vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
        if vision_start_indices.numel() == 0:
            continue

        vision_tokens = input_ids_i[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum().item())
        input_tokens = input_ids_i.tolist()
        st = 0

        batch_prompt_box = prompt_boxes[batch_index] if batch_index < len(prompt_boxes) else None
        batch_image_size = prompt_image_sizes[batch_index] if batch_index < len(prompt_image_sizes) else None

        for image_local_index in range(image_nums):
            try:
                ed = input_tokens.index(image_token_id, st)
            except ValueError:
                break

            if image_index >= image_grid_thw.shape[0]:
                break

            t, h, w = image_grid_thw[image_index]
            image_index += 1

            llm_grid_t = int(t.item())
            llm_grid_h = int(h.item()) // spatial_merge_size
            llm_grid_w = int(w.item()) // spatial_merge_size
            visual_token_count = llm_grid_t * llm_grid_h * llm_grid_w
            visual_positions = active_positions[ed : ed + visual_token_count]

            if image_local_index == 0 and llm_grid_t == 1:
                prompt_rows = estimate_prompt_rows(batch_image_size, batch_prompt_box, llm_grid_h)
                remapped = remap_prompt_band_positions(
                    remapped,
                    visual_positions,
                    llm_grid_h,
                    llm_grid_w,
                    prompt_rows,
                    mode=mode,
                    batch_index=batch_index,
                    offset=offset,
                )

            st = ed + visual_token_count

    return remapped
