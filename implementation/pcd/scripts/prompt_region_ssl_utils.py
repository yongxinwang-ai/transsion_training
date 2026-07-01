#!/usr/bin/env python3
"""Utilities for prompt-region self-supervised objectives."""

from __future__ import annotations

from contextlib import contextmanager
import math
import random
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw

from prompt_mrope_utils import estimate_prompt_rows


@contextmanager
def temporary_image_pixel_limits(
    processor: Any,
    *,
    image_max_pixels: int | None = None,
    image_min_pixels: int | None = None,
):
    if processor is None:
        yield
        return

    old_max = getattr(processor, "image_max_pixels", None)
    old_min = getattr(processor, "image_min_pixels", None)
    try:
        if image_max_pixels is not None:
            setattr(processor, "image_max_pixels", int(image_max_pixels))
        if image_min_pixels is not None:
            setattr(processor, "image_min_pixels", int(image_min_pixels))
        yield
    finally:
        if old_max is not None:
            setattr(processor, "image_max_pixels", old_max)
        elif hasattr(processor, "image_max_pixels"):
            delattr(processor, "image_max_pixels")

        if old_min is not None:
            setattr(processor, "image_min_pixels", old_min)
        elif hasattr(processor, "image_min_pixels"):
            delattr(processor, "image_min_pixels")


def _normalize_box(value: Any) -> tuple[int, int, int, int] | None:
    if torch.is_tensor(value):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def mask_prompt_region(
    image: Image.Image,
    prompt_box: Any,
    *,
    mask_ratio: float,
    block_size: int,
    seed: int,
    fill_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    prompt_box = _normalize_box(prompt_box)
    image = image.convert("RGB").copy()
    if prompt_box is None or mask_ratio <= 0.0 or block_size <= 0:
        return image

    left, top, right, bottom = prompt_box
    width, height = image.size
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    blocks: list[tuple[int, int, int, int]] = []
    for y in range(top, bottom, block_size):
        for x in range(left, right, block_size):
            blocks.append((x, y, min(x + block_size, right), min(y + block_size, bottom)))

    if not blocks:
        return image

    rng = random.Random(seed)
    num_mask = max(1, math.ceil(len(blocks) * min(mask_ratio, 1.0)))
    selected = rng.sample(blocks, k=min(num_mask, len(blocks)))

    draw = ImageDraw.Draw(image)
    for left_i, top_i, right_i, bottom_i in selected:
        draw.rectangle((left_i, top_i, right_i - 1, bottom_i - 1), fill=fill_color)
    return image


def build_prompt_band_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    prompt_boxes: list[Any] | None,
    prompt_image_sizes: list[Any] | None,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if image_grid_thw is None or prompt_boxes is None or prompt_image_sizes is None:
        return mask

    image_index = 0
    for batch_index in range(input_ids.size(0)):
        active_positions = torch.nonzero(attention_mask[batch_index] == 1, as_tuple=False).flatten()
        active_ids = input_ids[batch_index, active_positions]
        if active_ids.numel() == 0:
            continue

        vision_start_indices = torch.argwhere(active_ids == vision_start_token_id).squeeze(1)
        if vision_start_indices.numel() == 0:
            continue

        vision_tokens = active_ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == image_token_id).sum().item())
        token_list = active_ids.tolist()
        st = 0

        batch_prompt_box = prompt_boxes[batch_index] if batch_index < len(prompt_boxes) else None
        batch_image_size = prompt_image_sizes[batch_index] if batch_index < len(prompt_image_sizes) else None

        for image_local_index in range(image_nums):
            try:
                ed = token_list.index(image_token_id, st)
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
                prompt_token_count = min(int(visual_positions.numel()), prompt_rows * llm_grid_w)
                if prompt_token_count > 0:
                    mask[batch_index, visual_positions[:prompt_token_count]] = True

            st = ed + visual_token_count

    return mask


def seed_from_sample_id(sample_id: Any) -> int:
    text = str(sample_id or "")
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % (2**31)


def maybe_build_prmlp_image_views(
    *,
    enabled: bool,
    full_image_paths: list[str],
    crop_image_paths: list[str],
    prompt_box: Any,
    sample_id: Any,
    mask_ratio: float,
    block_size: int,
    load_rgb_image: Callable[[str], Image.Image],
    mask_fn: Callable[..., Image.Image] = mask_prompt_region,
) -> tuple[list[Image.Image], list[Image.Image]] | None:
    if not enabled:
        return None

    seed = seed_from_sample_id(sample_id)
    masked_images: list[Image.Image] = []
    for image_index, image_path in enumerate(full_image_paths):
        image = load_rgb_image(image_path)
        if image_index == 0:
            image = mask_fn(
                image,
                prompt_box,
                mask_ratio=mask_ratio,
                block_size=block_size,
                seed=seed,
            )
        masked_images.append(image)

    clean_crop_images = [load_rgb_image(image_path) for image_path in crop_image_paths]
    return masked_images, clean_crop_images


def distributed_boolean_and(
    local_enabled: bool,
    *,
    device: torch.device | str,
    dist_module: Any = None,
) -> bool:
    if dist_module is None:
        dist_module = torch.distributed

    if not getattr(dist_module, "is_available", lambda: False)():
        return local_enabled
    if not getattr(dist_module, "is_initialized", lambda: False)():
        return local_enabled

    tensor = torch.tensor(
        [1 if local_enabled else 0],
        device=device,
        dtype=torch.int32,
    )
    reduce_op = getattr(getattr(dist_module, "ReduceOp", None), "MIN", None)
    if reduce_op is None:
        dist_module.all_reduce(tensor)
    else:
        dist_module.all_reduce(tensor, op=reduce_op)
    return bool(int(tensor.item()))


def distributed_boolean_or(
    local_enabled: bool,
    *,
    device: torch.device | str,
    dist_module: Any = None,
) -> bool:
    if dist_module is None:
        dist_module = torch.distributed

    if not getattr(dist_module, "is_available", lambda: False)():
        return local_enabled
    if not getattr(dist_module, "is_initialized", lambda: False)():
        return local_enabled

    tensor = torch.tensor(
        [1 if local_enabled else 0],
        device=device,
        dtype=torch.int32,
    )
    reduce_op = getattr(getattr(dist_module, "ReduceOp", None), "MAX", None)
    if reduce_op is None:
        dist_module.all_reduce(tensor)
    else:
        dist_module.all_reduce(tensor, op=reduce_op)
    return bool(int(tensor.item()))
