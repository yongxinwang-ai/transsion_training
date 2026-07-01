#!/usr/bin/env python3
"""Minimal LlamaFactory wrapper for PVRD-SG.

This keeps LlamaFactory's SFT framework intact and monkeypatches only the narrow pieces needed to
carry training-only grounding metadata and add one auxiliary loss in compute_loss.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from llamafactory.data.collator import MultiModalDataCollatorForSeq2Seq
from llamafactory.data.converter import AlpacaDatasetConverter, SharegptDatasetConverter
from llamafactory.data.processor.processor_utils import infer_seqlen
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.tuner import run_exp

from prompt_region_ssl_utils import (
    build_prompt_band_mask,
    distributed_boolean_and,
    distributed_boolean_or,
    mask_prompt_region,
    maybe_build_prmlp_image_views,
    temporary_image_pixel_limits,
)
from shallow_mm_memory_utils import (
    ShallowMemoryAdapter,
    pool_hidden_states,
    replace_primary_tensor,
    resolve_decoder_layers,
    resolve_layer_list,
)

PVRD_SG_ENABLED = os.environ.get("PVRD_SG_ENABLED", "0") == "1"
PVRD_SG_LAMBDA = float(os.environ.get("PVRD_SG_LAMBDA", "0.15"))
PVRD_VTS_SFT_WEIGHT = float(os.environ.get("PVRD_VTS_SFT_WEIGHT", "1.0"))
PVRD_VTS_ANSWER_ONLY_LOSS = os.environ.get("PVRD_VTS_ANSWER_ONLY_LOSS", "0") == "1"
PVRD_VTS_ANSWER_INCLUDE_TAGS = os.environ.get("PVRD_VTS_ANSWER_INCLUDE_TAGS", "1") == "1"
PVRD_DISTILL_ENABLED = os.environ.get("PVRD_DISTILL_ENABLED", "0") == "1"
PVRD_DISTILL_LAMBDA = float(os.environ.get("PVRD_DISTILL_LAMBDA", "0.10"))
PVRD_HIDDEN_DISTILL_LAMBDA = float(os.environ.get("PVRD_HIDDEN_DISTILL_LAMBDA", "0.05"))
PVRD_DISTILL_TEMPERATURE = float(os.environ.get("PVRD_DISTILL_TEMPERATURE", "2.0"))
PVRD_DISTILL_ONLY_EXACT = os.environ.get("PVRD_DISTILL_ONLY_EXACT", "1") == "1"
PVRD_TEACHER_CACHE_PATH = os.environ.get("PVRD_TEACHER_CACHE_PATH", "")
PRMLP_ENABLED = os.environ.get("PRMLP_ENABLED", "0") == "1"
PRMLP_LAMBDA = float(os.environ.get("PRMLP_LAMBDA", "0.05"))
PRMLP_MASK_RATIO = float(os.environ.get("PRMLP_MASK_RATIO", "0.35"))
PRMLP_BLOCK_SIZE = int(os.environ.get("PRMLP_BLOCK_SIZE", "32"))
PRMLP_CUTOFF_LEN = int(os.environ.get("PRMLP_CUTOFF_LEN", "8192"))
PRMLP_EVERY_N_STEPS = max(1, int(os.environ.get("PRMLP_EVERY_N_STEPS", "1")))
PRMLP_MAX_MAIN_TOKENS = int(os.environ.get("PRMLP_MAX_MAIN_TOKENS", "0"))
PRMLP_MAX_EXTRA_TOKENS = int(os.environ.get("PRMLP_MAX_EXTRA_TOKENS", "0"))
PRMLP_LOG_EVERY_N_STEPS = int(os.environ.get("PRMLP_LOG_EVERY_N_STEPS", "0"))
PRMLP_ONLINE_VIEW = os.environ.get("PRMLP_ONLINE_VIEW", "masked").strip().lower()
PRMLP_PROMPT_TEXT = os.environ.get("PRMLP_PROMPT_TEXT", "Help me solve the problem")
PRMLP_DUMMY_ANSWER = os.environ.get("PRMLP_DUMMY_ANSWER", "0")
PRMLP_IMAGE_MAX_PIXELS = int(os.environ.get("PRMLP_IMAGE_MAX_PIXELS", "1048576"))
PRMLP_IMAGE_MIN_PIXELS = int(os.environ.get("PRMLP_IMAGE_MIN_PIXELS", "1024"))
PRMLP_DEBUG = os.environ.get("PRMLP_DEBUG", "0") == "1"
SKIP_ROOT_FINAL_SAVE = os.environ.get("SKIP_ROOT_FINAL_SAVE", "0") == "1"
SMMA_ENABLED = os.environ.get("SMMA_ENABLED", "0") == "1"
SMMA_SOURCE_LAYERS = os.environ.get("SMMA_SOURCE_LAYERS", "4,8,12")
SMMA_TARGET_LAYERS = os.environ.get("SMMA_TARGET_LAYERS", "20,24,28")
SMMA_HIDDEN_DIM = int(os.environ.get("SMMA_HIDDEN_DIM", "512"))
SMMA_INIT_GATE = float(os.environ.get("SMMA_INIT_GATE", "-4.0"))
SMMA_POOL_MODE = os.environ.get("SMMA_POOL_MODE", "mean").strip().lower()


def _prmlp_debug(message: str) -> None:
    if not PRMLP_DEBUG:
        return
    rank = os.environ.get("RANK", "?")
    print(f"[PRMLP][rank={rank}] {message}", file=sys.stderr, flush=True)


def _extract_sg_fields(example: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in example.items()
        if key.startswith("sg_") or key.startswith("pvrd_")
    }


def patch_converters() -> None:
    orig_share = SharegptDatasetConverter.__call__
    orig_alpaca = AlpacaDatasetConverter.__call__

    def patched_share(self, example: dict[str, Any]) -> dict[str, Any]:
        output = orig_share(self, example)
        output.update(_extract_sg_fields(example))
        return output

    def patched_alpaca(self, example: dict[str, Any]) -> dict[str, Any]:
        output = orig_alpaca(self, example)
        output.update(_extract_sg_fields(example))
        return output

    SharegptDatasetConverter.__call__ = patched_share
    AlpacaDatasetConverter.__call__ = patched_alpaca


def patch_supervised_processor() -> None:
    orig_preprocess = SupervisedDatasetProcessor.preprocess_dataset

    def patched_preprocess(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = orig_preprocess(self, examples)
        valid_indices = []
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                continue
            valid_indices.append(i)

        for key in list(examples.keys()):
            if key.startswith("sg_"):
                model_inputs[key] = [examples[key][i] for i in valid_indices]
        return model_inputs

    SupervisedDatasetProcessor.preprocess_dataset = patched_preprocess


def patch_collator() -> None:
    orig_call = MultiModalDataCollatorForSeq2Seq.__call__

    def patched_call(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sg_payload: dict[str, list[Any]] = {}
        stripped_features = []
        for feature in features:
            feature = dict(feature)
            for key in list(feature.keys()):
                if key.startswith("sg_"):
                    sg_payload.setdefault(key, []).append(feature.pop(key))
            stripped_features.append(feature)

        batch = orig_call(self, stripped_features)
        batch.update(sg_payload)
        return batch

    MultiModalDataCollatorForSeq2Seq.__call__ = patched_call


def _pool_masked_hidden(hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    mask = token_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return F.normalize(pooled, dim=-1)


def _distributed_int_max(value: int, device: torch.device | str) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(value)

    tensor = torch.tensor([int(value)], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def _unwrap_model(model):
    return getattr(model, "module", model)


def _extract_hidden_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    return None


@contextmanager
def _capture_module_outputs(modules):
    payload: dict[str, list[torch.Tensor | None]] = {"outputs": [None] * len(modules)}
    handles = []

    for idx, module in enumerate(modules):
        def hook(_module, _inputs, output, index=idx):
            payload["outputs"][index] = _extract_hidden_tensor(output)

        handles.append(module.register_forward_hook(hook))
    try:
        yield payload
    finally:
        for handle in handles:
            handle.remove()


def _pool_hidden_list(hiddens: list[torch.Tensor | None], token_mask: torch.Tensor) -> torch.Tensor | None:
    pooled = []
    for hidden in hiddens:
        if hidden is not None:
            pooled.append(_pool_masked_hidden(hidden.float(), token_mask))
    if not pooled:
        return None
    if len(pooled) == 1:
        return pooled[0]
    return F.normalize(torch.stack(pooled, dim=0).mean(dim=0), dim=-1)


def _smma_hidden_size(base_model) -> int:
    config = getattr(base_model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    decoder_layers = resolve_decoder_layers(base_model)
    sample_layer = decoder_layers[0]
    q_proj = getattr(getattr(sample_layer, "self_attn", None), "q_proj", None)
    if q_proj is not None and hasattr(q_proj, "in_features"):
        return int(q_proj.in_features)

    raise AttributeError("Could not infer hidden size for SMMA-Pool.")


def _ensure_smma_modules(model) -> None:
    if not SMMA_ENABLED:
        return

    base_model = _unwrap_model(model)
    if hasattr(base_model, "smma_target_adapters"):
        return

    decoder_layers = resolve_decoder_layers(base_model)
    total_layers = len(decoder_layers)
    source_indices = resolve_layer_list(SMMA_SOURCE_LAYERS, total_layers, "SMMA_SOURCE_LAYERS")
    target_indices = resolve_layer_list(SMMA_TARGET_LAYERS, total_layers, "SMMA_TARGET_LAYERS")
    if max(source_indices) >= min(target_indices):
        raise ValueError("SMMA_SOURCE_LAYERS must be shallower than SMMA_TARGET_LAYERS.")

    hidden_size = _smma_hidden_size(base_model)
    adapters = torch.nn.ModuleDict(
        {
            str(index): ShallowMemoryAdapter(
                hidden_size=hidden_size,
                adapter_hidden_dim=SMMA_HIDDEN_DIM,
                init_gate=SMMA_INIT_GATE,
            )
            for index in target_indices
        }
    )
    ref_param = next(base_model.parameters(), None)
    if ref_param is not None:
        adapters.to(device=ref_param.device, dtype=ref_param.dtype)
    base_model.smma_source_layers = source_indices
    base_model.smma_target_layers = target_indices
    base_model.smma_target_adapters = adapters


def _smma_capture_modules(model, layer_indices: tuple[int, ...]):
    decoder_layers = resolve_decoder_layers(_unwrap_model(model))
    return [decoder_layers[index] for index in layer_indices]


@contextmanager
def _smma_injection(model, attention_mask: torch.Tensor | None):
    if not SMMA_ENABLED:
        yield
        return

    base_model = _unwrap_model(model)
    source_indices = getattr(base_model, "smma_source_layers", ())
    target_indices = getattr(base_model, "smma_target_layers", ())
    adapters = getattr(base_model, "smma_target_adapters", None)
    if not source_indices or not target_indices or adapters is None:
        yield
        return

    source_outputs: list[torch.Tensor | None] = [None] * len(source_indices)
    handles = []

    for source_pos, module in enumerate(_smma_capture_modules(model, source_indices)):
        def source_hook(_module, _inputs, output, index=source_pos):
            source_outputs[index] = _extract_hidden_tensor(output)

        handles.append(module.register_forward_hook(source_hook))

    for target_index, module in zip(target_indices, _smma_capture_modules(model, target_indices), strict=True):
        adapter = adapters[str(target_index)]

        def target_hook(_module, _inputs, output, target_adapter=adapter):
            hidden = _extract_hidden_tensor(output)
            if hidden is None:
                return output
            memory = pool_hidden_states(source_outputs, attention_mask, SMMA_POOL_MODE)
            if memory is None:
                return output
            updated_hidden = target_adapter(hidden, memory.to(device=hidden.device, dtype=hidden.dtype))
            return replace_primary_tensor(output, updated_hidden)

        handles.append(module.register_forward_hook(target_hook))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _prmlp_capture_modules(model):
    base_model = _unwrap_model(model)
    return [base_model.model.language_model.norm]


def _build_user_content(prompt_text: str, num_images: int) -> str:
    return ("<image>" * num_images) + prompt_text


def _encode_example(template, tokenizer, processor, prompt_text: str, images: list[Any], answer_text: str) -> dict[str, Any]:
    prompt = [{"role": "user", "content": _build_user_content(prompt_text, len(images))}]
    response = [{"role": "assistant", "content": answer_text}]
    messages = template.mm_plugin.process_messages(prompt + response, images, [], [], processor)
    input_ids, labels = template.mm_plugin.process_token_ids([], [], images, [], [], tokenizer, processor)
    encoded_pairs = template.encode_multiturn(tokenizer, messages, None, None)

    total_length = len(input_ids) + (1 if template.efficient_eos else 0)
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= PRMLP_CUTOFF_LEN:
            break
        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), PRMLP_CUTOFF_LEN - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if template.efficient_eos and turn_idx != 0:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len
        target_label = target_ids

        input_ids += source_ids + target_ids
        labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "images": images,
    }


def _move_batch_to_device(batch: dict[str, Any], device: torch.device, model_dtype: torch.dtype) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=model_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _first_or_self(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    if not pattern:
        return -1
    max_start = len(sequence) - len(pattern)
    for idx in range(max(0, start), max_start + 1):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def _get_answer_tag_token_ids(self) -> tuple[list[int], list[int]]:
    cached = getattr(self, "_pvrd_answer_tag_token_ids", None)
    if cached is not None:
        return cached
    tokenizer = self.data_collator.tokenizer
    start_ids = tokenizer.encode("<answer>", add_special_tokens=False)
    end_ids = tokenizer.encode("</answer>", add_special_tokens=False)
    self._pvrd_answer_tag_token_ids = (start_ids, end_ids)
    return start_ids, end_ids


def _mask_labels_to_answer_span(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if input_ids is None or labels is None or input_ids.ndim != 2 or labels.ndim != 2:
        return labels

    answer_start_ids, answer_end_ids = _get_answer_tag_token_ids(self)
    if not answer_start_ids:
        return labels

    masked_labels = labels.clone()
    batch_size = labels.size(0)
    for batch_idx in range(batch_size):
        supervised = labels[batch_idx].ne(IGNORE_INDEX)
        supervised_positions = supervised.nonzero(as_tuple=False).flatten()
        if supervised_positions.numel() == 0:
            continue

        ids = input_ids[batch_idx].detach().cpu().tolist()
        first_supervised = int(supervised_positions[0].item())
        last_supervised = int(supervised_positions[-1].item())
        start = _find_subsequence(ids, answer_start_ids, start=first_supervised)
        if start < 0:
            # Keep the original labels if the answer tag is absent or tokenized
            # unexpectedly. This avoids silently dropping all VTS supervision.
            continue

        end_search_start = start + len(answer_start_ids)
        end = _find_subsequence(ids, answer_end_ids, start=end_search_start)
        if end < 0:
            end = last_supervised + 1
            keep_end = end
        else:
            keep_end = end + len(answer_end_ids) if PVRD_VTS_ANSWER_INCLUDE_TAGS else end

        keep_start = start if PVRD_VTS_ANSWER_INCLUDE_TAGS else start + len(answer_start_ids)
        keep = torch.zeros_like(supervised, dtype=torch.bool)
        keep[max(first_supervised, keep_start) : min(labels.size(1), keep_end)] = True
        masked_labels[batch_idx] = torch.where(supervised & ~keep, torch.full_like(labels[batch_idx], IGNORE_INDEX), labels[batch_idx])

    return masked_labels


def _build_prmlp_dummy_views() -> tuple[list[Image.Image], list[Image.Image], list[int], list[int]]:
    # Non-PII ranks still need to participate in DDP for the online PRMLP forward.
    image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    prompt_box = [0, 0, 64, 64]
    prompt_image_size = [64, 64]
    return [image.copy()], [image.copy()], prompt_box, prompt_image_size


def patch_trainer() -> None:
    orig_init = CustomSeq2SeqTrainer.__init__
    orig_compute_loss = CustomSeq2SeqTrainer.compute_loss
    orig_save_model = CustomSeq2SeqTrainer.save_model

    def patched_init(self, finetuning_args, processor=None, model_args=None, gen_kwargs=None, **kwargs) -> None:
        orig_init(self, finetuning_args=finetuning_args, processor=processor, model_args=model_args, gen_kwargs=gen_kwargs, **kwargs)
        self._pvrd_sg_processor = processor
        self._pvrd_sg_embedding_cache: dict[str, torch.Tensor] = {}
        self._pvrd_teacher_cache = {}
        self._pvrd_answer_tag_token_ids = None
        _ensure_smma_modules(self.model)
        if PVRD_DISTILL_ENABLED and PVRD_TEACHER_CACHE_PATH:
            self._pvrd_teacher_cache = torch.load(PVRD_TEACHER_CACHE_PATH, map_location="cpu")

    def patched_save_model(self, output_dir=None, _internal_call=False):
        if SKIP_ROOT_FINAL_SAVE:
            target_dir = Path(output_dir or self.args.output_dir).resolve()
            root_dir = Path(self.args.output_dir).resolve()
            if target_dir == root_dir:
                print(f"[PVRD-SG] SKIP_ROOT_FINAL_SAVE=1: skipping final root save to {target_dir}", flush=True)
                return None
        return orig_save_model(self, output_dir=output_dir, _internal_call=_internal_call)

    def _build_single_view_batch(
        self,
        prompt_text: str,
        images: list[Any],
        answer_text: str,
        device: torch.device,
        model_dtype: torch.dtype,
        image_max_pixels: int | None = None,
        image_min_pixels: int | None = None,
    ) -> dict[str, Any]:
        with temporary_image_pixel_limits(
            self.data_collator.processor,
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
        ):
            feature = _encode_example(
                template=self.data_collator.template,
                tokenizer=self.data_collator.tokenizer,
                processor=self.data_collator.processor,
                prompt_text=prompt_text,
                images=images,
                answer_text=answer_text,
            )
            batch = self.data_collator([feature])
        return _move_batch_to_device(batch, device=device, model_dtype=model_dtype)

    def _load_target_embedding(self, emb_path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._pvrd_sg_embedding_cache.get(emb_path)
        if cached is None:
            cached = torch.load(emb_path, map_location="cpu")
            if not torch.is_tensor(cached):
                cached = torch.tensor(cached)
            cached = cached.float().cpu()
            self._pvrd_sg_embedding_cache[emb_path] = cached
        return F.normalize(cached.to(device=device, dtype=dtype), dim=-1).unsqueeze(0)

    def _compute_grounding_loss(
        self,
        model,
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        emb_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_model = _unwrap_model(model)
        image_token_id = getattr(base_model.config, "image_token_id", None)
        if image_token_id is None or last_hidden is None:
            return torch.zeros((), device=device, dtype=dtype)

        token_mask = input_ids.eq(image_token_id)
        if token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0)
        if not token_mask.any():
            return torch.zeros((), device=device, dtype=dtype)

        hidden = last_hidden.float()
        image_repr = _pool_masked_hidden(hidden, token_mask)
        target_repr = _load_target_embedding(self, emb_path, device=device, dtype=image_repr.dtype)
        return 1.0 - F.cosine_similarity(image_repr, target_repr, dim=-1).mean()

    def _answer_positions_from_labels(labels: torch.Tensor) -> torch.Tensor:
        positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
        return positions[positions > 0]

    def _compute_logit_distill_loss(
        self,
        outputs,
        labels: torch.Tensor,
        teacher_item: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        answer_positions = _answer_positions_from_labels(labels)
        if answer_positions.numel() == 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        teacher_topk_token_ids = teacher_item.get("topk_token_ids")
        teacher_topk_logits = teacher_item.get("topk_logits")
        if teacher_topk_token_ids is None or teacher_topk_logits is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        if torch.is_tensor(teacher_topk_token_ids):
            teacher_topk_token_ids = teacher_topk_token_ids.cpu()
        if torch.is_tensor(teacher_topk_logits):
            teacher_topk_logits = teacher_topk_logits.cpu()

        max_steps = min(answer_positions.numel(), int(teacher_topk_token_ids.shape[0]))
        if max_steps <= 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        student_logits = outputs.logits[0].float()
        kl_terms = []
        for idx in range(max_steps):
            pos = int(answer_positions[idx].item())
            student_pos = pos - 1
            candidate_ids = teacher_topk_token_ids[idx].to(device=device, dtype=torch.long)
            teacher_logits = teacher_topk_logits[idx].to(device=device, dtype=student_logits.dtype)
            student_subset = student_logits[student_pos].index_select(0, candidate_ids)
            teacher_probs = F.softmax(teacher_logits / PVRD_DISTILL_TEMPERATURE, dim=-1)
            student_log_probs = F.log_softmax(student_subset / PVRD_DISTILL_TEMPERATURE, dim=-1)
            kl_terms.append(
                F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (PVRD_DISTILL_TEMPERATURE ** 2)
            )

        if not kl_terms:
            return torch.zeros((), device=device, dtype=torch.float32)
        return torch.stack(kl_terms).mean()

    def _compute_hidden_distill_loss(
        self,
        outputs,
        labels: torch.Tensor,
        teacher_item: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        teacher_answer_hidden = teacher_item.get("answer_hidden")
        if teacher_answer_hidden is None or not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        answer_positions = _answer_positions_from_labels(labels)
        if answer_positions.numel() == 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        student_hidden = outputs.hidden_states[-1][0].float()
        student_repr = F.normalize(student_hidden.index_select(0, answer_positions).mean(dim=0), dim=-1)
        teacher_repr = teacher_answer_hidden.to(device=device, dtype=student_repr.dtype)
        teacher_repr = F.normalize(teacher_repr, dim=-1)
        return 1.0 - F.cosine_similarity(student_repr.unsqueeze(0), teacher_repr.unsqueeze(0), dim=-1).mean()

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        user_return_outputs = kwargs.get("return_outputs", False)
        sg_enabled = inputs.pop("sg_enabled", None)
        sg_crop_images = inputs.pop("sg_crop_images", None)
        sg_full_images = inputs.pop("sg_full_images", None)
        sg_prompt_box = inputs.pop("sg_prompt_box", None)
        sg_prompt_image_size = inputs.pop("sg_prompt_image_size", None)
        sg_text_embedding_path = inputs.pop("sg_text_embedding_path", None)
        sg_sample_id = inputs.pop("sg_sample_id", None)
        pvrd_enabled = inputs.pop("pvrd_distill_enabled", None)
        pvrd_sample_id = inputs.pop("pvrd_sample_id", None)

        enabled = bool(sg_enabled[0]) if isinstance(sg_enabled, list) and sg_enabled else bool(sg_enabled)
        emb_path = sg_text_embedding_path[0] if isinstance(sg_text_embedding_path, list) and sg_text_embedding_path else sg_text_embedding_path
        model_inputs = dict(inputs)
        if enabled and PVRD_VTS_ANSWER_ONLY_LOSS and "input_ids" in model_inputs and "labels" in model_inputs:
            model_inputs["labels"] = _mask_labels_to_answer_span(self, model_inputs["input_ids"], model_inputs["labels"])

        # PVRD-SG only needs the final language hidden state. Capturing the last
        # layer by hook avoids materializing all decoder-layer hidden states.
        need_hidden_states = PVRD_DISTILL_ENABLED and PVRD_HIDDEN_DISTILL_LAMBDA > 0.0
        if need_hidden_states:
            model_inputs["output_hidden_states"] = True
        forced_kwargs = dict(kwargs)
        forced_kwargs["return_outputs"] = True
        main_capture_modules = (
            _prmlp_capture_modules(model)
            if PVRD_SG_ENABLED or (PRMLP_ENABLED and PRMLP_ONLINE_VIEW == "main")
            else []
        )
        with _capture_module_outputs(main_capture_modules) as main_capture:
            with _smma_injection(model, model_inputs.get("attention_mask")):
                base_loss, outputs = orig_compute_loss(self, model, model_inputs, *args, **forced_kwargs)

        total_loss = base_loss

        pvrd_last_hidden = None
        if PVRD_SG_ENABLED or (PRMLP_ENABLED and PRMLP_ONLINE_VIEW == "main"):
            for hidden in main_capture["outputs"]:
                if hidden is not None:
                    pvrd_last_hidden = hidden
                    break
            if pvrd_last_hidden is None and outputs is not None and hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states
                if hidden_states is not None:
                    pvrd_last_hidden = hidden_states[-1]

        if enabled and PVRD_VTS_SFT_WEIGHT != 1.0:
            # VTS rows can be used mostly as prompt-region alignment data.
            # This reduces damage from noisy teacher reasoning while keeping
            # original replay rows at full SFT weight.
            total_loss = total_loss * PVRD_VTS_SFT_WEIGHT

        if PVRD_SG_ENABLED and enabled and emb_path and pvrd_last_hidden is not None:
            aux_loss = _compute_grounding_loss(
                self,
                model,
                last_hidden=pvrd_last_hidden,
                input_ids=model_inputs["input_ids"],
                emb_path=emb_path,
                device=base_loss.device,
                dtype=base_loss.dtype,
            )
            total_loss = total_loss + PVRD_SG_LAMBDA * aux_loss.to(base_loss.dtype)

        distill_on = bool(pvrd_enabled[0]) if isinstance(pvrd_enabled, list) and pvrd_enabled else bool(pvrd_enabled)
        sample_id = pvrd_sample_id[0] if isinstance(pvrd_sample_id, list) and pvrd_sample_id else pvrd_sample_id
        teacher_item = self._pvrd_teacher_cache.get(sample_id) if distill_on and sample_id else None
        labels = model_inputs.get("labels")
        if (
            PVRD_DISTILL_ENABLED
            and distill_on
            and teacher_item is not None
            and labels is not None
            and outputs is not None
            and (teacher_item.get("teacher_exact_match", False) or not PVRD_DISTILL_ONLY_EXACT)
        ):
            if PVRD_DISTILL_LAMBDA > 0.0:
                logit_loss = _compute_logit_distill_loss(self, outputs, labels, teacher_item, base_loss.device)
                total_loss = total_loss + PVRD_DISTILL_LAMBDA * logit_loss.to(base_loss.dtype)
            if PVRD_HIDDEN_DISTILL_LAMBDA > 0.0:
                hidden_loss = _compute_hidden_distill_loss(self, outputs, labels, teacher_item, base_loss.device)
                total_loss = total_loss + PVRD_HIDDEN_DISTILL_LAMBDA * hidden_loss.to(base_loss.dtype)

        prmlp_on = bool(sg_enabled[0]) if isinstance(sg_enabled, list) and sg_enabled else bool(sg_enabled)
        full_images = _first_or_self(sg_full_images)
        crop_images = _first_or_self(sg_crop_images)
        prompt_box = _first_or_self(sg_prompt_box)
        prompt_image_size = _first_or_self(sg_prompt_image_size)
        sample_id_for_mask = _first_or_self(sg_sample_id)

        if PRMLP_ENABLED:
            capture_modules = _prmlp_capture_modules(model)
            base_model = _unwrap_model(model)
            model_dtype = getattr(base_model, "dtype", torch.bfloat16)
            image_token_id = getattr(base_model.config, "image_token_id", None)
            vision_start_token_id = getattr(base_model.config, "vision_start_token_id", None)
            spatial_merge_size = getattr(base_model.config.vision_config, "spatial_merge_size", 1)
            trainer_step = int(getattr(getattr(self, "state", None), "global_step", 0) or 0)
            step_allowed = trainer_step % PRMLP_EVERY_N_STEPS == 0
            local_main_tokens = int(model_inputs.get("attention_mask", model_inputs["input_ids"].ne(0)).sum().item())
            global_main_tokens = _distributed_int_max(local_main_tokens, base_loss.device)
            length_allowed = PRMLP_MAX_MAIN_TOKENS <= 0 or global_main_tokens <= PRMLP_MAX_MAIN_TOKENS
            batch_allowed = step_allowed and length_allowed
            if (
                PRMLP_LOG_EVERY_N_STEPS > 0
                and trainer_step % PRMLP_LOG_EVERY_N_STEPS == 0
                and int(os.environ.get("RANK", "0")) == 0
            ):
                print(
                    "[PRMLP] "
                    f"step={trainer_step} online_view={PRMLP_ONLINE_VIEW} step_allowed={step_allowed} "
                    f"main_tokens_max={global_main_tokens} "
                    f"max_main_tokens={PRMLP_MAX_MAIN_TOKENS} "
                    f"length_allowed={length_allowed}",
                    file=sys.stderr,
                    flush=True,
                )

            can_prmlp_local = (
                batch_allowed
                and prmlp_on
                and model_inputs["input_ids"].size(0) == 1
                and (
                    PRMLP_ONLINE_VIEW == "main"
                    or (isinstance(full_images, list) and len(full_images) > 0)
                )
                and isinstance(crop_images, list)
                and len(crop_images) > 0
                and isinstance(prompt_box, list)
                and len(prompt_box) == 4
                and isinstance(prompt_image_size, list)
                and len(prompt_image_size) == 2
                and image_token_id is not None
                and vision_start_token_id is not None
            )
            can_prmlp_all = distributed_boolean_and(can_prmlp_local, device=base_loss.device)
            can_prmlp_any = distributed_boolean_or(can_prmlp_local, device=base_loss.device)
            _prmlp_debug(
                "enter enabled="
                f"{prmlp_on} local={can_prmlp_local} all={can_prmlp_all} any={can_prmlp_any} "
                f"sample_id={sample_id_for_mask} "
                f"full={len(full_images) if isinstance(full_images, list) else full_images is not None} "
                f"crop={len(crop_images) if isinstance(crop_images, list) else crop_images is not None} "
                f"box={prompt_box} size={prompt_image_size} "
                f"step={trainer_step} step_allowed={step_allowed} "
                f"main_tokens_local={local_main_tokens} main_tokens_global={global_main_tokens} "
                f"length_allowed={length_allowed} "
                f"input_shape={tuple(model_inputs['input_ids'].shape)}"
            )

            try:
                active_prompt_box = prompt_box
                active_prompt_image_size = prompt_image_size
                prmlp_image_views = None
                if can_prmlp_local:
                    if PRMLP_ONLINE_VIEW == "main":
                        clean_crop_images = [_load_rgb_image(image_path) for image_path in crop_images]
                        prmlp_image_views = ([], clean_crop_images)
                    else:
                        prmlp_image_views = maybe_build_prmlp_image_views(
                            enabled=can_prmlp_any,
                            full_image_paths=full_images if isinstance(full_images, list) else [],
                            crop_image_paths=crop_images if isinstance(crop_images, list) else [],
                            prompt_box=prompt_box,
                            sample_id=sample_id_for_mask,
                            mask_ratio=PRMLP_MASK_RATIO,
                            block_size=PRMLP_BLOCK_SIZE,
                            load_rgb_image=_load_rgb_image,
                            mask_fn=mask_prompt_region,
                        )
                elif can_prmlp_any:
                    masked_images, clean_crop_images, active_prompt_box, active_prompt_image_size = _build_prmlp_dummy_views()
                    prmlp_image_views = ([], clean_crop_images) if PRMLP_ONLINE_VIEW == "main" else (masked_images, clean_crop_images)

                if prmlp_image_views is None:
                    _prmlp_debug("skip extra PRMLP forwards")
                else:
                    masked_images, clean_crop_images = prmlp_image_views
                    masked_batch = None
                    if PRMLP_ONLINE_VIEW != "main":
                        masked_batch = _build_single_view_batch(
                            self,
                            PRMLP_PROMPT_TEXT,
                            masked_images,
                            PRMLP_DUMMY_ANSWER,
                            base_loss.device,
                            model_dtype,
                            image_max_pixels=PRMLP_IMAGE_MAX_PIXELS,
                            image_min_pixels=PRMLP_IMAGE_MIN_PIXELS,
                        )
                    crop_batch = _build_single_view_batch(
                        self,
                        PRMLP_PROMPT_TEXT,
                        clean_crop_images,
                        PRMLP_DUMMY_ANSWER,
                        base_loss.device,
                        model_dtype,
                        image_max_pixels=PRMLP_IMAGE_MAX_PIXELS,
                        image_min_pixels=PRMLP_IMAGE_MIN_PIXELS,
                    )

                    if masked_batch is not None:
                        masked_batch = {
                            key: value
                            for key, value in masked_batch.items()
                            if key not in {"labels", "output_hidden_states", "use_cache"}
                        }
                    crop_batch = {
                        key: value
                        for key, value in crop_batch.items()
                        if key not in {"labels", "output_hidden_states", "use_cache"}
                    }
                    _prmlp_debug(
                        f"built online_shape="
                        f"{tuple(model_inputs['input_ids'].shape) if PRMLP_ONLINE_VIEW == 'main' else tuple(masked_batch['input_ids'].shape)} "
                        f"crop_shape={tuple(crop_batch['input_ids'].shape)}"
                    )
                    local_extra_tokens = int(crop_batch["input_ids"].numel())
                    if masked_batch is not None:
                        local_extra_tokens = max(local_extra_tokens, int(masked_batch["input_ids"].numel()))
                    global_extra_tokens = _distributed_int_max(local_extra_tokens, base_loss.device)
                    skip_extra_forward = PRMLP_MAX_EXTRA_TOKENS > 0 and global_extra_tokens > PRMLP_MAX_EXTRA_TOKENS
                    if skip_extra_forward:
                        _prmlp_debug(
                            f"skip extra PRMLP forwards extra_tokens_global={global_extra_tokens} "
                            f"max_extra_tokens={PRMLP_MAX_EXTRA_TOKENS}"
                        )
                        masked_capture = {"outputs": []}
                        crop_capture = {"outputs": []}
                    else:
                        if PRMLP_ONLINE_VIEW == "main":
                            masked_capture = {"outputs": [pvrd_last_hidden]}
                        else:
                            with _capture_module_outputs(capture_modules) as masked_capture:
                                _prmlp_debug("before masked forward")
                                model(**masked_batch, use_cache=False, logits_to_keep=1)
                                _prmlp_debug("after masked forward")
                        with torch.no_grad():
                            with _capture_module_outputs(capture_modules) as crop_capture:
                                _prmlp_debug("before crop forward")
                                base_model(**crop_batch, use_cache=False, logits_to_keep=1)
                                _prmlp_debug("after crop forward")

                    if PRMLP_ONLINE_VIEW == "main":
                        masked_token_mask = build_prompt_band_mask(
                            model_inputs["input_ids"],
                            model_inputs.get("attention_mask"),
                            model_inputs.get("image_grid_thw"),
                            [active_prompt_box],
                            [active_prompt_image_size],
                            image_token_id=image_token_id,
                            vision_start_token_id=vision_start_token_id,
                            spatial_merge_size=spatial_merge_size,
                        )
                    else:
                        masked_token_mask = build_prompt_band_mask(
                            masked_batch["input_ids"],
                            masked_batch.get("attention_mask"),
                            masked_batch.get("image_grid_thw"),
                            [active_prompt_box],
                            [active_prompt_image_size],
                            image_token_id=image_token_id,
                            vision_start_token_id=vision_start_token_id,
                            spatial_merge_size=spatial_merge_size,
                        )
                    if not masked_token_mask.any():
                        masked_token_mask = (
                            model_inputs["input_ids"].eq(image_token_id)
                            if PRMLP_ONLINE_VIEW == "main"
                            else masked_batch["input_ids"].eq(image_token_id)
                        )
                    crop_token_mask = crop_batch["input_ids"].eq(image_token_id)
                    _prmlp_debug(
                        f"token_mask masked={int(masked_token_mask.sum().item())} "
                        f"crop={int(crop_token_mask.sum().item())}"
                    )

                    masked_repr = None
                    crop_repr = None
                    if masked_token_mask.any() and crop_token_mask.any():
                        masked_repr = _pool_hidden_list(masked_capture["outputs"], masked_token_mask)
                        crop_repr = _pool_hidden_list(crop_capture["outputs"], crop_token_mask)
                        _prmlp_debug(
                            f"repr masked={masked_repr is not None} crop={crop_repr is not None}"
                        )

                    zero_anchor = None
                    for hidden in masked_capture["outputs"]:
                        if hidden is not None:
                            zero_anchor = hidden.float().sum()
                            break

                    if can_prmlp_local and masked_repr is not None and crop_repr is not None and PRMLP_LAMBDA > 0.0:
                        prmlp_loss = 1.0 - F.cosine_similarity(masked_repr, crop_repr.detach(), dim=-1).mean()
                        total_loss = total_loss + PRMLP_LAMBDA * prmlp_loss.to(base_loss.dtype)
                        _prmlp_debug(f"loss={float(prmlp_loss.detach().cpu().item()):.6f}")
                    elif zero_anchor is not None:
                        total_loss = total_loss + zero_anchor.to(base_loss.dtype) * 0.0
                        _prmlp_debug("dummy zero-loss sync")
            except Exception as exc:
                _prmlp_debug(f"exception {type(exc).__name__}: {exc}")
                raise

        return (total_loss, outputs) if user_return_outputs else total_loss

    CustomSeq2SeqTrainer.__init__ = patched_init
    CustomSeq2SeqTrainer.compute_loss = patched_compute_loss
    CustomSeq2SeqTrainer.save_model = patched_save_model


def main() -> None:
    patch_converters()
    patch_supervised_processor()
    patch_collator()
    patch_trainer()
    run_exp()


if __name__ == "__main__":
    main()
