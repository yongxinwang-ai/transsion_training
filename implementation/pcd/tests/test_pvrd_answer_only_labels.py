from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

LLAMAFACTORY_SRC = Path('/mnt/weka/home/yongxin.wang/workspace/LlamaFactory/src')
SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(LLAMAFACTORY_SRC))
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location('run_pvrd_sg_llamafactory', SCRIPT_DIR / 'run_pvrd_sg_llamafactory.py')
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == '<answer>':
            return [11, 12]
        if text == '</answer>':
            return [13, 14]
        raise AssertionError(text)


class DummyCollator:
    tokenizer = DummyTokenizer()


class DummyTrainer:
    data_collator = DummyCollator()
    _pvrd_answer_tag_token_ids = None


def test_mask_labels_to_answer_span_keeps_only_answer_tags_and_content() -> None:
    trainer = DummyTrainer()
    input_ids = torch.tensor([[1, 2, 11, 12, 42, 43, 13, 14, 99]])
    labels = torch.tensor([[-100, 2, 11, 12, 42, 43, 13, 14, 99]])

    masked = runner._mask_labels_to_answer_span(trainer, input_ids, labels)

    expected = torch.tensor([[-100, -100, 11, 12, 42, 43, 13, 14, -100]])
    assert torch.equal(masked, expected)
    assert torch.equal(labels, torch.tensor([[-100, 2, 11, 12, 42, 43, 13, 14, 99]]))


def test_mask_labels_to_answer_span_falls_back_when_answer_tag_missing() -> None:
    trainer = DummyTrainer()
    input_ids = torch.tensor([[1, 2, 42, 43, 99]])
    labels = torch.tensor([[-100, 2, 42, 43, 99]])

    masked = runner._mask_labels_to_answer_span(trainer, input_ids, labels)

    assert torch.equal(masked, labels)
