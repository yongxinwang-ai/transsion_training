from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "shallow_mm_memory_utils", SCRIPT_DIR / "shallow_mm_memory_utils.py"
)
assert SPEC is not None and SPEC.loader is not None
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def test_resolve_layer_list_parses_unique_indices() -> None:
    indices = utils.resolve_layer_list(" 4, 8,4, 12 ", total_layers=16, env_name="SMMA_SOURCE_LAYERS")
    assert indices == (4, 8, 12)


def test_pool_hidden_states_mean_respects_attention_mask() -> None:
    hidden_a = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    hidden_b = torch.tensor([[[5.0, 6.0], [7.0, 8.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])

    pooled = utils.pool_hidden_states([hidden_a, hidden_b], attention_mask, pool_mode="mean")

    expected_a = torch.tensor([[2.0, 3.0]])
    expected_b = torch.tensor([[6.0, 7.0]])
    expected = (expected_a + expected_b) / 2.0
    assert torch.allclose(pooled, expected)


def test_shallow_memory_adapter_gate_controls_update_strength() -> None:
    hidden = torch.randn(2, 3, 6)
    memory = torch.randn(2, 6)

    almost_off = utils.ShallowMemoryAdapter(hidden_size=6, adapter_hidden_dim=4, init_gate=-20.0)
    off_output = almost_off(hidden, memory)
    assert torch.allclose(off_output, hidden, atol=1e-6)

    on_adapter = utils.ShallowMemoryAdapter(hidden_size=6, adapter_hidden_dim=4, init_gate=20.0)
    on_output = on_adapter(hidden, memory)
    assert on_output.shape == hidden.shape
    assert not torch.allclose(on_output, hidden)
