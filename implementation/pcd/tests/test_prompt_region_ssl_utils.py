from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "prompt_region_ssl_utils", SCRIPT_DIR / "prompt_region_ssl_utils.py"
)
assert SPEC is not None and SPEC.loader is not None
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def test_mask_prompt_region_only_changes_inside_box() -> None:
    image = Image.new("RGB", (8, 8), (0, 0, 0))
    masked = utils.mask_prompt_region(
        image,
        [2, 2, 6, 6],
        mask_ratio=1.0,
        block_size=2,
        seed=0,
        fill_color=(255, 255, 255),
    )

    for y in range(8):
        for x in range(8):
            pixel = masked.getpixel((x, y))
            if 2 <= x < 6 and 2 <= y < 6:
                assert pixel == (255, 255, 255)
            else:
                assert pixel == (0, 0, 0)


def test_build_prompt_band_mask_marks_first_prompt_row_tokens() -> None:
    input_ids = torch.tensor([[98, 99, 99, 99, 99, 99, 99, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    image_grid_thw = torch.tensor([[1, 2, 3]])

    mask = utils.build_prompt_band_mask(
        input_ids,
        attention_mask,
        image_grid_thw,
        prompt_boxes=[[0, 0, 300, 100]],
        prompt_image_sizes=[[300, 200]],
        image_token_id=99,
        vision_start_token_id=98,
        spatial_merge_size=1,
    )

    expected = torch.tensor([[False, True, True, True, False, False, False, False, False]])
    assert torch.equal(mask, expected)


def test_temporary_image_pixel_limits_override_and_restore() -> None:
    class DummyProcessor:
        image_max_pixels = 1024
        image_min_pixels = 64

    processor = DummyProcessor()

    with utils.temporary_image_pixel_limits(processor, image_max_pixels=256, image_min_pixels=16):
        assert processor.image_max_pixels == 256
        assert processor.image_min_pixels == 16

    assert processor.image_max_pixels == 1024
    assert processor.image_min_pixels == 64


def test_maybe_build_prmlp_image_views_skips_disabled_sample_without_loading() -> None:
    calls: list[str] = []

    def fail_loader(_path: str) -> Image.Image:
        calls.append("load")
        raise AssertionError("loader should not run when PRMLP is disabled")

    views = utils.maybe_build_prmlp_image_views(
        enabled=False,
        full_image_paths=["full.png"],
        crop_image_paths=["crop.png"],
        prompt_box=[0, 0, 10, 10],
        sample_id="sample-1",
        mask_ratio=0.35,
        block_size=16,
        load_rgb_image=fail_loader,
    )

    assert views is None
    assert calls == []


def test_maybe_build_prmlp_image_views_masks_first_full_image_and_loads_crop() -> None:
    seen_paths: list[str] = []

    def fake_loader(path: str) -> Image.Image:
        seen_paths.append(path)
        color = (10, 20, 30) if "full" in path else (200, 210, 220)
        return Image.new("RGB", (8, 8), color)

    def fake_mask(
        image: Image.Image,
        prompt_box,
        *,
        mask_ratio: float,
        block_size: int,
        seed: int,
    ) -> Image.Image:
        masked = image.copy()
        masked.putpixel((0, 0), (255, 0, 0))
        assert prompt_box == [1, 1, 6, 6]
        assert mask_ratio == 0.5
        assert block_size == 4
        assert isinstance(seed, int)
        return masked

    views = utils.maybe_build_prmlp_image_views(
        enabled=True,
        full_image_paths=["full_a.png", "full_b.png"],
        crop_image_paths=["crop_a.png"],
        prompt_box=[1, 1, 6, 6],
        sample_id="sample-2",
        mask_ratio=0.5,
        block_size=4,
        load_rgb_image=fake_loader,
        mask_fn=fake_mask,
    )

    assert views is not None
    masked_images, crop_images = views
    assert seen_paths == ["full_a.png", "full_b.png", "crop_a.png"]
    assert len(masked_images) == 2
    assert len(crop_images) == 1
    assert masked_images[0].getpixel((0, 0)) == (255, 0, 0)
    assert masked_images[1].getpixel((0, 0)) == (10, 20, 30)
    assert crop_images[0].getpixel((0, 0)) == (200, 210, 220)


def test_distributed_boolean_and_returns_local_value_without_dist() -> None:
    class FakeDist:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_initialized() -> bool:
            return False

    assert utils.distributed_boolean_and(True, device="cpu", dist_module=FakeDist()) is True
    assert utils.distributed_boolean_and(False, device="cpu", dist_module=FakeDist()) is False


def test_distributed_boolean_and_uses_global_min() -> None:
    class FakeReduceOp:
        MIN = "min"

    class FakeDist:
        ReduceOp = FakeReduceOp

        def __init__(self, final_value: int) -> None:
            self.final_value = final_value
            self.calls = 0
            self.last_op = None

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        def all_reduce(self, tensor: torch.Tensor, op=None) -> None:
            self.calls += 1
            self.last_op = op
            tensor.fill_(self.final_value)

    all_true = FakeDist(final_value=1)
    one_false = FakeDist(final_value=0)

    assert utils.distributed_boolean_and(True, device="cpu", dist_module=all_true) is True
    assert all_true.calls == 1
    assert all_true.last_op == FakeReduceOp.MIN

    assert utils.distributed_boolean_and(True, device="cpu", dist_module=one_false) is False
    assert one_false.calls == 1
    assert one_false.last_op == FakeReduceOp.MIN
