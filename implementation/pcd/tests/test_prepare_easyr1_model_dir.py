from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_easyr1_model_dir.py"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_easyr1_model_dir_supplements_processor_sidecars() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "checkpoint-1000"
        fallback = root / "base_model"
        output = root / "compat"
        source.mkdir()
        fallback.mkdir()

        write_json(
            source / "config.json",
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "rope_parameters": {
                        "rope_type": "default",
                        "mrope_section": [16, 24, 24],
                        "mrope_interleaved": True,
                    }
                },
            },
        )
        write_json(source / "tokenizer_config.json", {"extra_special_tokens": ["<image>"]})
        (source / "model.safetensors").write_bytes(b"stub")

        for name in (
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
            "merges.txt",
        ):
            (fallback / name).write_text(f"{name}\n", encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source-model",
                str(source),
                "--output-dir",
                str(output),
                "--fallback-dir",
                str(fallback),
                "--overwrite",
            ],
            check=True,
        )

        for name in (
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
        ):
            assert (output / name).exists()

        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert config["text_config"]["rope_scaling"]["mrope_section"] == [16, 24, 24]

        tokenizer = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert tokenizer["additional_special_tokens"] == ["<image>"]
        assert tokenizer["extra_special_tokens"] == {}
