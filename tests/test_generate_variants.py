from __future__ import annotations
from conftest import REPO_ROOT, run_script

from pathlib import Path
import json


def test_generate_variants_cli_with_mocked_ollama(fake_ollama_dir, tmp_path: Path):
    script = REPO_ROOT / "scripts" / "generate_variants.py"
    assert script.exists(), "scripts/generate_variants.py not found"

    # --- Prepare input ---
    input_path = tmp_path / "prompts.json"
    output_path = tmp_path / "variants.json"

    prompts = [
        {"id": "p1", "text": "Explain recursion with an example."},
        {"id": "p2", "text": "How do I reverse a linked list?"},
    ]
    input_path.write_text(json.dumps(prompts), encoding="utf-8")

    # --- Run script ---
    result = run_script(
        script,
        [
            "--input", str(input_path),
            "--output", str(output_path),
            "--seed", "7",
        ],
        fake_ollama_dir,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists(), "Expected variants.json to be created"

    # --- Load output ---
    data = json.loads(output_path.read_text(encoding="utf-8"))

    # --- Basic structure checks ---
    assert isinstance(data, list)
    assert len(data) == len(prompts)

    expected_methods = {"noise", "adversarial", "emotional", "jailbreak", "irrelevant"}

    for record in data:
        # schema
        assert {"id", "base", "variants"} <= set(record.keys())
        assert set(record["variants"].keys()) == expected_methods

        base = record["base"]
        variants = record["variants"]

        # --- Behavioral sanity checks ---

        # noise should differ from base
        assert variants["noise"] != base

        # adversarial + jailbreak should include base prompt
        assert base in variants["adversarial"]
        assert base in variants["jailbreak"]

        # irrelevant should be longer than base
        assert len(variants["irrelevant"]) > len(base)

        # emotional should go through mocked ollama path
        emotional_text = variants["emotional"]
        assert emotional_text.startswith("MOCK_EMOTION::"), (
            "Expected emotional variant to be generated via mocked Ollama"
        )

