from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_ollama_dir(tmp_path: Path) -> Path:
    """
    Creates a fake `ollama` module that supports common APIs:
    - chat(...)
    - embed(...)
    - embeddings(...)
    - Client().chat/embed/embeddings
    """
    mod_dir = tmp_path / "fake_modules"
    mod_dir.mkdir(parents=True, exist_ok=True)

    (mod_dir / "ollama.py").write_text(
        # language=python
        """
from __future__ import annotations

class _Message:
    def __init__(self, content: str):
        self.content = content

class _ChatResponse:
    def __init__(self, content: str):
        self.message = _Message(content)

def _vec_for_text(text: str, dim: int = 8):
    # deterministic lightweight embedding
    base = sum(ord(c) for c in text) or 1
    return [((base + i * 17) % 100) / 100.0 for i in range(dim)]

def chat(model: str, messages: list[dict], **kwargs):
    user_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_text = m.get("content", "")
    return _ChatResponse(f"MOCK_EMOTION::{user_text}")

def embed(model: str, input, **kwargs):
    if isinstance(input, str):
        return {"embeddings": [_vec_for_text(input)]}
    return {"embeddings": [_vec_for_text(x) for x in input]}

def embeddings(model: str, prompt: str, **kwargs):
    return {"embedding": _vec_for_text(prompt)}

class Client:
    def chat(self, *args, **kwargs):
        return chat(*args, **kwargs)

    def embed(self, *args, **kwargs):
        return embed(*args, **kwargs)

    def embeddings(self, *args, **kwargs):
        return embeddings(*args, **kwargs)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return mod_dir


def run_script(script_path: Path, args: list[str], fake_module_dir: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(fake_module_dir)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )

