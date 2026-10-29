from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_kodak.py"
spec = importlib.util.spec_from_file_location("fetch_kodak", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def png(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (768, 512), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_fetch_manifest_and_unchanged_pixels(tmp_path) -> None:
    data = png("white")
    with patch.object(
        module.urllib.request,
        "urlopen",
        side_effect=lambda *a, **k: io.BytesIO(data),
    ):
        module.fetch(tmp_path, count=4)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["images"]) == 4
    assert [r["split"] for r in manifest["images"]] == ["development"] * 3 + ["heldout"]
    assert (tmp_path / "kodim01.png").read_bytes() == data


def test_fetch_refuses_changed_image(tmp_path) -> None:
    original = png("black")
    (tmp_path / "kodim01.png").write_bytes(original)
    with (
        patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(png("white"))),
        pytest.raises(FileExistsError),
    ):
        module.fetch(tmp_path, count=1)
    assert (tmp_path / "kodim01.png").read_bytes() == original


def test_fetch_rejects_non_image(tmp_path) -> None:
    with (
        patch.object(module.urllib.request, "urlopen", return_value=io.BytesIO(b"<html>")),
        pytest.raises(ValueError),
    ):
        module.fetch(tmp_path, count=1)
    assert not list(tmp_path.glob("*.png"))
