"""The Tecnick cohort rule is fixed by sorted names and never by results."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"


@pytest.fixture
def fetch(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import fetch_tecnick

    return fetch_tecnick


def test_selection_is_disjoint_and_name_ordered(fetch):
    names = [f"{fetch.PREFIX}{i:02}.png" for i in range(40)][::-1]
    native, cropped = fetch.select(names)
    assert native == [f"{fetch.PREFIX}{i:02}.png" for i in range(0, 36, 3)]
    assert cropped == [f"{fetch.PREFIX}{i:02}.png" for i in range(1, 36, 3)]
    assert not set(native) & set(cropped)
    with pytest.raises(ValueError):
        fetch.select(names[:39])


def test_center_crop_is_exact(fetch):
    pixels = np.arange(2400 * 2400 * 3, dtype=np.uint32).reshape(2400, 2400, 3)
    crop = fetch.center_crop(pixels)
    assert crop.shape == (512, 512, 3)
    assert np.array_equal(crop, pixels[944:1456, 944:1456])


def test_changed_archive_is_rejected(fetch, tmp_path):
    with pytest.raises(ValueError, match="archive changed"):
        fetch.prepare(b"not the pinned archive", tmp_path)


def test_unexpected_members_are_rejected(fetch):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tar:
        data = b"x"
        info = tarfile.TarInfo("SAMPLING/other.png")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="unexpected archive member"):
        fetch.read_sources(buffer.getvalue())
