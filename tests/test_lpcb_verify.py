from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"


@pytest.fixture
def verify(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    import verify_lpcb

    return verify_lpcb


def test_coded_files_are_checked_against_published_ppm_md5(verify, tmp_path, monkeypatch):
    pixels = np.random.default_rng(5).integers(0, 256, (5, 7, 3), np.uint8)
    ppm = b"P6\n7 5\n255\n" + pixels.tobytes()
    source = tmp_path / "photo.ppm"
    source.write_bytes(ppm)
    files = tmp_path / "files"
    files.mkdir()
    blob, _ = verify.experiment.compress(source, widths=(), hybrid=False)
    (files / "photo.coded").write_bytes(blob)
    assert verify.ppm_bytes(verify.experiment.decompress(blob)) == ppm
    sums = tmp_path / "md5sums.txt"
    for listed, code in (([hashlib.md5(ppm).hexdigest()], 0), (["0" * 32], 1)):
        sums.write_text("".join(f"photo.ppm {md5}\n" for md5 in listed))
        monkeypatch.setattr(sys, "argv", ["verify_lpcb", str(files), "--md5sums", str(sums)])
        with pytest.raises(SystemExit) as exit_:
            verify.main()
        assert exit_.value.code == code
    sums.write_text(f"photo.ppm {hashlib.md5(ppm).hexdigest()}\nother.ppm {'0' * 32}\n")
    monkeypatch.setattr(sys, "argv", ["verify_lpcb", str(files), "--md5sums", str(sums)])
    with pytest.raises(SystemExit) as exit_:
        verify.main()
    assert exit_.value.code == 1
