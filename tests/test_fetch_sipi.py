from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/local_probability"
spec = importlib.util.spec_from_file_location(
    "_sipi_research_fetch", SOURCE / "fetch_sipi_holdout.py"
)
assert spec is not None and spec.loader is not None
fetcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fetcher
spec.loader.exec_module(fetcher)


def archive(*, duplicate=False, extra_path=False):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for index, name in enumerate(fetcher.NAMES):
            encoded = io.BytesIO()
            pixels = np.full((256, 256, 3), index * 11, np.uint8)
            Image.fromarray(pixels).save(encoded, format="TIFF")
            data = encoded.getvalue()
            member = tarfile.TarInfo("misc/" + name + ".tif")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
            if duplicate and index == 0:
                tar.addfile(member, io.BytesIO(data))
        if extra_path:
            member = tarfile.TarInfo("../outside.txt")
            member.size = 7
            tar.addfile(member, io.BytesIO(b"changed"))
    return buf.getvalue()


def test_sipi_exact_conversion_is_immutable_and_does_not_extract_paths(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")
    blob = archive(extra_path=True)
    out = tmp_path / "cohort"
    manifest = fetcher.convert_archive(blob, out)
    assert len(manifest["images"]) == 14
    assert outside.read_bytes() == b"untouched"
    assert fetcher.convert_archive(blob, out) == manifest
    for index, record in enumerate(manifest["images"]):
        with Image.open(out / record["image"]) as image:
            assert np.array_equal(np.asarray(image), np.full((256, 256, 3), index * 11, np.uint8))


def test_sipi_changed_images_and_manifests_are_preserved(tmp_path):
    blob = archive()
    manifest = fetcher.convert_archive(blob, tmp_path)
    first = tmp_path / manifest["images"][0]["image"]
    first.write_bytes(b"user work")
    with pytest.raises(FileExistsError):
        fetcher.convert_archive(blob, tmp_path)
    assert first.read_bytes() == b"user work"
    manifest["archive_sha256"] = "changed-by-user"
    target = tmp_path / "manifest.json"
    changed = json.dumps(manifest).encode()
    target.write_bytes(changed)
    with pytest.raises(ValueError, match="changed"):
        fetcher.convert_archive(blob, tmp_path)
    assert target.read_bytes() == changed


def test_sipi_duplicate_member_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        fetcher.convert_archive(archive(duplicate=True), tmp_path)


def test_sipi_expansion_guard_precedes_skipping_unknown_member(tmp_path):
    member = tarfile.TarInfo("unknown/large.bin")
    member.size = fetcher.MAX_MEMBER + 1
    blob = gzip.compress(member.tobuf() + b"\0" * 1024)
    with pytest.raises(ValueError, match="expansion"):
        fetcher.convert_archive(blob, tmp_path)


def test_explicit_cohort_keeps_source_and_selection_provenance(tmp_path):
    cohort = fetcher.Cohort(
        tuple(fetcher.NAMES[:2]),
        "https://example.invalid/archive",
        "https://example.invalid/listing",
        "Fixed before measurements",
        fetcher.MAX_ARCHIVE,
    )
    manifest = fetcher.convert_archive(archive(), tmp_path, cohort=cohort)
    assert len(manifest["images"]) == 2
    assert manifest["source"] == cohort.url
    assert manifest["selection"] == cohort.selection
    assert manifest["source_page"] == cohort.source_page


def test_cohort_native_size_constraint(tmp_path):
    cohort = fetcher.Cohort(
        tuple(fetcher.NAMES[:2]),
        "https://example.invalid/archive",
        "https://example.invalid/listing",
        "Native 512 only",
        fetcher.MAX_ARCHIVE,
        sizes=((512, 512),),
    )
    with pytest.raises(ValueError, match="unexpected source pixels"):
        fetcher.convert_archive(archive(), tmp_path, cohort=cohort)
