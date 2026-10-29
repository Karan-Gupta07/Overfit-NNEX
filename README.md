# Overfit-NNEX

Research into lossless photo compression that beats the standard formats.
The photo coder overfits a small neural network to each image, stores the
weights in the file (an NNEX file), and arithmetic-codes every pixel with the
network's predicted probabilities. Decoding uses integer-only C++ and needs
nothing but the file. This is research code under
[`scripts/research/local_probability/`](scripts/research/local_probability/).

It is lossless: every decoded file is checked sample for sample against the
source. All file sizes are complete files, including headers, checksums and
every trained weight.

The supporting Python package is named `semanticcodec` because the project
began as a screenshot codec.

## Results at a glance

"Best baseline" is the smallest of PNG, WebP lossless and JPEG XL lossless
(efforts 9 and 10), chosen per image. "Fresh" images were never used for
tuning: the encoder was committed before it first ran on them, and later
changes were tuned on other images only. LPCB was last measured at commit
`2db19a3`; the fresh Tecnick numbers were re-measured at `da035fe`.

Commit hashes like these refer to the development history, which was
squashed into this repository's first commit when it was made public. The
photo coder's encoder and decoder are unchanged since `2db19a3` apart from
comments.

| codec | test images | vs best baseline | wins |
|---|---|---|---:|
| photo coder | 62 fresh LPCB camera photos, 10 to 16 Mpx | **16.24% fewer bytes** (median 15.81%) | 62/62 |
| photo coder | all 107 fresh LPCB images (photos and NASA images) | 15.21% fewer bytes | 99/107 |
| photo coder | 16 fresh Tecnick studio photos, 512×512 crops | **28.83% fewer bytes** (median 28.28%) | 16/16 |
| photo coder | 4 fresh Tecnick studio photos, 2400×2400 | **23.27% fewer bytes** (median 23.90%) | 4/4 |
| photo coder, round 2 settings | 24 Tecnick photos, fresh at the time | 24.68% fewer bytes | 24/24 |
| photo coder, round 1 settings | 12 fresh noisy aerial photos, 512×512 | 4.83% fewer bytes | 12/12 |

paq8px -9L, a research compressor, leads the Lossless Photo Compression
Benchmark (LPCB). Against its published LPCB sizes, the photo coder's files
are 1.21% smaller in total over all 107 images, and 3.91% smaller over the 62
camera photos, smaller on every one. They are also smaller in total than
those of cmix (3.28%) and GraLIC (9.76%), the next two entries. paq8px wins 22
of the 25 non-star NASA images. On the 20 fresh Tecnick photos the photo
coder's files are 1.06% (crops) and 1.75% (2400×2400) smaller than paq8px's,
which still wins 5 of the 16 crops.

The Tecnick photos are clean (about 1 to 1.4 bits per sample at the best
baseline) and the LPCB photos noisier (2.6), so the same absolute saving
shows up as a larger percentage on Tecnick. Every number above comes from
[benchmarks](docs/benchmarks.md), which links each stored result file.

## How it works

1. **Floor.** Encode the image with the conventional SCDX codec in
   `src/semanticcodec`. That complete file is the fallback. A photograph
   first gets a quick JPEG XL floor; the full one is built only if the
   network's early size estimate is not at least 10% smaller.
2. **Fit.** For every sample, solve two small least-squares predictions over
   its causal window (the rows above and the samples to its left). The decoder
   repeats these fits exactly.
3. **Train.** For each colour plane, train a small network on this image
   only, on the GPU. It sees decoded neighbours near and far, the planes
   decoded before it, recent prediction errors, and both fits. It outputs a
   mixture of four logistic distributions over the 256 possible values.
4. **Shrink.** Quantization-aware passes and a per-tensor grid search round
   the weights to whatever grid makes the file smallest. The stored weights
   take 2.4 to 7.7 kB per fresh crop and 16 to 18 kB per 2400×2400 photo.
5. **Code and check.** An integer-only C++ program runs the network for every
   sample and arithmetic-codes the true value. The file is then decoded
   independently to prove it is exact.
6. **Pick.** Keep the smallest complete file, NNEX or the SCDX floor.

No pretrained model or shared dictionary is involved. An ARM64 build and an
x86_64 build (under Rosetta on the same Mac) decoded every fresh file to the
same pixels and re-encoded it to the same bytes.

## Fresh Tecnick test, totals in bytes

16 centered 512×512 crops and four full 2400×2400 images (the same scenes as
four of the crops), from TESTIMAGES photographs that earlier rounds left
unused.

| codec | 16 crops | our saving | 4 native | our saving |
|---|---:|---:|---:|---:|
| PNG | 3,738,879 | 59.3% | 15,427,276 | 58.5% |
| WebP lossless | 2,733,334 | 44.3% | 11,848,508 | 46.0% |
| JPEG XL effort 9 | 2,147,237 | 29.1% | 8,905,825 | 28.1% |
| JPEG XL effort 10 | 2,140,363 | 28.8% | 8,346,560 | 23.3% |
| paq8px -9L | 1,539,595 | 1.06% | 6,518,112 | 1.75% |
| **ours** | **1,523,239** | | **6,404,016** | |

Against JPEG XL effort 10, per-image savings range from 21.6% to 37.2% on
the crops and 19.4% to 25.5% on the native images.

## LPCB, totals in bytes

The 107 images of the
[Lossless Photo Compression Benchmark](https://qlic.altervista.org/LPCB.html),
measured at `2db19a3` with no setting chosen on them: 62 camera photos (10 to
16 Mpx) and 45 NASA images, one file per image. Five images over 16.8 Mpx
were coded as two or three strips inside their file. paq8px's size is LPCB's
published result; the rest were measured here. Every file decoded exactly,
and all 107 are in the [`lpcb-2db19a3` release](https://github.com/Karan-Gupta07/Overfit-NNEX/releases/tag/lpcb-2db19a3) with a script
that checks them against LPCB's MD5 sums.

| codec | 62 camera photos | our saving | all 107 | our saving |
|---|---:|---:|---:|---:|
| PNG | 969,833,223 | 37.5% | 1,453,911,150 | 39.7% |
| WebP lossless | 807,732,126 | 25.0% | 1,198,793,394 | 26.8% |
| JPEG XL effort 9 | 743,950,384 | 18.5% | 1,071,591,474 | 18.2% |
| JPEG XL effort 10 | 723,529,248 | 16.2% | 1,034,707,323 | 15.2% |
| paq8px -9L | 630,710,768 | 3.91% | 887,823,381 | 1.21% |
| **ours** | **606,034,580** | | **877,080,059** | |

Ours is smaller than JPEG XL effort 10 and paq8px on every one of the 62
photos. paq8px wins 22 of the 25 non-star NASA images (renders, composites,
processed views), where ours is 22.7% larger in total. See the
[LPCB notebook](docs/experiments/2026-10-27-lpcb.md).

## Against paq8px

paq8px leads the
[Lossless Photo Compression Benchmark](https://qlic.altervista.org/LPCB.html).
It blends hundreds of adaptive models as it codes, so its decoder repeats all
of the encoder's work. This run used paq8px v217 at `-9L`, and every archive
decompressed exactly.

| | 16 crops | 4 native |
|---|---:|---:|
| ours vs paq8px, total | 1.06% smaller | 1.75% smaller |
| images where ours is smaller | 11/16 | 4/4 |
| per-image range | 4.2% smaller to 2.8% larger | 0.6% to 2.9% smaller |
| paq8px time per image | about 8 min each way | 2.2 to 2.7 hours each way |
| our time per image | 9 s to encode, 0.2 s to decode | 40 s to encode, 4.8 s to decode |

The stored weights are 2.0% to 6.8% of a crop file but only 1.0% to 1.1% of
a native file, which is most of why small images are the weak point.

## Speed

Measured on one Apple M4 Pro laptop (12 cores, 16-core GPU), one image at a
time:

| step | 512×512 | 2400×2400 | 12 Mpx LPCB photo |
|---|---:|---:|---:|
| encode | median 8.6 s | median 40.0 s | median 54 s |
| decode (NNEX v4, 8 threads) | median 0.18 s | median 4.77 s | median 9.3 s |

Training on the GPU is most of the encode time. Decoding is roughly four to
nine times slower than single-threaded JPEG XL. See the
[round 3 notebook](docs/experiments/2026-10-26-fresh-tecnick.md) and, for the
23 LPCB photos of 12.3 to 12.4 Mpx, the
[LPCB notebook](docs/experiments/2026-10-27-lpcb.md).

## Run it

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,cv,hybrid,dev]"
mkdir -p data/research-local-probability
c++ -O3 -std=c++17 -Wall -Wextra -Werror -pthread \
  scripts/research/local_probability/nnproto_integer.cpp \
  -o data/research-local-probability/nnproto-integer

python scripts/research/local_probability/compress.py photo.png \
  --out photo.coded --receipt photo.json
python scripts/research/local_probability/compress.py photo.coded \
  --decode --out photo-decoded.png
```

PyTorch is needed for training only. `cwebp` supplies the strongest WebP
floor when installed. Options, formats and limits are in the
[research README](scripts/research/local_probability/README.md) and
[design 15](docs/design/15-image-local-probability.md).

## Limits

- 8-bit RGB and RGBA stills only. Images with transparency skip training and
  use the SCDX floor.
- Encoding needs a GPU to be fast (about 10 s per 512×512 crop, under a
  minute per 2400×2400 image and about a minute per 10 to 16 Mpx photo on an
  M4 Pro), and decoding is slower than JPEG XL.
- Files decode only with the research decoder above (`compress.py --decode`).
- The current settings were tested fresh on 20 studio photos and on LPCB's
  107 images (62 camera photos from four cameras, 45 NASA images). Processed
  and composited images are a weak point: paq8px wins 22 of LPCB's 25
  non-star NASA images. Art with alpha and lossy targets are untested.
- cmix and GraLIC are compared only through LPCB's published sizes. The
  published neural coders have not been measured.
- Images over 16,777,216 pixels, up to four times that, are coded as
  horizontal strips inside one file. Each strip has its own network, and its
  first rows cannot see the strip above, so strips cost a little.

## Tests

```bash
pytest -q
ruff check src scripts tests
mypy src
```

Benchmark reproduction commands are in [benchmarks](docs/benchmarks.md), the
experiment notebooks, and the research README. Corpora download into the
ignored `data/` directory and are not redistributed.

## Repository layout

```
scripts/research/local_probability/
               photo coder: training, integer C++ backend, validation scripts
src/semanticcodec/
  core/        geometry (q2 fixed point), colour, scene graph, validation
  render/      deterministic rasteriser shared by encoder and decoder
  format/      .scdx container, serialisation, symbol tables
  fallback/    raster patch codecs (PNG, WebP, JPEG XL)
  encode/      SCDX floor encoder: symbol extraction, region partitioning
  decode/      SCDX decoder
scripts/       Kodak corpus downloader
tests/         pytest suite
docs/          benchmarks, design document, experiment notebooks, results
```

## Documentation

- [Benchmarks](docs/benchmarks.md): every measured result in one place.
- [Design 15, image-local probability coding](docs/design/15-image-local-probability.md).
- Photo coder notebooks: [round 1](docs/experiments/2026-10-23-local-probability.md),
  [round 2](docs/experiments/2026-10-24-neural-training.md),
  [encode speed](docs/experiments/2026-10-25-encode-speed.md),
  [round 3 and the fresh Tecnick test](docs/experiments/2026-10-26-fresh-tecnick.md),
  [LPCB](docs/experiments/2026-10-27-lpcb.md).

Where the design document and the measurements disagree, the measurements
win.

## Stack

Python 3.12+ with NumPy, Pillow, zstandard and CBOR. Optional: OpenCV (symbol
analysis in the floor), imagecodecs (JPEG XL), PyTorch (training only). The
backend needs a C++17 compiler with 128-bit integer support.
