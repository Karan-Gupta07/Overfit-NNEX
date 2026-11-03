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
changes were tuned on other images only. The current coder, NNEX v5 at
commit `2063b53`, adds a context-mixing stage that was tuned with LPCB
images, so its LPCB results are not fresh; the fresh LPCB results are
NNEX v4's at `2db19a3`. LPCB was measured at `de30ed3`, and `2063b53`
re-encoded the three images it changes to the same files. The fresh Tecnick
photos were re-measured at `2063b53`, and no v5 setting was chosen on them.

Commit hashes before `7eb6a2c` refer to the development history, which was
squashed into this repository's first commit when it was made public.

| codec | test images | vs best baseline | wins |
|---|---|---|---:|
| photo coder v5 | all 107 LPCB images (tuned with them) | **17.39% fewer bytes** | 107/107 |
| photo coder v5 | 62 LPCB camera photos (tuned with LPCB) | 16.35% fewer bytes (median 15.81%) | 62/62 |
| photo coder v4 | 62 fresh LPCB camera photos, 10 to 16 Mpx | **16.24% fewer bytes** (median 15.81%) | 62/62 |
| photo coder v4 | all 107 fresh LPCB images (photos and NASA images) | 15.21% fewer bytes | 99/107 |
| photo coder v5 | 16 fresh Tecnick studio photos, 512×512 crops | **29.02% fewer bytes** (median 28.28%) | 16/16 |
| photo coder v5 | 4 fresh Tecnick studio photos, 2400×2400 | **23.28% fewer bytes** (median 23.90%) | 4/4 |
| photo coder, round 2 settings | 24 Tecnick photos, fresh at the time | 24.68% fewer bytes | 24/24 |
| photo coder, round 1 settings | 12 fresh noisy aerial photos, 512×512 | 4.83% fewer bytes | 12/12 |

paq8px -9L, a research compressor, leads the Lossless Photo Compression
Benchmark (LPCB). Against its published LPCB sizes, v5's files are 3.75%
smaller in total over all 107 images and 4.04% smaller over the 62 camera
photos, smaller on every photo and every star field. They are also smaller
in total than those of cmix (5.76%) and GraLIC (12.08%), the next two
entries. paq8px still wins 17 of the 25 other NASA images, though over those
25 ours is 0.22% smaller in total. Fresh, v4 was 1.21% smaller than paq8px
over all 107 and 3.91% over the photos. On the 20 fresh Tecnick photos v5's
files are 1.32% (crops) and 1.75% (2400×2400) smaller than paq8px's, which
still wins 5 of the 16 crops.

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
   take 2.2 to 7.5 kB per fresh crop and 16 to 18 kB per 2400×2400 photo.
5. **Code and check.** An integer-only C++ program runs the network for every
   sample and arithmetic-codes the true value. The file is then decoded
   independently to prove it is exact.
6. **Mix (NNEX v5).** Code the samples again through a context-mixing model
   that takes the network's probability of every bit as one input, beside
   hashed contexts of the neighbours, runs of equal samples and the
   network's recent errors. It learns as it codes, so decoding repeats it.
   The mixed file is kept only if it saves at least 0.002 bytes per pixel.
7. **Pick.** Keep the smallest complete file, NNEX or the SCDX floor.

No pretrained model or shared dictionary is involved. An ARM64 build and an
x86_64 build (under Rosetta on the same Mac) decoded every fresh Tecnick file
and 17 of the v5 LPCB files to the same pixels and re-encoded each to the
same bytes.

## Fresh Tecnick test, totals in bytes

16 centered 512×512 crops and four full 2400×2400 images (the same scenes as
four of the crops), from TESTIMAGES photographs that earlier rounds left
unused, coded by v5 at `2063b53`.

| codec | 16 crops | our saving | 4 native | our saving |
|---|---:|---:|---:|---:|
| PNG | 3,738,879 | 59.4% | 15,427,276 | 58.5% |
| WebP lossless | 2,733,334 | 44.4% | 11,848,508 | 46.0% |
| JPEG XL effort 9 | 2,147,237 | 29.2% | 8,905,825 | 28.1% |
| JPEG XL effort 10 | 2,140,363 | 29.0% | 8,346,560 | 23.3% |
| paq8px -9L | 1,539,595 | 1.32% | 6,518,112 | 1.75% |
| ours, v4 at `da035fe` | 1,523,239 | 0.26% | 6,404,016 | 0.00% |
| **ours** | **1,519,289** | | **6,403,866** | |

Against JPEG XL effort 10, per-image savings range from 23.1% to 37.2% on
the crops and 19.5% to 25.5% on the native images. The native files are 150
bytes smaller than v4's in total.

## LPCB, totals in bytes

The 107 images of the
[Lossless Photo Compression Benchmark](https://qlic.altervista.org/LPCB.html):
62 camera photos (10 to 16 Mpx) and 45 NASA images, one file per image. Five
images over 16.8 Mpx were coded as two or three strips inside their file.
paq8px's size is LPCB's published result; the rest were measured here, and
every file decoded exactly. "v5" is the current coder, measured at
`de30ed3` (`2063b53` re-encoded the three images it changes to the same
files) and tuned with these images; "v4, fresh" is `2db19a3`, which had
never seen them.

| codec | 62 camera photos | v5 saving | all 107 | v5 saving |
|---|---:|---:|---:|---:|
| PNG | 969,833,223 | 37.6% | 1,453,911,150 | 41.2% |
| WebP lossless | 807,732,126 | 25.1% | 1,198,793,394 | 28.7% |
| JPEG XL effort 9 | 743,950,384 | 18.7% | 1,071,591,474 | 20.3% |
| JPEG XL effort 10 | 723,529,248 | 16.4% | 1,034,707,323 | 17.4% |
| GraLIC 1.11 (published) | 680,583,590 | 11.1% | 971,986,179 | 12.1% |
| cmix v16 (published) | 645,835,881 | 6.29% | 906,796,260 | 5.76% |
| paq8px -9L (published) | 630,710,768 | 4.04% | 887,823,381 | 3.75% |
| ours, v4, fresh | 606,034,580 | 0.14% | 877,080,059 | 2.57% |
| **ours, v5** | **605,200,445** | | **854,571,285** | |

v5 is smaller than JPEG XL effort 10 on all 107 images and smaller than
paq8px on all 62 photos and all 20 star fields. paq8px still wins 17 of the
25 other NASA images (renders, composites, processed views), mostly small
ones where our stored weights are most of the difference; over those 25 ours
is 0.22% smaller in total, against 22.7% larger for v4. The v5 files are in
the [`lpcb-2063b53` release](https://github.com/Karan-Gupta07/Overfit-NNEX/releases/tag/lpcb-2063b53)
and the v4 files in the
[`lpcb-2db19a3` release](https://github.com/Karan-Gupta07/Overfit-NNEX/releases/tag/lpcb-2db19a3),
each with a script that checks them against LPCB's MD5 sums. See the
[v5 notebook](docs/experiments/2026-10-31-lpcb-v5.md) and the
[v4 notebook](docs/experiments/2026-10-27-lpcb.md).

## Against paq8px

paq8px leads the
[Lossless Photo Compression Benchmark](https://qlic.altervista.org/LPCB.html).
It blends hundreds of adaptive models as it codes, so its decoder repeats all
of the encoder's work. This run used paq8px v217 at `-9L`, and every archive
decompressed exactly.

| | 16 crops | 4 native |
|---|---:|---:|
| ours vs paq8px, total | 1.32% smaller | 1.75% smaller |
| images where ours is smaller | 11/16 | 4/4 |
| per-image range | 4.2% smaller to 1.7% larger | 0.6% to 2.9% smaller |
| paq8px time per image | about 8 min each way | 2.2 to 2.7 hours each way |
| our time per image | 11 s to encode, 0.17 s to decode | 51 s to encode, 3.6 s to decode |

The stored weights are 2.0% to 6.7% of a crop file but only 0.5% to 1.1% of
a native file, which is most of why small images are the weak point.

## Speed

Measured on one Apple M4 Pro laptop (12 cores, 16-core GPU), one image at a
time:

| step | 512×512 | 2400×2400 | 10 to 16 Mpx LPCB photo |
|---|---:|---:|---:|
| encode | median 10.7 s | median 50.8 s | median 47.3 s |
| decode (8 threads) | median 0.17 s | median 3.57 s | median 6.0 s |

Training on the GPU is most of the encode time. Images up to 4 Mpx, and
cleaner larger ones, race two network widths and train both side by side
while they are close, as on all four 2400×2400 Tecnick images; since
`de30ed3` the narrower width stops after an eighth of training if the wider
one leads, so those encode in 51 s (40.0 s at `da035fe`, 64 s before
`de30ed3`). Most crops train one width; they took 8.6 s at `da035fe` and
now spend about 1.5 s more trying mixing. The LPCB photos encode and decode
faster than at `2db19a3` (54.1 s and 9.1 s). Files that keep the network
alone decode roughly three to six times slower than single-threaded JPEG XL;
mixed files are
slower again (the four mixed LPCB photos took 13 to 28 s). See the
[round 3 notebook](docs/experiments/2026-10-26-fresh-tecnick.md) and the
[LPCB v5 notebook](docs/experiments/2026-10-31-lpcb-v5.md).

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
- Encoding needs a GPU to be fast (about 11 s per 512×512 crop and about 50 s
  per 2400×2400 image or 10 to 16 Mpx photo on an M4 Pro), and
  decoding is slower than JPEG XL. `--device cpu` and `--device cuda`
  (NVIDIA) encode without Metal with the same settings, but much more slowly:
  247 to 344 s per 512×512 crop on the M4 Pro's CPU, 197 to 228 s on an
  NVIDIA L4, and 5 to 36 minutes for an LPCB photo on an L4 or 4-core CPUs,
  with files within 0.4% of Metal's
  ([Encoding without Metal](scripts/research/local_probability/README.md#encoding-without-metal)).
  The released LPCB files decode exactly on macOS, Linux and Windows. Files
  that mix (NNEX v5) decode several
  times slower again, and the largest need several GB of memory to decode.
- Files decode only with the research decoder above (`compress.py --decode`).
- v4 was tested fresh on 20 studio photos and on LPCB's 107 images (62
  camera photos from four cameras, 45 NASA images). v5 was tuned with LPCB
  images, so its LPCB results are not fresh; on the fresh Tecnick photos it
  was 0.26% smaller than v4 on the crops and the same size on the 2400×2400
  images (150 bytes smaller), and took about 1.2 and 1.3 times as long to
  encode. Small processed and composited images are a
  weak point: paq8px still wins 17 of LPCB's 25 non-star NASA images. Art
  with alpha and lossy targets are untested.
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
  [LPCB](docs/experiments/2026-10-27-lpcb.md),
  [LPCB with v5](docs/experiments/2026-10-31-lpcb-v5.md).

Where the design document and the measurements disagree, the measurements
win.

## Stack

Python 3.12+ with NumPy, Pillow, zstandard and CBOR. Optional: OpenCV (symbol
analysis in the floor), imagecodecs (JPEG XL), PyTorch (training only). The
backend needs a C++17 compiler with 128-bit integer support.
