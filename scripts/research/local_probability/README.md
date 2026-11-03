# Image-local learned probability experiment

Research code, **not a stable release or file format**.

A small network predicts RGB sample probabilities. All image-local trained
coefficients are transmitted with arithmetic-coded samples. A separate C++
process decodes using only that file, without PyTorch, the source image,
pretrained weights, external fonts, or other image-specific assets.

This is established predictive/neural entropy coding, not a claim to have
invented minimum-rate prediction or logistic mixtures. No third-party predictor
or arithmetic-coder implementation was copied.

## Current status

- The original frozen width-32 model decoded all 14 fresh SIPI miscellaneous
  images exactly. It won on six larger images but lost on eight smaller ones:
  only **0.24% aggregate savings** against the measured bare PNG/WebP/JXL best.
- Integer inference and a pixel checksum now replace floating inference by
  default. All 17 original models produce identical encoded bytes on ARM64 and
  x86_64 **under Rosetta on the same Mac**. This is not testing on two physical
  machines or a broad portability certification.
- Width/precision selection improves the **known-input regression**. Actual
  cached neural candidates plus newly encoded SCDX floors total 3,088,562 bytes
  on the 14 SIPI inputs, versus 3,240,758 bare-baseline bytes, with 9/14 wins.
  This replay includes all selected model bytes, but excludes training from
  timing. It is not a new holdout.
- A separate **fresh** twelve-image native aerial cohort uses the frozen
  [protocol](../../../docs/experiments/2026-10-23-neural-aerial-protocol.json).
  The full encoder won **12/12**, 5,242,156 vs 5,508,321 bare-best bytes:
  **4.83% aggregate / 4.43% median savings**, all RGBA exact. All eight learned
  candidates per image independently decoded, and all twelve winners
  cross-encoded identically under Rosetta.
  [Full fresh receipt](../../../docs/results/2026-10-23-neural-aerial.json).
- Round 2 (NNEX v3 model coding, far causal taps, four components, grouped
  cosine training, tiling above 1 Mpx) used only known development images.
  A separately frozen [Tecnick protocol](../../../docs/experiments/2026-10-24-tecnick-protocol.json)
  then measured 24 fresh studio photographs: **24/24 wins**, 30,359,801 vs
  40,308,146 best-baseline bytes, **24.68% aggregate / 24.01% median savings**
  against PNG, WebP, and JPEG XL efforts 9 and 10. All samples were exact, and
  all 24 files re-encoded identically under Rosetta. These images are very clean
  (about 1.5 bits per sample), which inflates percentages relative to noisier
  photos. [Durable summary](../../../docs/results/2026-10-24-tecnick.json).
  Those 24 images have been development data since.
- Round 3 (fused Metal trainer, per-column and per-row weight grids, input
  pruning, two causal-window least-squares fits) was first measured at commit
  `6c0131e` on a fresh cohort of 16 Tecnick crops and four 2400×2400 images
  (`fetch_tecnick_fresh.py`, `fresh_validation.py`), and re-measured at
  `a82b98a` and `da035fe` after changes tuned on development images only. At
  `da035fe`, against JPEG XL effort 10, the best baseline on all 20, it saved
  **28.83%** on the crops (16/16) and **23.27%** on the native images (4/4).
  Against paq8px -9L it was 1.06% smaller on the crops (11/16 images) and 1.75%
  smaller on the native images (4/4). All 20 files decoded exactly and
  re-encoded identically under Rosetta.
  [Durable summary](../../../docs/results/2026-10-27-tecnick-fresh-da035fe.json).
  NNEX v5 at `ea118a7`, never tuned on these images, saved 29.07% and
  23.31% against JPEG XL effort 10 and was 0.33% and 0.04% smaller than
  `da035fe`, but encoded 2.4 and 1.6 times slower
  ([summary](../../../docs/results/2026-10-31-tecnick-fresh-ea118a7.json)).
  `10fbde6` changed the width race (chosen on development crops and LPCB)
  and halved the crop encodes; one crop grew by 1,095 bytes. It saved
  **29.02%** and **23.31%** against JPEG XL effort 10, was 1.32% (11/16) and
  1.79% (4/4) smaller than paq8px and 0.26% and 0.04% smaller than
  `da035fe`, and took about 1.2 and 1.6 times as long to encode
  ([summary](../../../docs/results/2026-10-31-tecnick-fresh-10fbde6.json)).
  `de30ed3` and `2063b53` (a later width race and a grey-image screen, chosen
  on development images and LPCB) left the crop files unchanged and cut the
  native encodes from a median 63.8 s to 50.8 s; the native ducks grew by
  2,500 bytes. At `2063b53` it saves **29.02%** and **23.28%** against JPEG
  XL effort 10, is 1.32% (11/16) and 1.75% (4/4) smaller than paq8px and
  0.26% and 0.002% smaller than `da035fe`, and takes about 1.25 times as long
  to encode; all 20 files again decoded exactly and re-encoded identically
  under Rosetta.
  [Durable summary](../../../docs/results/2026-10-31-tecnick-fresh-2063b53.json).
- On the 107 images of the Lossless Photo Compression Benchmark (LPCB), never
  used before, commit `299bc4c` came out **1.21%** smaller in total than
  LPCB's published paq8px v216 -9L sizes (81/107 images), 3.28% smaller than
  cmix v16 and 9.76% smaller than GraLIC 1.11. On the 62 camera photos it was
  **3.91%** smaller than paq8px and **16.24%** smaller than JPEG XL effort 10,
  the best standard baseline, and smaller than both on all 62; paq8px won 22
  of the 25 non-star NASA images. All files decoded exactly. The re-run at
  `2db19a3` codes each image as one file (the five largest as strips inside
  it), one image at a time: 877,080,059 bytes, 54 s median encode and 9.1 s
  median decode per camera photo.
  [Notebook](../../../docs/experiments/2026-10-27-lpcb.md),
  [durable summary](../../../docs/results/2026-10-28-lpcb-2db19a3.json),
  [coded files](https://github.com/Karan-Gupta07/Overfit-NNEX/releases/tag/lpcb-2db19a3).
- NNEX v5 (context mixing on top of the network) was developed with LPCB
  images, mostly the NASA ones, so its LPCB results are **not fresh**. At
  `de30ed3` it coded the 107 images in 854,571,285 bytes: **3.75%** smaller
  than paq8px -9L (90/107 images), 5.76% smaller than cmix and 12.08%
  smaller than GraLIC. The photos came out 4.04% smaller than paq8px
  (62/62), the star fields 3.90% (20/20) and the other NASA images 0.22%
  (paq8px still smaller on 17 of 25); every image is smaller than its best
  standard baseline. A photo took a median 47.3 s to encode and 6.0 s to
  decode, faster both ways than v4 at `2db19a3`; mixed files decode several
  times slower than the network alone. All 107 files match LPCB's MD5 sums
  (`verify_lpcb.py`). `2063b53` re-encoded the three images it changes to
  the same files.
  [Notebook](../../../docs/experiments/2026-10-31-lpcb-v5.md),
  [durable summary](../../../docs/results/2026-10-31-lpcb-2063b53.json).

All results are collected in [benchmarks](../../../docs/benchmarks.md).
See [design](../../../docs/design/15-image-local-probability.md), notebooks for
[round 1](../../../docs/experiments/2026-10-23-local-probability.md),
[round 2](../../../docs/experiments/2026-10-24-neural-training.md) and
[round 3](../../../docs/experiments/2026-10-26-fresh-tecnick.md), and
[durable results](../../../docs/results/2026-10-23-local-probability.json).

## Build

From the repository root (use the project's Python 3.13 virtual environment):

```bash
pip install -e ".[ml,cv,hybrid,dev]"
mkdir -p data/research-local-probability
c++ -O3 -std=c++17 -Wall -Wextra -Werror -pthread \
  scripts/research/local_probability/nnproto_integer.cpp \
  -o data/research-local-probability/nnproto-integer
```

A C++17 compiler with signed/unsigned 128-bit integer extensions and
`std::thread` is required.
The `hybrid` extra supplies JPEG XL; `cwebp -lossless -z 9 -exact` supplies the
strongest measured WebP floor. Record fallback behavior if `cwebp` is absent.
PyTorch is needed only for training, not decoding.

## Encode and decode with actual file competition

```bash
python scripts/research/local_probability/compress.py image.png \
  --out data/probe/image.coded --receipt docs/results/raw/image.json
python scripts/research/local_probability/compress.py data/probe/image.coded \
  --decode --out data/probe/reconstructed.png
```

The output is the winning complete file, unchanged: either SCDX or integer
NNEX v5 (`--envelope 4` writes v4, `--envelope 3` v3). Older NNEX v2, v3 and
v4 files still decode.
There is no extra selection envelope. Use this research decoder for NNEX
outputs.

Defaults: feature set 4 (twelve far causal taps, the plane's level, row, column
and diagonal parity, and two causal-window least-squares fits, each with its
spread and errors), four logistic components, and causal error context. The
encoder gets the fits of the whole image once from `nnproto-integer local-fit`
and reuses them for every complete file it codes, so the backend must be built
before training with feature set 3 or 4.
Hidden widths follow image size: 12 and 32 up to 1,000,000 pixels, 24 and 48
up to 4,000,000, and 48 above, unless the order-0 entropy of MED residuals on
every fourth row puts the image under 4,000,000 bytes (24 and 48 again: weights
are a large share of such files once mixed). Training runs 800 epochs, capped at 51,200 × 8192
samples (73 epochs at 2400×2400), with Adam at 0.015, 5% warmup and cosine
decay to a 1% floor, seed 20261023, four CPU threads, and MPS when available.
Images up to 1,000,000 pixels use batch 4096, a per-plane clip norm of 30 and
first-layer input pruning; larger ones use batch 8192, clip norm 10 and no
pruning. The three color planes train together in one batched step with
independent weights (`sequential=True` in `compress()` restores the older
per-plane loop).

A fused trainer runs the steps (Metal kernels with MPS, otherwise a PyTorch
copy on a CUDA GPU or the CPU; see [Encoding without Metal](#encoding-without-metal)), then a quantization-aware
snapshot on 12-bit grids. An estimated per-plane search picks each tensor's
grid (one grid, or one per column or row) and codes the searched model
as a complete file, plus the best uniform-grid model when its estimate is within
0.05%. Estimates spend bits as the integer coder does: its logistic is exactly
0 or 1 beyond ±16, and every value keeps one of 2^20 counts. (Training's
objective clamps the mass of the coded range instead, which comes out negative
for means far below the range, and searches that used it chose such roundings.)
Parallel widths race: after a 32nd of their epochs (at least two) only widths
whose estimate (recent mean training bits plus 10-bit model bytes) is within
3% of the best keep training. For v5, so does a width within 3% of the best
estimate that counts 30% of those bits, and a width narrower than the one with
that best estimate does while that model's weights make up a tenth of it. The
widths left race again after an eighth of their epochs: a width stops there if
a wider one's estimate is lower, unless that wider model's weights make up 5%
of its estimate counting 30% of the bits. Each width that finishes then mixes.
The PyTorch path (`--torch-training`)
instead tries weight grids labelled 16/12/10/8/7/6 bits, in that order,
stopping after the first file that grows. Every file is independently
verified before it is kept. Quantized tensors are entropy-coded exact binary16
values; these labels are **not literal packed-integer tensor formats**.

With v5 (the default), images whose sample values leave gaps, and whose
planes each use at least 90% of the values all three use, train on and code
each sample's rank in that set. The smallest learned file then codes again
through a context-mixing model (`context_mixing.hpp`): the network's
probability of every bit is one input to gated mixers beside hashed context
nodes and bit histories, followed by a second mixer layer, adaptive
probability maps and a learned blend. Its contexts include the runs of equal
samples that end at the sample from the left and from above, and the network's
recent errors (coded sample less its most likely value) around the sample. A
plane that leaves at least eight sample values unused also skips every bit
that its used values decide. The mixed file is kept if it saves at least 0.002
bytes per pixel, because it decodes several times slower. The model it mixes
comes from a second weight search that counts 30% of each network stream bit,
since the context model makes up most of what coarser weights cost the
network; that model is also coded alone as one more finalist. Every width that
finished training mixes, and the candidates code side by side while their
context tables fit in half the machine's memory. Images over 2,000,000 pixels
first code a centred band of a 32nd of their rows, after 32 rows of lead-in
whose bytes are subtracted, with the best network alone and mixed, and skip
mixing unless it saves 0.001 bytes per pixel there. Images up to 1,000,000
pixels try ungated and gated mixing; up to 2,000,000 pixels, one band and two
column bands per plane. Larger images mix in one band (two over 8,000,000
pixels), and in one more if mixing saved under 10% of the network's bytes on
the screened rows: each band decodes on threads of its own, but its model
learns alone. Candidates whose encoders and decoders fit in memory at once
are checked as they are coded: a decoder reads each band stream as it is
written, on a canvas of its own, so the file needs no final decode, and a
strip file decodes only its strips that were not. R and B planes that nearly
repeat G (under one bit per sample of order-0 entropy in their difference
from G) are also coded with their networks cut to a constant, which wins on
grey images; when the best file is the network alone, the constant file also
mixes, over 2,000,000 pixels only if its own screened band saves 0.001 bytes
per pixel. A strip of an image over the pixel limit may code with another
strip's model instead of its own when that file comes out smaller.

Each width trains in its own spawned process while the parent builds the SCDX
floor; `--serial` trains one at a time and writes the same bytes. The native
backend evaluates encoder probabilities on all cores and decodes v4 and
network-only v5 files with a multithreaded row wavefront; a mixed v5 file
decodes its planes G, R and B (and their bands) on concurrent threads, each a
few rows behind the last. `--threads N` caps both (0 chooses).
`--compile-training` is faster on MPS but trains slightly different models, so
it is a different encoder setting rather than a speedup of the default. On six
known development images it cut encode time by about a third and changed the
total size by −0.04% (−0.26% to +0.12% per image); see the
[speed notebook](../../../docs/experiments/2026-10-25-encode-speed.md).

Choose the smallest complete candidate against an actual balanced hybrid SCDX
floor, no region partitioning. Ties retain the floor. A parallel fused encode
of an image with at least 4,096 sampled colours first builds a JPEG XL effort
7 floor and skips the full one when the learned models' first estimates are
at least 10% smaller (`SCREEN_FLOOR` in `compress.py`). `--partition-depth 2`
also enables the existing bounded spatial search. `--floor-only` skips training;
`--no-hybrid` disables JPEG XL. `--without-floor` is diagnostic, not a guarantee.

Only still, 8-bit RGB/RGBA inputs are accepted. Nonopaque images skip training
and retain the conventional SCDX path, which preserves alpha and hidden RGB.
Inputs are limited to 16,777,216 pixels and 64 MiB source files. Above the
training budget (8,388,608 pixels for the fused trainer, 1,048,576 for
PyTorch), training uses evenly spread 256×256 tiles (with six-pixel causal
margins) up to that budget; every pixel is still coded.
Supported source files are PNG, JPEG, TIFF, WebP, and BMP. Pixel losslessness
does not preserve original file bytes, EXIF/ICC metadata, or animation; grayscale
and higher bit depths are not accepted. Decoding is several times slower than
JPEG XL, and encoding trains a network per image. On an M4 Pro the fresh
Tecnick run at `2063b53` took a median 10.7 s to encode a 512×512 crop and
50.8 s for a 2400×2400 image, and 0.17 s and 3.57 s to decode them. These are
not statistically stable latency benchmarks.

## Encoding without Metal

`--device cpu` encodes on the CPU, and `--device cuda` on an NVIDIA GPU (or
an AMD GPU under PyTorch's ROCm build, which presents it as a CUDA device;
untested). `--device auto`, the default, picks Metal, then a CUDA GPU, then
the CPU:

```bash
python scripts/research/local_probability/compress.py image.png --device cpu \
  --out data/probe/image.coded --receipt data/probe/image.json
python scripts/research/local_probability/compress.py data/probe/image.coded \
  --decode --out data/probe/image-decoded.png
```

It uses the same settings as a Metal encode: training budget, batch, clip
norm, input pruning, width race, quantization-aware snapshots and weight
search. Only the step and the search's batched estimates run in PyTorch
(`cpu_training.py`) instead of the Metal kernels. The PyTorch step computes the
same objective with hand-written gradients. Tests check them against autograd
and check 20 steps against the Metal trainer, and against the CPU on a CUDA
GPU. The arithmetic rounds differently, so the models are close to Metal's,
not identical. `--torch-training` still selects
the older PyTorch path (1,048,576 training pixels, batch 8192, uniform weight
grids, no race). The files decode with the same backend, without PyTorch or
a GPU. `--threads N` caps only the native backend; each width trains on four
PyTorch threads.

### On the M4 Pro's CPU

Measured at `878abae` plus this change (Metal files are byte-identical to
before it) on the M4 Pro used for the benchmarks (12 cores: 8 performance and
4 efficiency, 24 GB), macOS 27.0.1, PyTorch 2.13.0, on AC power with other
apps open. Three development crops were measured first. Then the fresh
Tecnick baloons crop was encoded once, with nothing tuned on it.

| 512×512 crop | CPU file | Metal file | CPU vs Metal | encode, CPU / Metal | width kept, CPU / Metal |
|---|---:|---:|---:|---:|---:|
| apples (development) | 88,353 B | 88,438 B | −0.10% | 290 s / 12 s | 12 / 12 |
| flowers (development) | 89,435 B | 89,288 B | +0.16% | 252 s / 11 s | 12 / 12 |
| screws (development) | 140,466 B | 140,482 B | −0.01% | 344 s / 24 s | 32 / 32 |
| baloons (fresh) | 79,329 B | 79,017 B | +0.39% | 247 s / 11.0 s | 12 / 12 |

All four CPU files are complete NNEX v5 files and decoded to exactly the
source samples. On every crop the race kept the same width as on Metal, and
the file kept the network alone (no mixing). Encode times are wall time for
the whole command, including interpreter start and the file check. The
baloons Metal file is the published `2063b53` result; the other Metal files
were encoded on the same day as the CPU files.

On baloons, training took 191 s and the quantization-aware snapshot plus
weight search took 53 s, of a 247 s encode. The CPU file's weights take 2,720
bytes (2,894 for Metal). It decodes in 0.16 s in-process (median of three;
0.135 s for the Metal file). Peak memory was 2.0 to 2.1 GB on every crop,
summed over the encoder's process tree and sampled every 0.2 s. The file is
27.2% smaller than JPEG XL effort 10 (108,978 bytes; 27.5% for Metal).

The older `--torch-training` path, which `--device cpu` used before this
change, gave 81,749 bytes on baloons (3.46% over Metal) in 199 s. It kept
uniform 12-bit weights (5,262 bytes) and a mixed layout that decodes about
eight times slower. The fused settings take about a quarter longer on the CPU
and make a 2.96% smaller file.

### On Linux and Windows

A GitHub Actions workflow (`.github/workflows/cpu-portability.yml`, run on
demand) checks the CPU path at `f26792c` on x86_64 Linux (GCC 15.2) and
Windows (MinGW-Builds GCC 15.2 under Git Bash; MSVC lacks the 128-bit
integers the backend needs), on GitHub's 4-vCPU, 16 GB runners with PyTorch
2.13.0. On both, the backend builds with `-Wall -Wextra -Werror`, the trainer
tests pass, and all 107 released `lpcb-2063b53` files decode to PPMs whose
MD5s match LPCB's `md5sums.txt` (1.5 to 2.3 h per run). One LPCB photo,
`canon_eos_1100d_01` (recovered from its released file), was encoded on the
CPU with default settings and decoded exactly:

| machine | file | vs Metal (8,301,756 B) | encode | peak memory |
|---|---:|---:|---:|---:|
| Linux runner, first run | 8,294,695 B | −0.09% | 983 s | 8.66 GB |
| Linux runner, second run | 8,294,695 B | −0.09% | 1,542 s | 8.66 GB |
| Windows runner | 8,324,106 B | +0.27% | 2,186 s | 6.23 GB |
| M4 Pro, Metal (`de30ed3`) | 8,301,756 B | | 47.3 s | |

The two Linux runs gave the same file; the second ran beside three other
jobs. Peak memory is summed over the encoder's process tree.

### On an NVIDIA GPU

Measured at `659349a` plus the commit that captures the CUDA step as a graph,
on a Modal container: NVIDIA L4 (24 GB), 8 CPU cores, 32 GB, Linux, PyTorch
2.13.0 with CUDA 13.0. The crops used `--device cuda`; the photo used
`--device auto`, which picked the GPU.

| image | CUDA file | Metal file | CUDA vs Metal | encode, CUDA / M4 CPU / Metal | GPU memory |
|---|---:|---:|---:|---:|---:|
| apples crop | 88,395 B | 88,438 B | −0.05% | 109 s / 290 s / 12 s | 1.4 GB |
| flowers crop | 89,351 B | 89,288 B | +0.07% | 110 s / 252 s / 11 s | 1.4 GB |
| screws crop | 140,328 B | 140,482 B | −0.11% | 213 s / 344 s / 24 s | 2.0 GB |
| `canon_eos_1100d_01` | 8,301,209 B | 8,301,756 B | −0.01% | 256 s / – / 47.3 s | 6.0 GB |

Every file decoded exactly, kept the network alone and kept Metal's width
(12, 12, 32 and 48). On CUDA, `cpu_training.py` captures each step shape as a
CUDA graph and replays it, which gives the same parameters as running the
step's kernels one by one (a test checks this) and halved the crops' encode
times (197 to 202 s before; screws, which trains two widths side by side,
went from 228 s to 213 s, and the photo from 300 s to 256 s). Steps on
random data:

| step | one by one | graph | fused Metal |
|---|---:|---:|---:|
| width 12, batch 4096 | 2.96 ms | 1.47 ms | about 0.13 ms |
| width 48, batch 8192 | 3.06 ms | 2.89 ms | about 0.6 ms |

The width-48 step was already limited by the GPU's own work, not by
launches: PyTorch runs it as dozens of separate kernels that each read and
write whole intermediate tensors, where the Metal kernels fuse them.
### Limits

- About 20 times slower to encode than Metal on the M4 Pro's CPU, and 5 to 10
  times slower on the L4 GPU. Most of the time is the training step. A full
  LPCB run on a 4-core CPU would take days; none has been run.
- A few images on a few machines, not a stable latency benchmark. AMD (ROCm),
  Intel GPUs and ARM64 Linux are untested.
- Encoding is not reproducible across machines. Floating-point training can
  differ between CPUs, GPUs, operating systems, PyTorch builds and the Metal
  trainer, which changes the weights, the selected model and the file size.
  An encode off Metal is not expected to reproduce the published files or
  sizes; the files above were within 0.4% of them.
- Decoding is what must be reproducible: a given file must decode to the
  same pixels on every supported machine. All 107 LPCB files decode exactly
  on ARM64 macOS, x86_64 Linux and x86_64 Windows, and two were checked with
  an x86_64 build under Rosetta.
- To check a published size, decode the released file and compare its
  pixels with the source (`verify_lpcb.py` does this for LPCB files).
  Re-encoding the source on another machine is a separate measurement and
  may give a different size.
- No published benchmark uses encoding off Metal.

## Individual model and precision experiments

```bash
python scripts/research/local_probability/neural_probability_probe.py \
  data/corpora/kodak/kodim01.png --epochs 400 --hidden 16 --mixtures 4 \
  --causal-context --feature-set 1 --schedule cosine --lr 0.015 \
  --out docs/results/raw/local-probability-development.json
python scripts/research/local_probability/quantize_model.py \
  data/corpora/kodak/kodim01.png \
  --model docs/results/raw/local-probability-development-kodim01-model.nnm \
  --out data/research-local-probability/kodim01-precision.json
python scripts/research/local_probability/training_sweep.py \
  data/corpora/kodak/kodim01.png --config g-lr15-e400 \
  --work data/research-local-probability/sweep \
  --out docs/results/raw/sweep-g-lr15-e400.json
```

The trainer writes a v3 model (`-model.nnm`) and, for feature set 0, a v2 CBOR
model (`-model.cbor.zst`) that `neural_bitstream.py` can encode with baseline
measurements. Training reports are entropy estimates, **not real file
measurements**. The precision and sweep commands encode complete v3 files,
independently decode them, and check exact RGB. Training is not promised
byte-identical across GPU/runtime versions; deterministic decoding and
re-encoding use the transmitted model.

## Corpora and frozen validation

```bash
python scripts/research/local_probability/fetch_sipi_holdout.py
# This cohort is now known; reruns are regressions, never fresh validation.
python scripts/research/local_probability/neural_validation.py \
  --out docs/results/raw/local-probability-sipi-regression.json

python scripts/research/local_probability/fetch_sipi_aerial.py
# Also known now; pins the aerial protocol's training settings.
python scripts/research/local_probability/adaptive_validation.py \
  --out docs/results/raw/local-probability-aerial.json

python scripts/research/local_probability/fetch_tecnick.py
python scripts/research/local_probability/tecnick_validation.py \
  --cohort tecnick-crop512 --out docs/results/raw/tecnick-crop512.json
python scripts/research/local_probability/tecnick_validation.py \
  --cohort tecnick-2400 --out docs/results/raw/tecnick-2400.json

python scripts/research/local_probability/fetch_tecnick_fresh.py --archive ARCHIVE.zip
python scripts/research/local_probability/fresh_validation.py \
  --cohort tecnick-fresh-crop512 \
  --out docs/results/raw/local-probability-fs4-tecnick-fresh-crop512.json

python scripts/research/local_probability/fetch_lpcb.py
python scripts/research/local_probability/lpcb_validation.py \
  --out docs/results/raw/lpcb.json --files data/lpcb-coded $(ls data/corpora/lpcb/ppm)
python scripts/research/local_probability/verify_lpcb.py data/lpcb-coded
python scripts/research/local_probability/lpcb_baselines.py \
  --out docs/results/raw/lpcb-baselines.json
```

The first cohort is all 14 currently listed color miscellaneous images,
excluding withdrawn images. The aerial cohort is all twelve currently listed
native 512×512 color images, not the larger images. TIFF-to-PNG preserves every
RGB sample. Downloaders bound archive size/expansion, never extract filesystem
paths, and refuse changed existing images/manifests. Original assets stay under
ignored `data/` and remain subject to their owners' rights.

`fetch_tecnick.py` pins the TESTIMAGES 2400×2400 RGB archive by SHA-256. It
selects two disjoint cohorts by sorted file name only: twelve native images
(`tecnick-2400`) and twelve centered 512×512 crops of other images
(`tecnick-crop512`). The images are CC BY-NC-SA 4.0 and are not
redistributed. `tecnick_validation.py` follows the
[frozen protocol](../../../docs/experiments/2026-10-24-tecnick-protocol.json);
`--start` and `--limit` split a cohort across workers. Both cohorts are now
known, so reruns are regressions.

`fetch_tecnick_fresh.py` reads a local copy of the same archive and builds the
fresh cohort from the 16 images the first two left unused: a centered 512×512
crop of each (`tecnick-fresh-crop512`) and the full image of four of them
(`tecnick-fresh-2400`). `fresh_validation.py` measures the committed default
encoder on every image and refuses changed sources. It first ran at
`6c0131e`; the cohort is known from then on, so later runs (`a82b98a`,
`da035fe`, `ea118a7`, `10fbde6`, `2063b53`) re-measure settings tuned
elsewhere.

`fetch_lpcb.py` downloads LPCB's original PPM archive (from the Wayback
Machine), its MD5 list and its published per-image paq8px, cmix and GraLIC
results, refuses any image whose MD5 differs, and keeps each PPM plus a
manifest. It also writes PNG copies (images over the pixel limit as two or
three horizontal strips), which the `299bc4c` receipts coded.
`lpcb_validation.py` encodes and decodes the given PPMs, one file per image,
and `--files` keeps each `<name>.coded`; workers sharing `--claims` split the
list. `verify_lpcb.py` decodes a directory of those files, rebuilds each PPM
and checks it against LPCB's MD5 sums. `lpcb_baselines.py` measures PNG, WebP
lossless and JPEG XL effort 9 and 10 on the whole images, and
`summarize_lpcb.py` joins everything per image and per set.
`measure_checkout.py` measures another checkout's encoder, for example an old
commit in a worktree.

`replay_models.py` reproduces the specific existing known-input cache selection
after the prerequisite raw receipts and precision artifacts have been generated.
It recomputes the real SCDX floor, checks cache hashes and both decoders, but does
not measure full cold-encoder runtime.

## Numerical format and limits

NNEX v3: six-byte magic, two little-endian uint32 lengths, 32-byte SHA-256 of
row-major raw RGB, compact canonical model, arithmetic stream. NNEX v4 is
identical except that the stream is split into up to 64 row substreams (row y
in substream y mod S, default S = 16) behind a one-byte count and varint
lengths, so rows can decode in parallel. The probabilities are the v3
probabilities. NNEX v5 has the v3 header and model; its network bounds each
mixture component's mean to the sample range. The stream starts with a value
map (a zero byte, or a one byte and a 256-bit set of the sample values whose
ranks the network codes), then a flags byte. Flags 0 is followed by a v4 row
stream body. Flags 1 (mixing) or 3 (gated mixing) is followed by a band count
and the varint lengths and binary arithmetic streams of planes G, R and B,
band by band; R and B are coded as differences from G. Flag 4 (with 1 or 3)
puts a plane mask (bits 0 to 2 for G, R, B) and a 32-byte bitmap of the values
each masked plane uses after the band count; bits those values decide are not
coded. A strip file (magic `NNST`) holds complete SCDX or NNEX files, one per
horizontal strip, behind its width, height, strip count and each strip's rows
and length; a part with magic `NNSH` is a strip coded with another strip's
model: that strip's index, then the strip's v5 file less its model payload.
The model transmits each fixed-shape tensor as raw binary16 or as Rice-coded
integers on a power-of-two grid (one per tensor, column, or row) that decode to identical
binary16 values. Model format 2 codes an all-zero column or row with one
reserved grid shift; format 1 payloads still decode. V2 differs only in
its zstd-compressed canonical CBOR model and lack of feature sets. The 46-byte
header and all trained coefficients count in file size. The checksum detects
corruption; it does **not authenticate** files.

The backend uses integer Q24 guide/affine coefficients, Q16 neural operations,
deterministic integer exponential tables, Q30 probabilities, and 20-bit CDFs.
No `exp`/libm or floating inference runs in the v2/v3/v4/v5 decoder; the
context model uses wrapping 32-bit integer sums, so NEON and scalar builds
compute identical probabilities. Model payloads
are limited to 1 MiB (also 1 MiB expanded for v2); arithmetic streams are at most 128 MiB; model
dimensions, tensor names/shapes, coefficient magnitudes, and positive CDF counts
are checked. This is not a comprehensive hostile-input security audit or a
hard process-level memory/CPU sandbox.

The legacy floating v1 reference remains available only for historical work:

```bash
c++ -O3 -std=c++17 -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math \
  scripts/research/local_probability/nnproto.cpp \
  -o data/research-local-probability/nnproto
# neural_bitstream.py ... --float-reference
```

V1 lacks the v2 pixel checksum and normative integer arithmetic. Do not deploy
it or assume cross-platform interoperability.

On a Mac with Rosetta, the alternate-ISA check can be reproduced after generating
validation outputs:

```bash
c++ -O3 -arch x86_64 -std=c++17 -Wall -Wextra -Werror -pthread \
  scripts/research/local_probability/nnproto_integer.cpp \
  -o data/research-local-probability/nnproto-integer-x86_64
python scripts/research/local_probability/verify_cross_isa.py \
  data/probe/tecnick-*.coded \
  --binary data/research-local-probability/nnproto-integer-x86_64 \
  --out docs/results/raw/neural-cross-isa-tecnick.json
```

It uses only each in-band model and decoded pixels. It verifies alternate
decoding and byte-identical re-encoding; it does not retrain a model or establish
two-machine coverage.

## Prior art

- [Minimum-rate predictors](https://www.rs.tus.ac.jp/matsuda-lab/matsuda/mrp/index.html)
- [PixelCNN++ discretized logistic mixtures](https://arxiv.org/abs/1701.05517)

These references motivate the experiment; this directory does not vendor their
implementations or claim their results.
