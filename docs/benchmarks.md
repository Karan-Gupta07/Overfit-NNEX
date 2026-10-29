# Benchmarks

This page collects every measured compression result in one place. Each number
comes from a stored result file under [`results/`](results/), linked in each
section. Raw receipts live in the ignored `docs/results/raw/` directory.

How to read these numbers:

- **Sizes are complete files.** They include every header, checksum, and, for
  the learned photo coder, every trained weight.
- **Everything is lossless.** Every reported file was decoded and checked
  sample-for-sample against the source.
- **"Fresh" means never seen during development.** The cohort was fixed, and
  the encoder committed, before our encoder first ran on those images, and no
  setting changed afterwards. "Development" inputs were used for tuning, so
  their results are regressions, not proof.
- **"Best baseline" is chosen per image**: the smallest of PNG (Pillow
  optimize), WebP lossless (`cwebp 1.6.0 -lossless -z 9 -exact`), and JPEG XL
  lossless at effort 9 and 10 (libjxl 0.12.0, one thread). A saving is
  `1 - ours / reference`; positive means our file is smaller.
- **Timings come from one laptop** (Apple M4 Pro, 12 cores, 16-core GPU,
  24 GiB). Load is stated per table. They show scale, not isolated latency.

## How the photo coder works

The learned photo coder is a research path
([design](design/15-image-local-probability.md)).

1. **Floor.** Encode the image with the normal SCDX codec. This complete file
   is the fallback. A photograph first gets a quick JPEG XL floor; the full
   one is built only if the network's first size estimate, after a 32nd of
   training, is not at least 10% smaller.
2. **Fit.** For every sample, solve two small ridge least-squares predictions
   over its causal window (the rows above and the samples to its left). The
   decoder repeats these fits exactly from decoded pixels.
3. **Train.** For each color plane (green, then red, then blue), train a small
   network on this image only. It sees decoded neighbors near and far, the
   planes decoded before it, recent prediction errors, and both fits with
   their errors, and it outputs a mixture of four logistic distributions over
   the 256 sample values. The three planes train together on the GPU with
   custom Metal kernels. Images up to 1 Mpx race widths 12 and 32; after a
   32nd of training only a width whose estimated size (from its recent
   training loss) is within 3% of the better one continues.
   Larger images use width 48.
4. **Shrink.** Short quantization-aware passes give 12- and 10-bit snapshots.
   A search then picks each weight tensor's grid (one grid, or one per column
   or row) by estimated file size. The stored weights take 3.4 to 9.7 kB per
   fresh crop and 19 to 22 kB per 2400×2400 file.
5. **Code.** A deterministic integer C++ program replays the network for every
   sample and arithmetic-codes the true value with the predicted
   probabilities. The program then decodes the file independently to prove it
   is exact.
6. **Pick.** Keep the smallest complete file, NNEX or SCDX.

Decoding needs only the file: it reads the weights, repeats the fits, runs the
same integer network sample by sample, and arithmetic-decodes. No PyTorch, GPU,
or source image is involved.

## Photos: LPCB

Summary: [`results/2026-10-28-lpcb-2db19a3.json`](results/2026-10-28-lpcb-2db19a3.json)
(first run at `299bc4c`: [`results/2026-10-27-lpcb-299bc4c.json`](results/2026-10-27-lpcb-299bc4c.json)).
Notebook: [LPCB](experiments/2026-10-27-lpcb.md). Coded files:
[`lpcb-2db19a3` release](https://github.com/Karan-Gupta07/Overfit-NNEX/releases/tag/lpcb-2db19a3).

The [Lossless Photo Compression Benchmark](https://qlic.altervista.org/LPCB.html)
ranks lossless codecs on 107 24-bit RGB images (1.15 billion pixels): 62
camera photos converted from RAW samples (10 to 16 Mpx each) and 45 NASA
Photojournal images (20 star fields, 25 renders, composites and processed
views; 0.2 to 46 Mpx). `fetch_lpcb.py` downloads the benchmark's own PPM
archive from the Wayback Machine and checks all 107 against its published
MD5 sums, so the published sizes below are for the same pixels. No setting
was chosen on LPCB: `lpcb_validation.py` ran commit `2db19a3`, one file per
image, and every file decoded exactly. The coder is unchanged since the first
run at `299bc4c`, and 102 of the files are byte-identical to that run's. The
five images over one network's 16.8 Mpx limit hold two or three strips, each
with its own model, in one file; together they came out 4.8% larger than
paq8px.

The paq8px v216 `-9L`, cmix v16 and GraLIC 1.11 sizes are LPCB's published
per-image results (the top three entries). Negative means our file is
smaller:

| set | images | ours | vs paq8px | vs cmix | vs GraLIC | smaller than paq8px |
|---|---:|---:|---:|---:|---:|---:|
| Olympus XZ1 | 27 | 246,185,072 | −4.67% | −6.66% | −10.77% | 27/27 |
| Sony A55 | 12 | 163,744,413 | −3.45% | −5.57% | −11.25% | 12/12 |
| Fujifilm X100 | 12 | 96,454,087 | −3.72% | −6.18% | −10.31% | 12/12 |
| Canon 1100D | 11 | 99,651,008 | −2.97% | −5.87% | −11.53% | 11/11 |
| NASA, other | 25 | 76,254,223 | +22.68% | +21.02% | +0.40% | 3/25 |
| NASA, stars | 20 | 194,791,256 | −0.08% | −1.60% | −9.59% | 16/20 |
| **camera photos** | **62** | **606,034,580** | **−3.91%** | −6.16% | −10.95% | **62/62** |
| **all** | **107** | **877,080,059** | **−1.21%** | −3.28% | −9.76% | 81/107 |

Per camera photo, ours is 0.81% to 5.36% smaller than paq8px. Its stored
weights take 16 to 23 kB, 0.08% to 0.29% of the file. paq8px's largest wins
are processed or composited NASA images such as PIA13912 (ours +21%),
PIA13915 (+44%) and PIA13779 (3.1 times paq8px's 49 kB); ten of the 28 coded
parts from that group (counting strips) kept the SCDX floor.

Against the standard formats, measured here on the whole images with the
settings above:

| codec | 62 camera photos | our saving | all 107 | our saving |
|---|---:|---:|---:|---:|
| PNG | 969,833,223 | 37.51% | 1,453,911,150 | 39.67% |
| WebP lossless | 807,732,126 | 24.97% | 1,198,793,394 | 26.84% |
| JPEG XL effort 9 | 743,950,384 | 18.54% | 1,071,591,474 | 18.15% |
| JPEG XL effort 10 | 723,529,248 | 16.24% | 1,034,707,323 | 15.23% |
| paq8px -9L (published) | 630,710,768 | 3.91% | 887,823,381 | 1.21% |
| **ours** | **606,034,580** | | **877,080,059** | |

JPEG XL effort 10 was the best baseline on every camera photo; ours beat it
on all 62, by 12.52% to 20.70% (median 15.81%), at 2.15 bits per sample
against 2.57. Over all 107 images ours saved 15.21% against the per-image
best baseline and won 99. The eight losses are non-star NASA images, six of
them by 1.5% to 6.0%; PIA13779 is the worst (+55%). LPCB's own published
totals for WebP 0.6.1 and PNG (pngout), 1,194,936,668 and 1,432,319,902
bytes, are within 0.3% and 1.5% of the WebP and PNG totals measured here.

### Speed on LPCB

The `2db19a3` run coded one image at a time on AC power, with a chat app and
a browser open. Encode times include a full decode that checks the file:

| camera | photos | pixels | encode, median (range) | decode, median (range) |
|---|---:|---:|---:|---:|
| Olympus XZ1 | 27 | 10.2 M | 50.4 s (47.2 to 67.0) | 7.47 s (7.11 to 8.23) |
| Sony A55 | 12 | 16.2 M | 58.8 s (54.2 to 68.1) | 11.67 s (11.28 to 12.14) |
| Fujifilm X100 | 12 | 12.4 M | 54.5 s (50.6 to 58.6) | 9.33 s (8.91 to 9.75) |
| Canon 1100D | 11 | 12.3 M | 52.7 s (51.2 to 62.4) | 9.14 s (8.72 to 9.92) |
| **all 62** | | | **54.1 s** | **9.06 s** |

All 107 images took 1.64 hours to encode and 0.23 hours to decode. LPCB's
paq8px log, from its tester's machine (an Intel i7-1165G7), shows a median
of 2.56 hours to compress a camera photo and 2.58 hours to decompress it,
single-threaded.

## Photos: fresh Tecnick cohort

Summary: [`results/2026-10-27-tecnick-fresh-da035fe.json`](results/2026-10-27-tecnick-fresh-da035fe.json)
(first measurement: [`results/2026-10-26-tecnick-fresh.json`](results/2026-10-26-tecnick-fresh.json);
first re-measure at `a82b98a`: [`results/2026-10-27-tecnick-fresh.json`](results/2026-10-27-tecnick-fresh.json)).
Notebook: [round 3](experiments/2026-10-26-fresh-tecnick.md).

`fetch_tecnick_fresh.py` built this cohort from the 16 TESTIMAGES photographs
that the first Tecnick selection left unused: a centered 512×512 crop of every
one, and the full 2400×2400 image of four of them (positions 0, 4, 8 and 12 in
name order). The four native images are therefore the same scenes as four of
the crops. `fresh_validation.py` first measured the default encoder at commit
`6c0131e` with a clean tree. Later commits were tuned on the development
images only. It was re-measured at `a82b98a` (crop files byte-identical,
native total 42 bytes smaller), and the tables below re-measure it at
`da035fe`, also with a clean tree: the crops 0.87% and the native images 0.19%
smaller than at `a82b98a`. Every file decoded exactly, and an x86_64 build
under Rosetta decoded all 20 to the same pixels and re-encoded each to
identical bytes.

### Totals by codec (bytes)

| codec | 16 crops 512×512 | our saving | 4 native 2400×2400 | our saving |
|---|---:|---:|---:|---:|
| PNG | 3,738,879 | 59.26% | 15,427,276 | 58.49% |
| WebP lossless | 2,733,334 | 44.27% | 11,848,508 | 45.95% |
| JPEG XL effort 9 | 2,147,237 | 29.06% | 8,905,825 | 28.09% |
| JPEG XL effort 10 | 2,140,363 | 28.83% | 8,346,560 | 23.27% |
| paq8px -9L | 1,539,595 | 1.06% | 6,518,112 | 1.75% |
| **ours** | **1,523,239** | | **6,404,016** | |

JPEG XL effort 10 was the best baseline on all 20 files. Against it, the crops
saved 28.83% in total (median 28.28%, range 21.56% to 37.18%, 16/16 wins) and
the native images 23.27% (median 23.90%, range 19.45% to 25.46%, 4/4). That is
0.968 bits per sample against 1.361 on the crops, and 0.741 against 0.966 on
the native images.

### Per image: 512×512 crops

| image | ours | best baseline | saving | ours bits/sample | paq8px -9L | ours vs paq8px | weights (width) | encode s | decode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baloons | 79,047 | 108,978 (e10) | 27.47% | 0.804 | 77,733 | +1.69% | 2,970 (12) | 9.7 | 0.17 |
| billiard_balls_b | 66,880 | 106,456 (e10) | 37.18% | 0.680 | 67,017 | −0.20% | 3,122 (12) | 8.4 | 0.16 |
| cards_b | 111,773 | 157,633 (e10) | 29.09% | 1.137 | 116,509 | −4.06% | 7,405 (32) | 19.0 | 0.21 |
| clips | 113,649 | 169,984 (e10) | 33.14% | 1.156 | 116,074 | −2.09% | 7,702 (32) | 18.9 | 0.23 |
| ducks | 88,375 | 112,668 (e10) | 21.56% | 0.899 | 85,947 | +2.82% | 2,703 (12) | 8.6 | 0.19 |
| garden_table | 140,447 | 183,433 (e10) | 23.43% | 1.429 | 142,676 | −1.56% | 2,784 (12) | 8.6 | 0.18 |
| guitar_head | 114,491 | 163,557 (e10) | 30.00% | 1.165 | 116,684 | −1.88% | 2,657 (12) | 8.5 | 0.18 |
| lion | 87,416 | 132,928 (e10) | 34.24% | 0.889 | 90,820 | −3.75% | 2,501 (12) | 8.6 | 0.18 |
| pencils_b | 86,616 | 114,992 (e10) | 24.68% | 0.881 | 84,535 | +2.46% | 2,882 (12) | 8.5 | 0.19 |
| roof | 81,809 | 125,504 (e10) | 34.82% | 0.832 | 85,398 | −4.20% | 2,411 (12) | 8.8 | 0.18 |
| snails | 83,430 | 120,422 (e10) | 30.72% | 0.849 | 85,055 | −1.91% | 2,647 (12) | 8.6 | 0.18 |
| tomatoes_a | 93,258 | 122,597 (e10) | 23.93% | 0.949 | 92,100 | +1.26% | 2,564 (12) | 8.7 | 0.20 |
| tomatoes_b | 79,218 | 103,678 (e10) | 23.59% | 0.806 | 78,591 | +0.80% | 2,552 (12) | 8.6 | 0.18 |
| tools_a | 112,191 | 153,202 (e10) | 26.77% | 1.141 | 112,743 | −0.49% | 2,645 (12) | 8.6 | 0.19 |
| tools_b | 103,821 | 142,012 (e10) | 26.89% | 1.056 | 104,968 | −1.09% | 2,955 (12) | 8.5 | 0.19 |
| wood_game | 80,818 | 122,319 (e10) | 33.93% | 0.822 | 82,745 | −2.33% | 2,536 (12) | 8.8 | 0.19 |

### Per image: native 2400×2400

| image | ours | JXL e10 | saving | ours bits/sample | paq8px -9L | ours vs paq8px | weights (width) | encode s | decode s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baloons | 1,626,027 | 2,181,287 | 25.46% | 0.753 | 1,656,058 | −1.81% | 17,837 (48) | 39.8 | 4.88 |
| ducks | 1,559,084 | 1,935,470 | 19.45% | 0.722 | 1,586,294 | −1.72% | 16,003 (48) | 40.7 | 5.74 |
| pencils_b | 1,631,284 | 2,131,639 | 23.47% | 0.755 | 1,641,432 | −0.62% | 18,427 (48) | 40.2 | 4.64 |
| tomatoes_b | 1,587,621 | 2,098,164 | 24.33% | 0.735 | 1,634,328 | −2.86% | 17,130 (48) | 39.4 | 4.67 |

### Against paq8px

paq8px leads the [Lossless Photo Compression Benchmark](https://qlic.altervista.org/LPCB.html).
It is a context-mixing compressor: it blends hundreds of adaptive models and
updates them as it codes, so its decoder repeats all of the encoder's work.
This run uses paq8px v217 (commit `553024fc`, two small macOS build fixes)
built for arm64, at its strongest practical setting `-9L` (about 4.3 GB of
memory), on a binary PPM of the exact RGB samples. A size counts only after
its archive decompressed to exactly the source samples; all 20 did.

In total our files are 1.06% smaller on the crops and 1.75% smaller on the
native images. Per image, ours range from 4.20% smaller to 2.82% larger on the
crops (smaller on 11 of 16) and are 0.62% to 2.86% smaller on all four native
images. Our stored weights are 2.0% to 6.8% of a crop file but only 1.0% to
1.1% of a native file, which is most of why the native images fare better.
paq8px wins baloons, ducks, pencils_b, tomatoes_a and tomatoes_b, by 0.80% to
2.82%. billiard_balls_b, its largest win at `a82b98a` (+5.52%, when that crop
kept width 32 with 9.7 kB of weights), is now 0.20% smaller than paq8px.

### Speed

| step | 512×512 crops | 2400×2400 |
|---|---:|---:|
| our encode | median 8.6 s (8.4 to 19.0) | median 40.0 s (39.4 to 40.7) |
| our decode | median 0.18 s (0.16 to 0.23) | median 4.77 s (4.64 to 5.74) |
| paq8px compress | median 504 s (342 to 652) | median 9,791 s (9,548 to 10,110) |
| paq8px decompress | median 478 s (220 to 900) | median 7,888 s (7,712 to 8,041) |

Our encoder ran one image at a time, after the paq8px runs had finished. It
uses the GPU for training and every core for the fits and the arithmetic
coder; decoding uses the eight performance cores. paq8px is single-threaded and
ran several processes at once. Decode times include the Python wrapper and the
SHA-256 check. In round 2 (below), single-threaded JPEG XL effort 9 decoded a
crop in about 0.05 s and a native image in 0.56 to 1.05 s, so our decoder is
roughly four to nine times slower than JPEG XL.

A crop encode is mostly training: about 6.4 s of Adam steps, 1.1 s of
quantization-aware pass and weight search, 0.3 s of features, and 0.1 s for a
quick JPEG XL floor. Crops whose two widths finish within the race margin
(cards_b and clips here) train both and take about 19 s. A native encode spends
about 4.2 s on features, 23.6 s on Adam steps, 4.0 s on the quantization-aware
pass and search, 0.7 s on the floor, and 4.7 s on the final decode check.

## Photos: development Tecnick images

These 24 images were the fresh cohort of round 2 and have been used for
tuning since, so the current numbers are regressions. Round 2 summary:
[`results/2026-10-24-tecnick.json`](results/2026-10-24-tecnick.json); protocol:
[`experiments/2026-10-24-tecnick-protocol.json`](experiments/2026-10-24-tecnick-protocol.json).

| cohort | round 2 (validated) | current (`da035fe`) | paq8px -9L | current vs paq8px | current vs best baseline |
|---|---:|---:|---:|---:|---:|
| 12 crops 512×512 | 1,237,119 | 1,141,466 | 1,159,431 | −1.55% (10/12 smaller) | 29.38% saving |
| 12 native 2400×2400 | 29,122,682 | 26,884,291 | 28,513,871 | −5.72% (12/12 smaller) | 30.52% saving |

paq8px wins two development crops: chairs (ours +5.10%) and pencils_a
(+2.86%). All 24 paq8px archives decompressed exactly.
paq8px took about 200 s per crop and 70 to 105 min per native image, each way,
with other paq8px jobs running.

### Round 2 validation (historical)

Round 2's settings were frozen in the protocol file above before any codec
saw these images. They won all 24 against the best baseline: 23.46% on the
crops, 24.73% on the native images, and **24.68% overall** (median 24.01%,
range 5.68% to 36.09%), at 1.12 bits per sample against 1.49. All 24 files
decoded exactly and re-encoded byte-identically under an x86_64 build running
in Rosetta. That run took a median 110 s per crop and 642 s per native image
to encode, with up to three jobs at once; decoding took 0.62 to 0.79 s per
crop and 13.1 to 16.1 s per native image, against 0.045 to 0.052 s and 0.56
to 1.05 s for single-threaded JPEG XL effort 9. Per-image rows are in the
summary file and the [round 2 notebook](experiments/2026-10-24-neural-training.md).

Later work made encoding faster without changing those files
([notebook](experiments/2026-10-25-encode-speed.md),
[summary](results/2026-10-25-encode-speed.json)), then changed the model,
the features and the trainer ([round 3](experiments/2026-10-26-fresh-tecnick.md)).

## Photos: earlier cohorts

| cohort | status | result | source |
|---|---|---|---|
| 12 SIPI aerials, 512×512 (noisy) | fresh, round 1 settings | 12/12 wins, 5,242,156 vs 5,508,321 bytes, **4.83%** total, 4.43% median | [`neural-aerial`](results/2026-10-23-neural-aerial.json) |
| 14 SIPI miscellaneous | fresh, original width-32 model | 6/14 wins, **0.24%** total | [`local-probability`](results/2026-10-23-local-probability.json) |
| same 14 SIPI, adaptive width/precision | known (regression) | 9/14 wins, 3,088,562 vs 3,240,758 bytes, 4.70% total | [`local-probability`](results/2026-10-23-local-probability.json) |

These earlier runs predate the JPEG XL effort 10 baseline, so their best
baseline is the smallest of PNG, WebP lossless, and JPEG XL effort 9. The
Tecnick images are clean, needing about 1 to 1.5 bits per sample at the best
baseline, so a similar absolute saving shows up as a much larger percentage
than on the noisier aerial images. The current coder has not been run on the
aerials.

## Not measured

- Our own runs of other research compressors such as cmix, GraLIC, EMMA, BMF,
  LEA, and NBLI. Most ship only Windows binaries, and cmix is several times
  slower than paq8px. The LPCB section uses the published cmix and GraLIC
  sizes instead.
- Lossy coding, alpha-heavy art, high bit depths, and grayscale photographs.
- Hardware beyond one Mac (the x86_64 check ran under Rosetta on the same
  machine).
