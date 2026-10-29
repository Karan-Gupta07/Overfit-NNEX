# Image-local probability coding, round 2, 2026-10-24

## Outcome and scope

On 24 fresh Tecnick studio photographs, under a protocol frozen before any
codec opened them, the research encoder won **24/24** against the smallest of
PNG, WebP lossless, and JPEG XL lossless efforts 9 and 10. Total bytes were
**24.68% lower** (30,359,801 vs 40,308,146), with a **24.01% median** saving.
Every selected file is a learned NNEX v3 file whose model bytes count in its
size. Every sample decoded exactly, and every file re-encoded byte-identically
with an x86_64 build under Rosetta.

On the 12 crops, the same images were also encoded with the previous round's
settings. This round's files are **7.38% smaller** than those, and smaller on
all 12. The previous round could not train above 1,048,576 pixels, so on the
native 2400×2400 images its output would have been the SCDX fallback file. This
round's files are 28.11% smaller than that.

Read the percentages with care. These images are very clean: the best baseline
needs only 1.49 bits per sample, against 4.67 on the earlier aerial cohort. With
the previous settings, the learned model saved a similar absolute amount on both
cohorts (0.24 vs 0.23 bits per sample). That is a much larger share of a small
file. Per-image savings still range from 5.7% to 36.1%. This is one publisher's
studio photographs, and the baselines exclude research context-mixing coders.
Encoding takes about 11 minutes per 2400×2400 image and decoding is much slower
than JPEG XL (see below).

Production defaults and the SCDX asset registry are unchanged. This round
changes only the research encoder under `scripts/research/local_probability/`.
The [previous notebook](2026-10-23-local-probability.md) covers the first
round, and [design 15](../design/15-image-local-probability.md) has the current
format and numerical contract.

## What changed

- **NNEX v3 model coding.** Each fixed-shape tensor is raw binary16 or
  Rice-coded integers on a power-of-two grid that decode to identical binary16
  values. No tensor names or shapes are transmitted. Converting existing v2
  models without retraining saved 0.33% on a 512×512 aerial file and 1.3–2% on
  256×256 files.
- **Feature set 1.** Twelve far causal taps per plane, up to three samples away.
  The trainer and `nnproto_integer.cpp` must use the same tap list.
- **Four mixture components** instead of three.
- **Weight grids 16/12/10/8/7/6**, stopping after the first file that grows.
- **Grouped training.** All three planes train in one batched step with
  independent weights, optimizer state, and per-plane gradient clipping.
  This trains about three times faster than the old per-plane loop.
- **Schedule.** Adam at 0.015, 5% linear warmup, cosine decay to a 1% floor,
  400 epochs (previously constant 0.001, 200 epochs).
- **Widths 16 and 24** (previously 16 and 32).
- **Tiled training above 1,048,576 pixels.** Evenly spread 256×256 tiles with
  six-pixel causal margins, up to that budget. The decoder codes every pixel.
  The first round could not use the learned path on these images at all.

## Development measurements

All tuning used six known images: SIPI aerials 2.1.01, 2.1.05, and 2.1.09,
Kodak 01, and SIPI 4.2.03 and 4.1.05. Every number is a complete NNEX file that
decoded exactly, including the model. None of these images is validation
evidence.

| configuration | 6-image bytes | vs previous round |
|---|---:|---:|
| previous round (h16, k3, constant 0.001, e200, v2) | 2,383,726 | 0.00% |
| cosine 0.003 | 2,380,061 | −0.15% |
| + four components | 2,374,960 | −0.37% |
| + v3 model coding | 2,365,879 | −0.75% |
| + feature set 1 | 2,361,097 | −0.95% |
| + 400 epochs (sequential) | 2,349,782 | −1.42% |
| grouped, 0.006, e400 | 2,342,773 | −1.72% |
| grouped, 0.010, e400 | 2,337,333 | −1.95% |
| **grouped, 0.015, e400 (h16)** | **2,332,122** | **−2.16%** |
| grouped, 0.015, e400, h24 | 2,332,621 | −2.14% |
| grouped, 0.015, e400, feature set 0 | 2,340,207 | −1.83% |
| grouped, 0.020, e400 | 2,332,910 | −2.13% |
| grouped, 0.030, e400 | 2,331,041 | −2.21% |
| grouped, 0.015, e800 | 2,330,103 | −2.25% |
| **smaller of h16 and h24, 0.015, e400** | **2,328,581** | **−2.31%** |

Learning rates from 0.015 to 0.03 were within 0.1% of each other. I froze 0.015,
the middle of that plateau. Doubling to 800 epochs saved only another 0.09% for
twice the training time, so 400 stayed. Width 24 won on Kodak 01 and 4.2.03 and
lost elsewhere; offering both widths is worth 0.15 percentage points.

### Rejected

- **Warm start.** Pretraining hidden/head weights jointly on 13 other known
  images, then fine-tuning per image, made the first two images 0.07–0.29%
  larger. The code was removed.
- **Width 12.** Slightly worse than width 16 (−0.06% vs −0.15% under the same
  early settings).
- **Per-plane candidate selection.** Mixing planes from different candidates
  would save about 0.03% by estimate. Not worth a format change.

### Large known images

| image | SCDX floor | sequential 0.003, e200 | grouped 0.02, e400 |
|---|---:|---:|---:|
| 2.2.01 (1024×1024) | 1,880,573 | 1,766,577 | 1,757,906 |
| wash-ir (2250×2250, tiled) | 10,748,376 | 10,049,060 | not run |

The sequential column is the best of widths 16/24 and both feature sets with
cosine decay. The grouped column is the best of widths 16/24 with feature set 1
(374 s for 2.2.01). Wash-ir took 1,670 s with tiled training. These images
informed the tiling design and are known.

### Alternate ISA

Twelve v3 feature-set-1 development files decoded exactly and re-encoded
byte-identically with an x86_64 build under Rosetta. The full list is in the
[development results](../results/2026-10-24-neural-development.json).

## Fresh Tecnick validation

The [protocol](2026-10-24-tecnick-protocol.json) was frozen before any codec
opened a Tecnick image. It fixes the encoder settings above, the SCDX floor,
baselines (PNG, cwebp lossless `-z 9 -exact`, JPEG XL lossless efforts 9 and 10),
and two cohorts chosen by sorted file name only:

- `tecnick-2400`: twelve native 2400×2400 images, publisher PNG bytes unchanged.
- `tecnick-crop512`: twelve other images, one centered 512×512 crop each.

Sixteen images stay unused. On the crops, the protocol also retrains the
previous round's settings without the SCDX floor, to measure this round's change
on fresh data. The previous round could not train above 1,048,576 pixels, so on
the native cohort its result would simply have been the SCDX floor.

### Crops (512×512)

All twelve selected NNEX files. "Previous settings" is the learned-only file
from the prior round's settings (widths 16/32, three components, constant
0.001, 200 epochs, sequential), retrained here on the same crop and written as
v3.

| crop | best baseline | complete NNEX | saving | width / grid | previous settings | change |
|---|---:|---:|---:|---|---:|---:|
| apples | 123,546 (JXL e10) | 95,221 | 22.93% | 24 / q10 | 98,997 | 3.81% |
| billiard_balls_a | 87,459 (JXL e10) | 64,834 | 25.87% | 16 / q12 | 69,837 | 7.16% |
| cards_a | 158,723 (JXL e10) | 123,548 | 22.16% | 24 / q10 | 134,406 | 8.08% |
| chairs | 75,439 (JXL e10) | 71,150 | 5.68% | 16 / q16 | 77,843 | 8.60% |
| cushions | 149,894 (JXL e9) | 112,903 | 24.68% | 24 / q12 | 125,182 | 9.81% |
| flowers | 144,191 (JXL e10) | 98,104 | 31.96% | 16 / q10 | 108,596 | 9.66% |
| guitar_fret | 127,613 (JXL e10) | 93,603 | 26.65% | 16 / q10 | 99,754 | 6.17% |
| keyboard_b | 133,443 (JXL e10) | 99,488 | 25.45% | 16 / q10 | 106,336 | 6.44% |
| pencils_a | 118,198 (JXL e9) | 101,778 | 13.89% | 16 / q12 | 107,995 | 5.76% |
| plastic | 162,532 (JXL e9) | 113,611 | 30.10% | 16 / q10 | 122,427 | 7.20% |
| screws | 193,449 (JXL e10) | 149,431 | 22.75% | 24 / q10 | 159,032 | 6.04% |
| sweets | 141,877 (JXL e10) | 113,448 | 20.04% | 16 / q12 | 125,296 | 9.46% |

**12/12 wins**, 1,237,119 vs 1,616,364 bytes: **23.46% aggregate**, **23.80%
median**, 5.68–31.96%. That is 1.049 vs 1.370 bits per sample. The previous
settings total 1,335,701 bytes (17.36% below the baselines), but they would
have lost on `chairs`, as would their SCDX fallback (76,088 bytes). Selected
models are 4,765–9,082 bytes. Width 16 won 8 crops and width 24 won 4. Weight
grids were 10 bits for 7, 12 bits for 4, and 16 bits for 1.

### Native 2400×2400

JPEG XL effort 10 was the smallest baseline on all twelve.

| image | JXL e10 (best baseline) | complete NNEX | saving | width / grid |
|---|---:|---:|---:|---|
| almonds | 3,550,027 | 2,623,237 | 26.11% | 24 / q12 |
| bananas | 2,939,777 | 2,253,657 | 23.34% | 24 / q12 |
| building | 2,215,723 | 1,415,999 | 36.09% | 24 / q12 |
| carrots | 3,126,529 | 2,592,957 | 17.07% | 24 / q12 |
| coins | 3,998,383 | 3,109,465 | 22.23% | 24 / q12 |
| fence | 2,019,268 | 1,411,693 | 30.09% | 24 / q12 |
| guitar_bridge | 3,563,065 | 2,767,288 | 22.33% | 24 / q12 |
| keyboard_a | 2,879,787 | 2,006,319 | 30.33% | 24 / q12 |
| multimeter | 3,473,255 | 2,748,260 | 20.87% | 24 / q12 |
| pillar | 2,407,614 | 1,750,170 | 27.31% | 16 / q12 |
| scarf | 3,788,459 | 2,809,726 | 25.84% | 24 / q12 |
| socks | 4,729,895 | 3,633,911 | 23.17% | 24 / q12 |

**12/12 wins**, 29,122,682 vs 38,691,782 bytes: **24.73% aggregate**, **24.59%
median**, 17.07–36.09%. That is 1.124 vs 1.493 bits per sample. Training used
16 tiles (1,048,576 pixels, 18% of each image); the decoder codes all
5,760,000 pixels. Width 24 won 11 images and width 16 won one; all used 12-bit
weight grids. Models are 5,985–9,360 bytes. The SCDX fallback totals 40,508,317
bytes, larger than bare JPEG XL effort 10, so the gain is not inherited from it.

### Verification and cost

- Each selected file was decoded again by the runner and by the x86_64 build,
  then re-encoded from its in-band model to identical bytes. These are two
  builds on one Mac, not two machines.
- Encoding with three concurrent workers took a median 110 s per crop and 642 s
  per native image. The previous settings took a median 169 s per crop. Baseline
  measurement (all four codecs) added a median 138 s per native image.
- During the concurrent runs, decoding took a median 0.45 s per crop and 10.8 s
  per native image.
- A later quiet sequential pass, supplementary to the protocol, timed each
  selected file against a JPEG XL effort 9 encoding of the same pixels decoded
  in process on one thread (best of two runs). NNEX took a median 0.65 s per
  crop and 15.7 s per native image; JPEG XL took 0.049 s and 0.91 s. NNEX was
  12–29× slower (median 15×). Its time includes model parsing, temporary files,
  the subprocess, and the checksum. This pass ran slower than the concurrent
  observations above for reasons I did not isolate, so compare only same-pass
  ratios.
- The summary records implementation hashes. Those of `compress.py`, the
  trainer, and the runner match the hashes taken when the runs launched.

### Reproduction

```bash
python scripts/research/local_probability/fetch_tecnick.py
python scripts/research/local_probability/tecnick_validation.py \
  --cohort tecnick-crop512 --out docs/results/raw/tecnick-crop512.json
python scripts/research/local_probability/tecnick_validation.py \
  --cohort tecnick-2400 --limit 6 --out docs/results/raw/tecnick-2400-a.json
python scripts/research/local_probability/tecnick_validation.py \
  --cohort tecnick-2400 --start 6 --out docs/results/raw/tecnick-2400-b.json
for cohort in crop512 2400; do
  python scripts/research/local_probability/verify_cross_isa.py \
    data/probe/tecnick-$cohort-*.coded \
    --binary data/research-local-probability/nnproto-integer-x86_64 \
    --out docs/results/raw/neural-cross-isa-tecnick-$cohort.json
done
python scripts/research/local_probability/decode_timing.py \
  data/probe/tecnick-*.coded --repeats 2 \
  --out docs/results/raw/tecnick-decode-timing.json
python scripts/research/local_probability/summarize_tecnick.py \
  docs/results/raw/tecnick-{crop512,2400-a,2400-b}.json \
  --cross-isa docs/results/raw/neural-cross-isa-tecnick-*.json \
  --timing docs/results/raw/tecnick-decode-timing.json \
  --out docs/results/2026-10-24-tecnick.json
```

The recorded run split the native cross-ISA check into two six-file batches
(`-2400-a`, `-2400-b`); the summarizer accepts any number of those receipts.

The [durable summary](../results/2026-10-24-tecnick.json) holds every
per-image size, baseline, hash, selection, and timing, plus the hashes of the raw
receipts. Both cohorts are now known; reruns are regressions.

## Limits

- One publisher's clean studio photographs, half of them as center crops.
  Noisier camera images are likely to show smaller percentage gains, as the
  aerials did.
- Baselines are PNG, WebP lossless, and JPEG XL lossless efforts 9 and 10. Research
  context-mixing image coders were not measured.
- Encoding is minutes of training per image. Decoding needs the compiled C++
  backend and is much slower than JPEG XL.
- Opaque 8-bit RGB only for the learned path. The SCDX codec's defaults are
  unchanged.
