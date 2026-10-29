# Image-local probability coding, 2026-10-23

## Outcome and scope

This round establishes real self-contained learned probability streams, not just
entropy estimates. All trained coefficients count. Integer decoding is exact;
the model is transmitted, not silently assumed at the receiver.

It does **not** establish a general codec breakthrough. The original frozen
14-image fresh SIPI result was nearly aggregate-neutral and lost on eight
smaller images. Later width/precision/floor improvements are explicitly
known-input regressions. A separately frozen native aerial experiment won all
twelve inputs at **4.83% aggregate / 4.43% median savings**. It is reported
independently below, with its compute cost and limited coverage.

Production defaults and the SCDX asset registry are unchanged. The experimental
`compress.py` command selects a complete SCDX or NNEX file and its own decoder
dispatches by magic.

## Development and rejected approaches

Three Kodak inputs (01–03) were development photographs. Kodak 04–24 had already
been measured in the hybrid round and cannot become fresh evidence by being
renamed.

The local linear predictor probe used robust regression, transmitted models,
class labels, and residual contexts. Even estimated totals on Kodak 01–03
were 488,139 / 431,698 / 366,584 bytes, above JXL's
454,757 / 405,088 / 328,115. Reversible YCoCg did worse. Those were estimates,
not successful coded-file claims. Earlier lossy-base/exact-residual hypotheses
also lost, as recorded in the hybrid notebook and hypothesis receipts.

The neural path instead models the full conditional sample distribution using
three logistic components, causal spatial samples, previous decoded color
planes, and error contexts. There is no shared pretrained model. See
[design 15](../design/15-image-local-probability.md) for the numerical contract.

## Original fresh SIPI experiment

Before testing, fix width 32, 200 epochs, context enabled, and three components.
Select every currently publisher-listed color image in SIPI's miscellaneous
volume, excluding withdrawn images: 4.1.01–08, 4.2.01, 4.2.03, 4.2.05–07, house.
Use original 256²/512² dimensions, exact TIFF-to-PNG RGB, no crops or resampling.

All fourteen floating-reference v1 files independently decoded exactly. Six
larger inputs won by 3.21–5.59%; all eight smaller inputs lost by 3.70–27.80%.
Total: **3,232,939 vs 3,240,758 bytes**, only **0.241% aggregate saving**.
Median per-image saving was **−5.93%**, with 6/14 wins. A hypothetical conventional
floor was not an implemented result at this stage.

The model occupied about 16.5 kB per image. Counting it changes the conclusion.
This cohort is now known and all later use is a regression.

## Integer migration, not new fresh validation

Keep the same trained tensors. Replace floating inference and library
exponentials with integer coefficient expansion, Q16/Q24 inference,
deterministic exponential tables, Q30 probabilities, and 20-bit arithmetic CDFs.
V2 adds a 32-byte raw RGB SHA-256 checksum; its full header is 46 bytes.

All 17 original photo streams decoded exactly. SIPI totals became
**3,233,425 bytes**, 0.226% below the same bare baseline, still 6/14 wins.
Both the numerical changes and the additional checksum count in those totals.

ARM64 and x86_64 compiled executables decoded all 17 and re-encoded identical
bytes, using x86_64 under Rosetta on the same Mac. This is an architecture/compiler
check, **not two-machine validation**. Exhaustive accepted-binary16 conversion,
tiny O0/O2 comparisons, and ASan/UBSan boundary-model checks passed. This is not a
production security audit or a hard resource sandbox.

## Width and precision regressions

Try width 16 as well as 32, then actual model grids labelled 16/12/10/8 bits.
Only hidden/head weight matrices change. Every candidate becomes a real file
and independently decodes; guide, affine, and bias values remain in-band.

Precision is not a monotonic improvement. For Kodak 02, width-32 complete sizes
were 400,130 / 398,715 / 397,197 / **499,639** bytes at 16/12/10/8 bits.
The smaller weight payload at eight bits did not compensate for damaged
likelihood. The encoder must compare complete streams, not choose a precision
from model size alone.

Across the known SIPI set, selecting width and precision gives neural-only
files totaling 3,112,037 bytes, 3.972% below the bare best, 9/14 wins. Five still
lose without a floor. Smaller networks help but do not always win: Kodak 02/03
retain width 32, while Kodak 01 selects width 16.

## Actual complete-file selection replay

`replay_models.py` verifies cache/file/source hashes, independently decodes the
best cached neural file at each width, and newly encodes the complete balanced
hybrid SCDX floor with no spatial partitioning. It writes and decodes the winner.
This is **actual file competition**, not an unwrapped oracle, but it reuses
trained models and does **not measure cold end-to-end training latency**.

| SIPI image | best bare PNG/WebP/JXL | selected complete file | selection |
|---|---:|---:|---|
| 4.1.01 | 89,668 | 89,996 | SCDX floor |
| 4.1.02 | 85,173 | 85,501 | SCDX floor |
| 4.1.03 | 75,634 | 74,234 | NN, width 16, q8 |
| 4.1.04 | 83,067 | 83,395 | SCDX floor |
| 4.1.05 | 97,944 | 95,960 | NN, width 16, q8 |
| 4.1.06 | 116,492 | 111,483 | NN, width 16, q8 |
| 4.1.07 | 53,449 | 53,775 | SCDX floor |
| 4.1.08 | 61,154 | 61,480 | SCDX floor |
| 4.2.01 | 331,186 | 311,942 | NN, width 16, q8 |
| 4.2.03 | 565,624 | 541,605 | NN, width 16, q10 |
| 4.2.05 | 351,524 | 330,806 | NN, width 16, q8 |
| 4.2.06 | 495,056 | 460,479 | NN, width 16, q8 |
| 4.2.07 | 452,365 | 429,024 | NN, width 16, q10 |
| house | 382,422 | 358,882 | NN, width 16, q8 |

Total **3,088,562 vs 3,240,758 bytes**, **4.696% aggregate** and **3.136% median**
saving, 9/14 wins. All decoded RGBA samples are exact. The five losses retain
SCDX wrapping overhead; the floor is not a promise to beat every bare file.

The actual SCDX-only floor total is 3,245,349 bytes. Neural probability coding
removes 156,787 bytes from that existing implementation, so these gains are
not simply inherited from wrapping JXL.

The three Kodak development outputs total 1,148,032 vs 1,187,960 bare bytes
(3.361% aggregate); they remain development measurements.

## Separate frozen aerial protocol

Select all twelve native 512×512 color entries 2.1.01–12 from the
[publisher's separate aerial volume](https://sipi.usc.edu/database/database.php?volume=aerials).
Larger native images are outside this validation's compute budget. Do not use
previews, crop, resize, omit failures, or tune the model using results.

The [protocol JSON](2026-10-23-neural-aerial-protocol.json) was written before
compression measurements. It freezes both widths, all four precisions, 200
epochs, causal context, mixture count, seed, and the actual complete SCDX floor.
The full encoder trains from scratch on each new input and counts every selected
coefficient. The holdout concerns architecture/selection settings, not a ban on
image-local fitting whose weights are transmitted.

All three four-image batches completed without exclusions. Every one of the
96 learned candidate streams independently decoded exactly. All twelve winning
files were NNEX v2, not JXL wrappers:

| aerial | best bare PNG/WebP/JXL | selected complete NNEX | width / grid |
|---|---:|---:|---|
| 2.1.01 | 495,392 | 470,030 | 16 / q8 |
| 2.1.02 | 516,652 | 479,386 | 16 / q10 |
| 2.1.03 | 397,944 | 380,106 | 16 / q8 |
| 2.1.04 | 460,846 | 436,317 | 16 / q8 |
| 2.1.05 | 454,189 | 426,078 | 16 / q8 |
| 2.1.06 | 507,648 | 478,338 | 16 / q10 |
| 2.1.07 | 430,880 | 414,372 | 16 / q8 |
| 2.1.08 | 465,018 | 448,833 | 16 / q8 |
| 2.1.09 | 426,812 | 412,684 | 16 / q8 |
| 2.1.10 | 460,776 | 441,010 | 16 / q8 |
| 2.1.11 | 452,104 | 432,280 | 16 / q8 |
| 2.1.12 | 440,060 | 422,722 | 16 / q8 |

**12/12 wins**, **5,242,156 vs 5,508,321 bytes**, **4.832% aggregate** and
**4.434% median** saving. All RGBA samples are exact. Selected compressed models
range from 4,640 to 5,696 bytes and count inside the complete file sizes.
The actual SCDX floor totals 5,512,263 bytes, so this method removes another
270,107 bytes from the existing implementation.

All twelve final files decoded exactly and re-encoded byte-identically with the
x86_64 backend under Rosetta as well as ARM64. A subsequent sequential decoder
pass also verified RGBA and file hashes without importing PyTorch.
Median decode observations were **0.691 s NNEX vs 0.0467 s native JXL**.
NNEX includes its actual subprocess/config/temp-file overhead; JXL uses native
in-process decoding. This is a single pass, not a stable timing study. Full
encoder times under three concurrent workers ranged from 171 to 194 seconds per
image; do not interpret them as isolated latency.

The [full fresh receipt](../results/2026-10-23-neural-aerial.json) includes the
frozen protocol and hash, archive/source manifest and hashes, every candidate,
winner hashes, current implementation hashes, and cross-ISA/decode observations.
This cohort is now known; future tuning on it cannot be called fresh validation.

This is meaningful evidence beyond the earlier regression, but it remains one
native 512² aerial family from the same publisher, not broad proof for arbitrary
photos, larger resolutions, modern camera images, or all image classes.

## Reproduction and provenance

[Research README](../../scripts/research/local_probability/README.md) contains
build, encoder, decoder, training, precision, and corpus commands.
[Durable JSON](../results/2026-10-23-local-probability.json) preserves all original
and regression sizes, baseline sizes, hashes, and all precision trials, outside
the ignored raw directory.

Raw receipts include `neural-sipi-{first7,last7}.json`,
`neural-integer-all.json`, `neural-quant-{development,sipi-regression}.json`,
`neural-h16-regression.json`, `neural-floor-replay-{0,6,12}.json`, and
`neural-cross-isa.json`. The fresh run adds `neural-aerial-{0,4,8}.json` and
`neural-cross-isa-aerial.json`; selected known-input models are also covered by
`neural-cross-isa-selected.json`. Models and coded image artifacts remain under ignored
`data/`/`docs/results/raw/`, not vendored image assets. Source manifests record
archive, original TIFF, PNG, and raw RGB hashes. No external upload or commit was
made.

Environment: Apple ARM64 macOS, Apple Clang 21; Python 3.13.16, NumPy 2.5.1,
Pillow 12.3.0, torch 2.13.0/MPS, imagecodecs 2026.10.10/libjxl 0.12,
cwebp/dwebp 1.6. Baselines verify exact samples, use JXL effort 9 and cwebp
lossless `-z 9 -exact`. Maximum-strength JXL variants, FLIF, and all other codecs
are not covered. No lossy quality claim is made.

Training depends on floating optimization and is not promised byte-identical
across devices/runtime versions. Once a model is transmitted, the v2 CDF and
decoder are integer-specified. Parallel aerial workers share this machine;
their wall times are not isolated latency measurements.

Final checks: 146 pytest cases, Ruff across production/tests/research and
downloaders, mypy across 32 production files, dependency consistency, real
decoder CLI RGBA equality, durable JSON/link checks, and `git diff --check`.

## Remaining limits

Training is slow. Decoder deployment still needs a compiled C++17 backend with
128-bit integer support. Alpha-heavy images and images above 1,048,576 pixels do
not use the learned path. Pixel losslessness does not preserve original file
bytes, EXIF/ICC metadata, animation, grayscale, or higher bit depths.

The numerical checks do not constitute a comprehensive hostile-input review.
More independent imagery, resolutions, hardware, packaging work, and stronger
baseline coverage are needed before changing production defaults or making a
general-superiority claim.
