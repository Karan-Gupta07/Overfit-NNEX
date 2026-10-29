# 15. Image-local probability coding

This is an experimental photo path. It uses the package's conventional SCDX
codec as its floor. The implementation lives under
`scripts/research/local_probability/`.

## What is transmitted

The network does not regenerate an approximate image. It predicts a probability
distribution for each exact 8-bit sample, which is then arithmetic-coded.
Every image-local learned coefficient is in the file. The decoder needs only
ordinary decoder software, mathematical constants, and the file itself.

The encoder may train using the entire source. The decoder may not use unseen
source pixels. Independent decoding, with no training process or source image,
is therefore mandatory. Entropy estimates alone are insufficient evidence.

This uses established predictive coding and logistic-mixture ideas. It is not
a claim to have invented minimum-rate prediction or neural entropy models.

## Conditional model

Decode green, then red, then blue, each in row-major order. Twelve spatial
offsets use only past pixels in the current plane:

```text
(0,-1) (-1,0) (-1,-1) (-1,1) (0,-2) (-2,0)
(-2,-1) (-2,1) (-1,-2) (-1,2) (-2,-2) (-2,2)
```

Offsets are `(dy, dx)`. Prior decoded planes may use a nine-position local guide
including current and neighboring samples. These are fully decoded planes, not
a leak of future samples from the current plane.

Green uses a median-edge anchor. Chroma planes use green as their anchor.
Base features include local differences and gradients, with red-minus-green
guides for blue. Robust linear fits initialize the affine mean. The transmitted
binary16 guide predicts an integer base error; six causal errors and their
magnitudes, prior-plane error guides, selected low-bit phase information, and
the fractional guide prediction augment the context.

Feature set 1 (NNEX v3 and v4) adds twelve far causal taps per plane:

```text
(0,-3) (-3,0) (-1,-3) (-1,3) (-3,-1) (-3,1)
(-2,-3) (-2,3) (-3,-2) (-3,2) (-3,-3) (-3,3)
```

Green taps are samples minus the green anchor. Red and blue taps are their
red-minus-green or blue-minus-green differences. Positions outside the image
read as zero samples (for green, before the anchor is subtracted).

Base feature counts are 14/21/30 for G/R/B. With context they are 27/45/64, and
with feature set 1 they are 39/57/76. Missing causal samples are zero; other
guide boundaries clip to the edge as specified by `nnproto_integer.cpp`. Feature
normalization is by 32 except explicit phase features in that source.

Feature set 2 adds the plane's level (the green anchor, or the left sample for
red and blue) and the sample's row, column and diagonal parity: 43/61/80 inputs.

Feature set 3 adds a causal least-squares fit for every sample. Its inputs are
the twelve causal samples above (zero outside the image) and, for red and blue,
the 3×3 neighbourhood of each decoded plane (edge clamped): 12/21/30 inputs.
The window is rows y-12 to y-1 at columns x-12 to x+12, plus row y at columns
x-12 to x-1, clipped to the image. With G and r the window's exact integer
Gram matrix and input-times-sample sums, the fit solves (G + λI)β = r with
λ = max(1, trace(G)/n)/2^20 by an LDLᵀ factorization. It keeps the prediction
β·f in eighths of a sample (rounded, clipped to 0…2040) and 16 times the
window's RMS residual (at most 4095). The network sees the fit minus eight
times the anchor, 8·log2(1 + spread/16) (linear between powers of two), the fit
errors (8·sample − fit) at (0,-1), (-1,0), (-1,-1) and (-1,1), and each decoded
plane's fit error at the sample: 49/68/88 inputs. Fit errors at missing
positions are zero.

Feature set 4 replaces that fit with two. Both take each decoded plane's sample
and its four neighbours instead of the 3×3 neighbourhood (12/17/22 inputs). The
first keeps the 12-row, ±12-column window and λ scale 2^-20; the second uses
rows y-4 to y-1 at columns x-4 to x+4, plus row y at x-4 to x-1, with λ scale
2^-13, so it follows local structure the wide window averages away. Each fit
adds the same six to eight features as feature set 3: 55/75/96 inputs. These
fits use one LDLᵀ factorization of the bordered matrix [[G + λI, r], [rᵀ, t]],
where t is the sum of squared samples. Its last pivot is the residual sum, and
the fit inputs are eliminated alongside to give the prediction. On twelve
development crops feature set 4 made files 1.05% smaller than feature set 3,
and on twelve 2400×2400 natives 0.62% smaller; all 24 files shrank. Its wider
inputs cost some training time.

A linear mean plus two ReLU hidden layers predicts the weights, means, and
log-scales of one to four logistic mixture components. Initialization uses four
robust-fit iterations. This is slow image-local optimization, not a shared
pretrained model.

## Training

The current research defaults use feature set 4 and four components. The hidden
widths depend on image size: 12 and 32 up to 1,000,000 pixels, 24 and 48 up to
4,000,000, and 48 above that. Training runs 800 epochs, capped so that no model
sees more than 51,200 × 8192 samples (a 2400×2400 image trains for 73 epochs).
Adam runs at 0.015 with a 5% linear warmup and cosine decay to a 1% floor.
Images up to 1,000,000 pixels use batch 4096 and others 8192. Each plane's
gradient is clipped at norm 10, or 30 for the small images. Other settings are
seed 20261023, four CPU threads, and MPS when available.

On MPS a fused Metal kernel set computes the forward pass, the loss, and the
backward pass, reduces weight gradients, and applies the clipped Adam step. It
rounds differently from PyTorch autograd, so it trains different (not
byte-identical) models. After the float epochs, quantization-aware training
continues for a tenth of the epochs on 12-bit grids at learning rate 0.002.
The weight search below then picks each plane's weights from the float model
and this snapshot. A further 10-bit snapshot won 8 of 72 development planes
on estimates, but the files coded smaller without it (0.047% on the crops),
and it cost about 0.5 s per crop and 3 s per native image.

On small images training also prunes first-layer inputs. At 1/4, 3/8, 1/2, and
5/8 of the float epochs, each plane scores every live input by the estimated
stream bytes added when its first-layer weights are zeroed. It drops up to half
of its live inputs, cheapest first, among those under 100 bytes. The first round
also moves the constant input's weights into the first-layer bias. An input's
affine weight stays live. Pruned weights stay exactly zero through the rest of
training and quantization, and the model coder stores each as a zero slice.
Scoring draws no random numbers. Together with the looser clip, the first
version (three rounds, up to a third of the inputs, under 15 bytes) cut
estimated size on twelve development crops by 0.29% (three seeds each). The
score is taken before the network adapts, so it overstates the cost of a
removal: raising the limit to 100 bytes and adding the fourth round made the
development crop files another 0.26% smaller, and 16 fresh crops 0.21% smaller.
With no limit at all the development files grew 5.1%. On 2400×2400 natives
neither the clip nor pruning helped (pruning at the current settings: −0.01% on
three natives), so larger images keep the defaults.

All three planes train together in one batched step. They share no weights or
optimizer state. Each plane's inputs depend only on pixels and transmitted
guides, never on another plane's network, so grouped training does not change
the format. On the development images it trained about three times faster than
sequential training, with no measured loss in file size.

The research encoder trains each width in its own spawned process while the
parent process builds the SCDX floor. Every process seeds 20261023 exactly as
the serial loop does, so the models and files are byte-identical either way.
An opt-in `--compile-training` wraps the grouped loss in `torch.compile`. Its
fused kernels round differently, so it trains slightly different models and is
a separate encoder setting, not a faster copy of the default.

When several widths train in parallel with the fused Metal trainer, they race.
After one 32nd of their epochs every worker scores itself and waits for the
others. The score is an estimated file size: the planes' mean training bits
per sample over the last eighth of the epochs before the race (at least one
epoch) times the pixels, plus the model bytes of the current weights on
10-bit grids. A width keeps training only if its score is within 3% of the
lowest; the others stop. Scoring draws no random numbers, so
a surviving model and the file are byte-identical to an unraced run. When it
was introduced, racing left every Tecnick crop file unchanged and cut the
median encode from 31.8 s to 20.7 s. The wider net gains most late in
training: on the twelve development crops with feature set 4, width 32
trailed width 12 at the race by 0.49% to 10.4% and finished ahead only on the
closest image (screws, by 1.1%); every image it lost had trailed by at least
2.5%. With the 1.5% margin both widths finish on screws, whose file shrank
0.97% while its encode took 21.5 s instead of 10.6 s; the other eleven files
were unchanged. The race later moved from an eighth to a sixteenth of training.
There width 32 led on cards_a (by 2.0%), which it then finished 1.2% smaller,
was within the margin on screws, and trailed by 2.9% to 10.0% on the other ten;
those ten files and screws were unchanged, and the development crops' median
encode fell from 10.3 s to 9.7 s, because the dropped width trains half as
long.

Until then the score was an estimate of the checkpoint's weights on the
training pixels, and that one snapshot proved noisy. Between runs whose
learning rates differed by a millionth, the two widths' ratio of such
estimates at a 32nd of training moved by up to 15%. That flipped the race on
chairs, cards_a and screws and moved the development crops' total by up to
0.5%. The ratio of averaged training bits moved by at most 1.5%. From a 32nd
of training to the end, width 32 gained 1.7% to 5.4% on width 12. At the
race it trailed by 1.6% to 2.4% on cards_a and screws, which it won, by 2.9%
to 3.1% on cushions (a tie), and by 3.8% to 9.9% on the other nine, in each
of three runs. With averaged bits, a 32nd and the 3% margin, the mean
development crop total of three such runs fell 0.18% (1,140,817 against
1,142,898 bytes): width 32 now finishes and wins on cards_a and screws every
time. The score also no longer needs an estimator pass. A worker that fails
breaks the rendezvous, and the others then finish without racing.

Images above the training budget (8,388,608 pixels with the fused trainer,
1,048,576 with PyTorch) train on evenly spread 256×256 tiles whose interiors
total at most that budget. Each tile has a six-pixel margin, so every
interior feature equals the whole-image feature. Feature set 3 and 4 fits reach
much further, so the encoder computes them once for the whole image with the
native backend (`nnproto-integer local-fit`) and every tile crops them. The
complete-file encodes reuse the same fits instead of recomputing them. The decoder
still codes every pixel; tiling only limits training memory and time.

A warm start from weights pretrained on other known images was tried and
dropped. It made files 0.07–0.29% larger on the first development images.

## Actual rate selection

Binary16 coefficients initially cost about 16.5 kB compressed for width 32.
The original frozen model won on larger SIPI photos but lost on all eight
smaller ones. Model cost cannot be ignored.

With the fused trainer the encoder searches weight grids on estimates. Planes
share no weights, so each plane picks its own weights. A candidate's cost is its
exact model bytes plus its float-model bits on the training pixels (evenly
strided to at most 262,144 rows on large images), scaled to the image. Each
source (the float model and the snapshot) first tries uniform grids of
16/12/10/8/7/6 bits on the hidden and head weights. From the best of these,
two sweeps of per-tensor coordinate descent try, for each weight matrix, one
grid of 16 or 12 to 5 bits or per-column or per-row grids of 12 to 5 bits, and
for the affine and bias tensors 16 or 14 to 8 bits. The guide stays exact
because it defines the inputs. Only two finalists become complete streams: the
searched model and, when its estimate is within 0.05%, the best uniform model.
On development images the estimate tracked the integer coder within 0.04%, with
a near-constant offset per model. In 47 Tecnick encodes that coded both, the
uniform model never won; the smallest estimated gap was 0.094%.

Each weight matrix and the affine weights also try every grid but 16 bits
through GPTQ error-feedback rounding (Frantar et al., 2022) instead of
rounding to nearest. Columns are rounded one at a time, most curved first, and
each column's rounding error is spread over the columns not yet rounded
through the Cholesky factor of the inverse curvature. The curvature of a
matrix is a per-row empirical Fisher matrix: the Gram matrix of the layer's
inputs under the current best weights, each sample weighted by the squared
loss gradient of that row's output, plus a ridge of 1% of its mean diagonal.
Gradients are refreshed once per sweep and inputs at every matrix. On MPS the
per-sample gradients come from the trainer's fused forward-backward kernel
rather than autograd, about 12 times faster; that cut the weight search by
0.1 s per crop and 0.3 s per native image, and the development files moved by
at most 250 bytes (0.001% in total on the natives). The
matrix's bias joins as a constant input, so it absorbs error too; a chosen
compensated bias is what the bias's own grids then round. A first layer tries
only GPTQ grids and 16 bits, since rounding to nearest never won one in 45
searched development planes. On the 12 development crops this made the files
0.50% smaller (all 12 smaller) for about 0.8 s more search per crop. The plain
input Gram matrix (no gradient weights, no bias) gave 0.43%, and gradient
weights without the bias 0.48%. Model bytes fell by 15% (6,703 bytes) while
the coded pixels grew by 933 bytes: the search now affords coarser grids. On
native images the model is a smaller share of the file, and the 12
development natives came out 0.09% smaller in total, with the weight search
on 2^18 rows instead of 2^19.

The PyTorch path instead codes complete files for grids of 16/12/10/8/7/6 bits,
in that order, and stops after the first file larger than the previous one.
Either way, every grid is a per-tensor or per-slice power-of-two scale. Rounded
values remain exact binary16 values, so there is no untransmitted scale or
implicit model asset. Lower precision can reduce model cost while severely
damaging likelihood; Kodak 02's width-32 eight-bit candidate is much larger.
No fixed "eight bits is always best" rule is used.

The research multiplexer also encodes the actual complete SCDX floor, optionally
with the existing hybrid/spatial search. It writes the smallest file unchanged.
A parallel fused encode first builds a fast floor: the same SCDX document with
only a JPEG XL effort 7 asset (0.9 s at 2400×2400, against about 16 s for the
full floor on all cores beside training). If the image has at least 4,096
distinct sampled colours and the smallest learned model's first estimate (its
race score after a 32nd of training) is at least 10% below the fast
floor, the fast floor stands in for the full one, which is never built.
Otherwise the full floor is built as before. On the development crops the
first estimates were 10% to 19% above the final files, and effort 9 saved 3% to
4% over effort 7, so a screened image's learned file beats the full floor too.
NNEX and SCDX have distinct magic, so the decoder dispatches without an added
envelope. This floor guarantee is against **wrapped SCDX**, not bare JXL or every
possible codec, and on a screened image it is against the fast floor. Cached replay receipts explicitly exclude training from timing.

## Integer numerical contract

V2, v3, and v4 decoding do not call PyTorch or libm, and only feature sets 3
and 4 use floating point:

- Convert binary16 tensors to Q24 guide/affine and Q16 other coefficients using
  integer bit manipulation, round-to-nearest with ties to even.
- Neural layers use signed 64-bit accumulators, integer rounding, clipping at
  ±512 Q16, and ReLU. The affine sum uses signed 128-bit arithmetic.
- Logistic scales and exponentials use deterministic integer recurrences.
  The rounded Q62 step constants are 4611756387708439211 and
  4611615650220078421; the Q48 `exp(-4)` seed is 5155394029647.
- Mixture calculations produce Q30 probabilities, normalize to sample range
  0…255, and construct 20-bit integer CDFs with at least one count per symbol.
- A 32-bit arithmetic coder uses 64-bit products, E1/E2/E3 renormalization,
  bounded reads, and four terminal zero bytes.
- Feature set 3 and 4 window sums are exact 32-bit integers, updated column by
  column from the row above. The solves use IEEE 754 binary64 addition,
  subtraction, multiplication, division and square root, all correctly rounded,
  in the order of `nnproto_integer.cpp` (`solve` for feature set 3,
  `solve_bordered` for feature set 4), with no fused multiply-add or wider
  intermediates. Vectorized or unrolled builds may not change that order.
  The source refuses fast-math builds and requires `FLT_EVAL_METHOD` 0. A
  nonpositive pivot or a prediction outside ±10^6 falls back to the west
  sample with zero spread.

The encoder knows every pixel, so it computes every prediction error first and
then every sample's distribution in parallel threads. Only the arithmetic coder
runs in sample order. The decoder finds each symbol by searching the CDF outward
from the heaviest component's mean instead of by plain bisection. The CDF is
strictly increasing, so both searches find the same symbol; only the cost
changes. Neither change alters a v3 bitstream: v3 files from the older
single-threaded backend re-encode byte-identically (unit-test hashes and all 24
Tecnick validation files).

Dense-layer rounding is not the binary16 conversion rule; its exact signed
division behavior is specified by the C++ source. The normative experimental
implementation is `nnproto_integer.cpp` plus `arithmetic.hpp` and the strict
configuration expansion in `neural_bitstream.py`. Changes that alter any CDF
require a new format version.

Tests cover every finite binary16 word within the accepted coefficient range.
All 17 original real streams decoded exactly and re-encoded byte-identically
on ARM64 and an x86_64 build under Rosetta on one Mac, as did 12 v3 feature-set-1
development files. Feature set 3 fits of two development crops were identical
from ARM64 -O3 and -O0 builds and the x86_64 build. Tiny optimization-level
comparisons and extreme-parameter ASan/UBSan checks also passed. These checks
are useful evidence, not broad hardware coverage or a comprehensive audit.
Floating training is not promised to reproduce identical models across devices.

## Envelope and validation

```text
NNEX 00 03 (6 bytes)
model length, stream length (uint32 little endian each)
SHA-256 raw RGB (32 bytes)
compact canonical model
arithmetic stream
```

The v3 model starts with format byte 1 or 2, then varint width and height, one byte
each for hidden width and mixture count, and a flags byte (bit 0 context, bits
1+ feature set). Planes follow in G, R, B order. Each plane lists the eight
named tensors in a fixed order with fixed shapes, so no names or shapes are
transmitted. There is no affine bias.

Each tensor is either raw little-endian binary16 (kind 0) or Rice coded
(kind 1). A Rice-coded tensor stores a power-of-two grid shift, a Rice
parameter, a varint body length, and zigzag integers on the grid. Every decoded
value must be exactly representable in binary16. The encoder picks the smallest
Rice parameter and keeps raw bytes when they are not larger. The decoder rejects
any payload that does not re-encode to identical bytes, plus truncated data,
nonfinite or out-of-range values, unknown feature sets, and inconsistent
lengths. Converting v2 models to v3 saved 0.33% on a 512×512 aerial file and
1.3–2% on 256×256 files.

A weight matrix with more than one row and column may instead give each column
(kind 2) or each row (kind 3) its own grid. These tensors store a Rice parameter
and a varint body length; the body holds, slice by slice, a 5-bit grid shift
followed by that slice's zigzag integers, all with the one Rice parameter. The
encoder writes whichever of kinds 0 to 3 is shortest, preferring the lower kind
on ties. Without slice grids, one large weight (for example on an input with a
small range) forces a fine grid on the whole matrix; the weight search then
often kept 16-bit first layers. Its per-tensor choices now include column and
row grids of 12 to 5 bits. On the same trained models this cut estimated file
size by 1.0% (chairs) and 0.7% (pencils), and by at most 0.1% on seven other
development images. The native backend is unchanged: the payload still decodes
to binary16 tensors before conversion.

Model format 2 reserves slice shift 31, above any valid grid, for a slice of
zeros. Such a slice stores no integers and does not take part in choosing the
Rice parameter, and canonical coding requires the escape for every all-zero
slice. Format 1 payloads, which have no escape, still decode; the encoder
writes format 2. Pruned first-layer inputs (see Training) are zero columns, so
each costs five bits per plane instead of one Rice code per hidden unit.

V4 (`NNEX 00 04`) has the same header and model payload, and the same
probability for every sample. Only the arithmetic stream is split by row:

```text
stream count S (1 byte, 1 to 64, at most the image height)
S varint substream lengths
S substreams
```

Row y of every plane goes to substream y mod S, in plane order G, R, B and then
row order. Each substream is the v3 arithmetic coder's output without its four
terminal zero bytes; the decoder restores them. The encoder writes 16
substreams by default. On the 24 Tecnick files this added 36 to 57 bytes per
file.

The decoder runs a wavefront over rows. Row y may decode column x once row y-1
has decoded through column x+3, which covers the widest causal reach (far taps
at dx = +3; transitively also rows y-2 and y-3). With feature set 3 or 4 the
widest fit window reaches column x+12, so row y waits for row y-1 through x+12, and each
row passes its window column sums to the next. A row also waits for the
previous row of its substream, which holds that substream's coder state. Rows
are claimed in increasing order, so the lowest unfinished row can always
proceed; planes run one after another. The thread count never changes the
output. By default the decoder uses up to S threads, limited to performance
cores on Apple silicon: an efficiency core decoding one row would slow every
row below it.

V2 files (`NNEX 00 02`, zstd-compressed canonical CBOR model
`[width, height, hidden, mixtures, context, planes]`) and v3 files still decode.
The encoder writes v4 by default and v3 on request; the frozen validation
scripts pin v3.

Limits: 1 MiB model payload (and 1 MiB expanded v2 CBOR), 128 MiB arithmetic
stream, 16,777,216 canvas pixels, widths 1…64, mixtures 1…4. Training uses at
most 8,388,608 pixels with the fused trainer (1,048,576 with PyTorch), tiled as
above. The multiplexer bounds input files to
64 MiB, output-file reads to 256 MiB, and preserves nonopaque inputs via SCDX
only. Still RGB/RGBA at eight bits per sample is the supported source domain.

The SHA-256 checks decoded pixels, not authenticity or canonical coding.
Native image libraries and resource limits have not received a full security
audit. No hard subprocess memory/CPU sandbox is claimed.

## Evidence and next gate

Keep five distinct results separate:

1. Frozen fresh width-32 floating streams on the original 14 SIPI inputs.
2. Integer migration and adaptive width/precision experiments on those now-known
   inputs, including actual cached-candidate/SCDX replay.
3. The separately frozen twelve-image native aerial validation.
4. The frozen Tecnick validation of the round 2 settings: twelve native
   2400×2400 images and twelve 512×512 crops of other images.
5. The fresh Tecnick test of the current settings: sixteen 512×512 crops of
   the remaining Tecnick images and four of them at 2400×2400.

The frozen aerial profile won all 12 inputs at 4.83% aggregate / 4.43% median
savings, counting complete files and all trained weights.

The round 2 settings won all 24 Tecnick inputs against the smallest of PNG,
WebP lossless, and JPEG XL lossless efforts 9 and 10: 24.68% aggregate and
24.01% median savings (23.46% on crops, 24.73% native). On the crops they beat
the previous settings, retrained on the same images, by 7.38%. These clean
studio images need only about 1.5 bits per sample, so a similar absolute saving
is a much larger percentage than on the noisier aerials. All selected streams in
both cohorts re-encoded byte-identically under Rosetta. Those 24 images were
development data for everything since.

The round 3 settings, first measured at commit `6c0131e`, won all 20 fresh
images against the same baselines (JPEG XL effort 10 was the best on each).
Later commits were tuned on development images only. Re-measured at
`a82b98a`, the crop files were byte-identical and the native total 42 bytes
smaller. Re-measured at `da035fe`, the crops were 0.87% and the native images
0.19% smaller: 28.83% aggregate and 28.28% median savings on the crops, 23.27%
and 23.90% on the native images. Against paq8px -9L, the strongest coder
measured, the files were 1.06% smaller in total on the crops (smaller on 11 of
16) and 1.75% smaller on the native images (all four). All 20 files decoded
exactly and re-encoded byte-identically under Rosetta. A crop encoded in a
median 8.6 s and decoded in 0.18 s; a native image took 40.0 s and 4.77 s.

The notebooks for [round 1](../experiments/2026-10-23-local-probability.md),
[round 2](../experiments/2026-10-24-neural-training.md) and
[round 3](../experiments/2026-10-26-fresh-tecnick.md) record all phases.
Current results do not establish superiority across all codecs (only paq8px
among the research context-mixing coders was measured), camera noise levels,
alpha-heavy art, arbitrary photography, or lossy quality targets. Production integration needs
wider evidence, faster decoding, backend packaging, additional hardware tests,
and hostile-input review.
