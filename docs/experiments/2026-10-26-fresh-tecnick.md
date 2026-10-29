# Image-local probability coding: round 3 and a fresh Tecnick test, 2026-10-26

## Outcome and scope

Round 3 changed the trainer, the model coder and the features, then measured
the default encoder once on 20 Tecnick images it had never seen. On 16 fresh
512×512 crops its files were **28.21% smaller than JPEG XL effort 10** in total
(median 27.54%, every image smaller) and **0.20% smaller than paq8px -9L**
(10 of 16 images smaller). On four fresh 2400×2400 images they were 23.13%
smaller than JPEG XL effort 10 and **1.57% smaller than paq8px** (all four).
Every file decoded exactly, and an x86_64 build decoded all 20 to the same
pixels and re-encoded them to the same bytes.

On an idle laptop a crop encoded in a median 11.2 s and a native image in
66.1 s. Decoding took 0.21 s and 5.27 s. paq8px took about 8 minutes per
crop and 2.2 to 2.7 hours per native image, each way.

Two later commits, tuned on development images only, cut encoding to a median
10.3 s per crop and 51.3 s per native image. Re-measured at `a82b98a`, the 16
crop files were byte-identical and the four native files 42 bytes smaller in
total ([re-measure](#re-measure-at-a82b98a)).

The commits after that, also tuned on development images only, cut encoding
to a median 8.6 s per crop and 40.0 s per native image. Re-measured at
`da035fe`, the crops coded 0.87% smaller, **28.83% below JPEG XL effort 10
and 1.06% below paq8px** (11 of 16 images smaller), and the native images
0.19% smaller, 23.27% below JPEG XL effort 10 and **1.75% below paq8px**
([second re-measure](#re-measure-at-da035fe)).

For scale: round 2's validated files on its own 24 images were 6.70% larger
than paq8px on the crops and 2.14% larger on the native images. Those images
have been development data since, and the round-3 encoder makes their files
6.88% (crops) and 7.64% (native) smaller. The fresh cohort is the evidence;
the development numbers are regressions.

## What changed since round 2

Each row is one commit. Sizes are complete files on the 24 round-2 Tecnick
images (now development data), every file decoded exactly. "vs paq8px" is
ours / paq8px − 1, so negative means ours is smaller.

| commit | change | 12 crops | vs paq8px | 12 native | vs paq8px |
|---|---|---:|---:|---:|---:|
| round 2 | validated v3 files | 1,237,119 | +6.70% | 29,122,682 | +2.14% |
| `09f1b13`, `e95fa29`, `75c9428` | fused Metal trainer and weight search, feature set 2, widths 12 and 32 with batch 4096 on crops, per-column and per-row weight grids, racing widths | 1,190,674 | +2.69% | | |
| `7a62fa9`, `d2d714b` | faster features, search and guide fit (crops byte-identical) | 1,190,674 | +2.69% | | |
| `690866e` | first-layer input pruning and clip norm 30 on crops, zero-slice escape | 1,187,393 | +2.41% | 27,705,210 | −2.84% |
| `4b40651` | feature set 3: one causal-window least-squares fit | 1,164,250 | +0.42% | 27,066,813 | −5.07% |
| `6c0131e` | feature set 4: two fit windows | **1,152,000** | **−0.64%** | **26,898,389** | **−5.67%** |
| `a1ed669` | faster trainer kernels | byte-identical | | byte-identical | |
| `f4a727f` | parallel feature building, faster Adam step, weight search on large images strided to 2^19 rows | byte-identical | | 26,898,027 | −5.67% |
| `a82b98a` | both crop widths finish when the race is within 1.5% | 1,150,602 | −0.76% | unchanged | |
| `6ba3831` | pruning drops up to half the live inputs under 100 estimated bytes, in four rounds; half4 input loads in the trainer | 1,147,653 | −1.02% | unchanged | |
| `e48114b` | GPTQ weight rounding against per-row gradient-weighted curvature, with bias compensation; weight search on 2^18 rows | **1,141,883** | **−1.51%** | 26,874,914 | −5.75% |
| `86ac778` | large images skip the 10-bit snapshot | unchanged | | **26,874,225** | **−5.75%** |
| `0e1069b` | crops skip the 10-bit snapshot too | **1,141,348** | **−1.56%** | unchanged | |
| `ee14016` | the width race meets after a sixteenth of training instead of an eighth | **1,139,923** | **−1.68%** | unchanged | |
| `5329246` | the weight search takes its output gradients from the trainer kernel | 1,139,935 | −1.68% | 26,874,502 | −5.75% |
| `a0001ce` | features built in parallel row bands | byte-identical | | byte-identical | |
| `b715615` | weight candidates scored from cached layer outputs | byte-identical | | byte-identical | |
| `e3333b0` | GPTQ curvatures on every second estimator row, Gram matrices summed in a Metal kernel | 1,139,942 | −1.68% | 26,875,806 | −5.74% |
| `217ea41` | binary16 training scratch rows at width 48 | unchanged | | 26,885,193 | −5.71% |
| `3c277d5` | a quick JPEG XL floor screens the full floor on photographs | byte-identical | | byte-identical | |
| `c5f3f91` | the width race scores averaged training bits after a 32nd of training, 3% margin | 1,139,942 | −1.68% | unchanged | |
| `da035fe` | the trainer's mixture loss shares its exponentials | 1,141,466 | −1.55% | 26,884,291 | −5.72% |

The full native set was first re-measured just before feature set 3;
earlier commits checked two native images each. `a1ed669` was checked on five
development images (four crops, one native), which re-encoded to identical
files. `f4a727f` and `e48114b` were measured on all 24; `a82b98a` and `6ba3831`
change only crops (native images train one width and do not prune) and were
measured on all 12 of them, `86ac778` changes only large images and was
measured on the 12 natives, `0e1069b` and `ee14016` change only crops, and
`5329246`, `a0001ce`, `b715615` and `e3333b0` were measured on all 24.
`217ea41` changes only width 48 and was measured on the natives; `3c277d5`
and `c5f3f91` were measured on all 24, and `c5f3f91` on the crops at two
more learning rates as well ([noise](#run-to-run-noise)). `da035fe` was
measured the same way as `c5f3f91`.
[Design 15](../design/15-image-local-probability.md) describes every setting.

### Trainer

The fused Metal trainer runs one kernel per batch for the forward pass, the
mixture loss and the backward pass of all three planes, then reduces weight
gradients with simdgroup matrix tiles and applies a clipped Adam step. It
rounds differently from PyTorch autograd, so its models are different (not
better by construction). It made training fast enough to train on every
pixel of a 2400×2400 image within a 51,200-step cap instead of on tiles of
1 Mpx, and to run 800 epochs on crops. A forward-only kernel scores many
weight candidates at once for the weight search.

Crop-sized images train widths 12 and 32 in parallel, and after an eighth of
training (a sixteenth since `ee14016`) only the better one continues, or
both if they are within 1.5% (since `a82b98a`). Native images train width 48.

### Model coding

A weight matrix may give each column or each row its own grid, so one large
weight no longer forces a fine grid on the whole matrix. A zero slice costs
one escape code. On crops, training prunes the first-layer inputs whose
removal costs fewest estimated bytes (up to a third of each plane's inputs
under 15 bytes, three times, until `6ba3831` raised this to half under 100
bytes, four times); the pruned weights stay zero through quantization.

### Least-squares features

Feature set 3 gives every sample a ridge least-squares prediction fitted over
its causal window (12 rows above at ±12 columns, plus 12 samples to the left),
from 12 causal samples of its own plane and the neighbourhood of each decoded
plane. The network sees the fit, the window's residual spread, and the fit's
errors at four neighbours and on the decoded planes. Feature set 4 adds a
second, 4-row ±4-column window with a stronger ridge, which follows local
structure that the wide window averages away. The window sums are exact
integers and the solve is a fixed-order binary64 LDLᵀ, so the decoder
reproduces every fit bit for bit; the backend refuses fast-math builds.

Together, sets 3 and 4 made the files 2.98% smaller on the crops and 2.91%
smaller on the native images. They also cost decode time, because the decoder
repeats every fit (see Speed).

## Fresh cohort and protocol

`fetch_tecnick_fresh.py` builds the cohort from the same pinned TESTIMAGES
2400×2400 RGB archive as round 2. It takes the 16 photographs that round 2
left unused, makes a centered 512×512 crop of every one, and keeps the full
image of four of them (positions 0, 4, 8 and 12 in name order: baloons,
ducks, pencils_b and tomatoes_b). The four native images are therefore the
same scenes as four of the crops. The manifest's SHA-256 is `0798799c…`.

`fresh_validation.py` measures every image in the manifest with the default
encoder, checks the source hashes, decodes every file and compares it with
the source, and records PNG, WebP, JPEG XL effort 9 and JPEG XL effort 10
sizes. It ran on commit `6c0131e` with a clean tree (the receipts record
both). Our encoder had never run on these images before that, and nothing
changed after it ran. paq8px compressed these images earlier the same day,
while development used only the round-2 images; its sizes did not feed any
setting.

## Fresh results

Summary file: [`2026-10-26-tecnick-fresh.json`](../results/2026-10-26-tecnick-fresh.json).
Per-image tables are in [benchmarks](../benchmarks.md#photos-fresh-tecnick-cohort).

| codec | 16 crops | our saving | 4 native | our saving |
|---|---:|---:|---:|---:|
| PNG | 3,738,879 | 58.90% | 15,427,276 | 58.41% |
| WebP lossless | 2,733,334 | 43.78% | 11,848,508 | 45.85% |
| JPEG XL effort 9 | 2,147,237 | 28.44% | 8,905,825 | 27.96% |
| JPEG XL effort 10 | 2,140,363 | 28.21% | 8,346,560 | 23.13% |
| paq8px -9L | 1,539,595 | 0.20% | 6,518,112 | 1.57% |
| **ours** | **1,536,577** | | **6,415,937** | |

JPEG XL effort 10 was the best baseline on all 20 files. Per image, the crops
saved 22.76% to 34.24% against it and the native images 19.50% to 25.03%. In
bits per sample: 0.977 against 1.361 on the crops (paq8px 0.979), and 0.743
against 0.966 on the native images.

Against paq8px, per image:

- **Crops.** Ours were smaller on 10 of 16, from 0.13% (tools_b) to 3.36%
  (roof). paq8px was smaller on billiard_balls_b (ours +5.52%), pencils_b
  (+3.38%), baloons (+2.39%), tomatoes_a (+2.35%), tomatoes_b (+1.90%) and
  ducks (+1.22%). billiard_balls_b is the only crop that kept width 32; its
  9.7 kB of weights are 13.7% of the file.
- **Native images.** Ours were smaller on all four, by 0.61% (pencils_b) to
  2.64% (tomatoes_b). Their weights take 19 to 22 kB, 1.2% to 1.4% of the
  file. paq8px won the crop of all four of these scenes, so at full size the
  weights' smaller share turned each result around.

### Fresh against development

| | development (`6c0131e`) | fresh (`6c0131e`) |
|---|---:|---:|
| crops: saving vs best baseline | 28.73% | 28.21% |
| crops: ours vs paq8px | −0.64% | −0.20% |
| native: saving vs best baseline | 30.48% | 23.13% |
| native: ours vs paq8px | −5.67% | −1.57% |

The crops generalized almost exactly. The native gap to paq8px shrank from
5.67% to 1.57%. With four native images, all from scenes different from the
development set, this run cannot separate tuning on the development images
from differences between scenes; the fresh number is the one to quote.

### Re-measure at `a82b98a`

Two commits followed the first fresh run, both developed on the development
images only:

- `f4a727f` builds the three planes' features in parallel, starts the
  training workers while the least-squares fits run, and moves the Adam
  kernel's chunk sums onto every GPU core, none of which changes a file. It
  also strides the weight search on large images over 2^19 rows instead of
  2^21, which moved the development native total by −0.001%.
- `a82b98a` lets both crop widths finish training when the race at an
  eighth of training is within 1.5%. Of the development crops only screws
  came that close (0.97% smaller, at twice the encode time).

`fresh_validation.py` then re-ran on the fresh cohort with a clean tree
(summary: [`2026-10-27-tecnick-fresh.json`](../results/2026-10-27-tecnick-fresh.json)).
No fresh crop came within the race margin, so all 16 crop files are
byte-identical to the first run. Only the weight-search stride changed the
native files: baloons −187 bytes, ducks +43, pencils_b +113 and tomatoes_b
−11, for a total of 6,415,895 bytes (42 fewer). Every total percentage above
still holds; the native median saving against JPEG XL effort 10 moved from
23.82% to 23.81%. All 20 files decoded exactly, and the x86_64 build again
decoded them to the same pixels and re-encoded them to identical bytes.

| | `6c0131e` | `a82b98a` |
|---|---:|---:|
| encode, 512×512 | median 11.2 s (10.9 to 18.1) | median 10.3 s (10.1 to 16.1) |
| encode, 2400×2400 | median 66.1 s (64.8 to 67.0) | median 51.3 s (49.6 to 52.5) |
| decode, 512×512 | median 0.21 s (0.18 to 0.32) | median 0.21 s (0.18 to 0.27) |
| decode, 2400×2400 | median 5.27 s (5.07 to 5.49) | median 5.14 s (4.98 to 5.58) |

The decoder did not change, so the decode differences are run-to-run noise.
Inside a native training process, features now take about 7 s (was 8.6 s),
Adam steps 28 s (31 s), and the quantization-aware passes and weight search
7.6 s (14.4 s). Inside a crop's, they take 0.4 s, 7.4 s and 1.6 s.

### Pruning check at `6ba3831`

`6ba3831` retuned input pruning on the development crops only: byte limits
from 8 to 200 bytes and none, fractions from 0.34 to 0.7, and three or four
rounds. The cost estimate is taken before the network adapts, so it
overstates what a removal costs, and the old 15-byte limit was too strict.
The development crop files came out 0.26% smaller. After that choice the 16
fresh crops were encoded once as a check (sizes and each encode's own decode
check, not a receipt run): 1,536,577 → 1,533,285 bytes (−0.21%, 12 of 16
smaller). In total the crops are now 0.41% smaller than paq8px (was 0.20%),
but on 9 of 16 images instead of 10: tools_a grew 0.60% and now trails
paq8px by 0.33%. The median crop encode was 9.8 s (10.3 s at `a82b98a`) on a
machine that was not idle. Native images do not prune and are unchanged.

### Weight rounding at `e48114b`

`e48114b` rounds weights in the weight search by GPTQ error feedback
([design 15](../design/15-image-local-probability.md#actual-rate-selection)),
developed on the development crops only:

| variant, 12 development crops | total | vs `6ba3831` |
|---|---:|---:|
| `6ba3831` | 1,147,653 | |
| first layer only, input Gram matrix | 1,144,167 | −0.30% |
| all four matrices, input Gram matrices | 1,142,772 | −0.43% |
| with per-row gradient weights | 1,142,177 | −0.48% |
| with the bias compensated too (`e48114b`) | 1,141,883 | −0.50% |

Every crop was smaller, by 0.31% (cushions) to 0.82% (screws). Model bytes
fell from 44,553 to 37,850 while the coded pixels grew by 933 bytes. The
search now substitutes one tensor into the current best state per candidate
instead of re-rounding every tensor (the same candidates), so each matrix's
roundings are computed once per sweep. The median crop encode went from 9.8 s
to 10.7 s on a machine that was not idle. Large images now search on 2^18
rows instead of 2^19: on three development natives with GPTQ that cost
0.024% and took the weight search from 15 s to 10 s. The 12 development
natives came out 0.09% smaller in total (26,874,914 bytes); their encodes
took a median 56.8 s here, while background system processes were using
several cores.

The 16 fresh crops were then encoded once as a check (sizes and each
encode's own decode check, not a receipt run): 1,533,285 → 1,525,528 bytes
(−0.51%, all 16 smaller), in line with the development crops. In total the
fresh crops are now 0.91% smaller than paq8px (was 0.41%), on 10 of 16
images: tools_a is ahead again, by 0.02%. The median crop encode was 10.9 s
(9.8 s at `6ba3831`).

### One quantization-aware snapshot at `86ac778` and `0e1069b`

With GPTQ rounding, the weight search picked the 10-bit quantization-aware
snapshot for 1 of 36 development native planes (multimeter's green), and 7
of 36 crop planes. `86ac778` dropped it on large images. On the 12
development natives only multimeter changed, coding 0.03% smaller without
the 10-bit snapshot (the estimate had preferred it). In total the natives
came out 0.003% smaller. The quantization-aware passes and weight search took
a median 7.2 s instead of 10.1 s, and the median encode was 51.9 s (56.8 s
in the run before, with the machine loaded in both).

The crops behaved the same way. Without the 10-bit snapshot (`0e1069b`) the
seven planes that had picked it chose the float model or the 12-bit
snapshot, and the 12 development crops coded 0.047% smaller (screws −0.27%;
apples, +0.03%, was the only crop that grew). The estimates had favored the
10-bit weights by slightly more than the coder confirmed. The weight-search
phase fell from 2.5 s to 1.9 s and the median crop encode from 10.7 s to
10.2 s. A single 11-bit snapshot instead cost 0.016%.

### Race timing and kernel gradients at `ee14016` and `5329246`

`ee14016` scores the width race after a sixteenth of training instead of an
eighth. At an eighth, width 32 trailed on cards_a by more than the 1.5%
margin and stopped; at a sixteenth it led there by 2.0% and finished 1.2%
smaller. It stayed within the margin on screws, whose file did not change,
and trailed by 2.9% to 10.0% on the other ten crops, which also did not
change. The development crops came out 0.12% smaller, and the median crop
encode fell from 10.3 s to 9.7 s because a dropped width now stops sooner.

`5329246` takes the per-sample output gradients that weight GPTQ's curvature
from the trainer's forward-backward kernel instead of PyTorch autograd. The
kernel rounds differently, so the files moved slightly: +12 bytes on the
development crops and +277 bytes (+0.001%) on the natives. The weight search
went from 1.80 s to 1.70 s per crop and from 7.1 s to 6.8 s per native image.
The median encode was 9.5 s per crop and 51.2 s per native image, again with
background system processes busy.

`a0001ce` builds each plane's features in up to 16 row bands on threads and
writes the trainer's binary16 rows directly. All 24 development files were
byte-identical. On the natives, features and upload took about 3.7 s instead
of 5.7 s, the training phase a median 41.0 s instead of 43.2 s, and the
whole encode 48.7 s instead of 51.2 s. Crops did not change (9.4 s).

The floor competes with training for the CPU. Three development natives
encoded without it trained in 37.5 to 38.5 s instead of 41.1 to 44.9 s; the
one pair measured without other benchmarks running (coins) took 3.4 s
longer with the floor beside it.

### Weight search speed at `b715615` and `e3333b0`

`b715615` scores weight candidates from the current best state's cached
layer outputs: a candidate changes one tensor, so its forward pass starts at
the first layer it changes. The estimates are the same numbers, and all 24
development files were byte-identical. The weight search took 1.55 s
instead of 1.72 s per crop and 6.0 s instead of about 6.7 s per native image.

`e3333b0` sums the GPTQ curvatures' weighted Gram matrices in a Metal kernel
(simdgroup tiles of the upper triangle, about three times faster than the
batched PyTorch products) over every second estimator row. The development
crops grew by 7 bytes in total (+0.0006%) and the natives by 1,304 bytes
(+0.005%, at most 0.024% on one image). The weight search took 1.11 s per
crop and 4.5 s per native image.

### Run-to-run noise

Each encode is deterministic, so a re-run says nothing about noise. A
learning rate changed by a millionth (0.014999985 or 0.01500003 instead of
0.015) acts as a different seed. At `e3333b0` the development crops then
coded 1,139,942, 1,145,655 and 1,143,098 bytes: a spread of 0.5%, nearly all
from the width race flipping on chairs, cards_a and screws. The natives
moved by −0.09% to +0.16% per image and +0.007% in total. One run's crop
change under about 0.3%, or native change under about 0.05%, is not evidence
by itself, so the crop decisions below use the mean of three runs.

### Binary16 scratch, floor screen and race scoring at `217ea41`, `3c277d5` and `c5f3f91`

`217ea41` keeps the trainer's per-sample scratch rows (the activations and
gradients that pass from the forward-backward kernel to the gradient
reduction) in binary16 for nets at least 48 wide, with the gradients scaled
by 2^6 to keep them out of the subnormal range. A step got 14% faster at
width 48 (batch 8192), 9% at width 32 and 3% at width 12. On the crops,
over three runs each, binary16 rows coded 0.11% larger with the scale and
0.27% larger without it, so widths 12 and 32 keep float rows and the same
arithmetic as before. The natives coded 0.035% larger in total (per image
−0.13% to +0.19%), the size of the noise; unscaled rows had come out 0.034%
smaller. Native optimization took a median 25.6 s instead of 31.2 s.

`3c277d5` builds a quick floor first: the SCDX document with only a JPEG XL
effort 7 asset. It skips the full floor, which took 16.6 to 25.9 s of all
cores beside a native image's training, when the image has at least 4,096
sampled colours and the first learned estimate is at least 10% smaller.
Training does not depend on the floor, and no development file changed.
All 12 natives and 11 of the 12 crops were screened; chairs, whose estimate
was within 10%, built the full floor.

`c5f3f91` scores the width race by each width's mean training bits over the
last eighth of the epochs before the race, plus its model bytes on 10-bit
grids, and meets after a 32nd of training with a 3% margin
([design 15](../design/15-image-local-probability.md) has the details). The
race flips came from scoring one estimate of the checkpoint's weights: the
two widths' ratio of such estimates moved by up to 15% between runs, the
ratio of averaged bits by at most 1.5%. Over the three learning rates the
development crops coded 1,139,942, 1,140,753 and 1,141,756 bytes, a mean
0.18% smaller (1,140,817 against 1,142,898), and width 32 finished and won
on cards_a and screws every time. At the default rate all 12 files were
unchanged.

Timing with all three, one image at a time:

| | `e3333b0` | `c5f3f91` |
|---|---:|---:|
| crop encode, median | 9.0 s | 8.7 s |
| crop encode, mean | 10.5 s | 11.3 s |
| crop floor phase | 1.3 to 2.3 s | 0.10 to 0.18 s (chairs 1.9 s) |
| native encode, median | 48.4 s | 39.5 s |
| native training process, median | 39.3 s | 31.4 s |
| native floor phase | 16.6 to 25.9 s | 0.8 to 1.0 s |

The crops' mean rose because both widths now finish on cards_a and screws
(and, at the default rate, on cushions), at 19 to 20 s each. Two native
encodes (coins and fence) ran while a system process held a core and took
78 to 80 s; the other ten took 38.1 to 41.3 s.

### Shared loss exponentials at `da035fe`

`da035fe` computes the mixture loss's sigmoid derivatives, its 1 / expm1
term and the component posteriors from exponentials the loss already
takes, instead of separate ones. Alternating the two kernels in one
process, the forward-backward kernel took 49.2 instead of 52.3 µs at width
12 and 102.7 instead of 106.5 µs at width 32 (batch 4096), and 277 instead
of 303 µs at width 48 (batch 8192). That is 3% to 6% of a training step,
less than whole encodes vary on this machine, and no development encode
measured faster. (On battery, in Low Power Mode, the same width-48 kernel
took anywhere from 290 to 540 µs between runs, so no timing here comes from
battery power.) The sizes moved by rounding only: the natives coded 26,884,291
bytes (−0.003%), and the crops a mean 1,141,748 over the three learning
rates (+0.08% against 1,140,817), with chairs taking its larger race
outcome in all three runs.

### Re-measure at `da035fe`

Every commit from `cdedf8e` to `da035fe` was developed on the development
images only. `cdedf8e` solves the two fit windows together in the backend
(bit-identical fits, a slightly faster decode) and `1ffa95a` starts the floor
once the training features are on the GPU (byte-identical); the others are in
the table above. `fresh_validation.py` then re-ran on the fresh cohort at
`4469a93` (the notes commit after `da035fe`, the same code) with a clean
tree, one image at a time, on AC power (summary:
[`2026-10-27-tecnick-fresh-da035fe.json`](../results/2026-10-27-tecnick-fresh-da035fe.json)).

| | `a82b98a` | `da035fe` |
|---|---:|---:|
| 16 crops, bytes | 1,536,577 | **1,523,239** (−0.87%) |
| crops vs JPEG XL effort 10 | 28.21% smaller | 28.83% smaller |
| crops vs paq8px -9L | 0.20% smaller (10/16) | 1.06% smaller (11/16) |
| 4 natives, bytes | 6,415,895 | **6,404,016** (−0.19%) |
| natives vs JPEG XL effort 10 | 23.13% smaller | 23.27% smaller |
| natives vs paq8px -9L | 1.57% smaller (4/4) | 1.75% smaller (4/4) |
| encode, 512×512 | median 10.3 s (10.1 to 16.1) | median 8.6 s (8.4 to 19.0) |
| encode, 2400×2400 | median 51.3 s (49.6 to 52.5) | median 40.0 s (39.4 to 40.7) |
| decode, 512×512 | median 0.21 s (0.18 to 0.27) | median 0.18 s (0.16 to 0.23) |
| decode, 2400×2400 | median 5.14 s (4.98 to 5.58) | median 4.77 s (4.64 to 5.74) |

The same commits had made the development crops 0.79% smaller and the
development natives 0.05% smaller (1,150,602 to 1,141,466 and 26,898,027 to
26,884,291 bytes), so the fresh cohort moved the same way. 15 of the 16
fresh crops got smaller. billiard_balls_b, which kept width 32 at `a82b98a`
and lost to paq8px by 5.52%, kept width 12 with 3.1 kB of weights instead of
9.7 kB, came out 3,835 bytes (5.4%) smaller, and now beats paq8px by 0.20%.
cards_b and clips now finish both widths and keep width 32 (1,547 and 1,324
bytes smaller), which is why their encodes took 19 s. ducks, 1,382 bytes
(1.6%) larger, is the only crop that grew. Among the natives baloons came out
9,201 bytes smaller, tomatoes_b 3,522 and pencils_b 205, and ducks 1,049
larger. paq8px still wins five crops (baloons, ducks, pencils_b, tomatoes_a
and tomatoes_b), by 0.80% to 2.82%. The stored weights take 2.4 to 7.7 kB per
crop (2.0% to 6.8% of a file) and 16.0 to 18.4 kB per native image (1.0% to
1.1%).

All 20 files decoded exactly, and the x86_64 build decoded them to the same
pixels and re-encoded each to identical bytes.

Inside a native training process, features took about 4.2 s (7 s at
`a82b98a`), Adam steps 23.6 s (28 s), and the quantization-aware pass and
weight search 4.0 s (7.6 s). The floor took 0.6 to 0.8 s instead of building
beside training, and the final decode check 4.7 s. Inside a crop's training
process, the three took a median 0.29 s, 6.4 s and 1.1 s (0.36 s, 7.5 s and
1.6 s).

### Against the first neural build (`f53cb0f`)

`f53cb0f` is the commit that added the per-image neural photo codec, as it
stood on Oct 26 before rounds of training, coding and speed work. It had
never run on the fresh cohort. It ran from its own worktree with its own
native backend build, on the same 20 images, one at a time on AC power,
after the `da035fe` re-measure. Each image went through `compress.compress`
and `compress.decompress` (no baselines re-run). Summary:
[`2026-10-27-tecnick-fresh-f53cb0f.json`](../results/2026-10-27-tecnick-fresh-f53cb0f.json).

| | `f53cb0f` | `da035fe` |
|---|---:|---:|
| 16 crops, bytes | 1,652,665 | **1,523,239** (−7.83%) |
| crops vs paq8px -9L | 7.34% larger (0/16 smaller) | 1.06% smaller (11/16) |
| 4 natives, bytes | 7,011,154 | **6,404,016** (−8.66%) |
| natives vs paq8px -9L | 7.56% larger (0/4 smaller) | 1.75% smaller (4/4) |
| encode, 512×512, median | 49.5 s | 8.6 s |
| encode, 2400×2400, median | 205.9 s | 40.0 s |
| decode, 512×512, median | 0.085 s | 0.18 s |
| decode, 2400×2400, median | 1.45 s | 4.77 s |

Every one of the 20 images got smaller, by 4.5% to 12.6% (crops) and 7.1% to
10.6% (natives). All 20 `f53cb0f` files decoded exactly. `f53cb0f` used
feature set 1, kept width 16 or 24, and trained on at most 1,048,576 sampled
pixels. Today's files use feature set 4, whose two least-squares fits the
decoder repeats per sample, and width 48 on the natives, so they decode two
to three times more slowly even though encoding got five times faster.

## Speed

All timings are from one Apple M4 Pro (8 performance and 4 efficiency cores,
16-core GPU). The fresh encodes ran one image at a time after the paq8px
runs had finished.

| | round 2 validation | fresh, round 3 |
|---|---:|---:|
| encode, 512×512 | median 110 s | median 11.2 s (10.9 to 18.1) |
| encode, 2400×2400 | median 642 s | median 66.1 s (64.8 to 67.0) |
| decode, 512×512 | 0.62 to 0.79 s (v3) | median 0.21 s (0.18 to 0.32) |
| decode, 2400×2400 | 13.1 to 16.1 s (v3) | median 5.27 s (5.07 to 5.49) |

Round 2 ran up to three encoders at once and decoded the single-stream v3
format, so this is not a controlled comparison. Round 2's files, transcoded
to v4, decoded in 0.12 s per crop and 2.2 s per native image
([speed notebook](2026-10-25-encode-speed.md)). Today's files decode more
slowly than that because the decoder repeats two least-squares fits per
sample and the native images use width 48.

Where an encode spends its time now (fresh receipts; phases overlap):

| phase | crop | native |
|---|---:|---:|
| least-squares fits (backend, all cores) | 0.1 s | 1.7 s |
| SCDX floor, beside training | 1.3 to 2.3 s | 11 to 16 s |
| training process: features, Adam, quantization-aware passes, weight search | about 10.2 s | 54 to 57 s |
| complete-file encode of the finalists | 0.08 s | 2.3 s |
| final decode check | 0.2 s | 5.3 s |

Inside a native training process, features take about 8 s, Adam steps about
31 s, and the quantization-aware passes and weight search about 14 s. Inside
a crop's, features take about 0.4 s, Adam 7 to 8 s, and the
quantization-aware passes and weight search about 1.7 s.

### Trainer kernels (`a1ed669`)

Three changes to the fused kernels, none of which changes any arithmetic:
the parameter block sits in threadgroup memory with its affine column split
off, so hidden and head rows load as float4 runs; each simdgroup of the
gradient reduction handles two tile rows and shares each error-tile load
between them; and every plane passes its own input count, so the first layer
stops at that count and the reduction skips the narrower planes' all-zero
tiles. Each accumulator sees the same multiply-adds in the same order.
Gradients, bits, 40 steps with quantization-aware training, and evaluator
scores matched the previous kernels exactly.

| step, feature set 4, three planes | before | after |
|---|---:|---:|
| 512×512, width 12 | 108.8 µs | 103.6 µs |
| 512×512, width 32 | 214 µs | 191 µs |
| 2400×2400, width 24 | 341 µs | 312 µs |
| 2400×2400, width 48 | 588 µs | 561 µs |

## Tried and dropped

Unless stated, these were measured as estimated bytes (model bytes plus
float-model bits on the image) on development crops, which tracks the coded
files within about 0.04%.

- **Decoder-side secondary estimation** (an adaptive correction of the coded
  probabilities, as in paq8px's SSE/APM stages): at best 0.07% smaller. Not
  worth a format change.
- **Coder overhead.** The arithmetic coder and stream table cost about 42
  bytes per file. Nothing to gain.
- **Other weight coders** saved 0.02% and 0.29% of the model bytes, a
  negligible share of the file.
- **Rate-aware GPTQ rounding**: choosing between the two nearest grid points
  by curvature-weighted error plus the weight's Rice code length, at 0.3 to 3
  times the second-order exchange rate and with a smooth rate model: every
  variant within 0.004% of plain GPTQ on the development crops.
- **An L1 rate penalty on the weights** during training: +0.086% at α = 1 and
  +3.12% at α = 4.
- **Width 8** on crops was worse than width 12.
- **Native batch size, learning rate, and a doubled sample budget**, on
  development native images: at most 0.5% smaller, not enough for their cost.
- **A third least-squares window** on top of feature set 4, five shapes and
  ridge scales, 16 runs each: from −0.20% to +0.23%. The best (−0.20%) was
  mixed per image and adds features to every plane; the direction looks
  saturated.
- **Quantization-aware schedules**, on eight development crops at both
  widths: adding an 8-bit snapshot changed nothing (it was never chosen);
  skipping the 12-bit snapshot cost 0.53%; doubling the quantization-aware
  epochs changed size by +0.02%; doubling their learning rate by +0.03%.
- **Pruning variants**, on the development crops: pruning a column's affine
  weight with it cost 0.33%; pruning the native images at the crop settings
  saved 0.01% on three development natives.
- **Halving the quantization-aware epochs**: +0.015% on the development crops
  and +0.011% on three development natives, for about 5% of encode time.
  That is about the exchange rate of the epoch budget itself, so it stayed.
- **Width 16** on crops cost 0.31%; choosing each crop's best width by
  oracle would save only 0.14% over the race.
- **Trainer kernel variants** with no gain: a feature-major scratch layout,
  threadgroups other than 256 threads, reduction chunks other than 512
  rows, summing gradient chunks inside the Adam kernel, and a simdgroup
  matrix forward pass (correct but slower).
- **Averaging the weights**: an exponential moving average over the last
  quarter or eighth of float training (decay 0.9 to 0.97): complete files
  within ±0.015% of the final weights' on the development crops.
- **Learning rate** 0.01 or 0.022 instead of 0.015: complete files +0.84%
  and +0.26% on the development crops. Picking each crop's best of every
  training variant in this list would have saved only 0.10%.
- **Other race partners for width 12.** Alone, widths 20 and 24 cost 0.28%
  and 0.30% against the race. Racing 12 against 24 instead of 32 saved 0.08%
  on the development crops; on those and 12 more crops from the centers of
  the development natives, with a 3% margin, 0.10%, while the mean encode
  grew from 11.2 s to 13.0 s. Racing 12, 20 and 32 changed nothing.
- **Pruning hidden units** with the input-pruning estimate (up to a quarter
  of each layer per round, under 30 or 100 bytes): complete files +0.07%
  and +0.10%.
- **Decoder dense layers** stored transposed for NEON multiply-accumulates:
  the same output, and no faster on one thread (28 to 29 s for a native
  image either way).
- **A Kodak-trained prior** as the starting weights of width 12 instead of
  the per-image initialization: +0.57% on the development crops, and worse
  with fewer epochs.
- **Longer crop training or a smaller crop batch**: 1,200 epochs instead of
  800 coded the development crops 0.26% larger in total (the race chose
  width 12 on cards_a and screws; at the same width, 0.11% smaller) for 1.4
  times the encode time; 1,600 epochs 0.31% smaller at 2.4 times; batch 2,048
  0.014% smaller. None is worth its time.
- **Skipping pruned inputs in the trainer kernel**: the same parameters to
  the bit, but no faster at widths 12 and 32 (5% at width 24).
- **A float multiply-add decoder** for the dense layers: no faster than the
  integer one.
- **The floor in its own process**, at normal or lowest CPU priority, so
  that it competes less with the training process: on four development
  natives the same files, and whole encodes within about a second of the
  in-process floor either way (46.5 to 49.2 s against 46.5 to 48.7 s).
- **Binary16 scratch rows at crop widths** 12 and 32: 0.11% larger with the
  2^6 gradient scale and 0.27% without it (means of three runs), for steps
  3% to 9% faster.
- **Decoder progress counters on their own 128-byte cache lines**, published
  after every 1, 4 or 16 samples: no change above the noise (3.8 to 5.3 s
  for one native decode with every variant).
- **Earlier in the round**: the uniform-grid finalist never beat the searched
  model in 47 encodes, so it is now coded only within 0.05% of it; 8- and
  7-bit snapshots were never picked on 20 images and were dropped; a guide fit
  on subsampled rows cost 0.16%.

## Verification

- All 20 fresh files decoded exactly in the validation run.
- An x86_64 build of the unchanged backend, run under Rosetta, decoded all 20
  to the same pixels and re-encoded each to identical bytes
  (`verify_cross_isa.py`; recorded in the summary file). This is two builds
  on one Mac.
- Every paq8px archive decompressed to exactly the source samples before its
  size was counted.
- The test suite (295 tests) passed at `a1ed669` and again at `a82b98a`, and
  (320 tests) at `da035fe`.

## Reproduction

```bash
LP=scripts/research/local_probability
R=docs/results/raw
python $LP/fetch_tecnick_fresh.py --archive PATH/TO/TESTIMAGES-2400-RGB.zip
git checkout 6c0131e   # the measured commit
for c in tecnick-fresh-crop512 tecnick-fresh-2400; do
  python $LP/fresh_validation.py --cohort $c --out $R/local-probability-fs4-$c.json
done
# paq8px v217 at data/research-local-probability/paq8px-v217; split as you like
python $LP/paq8px_baseline.py --phase compress --cohort tecnick-fresh-crop512 \
  --start 0 --limit 8 --out $R/paq8px-tecnick-fresh-crop512-compress-0.json
python $LP/paq8px_baseline.py --phase verify --cohort tecnick-fresh-crop512 \
  --start 0 --limit 8 --out $R/paq8px-tecnick-fresh-crop512-verify-0.json
# ...the same for --start 8, and for tecnick-fresh-2400 one image per process
python $LP/verify_cross_isa.py data/probe/fresh/*.coded \
  --binary data/research-local-probability/nnproto-integer-x86_64 \
  --out $R/neural-cross-isa-tecnick-fresh.json
(cd $LP && python summarize_fresh.py \
  --learned ../../../$R/local-probability-fs4-tecnick-fresh-*.json \
  --paq8px ../../../$R/paq8px-tecnick-fresh-*.json \
  --cross-isa ../../../$R/neural-cross-isa-tecnick-fresh.json \
  --out ../../../docs/results/2026-10-26-tecnick-fresh.json)
# the re-measure: the same with the newer commit and receipt names
git checkout a82b98a
for c in tecnick-fresh-crop512 tecnick-fresh-2400; do
  python $LP/fresh_validation.py --cohort $c --save data/probe/fresh-a82b98a \
    --out $R/local-probability-a82b98a-$c.json
done
python $LP/verify_cross_isa.py data/probe/fresh-a82b98a/*.coded \
  --binary data/research-local-probability/nnproto-integer-x86_64 \
  --out $R/neural-cross-isa-a82b98a-tecnick-fresh.json
(cd $LP && python summarize_fresh.py \
  --learned ../../../$R/local-probability-a82b98a-tecnick-fresh-*.json \
  --paq8px ../../../$R/paq8px-tecnick-fresh-*.json \
  --cross-isa ../../../$R/neural-cross-isa-a82b98a-tecnick-fresh.json \
  --out ../../../docs/results/2026-10-27-tecnick-fresh.json)
# the second re-measure
git checkout 4469a93   # da035fe plus notes
for c in tecnick-fresh-crop512 tecnick-fresh-2400; do
  python $LP/fresh_validation.py --cohort $c --save data/probe/fresh-da035fe \
    --out $R/local-probability-da035fe-$c.json
done
python $LP/verify_cross_isa.py data/probe/fresh-da035fe/*.coded \
  --binary data/research-local-probability/nnproto-integer-x86_64 \
  --out $R/neural-cross-isa-da035fe-tecnick-fresh.json
(cd $LP && python summarize_fresh.py \
  --learned ../../../$R/local-probability-da035fe-tecnick-fresh-*.json \
  --paq8px ../../../$R/paq8px-tecnick-fresh-*.json \
  --cross-isa ../../../$R/neural-cross-isa-da035fe-tecnick-fresh.json \
  --out ../../../docs/results/2026-10-27-tecnick-fresh-da035fe.json)
# the first neural build, from its own worktree with its own backend build
# (measure_checkout.py is newer; it imports the checkout's compress.py)
git worktree add tmp/base-wt f53cb0f
mkdir -p tmp/base-wt/data/corpora
ln -s $PWD/data/corpora/tecnick-fresh-* tmp/base-wt/data/corpora/
(cd tmp/base-wt && python ../../$LP/measure_checkout.py \
  ../../$R/local-probability-f53cb0f-tecnick-fresh.json \
  data/corpora/tecnick-fresh-crop512/*.png data/corpora/tecnick-fresh-2400/*.png)
```

The x86_64 backend is built as in the
[research README](../../scripts/research/local_probability/README.md).

## Limits

- Twenty fresh images from one studio set. Four native images cannot pin
  down how the native result varies by scene.
- paq8px still wins 5 of the 16 fresh crops at `da035fe`, by up to 2.8%.
  Model bytes are 2.0% to 6.8% of a crop file, so small images remain the
  weak point.
- The fresh cohort has now been measured three times (`6c0131e`, `a82b98a`,
  `da035fe`), its crops checked once more after a choice at `6ba3831`, and
  the old build `f53cb0f` run on it for comparison. No setting was chosen
  from its results, but each further look makes it slightly weaker evidence
  than the first. The 62 LPCB camera photos
  ([notes](2026-10-27-lpcb.md)) are a newer fresh test.
- Decoding is still several times slower than JPEG XL, and the fits make it
  slower than round 2's files.
- Timings come from one laptop. Training is not promised to reproduce
  identical models on other GPUs or PyTorch versions; decoding is.
