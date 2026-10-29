# Image-local probability coding: encode speed and threads, 2026-10-25

## Outcome and scope

With `--envelope 3`, the research encoder writes the same bytes as before in
about half the time. On Kodak 01, a full NNEX v3 encode went from 152.6 s to 83.1 s and
produced the identical file (SHA-256 `64b40307…`). The Tecnick building image
(2400×2400) re-encoded to its exact round-2 file in 247 s; round 2 had taken
624 s under different load. Training the two widths at once, overlapping the
SCDX floor, and computing encoder probabilities on all cores account for the
saving. Training is now nearly all of the remaining time.

A new NNEX v4 envelope splits the arithmetic stream by row so the decoder can
use several threads. It carries the same model and the same per-sample
probabilities as v3. Transcoding all 24 Tecnick files added 36 to 57 bytes per
file (0.0036% in total), and the v4 files decoded 3.4 to 5.1 times faster than
the v3 files did with the old backend in the same pass: a median 0.12 s per
crop and 2.2 s per native image. The encoder writes v4 by default;
`--envelope 3` still writes v3, and the frozen validation scripts pin v3.

`--compile-training` is an opt-in that trains through `torch.compile`. It does
not reproduce the default's models, so it is a separate encoder setting.
On six known development images it cut total encode time from 357 s to 235 s
and changed the total size by −0.04%. Training 200 or 100 epochs instead of 400
cut the time further, to 135 s and 79 s, for files 0.28% and 0.62% larger.
None of these is the default.

Every image here is known. These are speed regressions, not compression
evidence, and the frozen Tecnick results are unchanged. Unless stated
otherwise, timings were taken while paq8px jobs occupied three or four of the
twelve cores, so they show scale, not isolated latency.

## Where the time went

Kodak 01 (768×512) before these changes, one encode, one process:

| phase | seconds |
|---|---:|
| SCDX floor | 9.9 |
| train width 16 | 63.3 |
| precision search, width 16 | 4.7 |
| train width 24 | 67.2 |
| precision search, width 24 | 5.8 |
| final decode check | 1.1 |
| **total** | **152.6** |

Training runs about 19,200 Adam steps per width (48 batches of 8,192 samples,
400 epochs). On the laptop GPU (MPS) a step took about 3 ms and was bound by
kernel dispatch, not arithmetic: the same step took 7.1 ms on the CPU.

## Byte-identical changes

- **Widths train in parallel.** Each width trains in its own spawned process
  while the parent builds the SCDX floor. Each process seeds 20261023 exactly as
  the serial loop does. On Kodak 01 the width-16 and width-24 models from the
  concurrent run matched the sequential ones byte for byte (`0a18b7e5…`,
  `40b875ab…`). `--serial` restores one-at-a-time training. Two processes share
  the GPU, so each width trains more slowly (about 77 s instead of 65 s) but
  both finish in that time.
- **Encoder probabilities on all cores.** The encoder knows every pixel, so it
  computes every prediction error first, then every sample's distribution in
  parallel threads. Only the arithmetic coder runs in sample order.
- **Faster symbol search when decoding.** The decoder searches the CDF outward
  from the heaviest mixture component's mean instead of bisecting. The CDF is
  strictly increasing, so both searches return the same symbol.

Backend timings on the Kodak 01 width-24 model (one run each, under load):

| step | before | after, 1 thread | after, all threads |
|---|---:|---:|---:|
| encode v3 | 1.21 s | 0.90 s | 0.15 s |
| decode v3 | 0.91 s | 0.68 s | (single stream) |
| decode v4, 16 streams | | 0.72 s | 0.21 s |

Every thread count wrote the same bytes for its format and decoded the same
pixels. With these changes Kodak 01 encoded
in 83.1 s with `--envelope 3`, and the file matched the earlier one exactly.

## NNEX v4: decoding on several threads

A v3 file is one arithmetic stream, so decoding is strictly sequential. V4 keeps
the v3 header and model and gives row y of every plane to substream y mod S
(S = 16 by default, at most 64). A one-byte count and S varint lengths precede
the substreams, and each substream drops the four terminal zero bytes that v3
writes.

The decoder runs a wavefront. Row y can decode column x once row y-1 has
decoded through column x+3, the widest causal reach. A row also waits for the
previous row of its substream, which holds that substream's coder state. The
thread count never changes the output; the format fixes S. By default the
decoder uses up to S threads, limited to the performance cores on Apple
silicon, because an efficiency core working on one row slows every row below
it. Planes still decode one after another.

V4 encoding writes the same probabilities, so the size difference is only the
stream table and the per-stream flushes. With S = 1 a v4 stream equals the v3
stream minus its four terminal zero bytes, which the tests check.

### Tecnick files, transcoded

Every round-2 Tecnick file was decoded, re-encoded as v3 with the new backend,
and transcoded to v4 using only its in-band model
(`v4_transcode.py`). All 24 v3 files re-encoded byte-identically, at one
thread and with all threads. Timings are best of two, in one pass, with three
paq8px jobs running.

| | 12 crops 512×512 | 12 native 2400×2400 |
|---|---:|---:|
| v4 extra bytes per file | 36 to 39 | 52 to 57 |
| old backend, v3 decode | 0.47 s (0.42–0.60) | 10.5 s (8.7–11.1) |
| new backend, v3 decode | 0.34 s (0.31–0.41) | 8.1 s (6.7–8.5) |
| new backend, v4 decode | **0.12 s** (0.12–0.14) | **2.2 s** (1.9–2.4) |
| v4 decode speedup over old v3 | 3.4–4.3× (median 3.6×) | 4.3–5.1× (median 4.8×) |
| backend encode, 1 thread | 0.36 s | 8.6 s |
| backend encode, all threads | 0.08 s | 1.5 s |

Medians, with ranges in parentheses. The totals are 30,360,894 bytes in v4
against 30,359,801 in v3 (+1,093). Decode times include model parsing,
temporary files, the subprocess, and the pixel checksum.

A separate pass under the same load timed each v4 file against JPEG XL effort 9
on the same pixels (`decode_timing.py`, best of two). V4 took 2.9 to 4.6
times as long as single-threaded JPEG XL (median 3.8×: 4.0× on crops, 3.6× on
native images). JPEG XL also speeds up with threads: with eight, it decoded
12.6 to 23.6 times faster than v4 (median 17×). The validation run had measured
v3 at 12 to 29 times single-threaded JPEG XL in a quieter pass whose absolute
times ran slower, so compare only ratios from the same pass.

Thread scaling on the building file (best of two, same load):

| threads | v3 encode | v4 encode | v4 decode |
|---:|---:|---:|---:|
| 1 | 8.77 s | 8.62 s | 8.14 s |
| 2 | 4.48 s | 4.41 s | 4.29 s |
| 4 | 2.46 s | 2.39 s | 2.69 s |
| 8 | 1.72 s | 1.57 s | 2.19 s |
| default | 1.38 s | 1.22 s | 2.18 s |

V3 decoding stayed at 8.0 s at every setting because it has one stream. The
default uses every core for encoding and the eight performance cores for
decoding. Three of those cores were busy with paq8px, which likely explains why
decoding stopped improving after four threads; a quiet run is still to do.
On the apples crop, v4 decoding went from 0.42 s on one thread to 0.14 s.

## Optional: compiled training and shorter schedules

The six known development images from round 2 (SIPI aerials 2.1.01, 2.1.05,
2.1.09; Kodak 01; SIPI 4.2.03 and 4.1.05), one encode each, v4 files. Every
file decoded exactly. The default row uses the identical-model encoder above.

| setting | total bytes | vs default | per image vs default | total s | median s |
|---|---:|---:|---:|---:|---:|
| default | 2,328,887 | | | 357 | 64.6 |
| compiled, 400 epochs | 2,328,007 | −0.04% | −0.26% to +0.12% | 235 | 40.5 |
| compiled, 400 epochs, width 16 only | 2,332,019 | +0.13% | −0.21% to +0.48% | 166 | 28.9 |
| compiled, 200 epochs | 2,335,452 | +0.28% | +0.09% to +1.02% | 135 | 23.2 |
| compiled, 100 epochs | 2,343,214 | +0.62% | +0.27% to +1.76% | 79 | 13.7 |

For scale, Kodak 01 took 152.6 s before any of this work and 86.9, 59.4, 41.0,
34.1, and 18.7 s in the rows above, in order.

- **Compiled training.** `torch.compile` fuses the loss and its backward pass
  into fewer GPU kernels: a training step fell from about 3.0 ms to 2.1 ms. The
  timings include compilation. Fused kernels round differently, so the
  models differ; per-image sizes moved both ways, by up to 0.26%. Six images
  cannot tell that apart from zero. PyTorch's fused Adam also changed the
  models and was dropped; its foreach and per-tensor paths matched each other
  and gave no speedup.
- **Width 16 only.** Training one model leaves the GPU to one process. Kodak 01
  and 4.2.03 pick width 24 and grew 0.48% and 0.17% against the default. The
  four images that pick width 16 produced exactly the compiled 400-epoch files.
- **Fewer epochs.** Size grows steadily as the schedule shortens; Kodak 01 is the
  most sensitive (+1.0% at 200 epochs, +1.8% at 100).

The encoder keeps the measured default. A faster preset would need its own
frozen validation before any claim about it.

## Native-size check

The Tecnick building image (2400×2400), known since round 2, encoded three ways
with three paq8px jobs running:

| run | bytes | total s | floor | training, each width | precision search | final check |
|---|---:|---:|---:|---:|---:|---:|
| round 2, old code | 1,415,999 | 624 | | | | |
| v3, current code | 1,415,999 | 247 | 104 | 205 | 29–33 | 8.2 |
| v4 default | 1,416,054 | 225 | 104 | 203–204 | 18 | 2.3 |
| v4, compiled training | 1,423,669 | 166 | 107 | 143–144 | 18 | 2.3 |

The v3 file matched the round-2 file exactly (SHA-256 `abb255f5…`). Round 2
ran three encoders at once on the GPU; this run shared the CPU with paq8px
instead, so 624 s against 247 s is not a controlled comparison. Kodak 01 above
is the controlled one. The phases overlap: the floor builds while both widths
train, and each width runs its own precision search.

Compiled training made this file **0.54% larger** than the default, a larger
loss than on any development image. That is why it stays opt-in.

## Verification

- The full test suite (230 tests) passed. The 118 in
  `test_local_probability.py` and `test_probability_compress.py` include golden
  hashes from the pre-change backend for v3 at one and four threads, v4 round
  trips across shapes, stream counts, and thread counts, truncated substreams
  under threads, and identical bytes from serial and parallel training.
- ThreadSanitizer, AddressSanitizer, and UndefinedBehaviorSanitizer builds of
  the backend ran the v3 and v4 paths cleanly, including the truncated-stream
  failure path.
- An x86_64 build under Rosetta matched the v3 golden hashes and produced the
  same v4 bytes as the arm64 build in the tests. It also decoded all 24 Tecnick
  v4 files exactly and re-encoded each to identical bytes. This is two builds
  on one Mac.

## Reproduction

```bash
LP=scripts/research/local_probability
# One development encode; add --compile-training, --epochs 200 or 100, or
# --hidden 16 for the other rows of the sweep.
python $LP/compress.py data/corpora/kodak/kodim01.png \
  --out data/probe/kodim01.coded --receipt docs/results/raw/kodim01.json
# Identical bytes against the round-2 file
python $LP/compress.py data/corpora/tecnick-2400/building.png --envelope 3 \
  --out data/probe/building-v3.coded --receipt docs/results/raw/building-v3.json
cmp data/probe/building-v3.coded data/probe/tecnick-2400-building.coded
# Transcode the round-2 files and time the backends
python $LP/v4_transcode.py data/probe/tecnick-*.coded --repeats 2 \
  --legacy-binary OLD_BACKEND --save-v4 data/probe/v4 \
  --out docs/results/raw/nnex-v4-transcode-tecnick.json
python $LP/decode_timing.py data/probe/v4/*.v4.coded --repeats 2 --jxl-threads 8 \
  --out docs/results/raw/tecnick-decode-timing-v4.json
```

`OLD_BACKEND` is the pre-change backend. It was built from the round-2
`nnproto_integer.cpp`, which is not kept in the repository; the summary records
its binary hash. The [summary](../results/2026-10-25-encode-speed.json) holds
every receipt above: sizes, file hashes, phase timings, the transcode rows, and
the thread-scaling runs.

## Limits

- Training still dominates encoding. The remaining large speedups change the
  trained model (fewer epochs, one width, compiled kernels), so they trade some
  size for time.
- V4 decoding gained little past four threads in these runs, and it is still
  about 4× slower than single-threaded JPEG XL.
- Timings come from one laptop with other jobs running.
- `torch.compile` output depends on the PyTorch version and device, as does
  ordinary training; neither is promised to reproduce models across machines.
