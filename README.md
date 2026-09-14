# printscale

Upscale scanned photographs for large-format print, with measured grain restoration.

The model is one stage of six. Most of the perceived quality comes from the
finishing chain, and most of the wasted time comes from committing compute
before a cheap crop test says it will pay.

```
printscale inspect scan.jpg                       # measure before deciding
printscale bake    scan.jpg -o out/ --crop 860,470,384
printscale run     scan.jpg -o out/ --grain 0.55 --width-mm 800
printscale finish  scan.jpg -o out/ --grain 0.8   # retune, model output cached
```

## Before you start

**A better scan beats every technique here.** A flatbed at 1200 dpi on an
original 8×10 print gives roughly 9,600 px on the long edge of genuinely
measured detail. Nothing in this repository competes with that. `printscale
inspect` prints the ppi table so the size decision stays on numbers:

```
   print width   effective   reading
       800 mm      173 dpi   comfortable
      1200 mm      115 dpi   fine
      1500 mm       92 dpi   mural territory
```

**Everything above the source Nyquist is synthesised, not recovered.** It is
constrained by the surrounding pixels and by what photographs look like, but no
measurement establishes that a particular detail was there. Do not present the
output as an evidential or forensic record. Faces below roughly 150 px across
are the worst case — the model invents structure rather than interpolating
texture, and will confidently produce someone else.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
printscale weights --fetch x2plus
```

Torch comes from the default index. For a CPU-only box:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## What each stage does

| Stage | Why it exists |
|---|---|
| Neutral greyscale | Colour management finds no cast to correct |
| Grain restoration | GAN output is locally smooth and reads as plastic |
| Local contrast | Large-radius unsharp; adds presence without haloing |
| Print sharpen | Edge-radius unsharp, threshold-gated so it does not amplify stage 2 |
| Dust removal | Last, on the 16-bit master, after quantisation cannot reshape a speck |

Order is load-bearing. Grain goes back before sharpening because the threshold
gate is calibrated against the grain that will be there.

## Choosing a model

Run `printscale bake` first. It takes about ninety seconds and produces a
contact sheet of a detail region at 1:1 — look at it, because the timings will
not tell you about waxy skin.

| Model | Params | Cost | Character |
|---|---|---|---|
| `lanczos+usm` | — | instant | Baseline. Keeps grain, stays soft. Sometimes wins |
| `x2plus` | 16.7M | ~116 s/MP | Default. Best photographic detail at 2× |
| `x4plus` | 16.7M | ~478 s/MP | Same body; only worth it if you truly need 4× |
| `general-x4v3` | 1.2M | ~33 s/MP | Fast, denoises film grain into plastic |

Measurements are on two 2.1 GHz cores. On a GPU raise `--tile` until VRAM
complains and leave `--threads` at 0.

**Prefer the native scale.** Running x4 and downsampling to 2× discards three
quarters of what the model synthesised and adds a resampling step on top.

**x2plus costs about a quarter of x4plus per input megapixel.** Same 23-block
body; the x2 weights apply `pixel_unshuffle` first, so `conv_first` takes 12
channels and the body runs at a quarter of the pixel count. If your measured
ratio is not near 4.0, something is wrong with the setup.

## Traps this package already handles

- **`basicsr` and `realesrgan` pin `torchvision.transforms.functional_tensor`**,
  removed in torchvision 0.17. The architectures are reimplemented in
  `arch.py` instead — about 120 lines, loads the official weights with
  `strict=True`.
- **Architecture is inferred from the checkpoint**, not passed in.
  `conv_first` input channels give the scale (3 → ×4, 12 → ×2, 48 → ×1); the
  highest `body.N` index gives the block count.
- **Tiled inference divides by accumulated weight, not an average.** With
  `step = tile - 2*overlap` the raised-cosine windows do *not* form a partition
  of unity — the accumulated weight reaches about 1.49, so a naive sum is up to
  49% too bright in the overlap zones. `tests/test_tiling.py` pins this by
  running an identity callable through the tiler and demanding exact
  reconstruction.
- **OpenCV's `medianBlur` only accepts ksize ≤ 5 above 8-bit depth.** The
  16-bit repair path iterates a 5×5 median and combines it with a greyscale
  opening.
- **A morphological top-hat finds eyelashes as readily as dust.** Both are
  bright structures smaller than the structuring element. The detector is gated
  on local standard deviation over a 21×21 window: dust sits on smooth tone,
  eyelashes sit in hair. On a real press scan this took candidates from 16,768
  to 163. `despeckle()` returns the mask — render it and look before applying.

## Security

Checkpoints are pickles. Two controls, both needed:

- `torch.load(..., weights_only=True)` refuses arbitrary object construction.
- Every download is verified against a SHA-256 digest pinned in
  `models.WEIGHTS` before the file is opened, written atomically via a temp
  file in the destination directory, and discarded on mismatch.

Weights are not redistributed here. `printscale weights --fetch` pulls them
from the upstream Real-ESRGAN release assets over HTTPS.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

No test needs the network or a real photograph; every fixture is synthetic.

## Licence

Apache-2.0. The architectures follow BasicSR (Apache-2.0) closely enough to be
treated as a derivative work; the pretrained weights are published by
Real-ESRGAN under BSD-3-Clause and remain subject to its terms. See `NOTICE`.

**Copyright in the photograph is separate from all of this.** A photograph
being public domain in the United States says nothing about the UK or EU, where
it runs for the photographer's life plus 70 years, and a commercial premises
needs a different licence from a print at home. Check before you send anything
to a printer.
