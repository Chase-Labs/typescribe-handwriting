# typescribe-handwriting

Convert typed text into realistic handwriting, rendered as SVG (optionally PNG).

This is a fork of [rudyoactiv/typescribe-handwriting](https://github.com/rudyoactiv/typescribe-handwriting),
restructured as an installable, inference-only Python library intended to back a
web service. The Tkinter GUI, the model-training code, and the conda environment
of the upstream project have been removed; the pretrained model and the
generation pipeline are unchanged.

## How it works

The package is organised around a three-stage pipeline. Stages 1 and 3 are plain
Python (strings and SVG geometry) — you can work on them without any ML
knowledge. Stage 2 wraps the neural network behind a small API.

| Stage | Module | What it does |
|---|---|---|
| 1. Preprocess | `handwriting_synthesis/preprocessing.py` | Sanitize text to the model's 73-character alphabet, wrap it into ≤75-char lines, group lines into pages. Pure string handling. |
| 2. Generate | `handwriting_synthesis/synthesizer.py` | Feed lines through the pretrained RNN, which "writes" them one pen movement at a time. Returns `(N, 3)` NumPy arrays of pen offsets `(dx, dy, pen_up)` per line. |
| 3. Render | `handwriting_synthesis/rendering.py` | Clean up the strokes (`strokes.py`) and draw them line-by-line onto a ruled, A4-proportioned SVG page. Pure drawing code. |

The ML internals live in `handwriting_synthesis/network/` — a three-layer LSTM
with an attention window over the text, from Alex Graves'
[*Generating Sequences with Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850).
You should not need to modify anything in there; each file has a docstring
explaining its role if you're curious.

Two knobs control the output, both accepted by every generation call:

- **`styles`** (0–12) — which reference handwriting to imitate. The network is
  "primed" by replaying a stored sample of that style before writing your text.
- **`biases`** (0 to ~1) — neatness. The model *samples* each pen movement from a
  probability distribution; bias sharpens that distribution. Low bias is wild and
  scrawly, ~0.95 is tidy.

The pretrained weights and per-style priming data ship in `model/` and are
loaded at startup (override the location with `$HANDWRITING_MODEL_DIR` or the
`model_dir` argument).

## Installation

Requires Python 3.11 (pinned in `.python-version`; it is the newest Python that
TensorFlow 2.15 supports).

With [uv](https://docs.astral.sh/uv/) (recommended — `uv.lock` pins the exact
dependency set this was tested with):

```bash
uv sync                   # create .venv and install everything
uv run typescribe --help  # run tools inside it
```

Or with plain pip:

```bash
pip install -e .

# Either way, PNG export support is an extra (it also needs system cairo:
# brew install cairo / apt-get install libcairo2):
uv sync --extra png       # or: pip install -e '.[png]'
```

## Usage

### CLI

```bash
typescribe examples/sample.txt -o output --style 8 --bias 0.9
```

Run `typescribe --help` for all options (page geometry, pen colour/width, PNG
export, model location).

### Python API

```python
from handwriting_synthesis import HandwritingSynthesizer, prepare_text

synthesizer = HandwritingSynthesizer()  # loads the model — do this once at startup

# One-call convenience: text lines -> SVG page on disk
pages = prepare_text(open("examples/sample.txt").read())
for i, lines in enumerate(pages, start=1):
    synthesizer.write_page(lines, f"output/page_{i}.svg", styles=1, biases=0.95)

# Or drive the stages yourself, e.g. to return SVG from an HTTP handler:
from handwriting_synthesis import render_page

stroke_offsets = synthesizer.generate(["Hello world!"], styles=5)
svg_string = render_page(stroke_offsets, ["Hello world!"]).tostring()
```

Notes for service use:

- Model loading takes a few seconds; keep one `HandwritingSynthesizer` alive and
  reuse it. Generation itself is CPU-bound, roughly a few seconds per line.
- Generation is stochastic: the same input produces a different (equally valid)
  page every call.
- Input constraints (enforced with clear `ValueError`s): max 75 characters per
  line, characters limited to the model's alphabet — no `Q`, `X`, `Z`, and only
  basic punctuation. `prepare_text` handles both constraints for you.
- Importing the package puts TensorFlow into v1-compatibility graph mode for the
  whole process (the pretrained model predates TF2).

## Repository layout

```
├── src/handwriting_synthesis/   the package (see pipeline table above)
│   ├── alphabet.py              the 73 characters the model can draw
│   ├── preprocessing.py         stage 1: text -> model-ready lines
│   ├── synthesizer.py           stage 2: lines -> pen strokes (public API)
│   ├── strokes.py               stroke-geometry helpers for stage 3
│   ├── rendering.py             stage 3: pen strokes -> SVG page
│   ├── cli.py                   the `typescribe` command
│   ├── config.py                where the model assets live
│   └── network/                 ML internals (TensorFlow) — no need to touch
├── model/                       pretrained weights + style priming data
└── examples/sample.txt          sample input text
```

## Acknowledgements

- Upstream project: [rudyoactiv/typescribe-handwriting](https://github.com/rudyoactiv/typescribe-handwriting)
- Which builds on [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis),
  an implementation of Graves (2013), [*Generating Sequences with Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850).
  The pretrained model in `model/` comes from that lineage.
