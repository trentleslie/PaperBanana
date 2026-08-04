---
name: paperbanana
description: Generate publication-quality academic diagrams from paper methodology text
license: MIT-0
dependencies:
  env:
    - OPENROUTER_API_KEY (recommended)
    - GOOGLE_API_KEY (alternative)
  runtime:
    - python3
    - uv
---

# PaperBanana

Generate publication-quality academic diagrams and pipeline figures from a paper's methodology section and figure caption. PaperBanana orchestrates a multi-agent pipeline (Retriever, Planner, Stylist, Visualizer, Critic) to produce camera-ready figures suitable for venues like NeurIPS, ICML, and ACL.

## Environment Setup

```bash
cd <repo-root>
uv pip install -r requirements.txt
```

Set your API key via environment variable or in `configs/model_config.yaml`.

**Option 1 (Recommended): OpenRouter API key** — one key for both text reasoning and image generation:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

**Option 2: Google API key** — direct access to Gemini API:
```bash
export GOOGLE_API_KEY="your-key-here"
```

If both keys are configured, OpenRouter is used by default.

> **Figure-size warning.** Only the direct Gemini branch is known to honour
> `--figure-size` end to end: it passes the resolved image size through
> `types.ImageConfig`. On the OpenRouter branch the image size is an unverified
> passthrough field the upstream model may ignore, and on a `gpt-image` model the
> Visualizer hardcodes `1536x1024` and ignores it entirely. `OPENROUTER_API_KEY`
> takes precedence over the config file, so setting it silently changes which
> branch runs. Check the `[dimensions]` lines on stderr, or
> `candidates[].dimensions` in the manifest, to see what was actually produced.

## Usage

```bash
python skill/run.py \
  --content "METHOD_TEXT" \
  --caption "FIGURE_CAPTION" \
  --task diagram
```

With `--output` omitted, images and the run manifest are written to a fresh
timestamped directory under `results/skill_runs/`, so a repeat invocation cannot
destroy a prior run. Pass `--output path/to/figure.png` to control placement.

## Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `--content` | Yes* | | Method section text to visualize |
| `--content-file` | Yes* | | Path to a file containing the method text (alternative to `--content`) |
| `--caption` | Yes | | Figure caption or visual intent |
| `--task` | No | `diagram` | Task type. `diagram` is the only working value; `plot` is accepted by the parser but non-functional (the plot dataset is downloaded and the diagram pipeline runs anyway) |
| `--output` | No | *(timestamped directory)* | Output image file path. Omitted, artifacts are written to a fresh `results/skill_runs/run_<YYYYmmdd_HHMMSS>/` directory |
| `--aspect-ratio` | No | `16:9` | Aspect ratio: `21:9`, `16:9`, or `3:2` |
| `--figure-size` | No | `14-17cm` | Target printed figure width, which sets the provider render resolution: `1-3cm` and `4-6cm` → `1k`, `7-9cm` and `10-13cm` → `2k`, `14-17cm` → `4k` |
| `--max-critic-rounds` | No | `3` | Maximum critic refinement iterations |
| `--num-candidates` | No | `10` | Number of parallel candidates to generate |
| `--retrieval-setting` | No | `auto` | Retrieval mode: `auto`, `manual`, `random`, or `none` |
| `--planner-metaphor` | No | off | Flag. Diagram-only Planner visual-metaphor discovery before the detailed description is produced |
| `--main-model-name` | No | `gemini-3.1-pro-preview` | Main model for VLM agents. Provider auto-detected from configured API key |
| `--image-gen-model-name` | No | `gemini-3.1-flash-image-preview` | Model for image generation. Also supports `gemini-3-pro-image-preview` |
| `--exp-mode` | No | `demo_full` | Pipeline: `demo_full` (with Stylist) or `demo_planner_critic` (without Stylist) |

*One of `--content` or `--content-file` is required.

### Figure size

`--figure-size` is not cosmetic: it selects the resolution the image model
renders at. A bare invocation requests `4k` at `16:9`, which is double-column
publication width. On the Gemini branch that decodes to roughly 5504x3072.

Because the requested size is a request rather than a guarantee, the CLI reports
the decoded pixel dimensions of every saved image on stderr and flags a material
shortfall or aspect-ratio drift rather than downgrading silently.

### Output naming

With `--output` omitted, every image is named from its candidate identity inside
the run directory: `run_<timestamp>/skill_candidate_0.png`, and so on. With an
explicit `--output`, a single candidate is written to exactly that path, and
multiple candidates become `<stem>_skill_candidate_0.png`, and so on.

Names derive from candidate identity, never from completion order, so the Nth
image and the Nth manifest entry always describe the same candidate.

## Output

**Stdout contract:** stdout carries the absolute path of each saved image, one
per line, and nothing else. Pipeline progress, warnings, dimension reports and
the manifest path all go to stderr. Piping stdout to a naive line reader yields
only image paths.

### Run manifest

Every run writes `<stem>.manifest.json` beside its images. There is no flag to
enable it and no flag to suppress it. Its path is printed to stderr on
completion.

The manifest records, for the run: status (`complete`, `partial` or `failed`),
start and end timestamps, the resolved main and image-generation model names,
the derived image-generation backend (`gemini`, `openrouter` or `openai`), the
resolved image-size tier, and the repository commit with an explicit dirty flag.
For each candidate it records that candidate's identity, the path of the image it
produced, the decoded dimensions, and the planner, stylist and per-critic-round
reasoning trace.

It never contains credential material: it is assembled from an explicit
allowlist of already-resolved values, and base64 image payloads are stripped, so
a manifest stays small enough to keep indefinitely.

Candidate statuses distinguish `succeeded`, `no_image` (the candidate finished
but produced nothing), `failed` (the candidate raised) and `missing` (the
candidate never returned). A failing candidate no longer aborts the batch: the
surviving images are still written, and the manifest states what was lost.

## Examples

### Diagram

```bash
python skill/run.py \
  --content "We propose a transformer-based encoder-decoder architecture. The encoder consists of 12 self-attention layers with residual connections. The decoder uses cross-attention to attend to encoder outputs and generates the target sequence autoregressively." \
  --caption "Figure 1: Overview of the proposed transformer architecture" \
  --task diagram \
  --output architecture.png
```


## Important Notes

- **Runtime**: A single candidate typically takes 3-10 minutes depending on model and network conditions. With the default 10 candidates running in parallel, expect ~10-30 minutes total. Plan accordingly.
- **API calls**: Each candidate involves multiple LLM calls (Retriever + Planner + Stylist + Visualizer + up to 3 Critic rounds). Candidates run in parallel for efficiency.
- **Image generation**: The Visualizer agent calls an image generation model (Gemini Image) to render diagrams.

## About

PaperBanana is based on the **PaperVizAgent** framework, a reference-driven multi-agent system for automated academic illustration. It was developed as part of the research paper:

> **PaperBanana: Automating Academic Illustration for AI Scientists**
> Dawei Zhu, Rui Meng, Yale Song, Xiyu Wei, Sujian Li, Tomas Pfister, Jinsung Yoon
> arXiv:2601.23265

The framework introduces a collaborative team of five specialized agents — Retriever, Planner, Stylist, Visualizer, and Critic — to transform raw scientific content into publication-quality diagrams. Evaluation is conducted on the **PaperBananaBench** benchmark.

