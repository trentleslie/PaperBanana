---
date: 2026-08-04
topic: headless-cli-fidelity
---

# Headless CLI Fidelity: Figure Size and Run Records

## Problem Frame

`skill/run.py` is the headless entry point for the PaperBanana multi-agent figure
pipeline. It is the interface an agent uses to generate figures without a human
driving the Gradio UI. Today it diverges from the UI in ways that make it unsuitable
for real publication work.

First, it cannot request a figure size at all. `run.py:141` builds `additional_info`
as `{"rounded_ratio": args.aspect_ratio}`, omitting figure size entirely. Figure size
is not cosmetic: `image_size_for_figure_size` in `utils/legacy_generation_options.py`
maps it to the provider's render resolution (`1-3cm`/`4-6cm` to `1k`,
`7-9cm`/`10-13cm` to `2k`, `14-17cm` to `4k`) and falls back to `1k` when unset. The
Gradio UI exposes the choice and defaults to `7-9cm`; the CLI is pinned at the `1k`
fallback and cannot reach `2k` or `4k` by any invocation.

The observed baseline is `4k`, not the UI default. All three UI runs on disk
(`results/demo/demo_20260803_*.json`) record `figure_size: 14-17cm`,
`image_size: 4k`, `rounded_ratio: 16:9` across all ten candidates each, and their
PNGs are 5504x3072. So 5504x3072 is what `4k` at `16:9` produces. The operator's
demonstrated working configuration is double-column `4k`, which is four tiers above
what the CLI currently produces.

Second, an expensive run leaves no record of itself. `run.py` writes PNGs and nothing
else. A default run is ten parallel candidates. The Retriever runs once for the whole
batch and its results are shared across candidates; Planner, Stylist, Visualizer and
up to three Critic rounds then run per candidate, against paid models, for ten to
thirty minutes. The reasoning that produced each image, the parameters that would
reproduce it, and the identity of each candidate are all discarded at exit. This
violates the operator's standing artifact-hygiene rule that expensive runs persist
their results by default rather than behind an opt-in flag.

A third defect compounds the second. Output PNGs are named from an enumeration index
over results drained from an async generator. `process_queries_batch` yields via
`asyncio.as_completed`, so completion order is not submission order: this is an active
bug, not a latent one, and every multi-candidate run to date has mislabelled outputs.
Each result already carries `"filename": "skill_candidate_<i>"`. Any record pairing
reasoning with images would therefore pair them incorrectly. This is treated as part
of the record work rather than a separate concern, because a record with untrustworthy
filename correspondence has no value.

## Requirements

**Figure size fidelity**
- R1. The CLI accepts a figure size argument restricted to the same choices the UI offers (`1-3cm`, `4-6cm`, `7-9cm`, `10-13cm`, `14-17cm`).
- R2. Figure size reaches the pipeline through the existing `generation_additional_info` helper so that both `figure_size` and the derived `image_size` are populated, rather than the CLI constructing `additional_info` itself.
- R3. When the argument is omitted the CLI defaults to `14-17cm`, resolving to `4k`. This matches the operator's demonstrated working configuration across all three observed runs rather than the UI dropdown's `7-9cm`. A bare invocation therefore produces publication-width output without the caller needing to know the flag exists.
- R4. The CLI reports the figure size requested, the image-size tier it resolved to, and the actual decoded pixel dimensions of each saved image, so a downgrade can never be silent. Actual dimensions are required because `image_size` is a request rather than a guarantee: `agents/visualizer_agent.py` hardcodes `1536x1024` on the `gpt-image` branch and passes `image_size` as an optional `image_config` field on the OpenRouter branch that the upstream model may ignore. `run.py` already opens each image with PIL, so the true dimensions are in hand.

**Run record**
- R5. Every run writes a manifest by default. There is no flag required to obtain it and no flag that suppresses it.
- R6. The manifest is written beside the images, sharing the `--output` stem, so that it travels with them and requires no knowledge of repo internals to find.
- R6a. Runs default to a timestamped output directory so that a repeat invocation cannot destroy a prior run's images or manifest. An explicit `--output` path remains honoured as an override. This follows the artifact-hygiene rule that expensive runs default to a timestamped path, with `--output` as an override rather than the only way to control placement.
- R7. The manifest pins what is needed to reproduce the run: the CLI arguments including resolved defaults, the main and image-generation model names actually used, the resolved image-size tier, the image-generation backend actually selected (`openai`, `openrouter`, or `gemini`, chosen at call time in `visualizer_agent` from ambient config rather than from any CLI argument), the repository commit, and start and end timestamps.
- R7a. The manifest must never contain credential material. Only already-resolved model-name strings may be recorded. The contents of `configs/model_config.yaml`, any `api_keys` section, and any environment variable values are excluded by construction rather than by filtering.
- R8. The manifest carries the per-candidate reasoning trace produced by the pipeline. It is built from `utils.legacy_ui_results.build_evolution_stages()`, using each stage's `desc_key` and `suggestions_key` to reach planner, stylist and every critic round, and it strips any key ending in `legacy_ui_results.BASE64_SUFFIX`. Retrieval output is recorded once for the run rather than per candidate, because the Retriever runs once per batch.
- R9. The manifest records, for each candidate, that candidate's own identity alongside the path of the image it produced, including candidates that produced no image.
- R10. The manifest path is written to stderr on completion. Stdout continues to carry image paths only, one per line, preserving the contract `skill/SKILL.md` publishes so existing agent callers keep working unchanged.
- R10a. A run that loses candidates still produces the surviving images and a manifest. A failing candidate is recorded as failed and skipped rather than aborting the batch, and the manifest is written on the way out regardless of how the run ends. Delivering this requires one guarded construct in `utils/paperviz_processor.py`; see the widened scope boundary.
- R3a. The `--aspect-ratio` default changes from `21:9` to `16:9`. All three observed runs used `16:9`, so leaving the default at `21:9` while changing figure size would put a bare invocation on an uncalibrated pair, and `21:9` is specifically wrong for the target figure, whose vertical certification branch needs the height.

**Candidate identity**
- R11. Output image filenames derive from each candidate's own identity rather than from completion order, so that the Nth image and the Nth manifest entry describe the same candidate regardless of the order results arrive.

**Interface contract**
- R14. `skill/SKILL.md` is updated in the same change to document the figure-size argument and its choices, the default and its resolved image size, the manifest artifact and its location, and any revision to the stdout contract. `skill/SKILL.md` is the discovery surface agents read before invoking this CLI, so a capability absent from it is unreachable in practice.

**Provenance of the change**
- R12. The work lands on a local branch that the operator runs from, leaving tracked upstream files unmodified on `main`. Note that `docs/` is untracked on `main` today, including this document.
- R13. The figure-size fix is separable from the run-record work so it can be offered upstream on its own.

## Success Criteria

- The CLI can reach every resolution the UI can, including `4k`, which is currently impossible.
- A bare invocation's resolution is a stated, documented value rather than an accident of an omitted field.
- Actual saved pixel dimensions are recorded and match the requested tier, or the mismatch is visible.
- After any run, the operator can answer "what produced this image, with what parameters, on what backend, at what commit" from the artifacts alone, without the session transcript.
- Every image in an output directory can be matched to its candidate entry in the manifest with no ambiguity.
- The manifest is small enough to keep indefinitely, in contrast to the roughly 210-234MB the UI writes per run across its results JSON and candidates zip.
- No credential material appears in any manifest.
- Two consecutive bare invocations both retain their images and manifests; neither destroys the other.
- A run in which one candidate fails still produces the remaining images and a manifest that records the failure.
- An existing caller that reads stdout as a list of image paths continues to work unchanged.
- `git status` shows no modifications to tracked files outside the working branch.

## Scope Boundaries

- The broken `--task plot` path is out of scope. `ExpConfig` is constructed without `task_name`, so `--task plot` downloads the plot dataset and then runs the entire diagram pipeline; `extract_final_image_b64` additionally hardcodes `diagram`. Deferred solely because there is no current plot use case. Note the earlier rationale that this would disturb shared extraction logic was wrong: `extract_final_image_b64` is private to `skill/run.py`, and the shared `utils.legacy_ui_results.resolve_final_output` is already task-generic with plot coverage in `tests/test_legacy_ui_result_keys.py`.
- Scope widened by decision on 2026-08-04: one `try/except` around the single `await future` in `utils/paperviz_processor.py` is in scope, because R10a is not deliverable from `run.py` alone. That generator's unguarded await means a raising candidate closes it permanently and the remaining tasks are cancelled, so no consumer-side handling can salvage the batch. No other pipeline change, no prompt change, no retrieval-behavior change, no UI change.
- No attempt to reproduce the UI's full trace or its candidates zip.
- No upstream pull request is opened as part of this work. R13 only requires that the change be shaped so one is possible later.
- Automated selection of a winning candidate is out of scope. After this work the operator still chooses among candidates by eye; `do_eval=False` means no scores are produced.

## Key Decisions

- **Slim manifest rather than mirroring the UI's JSON**: the final image is already on disk, so its base64 payload is duplicated mass. Note this justification is partial: the pipeline also produces planner and stylist intermediate images that the CLI never saves, so stripping all payloads does discard images the preserved critiques refer to. Accepted on size grounds with that cost acknowledged.
- **Manifest beside the output rather than in `results/`**: the operator is already looking at the output directory, and an artifact that travels with its images survives being moved.
- **Candidate identity folded into the record work rather than tracked separately**: a manifest whose filename correspondence may be wrong is worse than no manifest, because it looks authoritative.
- **Local branch with only the figure-size fix upstreamable**: `tests/test_legacy_generation_options.py` already asserts that `generation_additional_info` preserves figure size and image size, so the helper and its coverage exist upstream and only the caller is wrong. The run record encodes this operator's artifact policy and is not upstream's concern.

## Dependencies / Assumptions

- The checkout is on `main` with upstream `dwzhu-pku/PaperBanana`. Tracked files are unmodified; `docs/` is untracked. Verified.
- `configs/model_config.yaml` has a populated `google_api_key`. Verified.
- The image-generation backend is selected at call time, not by CLI argument: `visualizer_agent` branches to OpenAI on `"gpt-image" in model_name`, else OpenRouter when `openrouter_client is not None`, else direct Gemini. `skill/SKILL.md` recommends `OPENROUTER_API_KEY` and states OpenRouter wins when both keys are present. Only the Gemini branch is known to honor `image_size` through `types.ImageConfig`; the OpenRouter branch passes it as a passthrough field whose honoring is unverified. **Which backend the 4k criterion is validated against is unresolved, see Outstanding Questions.**
- `image_size` reaches every image-producing call including each critic round, because `_run_critic_iterations` re-enters `visualizer_agent.process`, which reads `image_size_from_data(data)` on every invocation. Verified.
- All three observed UI runs recorded empty `top10_references` and `retrieved_examples` under the default `auto` retrieval setting, so retrieval contributed nothing to the baseline runs.

## Outstanding Questions

All blocking questions are resolved. Their answers are recorded in R3, R6a, R10 and R10a.

### Deferred to Planning
- [Affects R7][Technical] How to obtain the repository commit when the checkout may be dirty, and how to represent that.
- [Affects R7][Technical] When `--content-file` is used the manifest would pin a path, not the bytes that path held at run time, so an edited or moved file voids reproducibility. Should the manifest embed the resolved input text, a hash, or a copy beside the images? Note `--content` is inline and already captured, so the gap is specific to the file path.
- [Affects R11][Technical] `run.py` currently writes to `--output` verbatim when `num_candidates == 1` and only appends `_{idx}` for N>1. Deriving names from candidate identity forces a choice about whether the single-candidate path keeps its exact-path contract.
- [Affects R8][Technical] Whether to also adopt `utils.legacy_ui_results.resolve_final_output` in place of the local `extract_final_image_b64`, which ignores the `eval_image_field` rollback pointer and `polished_*` keys that the UI honors, and can therefore select a different image than the UI would from the same result dict.

## Next Steps

-> `/ce:plan` for structured implementation planning.
