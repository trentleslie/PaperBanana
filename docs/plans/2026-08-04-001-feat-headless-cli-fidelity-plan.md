---
title: "feat: Headless CLI figure-size fidelity and run records"
type: feat
status: active
date: 2026-08-04
origin: docs/brainstorms/2026-08-04-headless-cli-fidelity-requirements.md
deepened: 2026-08-04
---

# feat: Headless CLI figure-size fidelity and run records

## Overview

`skill/run.py` is the headless entry point an agent uses to drive the PaperBanana
figure pipeline without a human in the Gradio UI. It cannot request a figure size, so
it is pinned at the `1k` fallback while the operator's real runs use `14-17cm`/`4k` at
`16:9`. It writes nothing but PNGs, so a ten-to-thirty-minute paid run leaves no record
of what produced it. Its filenames come from async completion order rather than
candidate identity, so they are already mislabelled. And a single failing candidate
destroys the entire batch.

This plan brings the CLI to parity with the calibrated UI configuration, makes candidate
identity authoritative, defaults runs to a timestamped directory, persists a slim
credential-free manifest that survives partial failure, and makes the stdout contract
true rather than merely asserted.

## Problem Frame

See origin document for the full frame. Verified defects:

1. `skill/run.py:141` hardcodes `additional_info` to `{"rounded_ratio": ...}`. `generation_additional_info` is called only from `app.py:122` and `demo.py:130`; `run.py` is the missing caller. Its consumers are live: `image_size_from_data` is read at `agents/visualizer_agent.py:123`, `agents/polish_agent.py:167`, `agents/vanilla_agent.py:93`.
2. No run record; expensive paid runs are unreproducible from artifacts alone.
3. Filenames come from `enumerate()` over a generator yielding via `asyncio.as_completed`, so image N is not candidate N. Active, not latent.
4. `utils/paperviz_processor.py` does an unguarded `result_data = await future` inside the async generator. One raising candidate closes the generator permanently; remaining tasks are cancelled when the loop closes. **A `try/except` in `run.py` cannot recover from this**, which is why the scope boundary below was deliberately widened.
5. Stdout is not clean today. There are 38 `print()` calls on the run path across `utils/paperviz_processor.py`, `agents/visualizer_agent.py` and `utils/generation_utils.py`, plus the dataset-download banner in `run.py:55`. Any consumer treating stdout as a list of image paths is already broken.

## Requirements Trace

- R1-R4. Figure size argument through the existing helper, defaulting to `14-17cm`/`4k`, with requested tier and actual decoded dimensions recorded.
- R5, R6, R6a. Manifest by default, beside the images, in a timestamped directory by default.
- R7, R7a. Run parameters, model names, resolved backend, commit, timestamps pinned; no credential material.
- R8. Per-candidate trace via `build_evolution_stages()`, base64 stripped, retrieval recorded once per run.
- R9. Per-candidate identity paired with its image path, including candidates that produced none.
- R10. Manifest path to stderr; stdout genuinely image paths only.
- R10a. A run losing candidates still produces the surviving images and a manifest.
- R11. Filenames derived from candidate identity.
- R12, R13. Isolated working copy, tracked files clean, figure-size fix separable for upstream.
- R14. `skill/SKILL.md` matches actual behavior.

## Scope Boundaries

- `--task plot` remains broken and out of scope.
- **Scope deliberately widened (decision 2026-08-04):** one `try/except` around the single `await future` in `utils/paperviz_processor.py` is in scope. This is the minimum change that makes R10a achievable; confining the work to `run.py` cannot deliver it. No other pipeline change, no prompt change, no retrieval-behavior change.
- No reproduction of the UI's full trace or candidates zip.
- Automated candidate selection is out of scope.

### Deferred to Separate Tasks

- Upstream pull request for the figure-size fix. Unit 1 is the only unit shaped to be upstreamable; the widened-scope processor change and the operator-policy units are not offered upstream.

## Context & Research

### Relevant Code and Patterns

- `utils/legacy_generation_options.py` — `generation_additional_info(aspect_ratio, figure_size)`, `image_size_for_figure_size`. The intended seam, verified on the live call path.
- `utils/legacy_ui_results.py` — `build_evolution_stages()`, `BASE64_SUFFIX`, `resolve_final_output()`.
- `app.py:111-122` — reference for how the UI builds `additional_info`.
- `utils/paperviz_processor.py` — the drain loop and the run-once Retriever sharing.
- `tests/test_legacy_generation_options.py` — existing helper coverage and the repo's test style.

### Institutional Learnings

- `fail-closed-guards-must-not-no-op-on-absent-input-2026-07-13` — a default at the "never trip" end is the exact shape of the `1k` fallback. Tests must exercise the unguarded path, because tests naturally supply the input that keeps a guard armed. Applied to the R7a sentinel test and the omitted-flag test.
- `nightshift-salvage-timeout-committed-work-discarded-2026-08-03` — route normal and failure paths through one shared emitter; encode incompleteness explicitly so a degraded run cannot look complete.
- `per-request-key-injected-into-unused-helper-2026-07-11` — a green unit test once validated a helper no caller used. Mitigated: call sites verified by grep before relying on the seam, and Unit 1 asserts at the `image_size_from_data` boundary.
- `canonical-dedup-key-shared-canonicalization-2026-08-03` — one shared identity function, never two derivations that must agree.
- `shared-clone-concurrent-harness-branch-switch-2026-08-04` — treat a live checkout as hostile shared state.

### Environment Constraints

- `paperbanana.service` and `paperbanana-serve.timer` are **active** with `WorkingDirectory=/home/trentleslie/projects/PaperBanana`. Work in a `git worktree` so the live Gradio service keeps serving `main`.
- `.gitignore` excludes `configs/model_config.yaml`, `.venv/`, `data/` and `results/`, so **a fresh worktree has no API key, no interpreter and no dataset**. A bootstrap step is mandatory (see Documentation / Operational Notes).
- Backend resolved: `openrouter_api_key` is empty, `OPENROUTER_API_KEY` is unset, and the service passes only `GRADIO_SERVER_NAME`. The three calibrated runs therefore used the **Gemini** branch, which honors `image_size` through `types.ImageConfig`. The 4k evidence is valid for the path the CLI will take on this machine.
- The venv is uv-pinned (`cpython-3.12.9`), so the known orphaned-venv trap does not apply.
- `pytest` is **not** installed and is not in `requirements.txt`. Existing tests are plain `unittest.TestCase`. Async scenarios must use `unittest.IsolatedAsyncioTestCase` rather than introducing a dependency.
- `data/` does not exist, so the first run triggers a HuggingFace `snapshot_download`. Budget for it before any timed verification.

## Key Technical Decisions

- **Route through `generation_additional_info`** rather than building `additional_info` inline: already derives `image_size`, already test-covered, consumers verified live.
- **Default `14-17cm` and `16:9`** rather than the UI dropdown's `7-9cm` and argparse's `21:9`: together these reproduce the operator's calibrated configuration end to end. Changing only figure size would leave a bare run on an untested `21:9`/`4k` pair, and `21:9` is specifically wrong for the target figure, whose vertical certification branch needs the height.
- **Record actual decoded pixel dimensions against a tolerance, not equality**: the one calibration point, 4k at 16:9, measured 5504x3072, a ratio of 1.7917 rather than 1.7778, so the provider already does not honor the requested ratio exactly.
- **Seed manifest entries from `data_list` before draining**, so completeness is evaluated against the number of candidates requested rather than the number observed. Otherwise a run that dies after four yields records four successes and stamps itself complete.
- **One shared emitter invoked from a single exit path**, per the salvage learning.
- **One shared identity function** for both filename and manifest key, per the canonical-key learning.
- **Manifest built by allowlist**, satisfying R7a by construction rather than by filtering.
- **Make stdout clean rather than merely declaring it clean**: redirect pipeline chatter to stderr inside `run.py` and emit image paths after, so Unit 6 documents a guarantee that actually holds.

## Open Questions

### Resolved During Planning

- Is `generation_additional_info` on the live call path? Yes, three agent call sites verified.
- Is R11 active or latent? Active; `as_completed` yields in completion order.
- Which backend is the 4k criterion validated against? Gemini, verified from empty OpenRouter config and the service environment.
- Can R10a be delivered inside `run.py`? No. Scope widened by explicit decision.
- Worktree or in-place branch? Worktree plus a bootstrap step.

### Deferred to Implementation

- Exact manifest schema field names.
- How to represent a dirty checkout in the commit field.
- Whether `--content-file` text is embedded, hashed, or copied, pending observed manifest size.
- Whether to adopt `resolve_final_output()` in place of the local `extract_final_image_b64`.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
argv ──> resolve_run_settings()      # figure_size 14-17cm, aspect 16:9, timestamped outdir
          │
          ├─> generation_additional_info(aspect_ratio, figure_size)
          │        └─> {rounded_ratio, figure_size, image_size} ──> per-candidate data dict
          │
          ├─> entries = {identity(d): status "missing" for d in data_list}   # SEEDED UP FRONT
          │
          └─> redirect_stdout(stderr):
                 async for result in process_queries_batch(...):
                     # processor now yields either a result or an error record
                     id = candidate_identity(result)          # ONE shared fn
                     if error record:  entries[id] = failed + error
                     else:             save PNG; entries[id] = succeeded + dims
          │
          └─ finally ──> write_manifest(outdir, run_meta, entries)   # SINGLE emitter
                            run status = complete iff no entry is missing or failed,
                                         evaluated against len(data_list)
             then, outside the redirect:
                 stdout: image paths only, one per line
                 stderr: manifest path
```

## Implementation Units

- [ ] **Unit 0: Worktree bootstrap**

**Goal:** An isolated working copy that can actually run.

**Requirements:** R12

**Dependencies:** None.

**Files:** None in-repo; environment setup only.

**Approach:**
- Create a `git worktree` so the live Gradio service keeps serving `main`.
- Symlink `configs/model_config.yaml`, `data/` and `.venv` from the main checkout into the worktree, since all three are gitignored and absent otherwise.
- Assert `ensure_model_config()` did not fall back to the key-less template before any paid run.

**Test scenarios:**
- Test expectation: none — environment setup, no behavioral change.

**Verification:**
- A one-candidate smoke run completes in the worktree, and the resolved config carries a non-empty key rather than the template's empty one.

---

- [ ] **Unit 1: Figure-size passthrough, calibrated defaults, dimension recording**

**Goal:** The CLI can request any figure size the UI can, defaults to the calibrated `14-17cm`/`16:9`, and records what was actually produced.

**Requirements:** R1, R2, R3, R4

**Dependencies:** Unit 0. Otherwise self-contained, and the only unit shaped for upstream (R13).

**Files:**
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_figure_size.py`

**Approach:**
- Add `--figure-size` with the five cm choices, defaulting to `14-17cm`.
- Change the `--aspect-ratio` default from `21:9` to `16:9`.
- Replace the inline `additional_info` literal with `generation_additional_info(aspect_ratio, figure_size)`.
- Read true dimensions from the PIL object already opened per image; compare against a per-(tier, ratio) expected envelope with tolerance, seeded with the one known point, 4k at 16:9 measuring 5504x3072. Report requested tier and actual dimensions, and flag a material shortfall rather than printing two numbers and leaving interpretation to the reader.

**Execution note:** Test-first, writing the omitted-flag case first so the default is proven to trip. The fail-closed learning is that tests naturally supply the input that keeps a guard armed.

**Patterns to follow:** `app.py:111-122`; `tests/test_legacy_generation_options.py` for `unittest` style.

**Test scenarios:**
- Happy path: `--figure-size 14-17cm` yields `additional_info` carrying `figure_size: 14-17cm` and `image_size: 4k`.
- Happy path: each of the five choices maps to its documented tier.
- Edge case: both flags omitted yield `14-17cm`/`4k` at `16:9`, never `1k`, never `21:9`. This is the guard-trips case and must not be skipped.
- Error path: an unaccepted value such as `20cm` is rejected by argparse rather than falling through to the `1k` default.
- Edge case: a decoded image materially below the tier's expected area is flagged, not silently accepted; the 1.7917-versus-1.7778 ratio drift does not trigger a false alarm.
- Integration: the assembled dict is shaped so `image_size_from_data` returns the expected tier, asserted at that boundary rather than only against the helper.

**Verification:**
- A bare run requests `4k` at `16:9`.
- Requested tier, actual dimensions, and any shortfall flag are all reported.
- The diff touches only argument parsing, `additional_info` construction, and reporting.

---

- [ ] **Unit 2: Candidate identity as the source of filenames**

**Goal:** Filenames derive from candidate identity, not completion order.

**Requirements:** R11

**Dependencies:** Unit 1.

**Files:**
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_candidate_identity.py`

**Approach:**
- One identity function over the result's own `filename` field, used for the PNG name and later as the manifest key.
- When identity cannot be derived, skip and warn rather than emitting a fabricated name.
- The exact-path contract for `--num-candidates 1` applies only when `--output` is supplied (see Unit 3).

**Test scenarios:**
- Happy path: results arriving out of submission order produce filenames matching their own identity, not arrival position.
- Edge case: `--num-candidates 1` with an explicit `--output` writes exactly that path.
- Edge case: a result missing `filename` is skipped with a warning, not written under a fabricated name.

**Verification:**
- Shuffled completion order produces correctly labelled files.

---

- [ ] **Unit 3: Timestamped output directory by default**

**Goal:** A repeat invocation cannot destroy a prior run.

**Requirements:** R6a

**Dependencies:** Unit 2.

**Files:**
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_output_placement.py`

**Approach:**
- Change `--output` to `default=None` so an omitted flag is distinguishable from an explicit `output.png`; argparse cannot tell them apart while the default remains a literal path.
- With `--output` omitted, write into a timestamped directory; define the N=1 omitted-flag path as `<timestamped-dir>/<identity>.png`.
- Handle same-second collisions rather than silently merging two runs' artifacts.

**Test scenarios:**
- Happy path: two consecutive default invocations write into distinct directories and both artifact sets survive.
- Happy path: an explicit `--output` path is honoured verbatim.
- Edge case: two runs starting within the same second do not merge.
- Edge case: the timestamped directory is created when absent, idempotently.

**Verification:**
- Running twice with no arguments leaves two complete, independent result sets.

---

- [ ] **Unit 4: Manifest emitter and clean stdout**

**Goal:** Every run persists a slim, credential-free, reproducible record, and stdout becomes genuinely parseable.

**Requirements:** R5, R6, R7, R7a, R8, R9, R10

**Dependencies:** Units 2 and 3.

**Files:**
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_manifest.py`

**Approach:**
- Seed one entry per requested candidate from `data_list` before draining, keyed by the shared identity function, initialized to `missing`. The drain loop only upgrades entries. Run status is `complete` only when no entry remains `missing` or `failed`, evaluated against `len(data_list)`, never against the number of observed results.
- Build the manifest from an explicit allowlist so credentials are excluded by construction.
- Per-candidate trace from `build_evolution_stages()`, reading `stage.get("suggestions_key")` because only Critic stages carry it, and skipping falsy `desc_key`. A literal `stage["suggestions_key"]` would raise on the first stage of every candidate, inside the very `finally` path Unit 5 depends on.
- Strip any key ending in `BASE64_SUFFIX`, and additionally assert no field value is a base64 payload, since a key-name check alone would admit a blob stored under another name.
- Record retrieval once per run, since the Retriever runs once per batch.
- Record the image-generation backend derived from the resolved `exp_config.image_gen_model_name` and `generation_utils.openrouter_client`, named to reflect that it is derived rather than observed.
- Wrap pipeline execution in `contextlib.redirect_stdout(sys.stderr)` so the 38 existing print calls land on stderr, then emit image paths to the real stdout after the redirect exits. Manifest path to stderr.

**Test scenarios:**
- Happy path: a completed run yields candidate entries matching the saved images one-for-one.
- Happy path: planner, stylist and every critic round appear in a candidate's trace.
- Edge case: a result whose only stages are Planner and Stylist does not raise in the emitter.
- Edge case: with a non-empty sentinel key injected into config, that sentinel is absent from the serialized manifest, and the test asserts the sentinel is non-empty so it fails loudly rather than no-opping when no key is configured.
- Edge case: no base64 payload appears under any field name, and manifest size stays orders of magnitude below the UI's per-run output.
- Error path: a candidate producing no image still appears, marked, with a null image path.
- Integration: every line of captured stdout is an existing image path, and the manifest path appears only on stderr.
- Integration: the identity used for each filename is byte-identical to its manifest key.

**Verification:**
- A manifest answers what produced each image, with what parameters, on what backend, at what commit.
- Piping stdout to a naive line reader yields only image paths.

---

- [ ] **Unit 5: Survive partial failure**

**Goal:** One failing candidate no longer aborts the batch or discards the rest of the run's paid work.

**Requirements:** R10a, R5, R9

**Dependencies:** Unit 4.

**Files:**
- Modify: `utils/paperviz_processor.py`
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_partial_failure.py`

**Approach:**
- In `process_queries_batch`, wrap the single `await future` so a raising candidate yields an explicit error record carrying that candidate's identity instead of terminating the generator. This is the widened-scope change; keep it to that one construct.
- In `run.py`, treat an error record as a `failed` entry and continue draining. Route both normal and failure exits through the single emitter from Unit 4.
- Distinguish `failed` (raised) from `missing` (never yielded) from `no_image` (completed but produced nothing). The last is the most common real degradation: `generation_utils` exhausts retries and returns empty rather than raising, and `visualizer_agent` then continues past the key.

**Execution note:** Test-first, injecting a genuinely failing candidate. The salvage precedent is a real incident where the happy path was green while the failure path discarded thirty minutes of work.

**Test scenarios:**
- Happy path: an all-successful run is marked complete.
- Error path: one candidate of ten raising still saves the other nine images and writes a manifest.
- Error path: that manifest marks the run partial and names the failed candidate with its error.
- Error path: a candidate that completes with no image is recorded as `no_image`, distinctly from a raised failure.
- Edge case: every candidate failing still writes a manifest rather than exiting with nothing.
- Edge case: a run that dies after four of ten yields is marked partial, not complete, because entries were seeded up front.
- Integration: the manifest write happens exactly once regardless of exit path.

**Verification:**
- An injected mid-run failure leaves the surviving images plus a manifest that states what was lost.

---

- [ ] **Unit 6: Update the agent-facing contract**

**Goal:** `skill/SKILL.md` matches actual behavior.

**Requirements:** R14

**Dependencies:** Units 1 through 5.

**Files:** Modify: `skill/SKILL.md`

**Approach:**
- Add the figure-size row with choices and the `14-17cm` default; correct the `--aspect-ratio` default to `16:9`.
- Add `--planner-metaphor`, which argparse accepts but the file omits entirely.
- Correct the `--task` row, which documents only `diagram` while argparse accepts `plot`; state that `plot` is accepted but non-functional.
- Document the manifest artifact, its location, and the timestamped default directory.
- State the stdout contract as an explicit guarantee, now that Unit 4 makes it true.
- Warn that setting `OPENROUTER_API_KEY` routes image generation to a branch where `image_size` is an unverified passthrough, since the file currently recommends OpenRouter and the env var takes precedence over the config file.

**Test scenarios:**
- Test expectation: none — documentation only, no behavioral change.

**Verification:**
- Every flag argparse accepts appears in the parameter table with its real default.

## System-Wide Impact

- **Interaction graph:** `skill/run.py` plus one guarded construct in `utils/paperviz_processor.py`. `generation_additional_info` gains a third caller; the UI path is untouched.
- **Error propagation:** Today a candidate exception unwinds through the generator and out of `asyncio.run`. Unit 5 converts it into a yielded error record at the single point where it is raised.
- **State lifecycle risks:** Partial runs, addressed by seeded entries, a single emitter, and an explicit `partial` status.
- **API surface parity:** Brings the CLI to parity with the UI's existing figure-size capability.
- **Integration coverage:** The seam unit tests will not prove is that `additional_info` reaches `image_size_from_data` in the agents; asserted at that boundary in Unit 1.
- **Unchanged invariants:** The single-candidate explicit-`--output` contract is preserved. `--task plot` stays broken. The Gradio UI is untouched and keeps serving `main` from the worktree arrangement.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| A fresh worktree lacks the API key, venv and dataset, so runs silently use the key-less template | Unit 0 bootstrap symlinks all three and asserts the config did not fall back to the template |
| Branch checkout swaps files under the live `paperbanana.service` | Worktree rather than in-place branch |
| Widened scope makes the change less upstreamable | Unit 1 is isolated and remains the only piece offered upstream |
| `image_size` unverified on OpenRouter and `gpt-image` branches | Backend confirmed Gemini on this machine; Unit 1 records actual dimensions with a shortfall flag, and Unit 6 warns that setting `OPENROUTER_API_KEY` changes the path |
| Defaults change raises cost and latency on every bare invocation | Deliberate; reported tier and dimensions make it self-evident on first run |
| Manifest larger than expected once real traces land | Size asserted in Unit 4; `--content-file` embedding stays deferred until sizes are observed |
| First run triggers a multi-minute HuggingFace dataset download | Budget for it in Unit 0's smoke run, before any timed verification |
| No pytest in the venv; async tests need different scaffolding | Use `unittest.IsolatedAsyncioTestCase`, matching the repo's existing `unittest` style, rather than adding a dependency |

## Documentation / Operational Notes

- Worktree bootstrap: create the worktree, then symlink `configs/model_config.yaml`, `data/` and `.venv` from the main checkout. All three are gitignored.
- Invoke the interpreter explicitly rather than relying on an activated shell.
- The live Gradio service keeps serving whatever is checked out in the main working directory; leave it on `main`.
- `skill/SKILL.md` is the contract agents read and is updated in Unit 6.

## Sources & References

- **Origin document:** `docs/brainstorms/2026-08-04-headless-cli-fidelity-requirements.md`
- Related code: `skill/run.py`, `utils/paperviz_processor.py`, `utils/legacy_generation_options.py`, `utils/legacy_ui_results.py`, `agents/visualizer_agent.py`, `app.py`
- Upstream: `dwzhu-pku/PaperBanana`
