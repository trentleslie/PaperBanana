---
title: "fix: PaperBanana open items, sliced into independent PRs"
type: fix
status: active
date: 2026-08-04
origin: docs/brainstorms/2026-08-04-headless-cli-fidelity-requirements.md
---

# fix: PaperBanana open items

**Target repos and artifacts.** Code paths are relative to the PaperBanana worktree
(branch `feat/headless-cli-fidelity`). Slices E, F and G act outside any repo and are
marked `[out-of-repo]`: the skill at `~/.claude/skills/paperbanana/`, and the vault note
`Active 🎯/Work/BioMapper Preprint - Figures & Tables Plan.outline.md`.

**Structure.** Sliced into independently mergeable PRs rather than one branch. Slices A
through D are code and each stands alone; E through G are not PRs at all. Slice H is the
remote setup that must precede any push.

## Overview

The headless CLI works and produced a publication-grade Figure 1, but carries a punch
list: reference retrieval that has never functioned, an ~18-minute dead download per run,
two unpinned review findings plus a misleading manifest field, an oversized and partly
inaccurate skill, an undecided branch disposition, a stale vault float manifest, and an
open venue question about raster versus vector.

## Problem Frame

**Retrieval has never worked, for a confirmed reason.** The HF dataset
`dwzhu/PaperBananaBench` contains exactly two files: `.gitattributes` and
`PaperBananaBench.zip` (265,846,711 bytes, LFS/Xet-backed). There is no `diagram/`
directory. `ensure_dataset` calls `snapshot_download(allow_patterns=[f"{task_name}/*"])`,
which matches zero files by construction. Every run recorded `top10_references_count: 0`.
PaperBanana's stated premise is reference-driven generation; that mechanism has been inert
here throughout, and the Planner, Stylist and Critic loop carried every figure.

**The ~18 minutes sits inside `ensure_dataset`, but its cause is not established.** Run
directory created 08:49:53, HF tree cache written 09:07:53.757, manifest `started_at`
09:07:54. So essentially the whole gap is inside the acquisition call, before any bytes
transfer. The call *succeeded* after 18 minutes rather than hanging, which is a weaker
match to the documented Python-to-Cloudflare stall on this box than first assumed;
`huggingface_hub` retry/backoff or Xet CAS resolution fit an 18-minutes-then-success
profile equally well. Treat the cause as undiagnosed.

**Two unpinned review findings plus one misleading field.** A duplicated prose/base64
guard where neither half is individually tested; a dead key-dropping branch in
`scrub_payloads` whose assertions cannot fail; and a manifest that records the *requested*
retrieval setting, so it asserts `"setting": "auto"` for runs that performed no retrieval.

**Two corrections to earlier framing, both verified.** The skill does already carry
recovery guidance (`git worktree list`); its real fragilities are a missing search root, a
second hardcoded path to the main checkout's venv, and a discovery predicate
(`--figure-size` present) that stops discriminating the moment the branch merges. And
`data` and `configs/model_config.yaml` in this worktree are **symlinks into the main
checkout**, which is where the live `paperbanana.service` runs. There is no filesystem
isolation between this branch and the running service for those paths.

## Requirements Trace

- R1. The two surviving review findings are fixed and pinned by tests that fail under mutation.
- R2. The manifest records what retrieval *did*, pinned to the same mutation standard.
- R3. No run pays an acquisition cost it cannot benefit from, and acquisition cost is visible rather than silent.
- R4. `ensure_dataset` can actually acquire the dataset, so retrieval is available.
- R5. Enabling retrieval does not silently change cost or output for existing callers.
- R6. Enabling retrieval does not change the behaviour of the live Gradio service as a side effect.
- R7. The skill is within its word guideline and its real fragilities are addressed.
- R8. Work reaches review under the correct Greptile org; `origin` is never a push target.
- R9. The vault float manifest reflects reality, including the vector decision.
- R10. No litter.

## Scope Boundaries

- `origin` (`dwzhu-pku/PaperBanana`) is never a push target. Any upstream contribution is a separate, explicitly approved act.
- Do not change the behaviour of the running `paperbanana.service`. Note this requires positive action, not inaction, because `data/` is shared by symlink.
- Not fixing the machine-wide Python-to-Cloudflare stall as a general problem.
- Not changing the agent pipeline, prompts, or the Gradio UI beyond what R2 requires.
- `--task plot` stays broken.

### Deferred to Separate Tasks

- **Does retrieval improve figures?** Answerable only by a paid A/B run once Slice D lands. Reuse the pinned inputs from `run_20260804_084953`.
- **Making retrieval the default.** Explicitly deferred until that A/B answers it. See Slice D.
- **Upstream contribution of commit `2ca63da`** to `dwzhu-pku/PaperBanana`: shaped, sent only on explicit approval.

## Key Technical Decisions

- **Slice into independent PRs** (Trent, 2026-08-04). Slices A through C are low-risk and independently valuable; D carries all the cost and behavioural risk and should be reviewable on its own.
- **Guard acquisition on the *requested* flag, never the effective mode.** The effective mode is derived by `RetrieverAgent` from whether `ref.json` exists, which only becomes true after acquisition. Keying the guard to it is circular and would permanently prevent bootstrap.
- **Diagnose the stall before downloading 266 MB through it.** Instrumentation and any IPv4 workaround land in Slice C, before Slice D transfers a quarter gigabyte.
- **Extract to a branch-local data root, not the shared symlink.** `data` points into the live service's checkout. Writing the dataset there switches on paid retrieval in the running Gradio UI with no merge and no deploy.
- **Retrieval stays opt-in after Slice D.** `--retrieval-setting` currently defaults to `auto`, so enabling acquisition would otherwise turn on a 200-candidate LLM call plus roughly ten reference-image attachments per candidate for every default run, before anyone has shown it helps.
- **Acceptance for R4 uses `--retrieval-setting random`**, which reads `ref.json` without a retrieval model call, at one candidate and zero critic rounds. This is the cheapest honest gate; the manifest is only written after a generation, so some paid run is unavoidable.

## Open Questions

### Resolved During Planning

- Why does `allow_patterns` match nothing? The dataset ships one zip; there is no per-task directory. Confirmed against the live HF file listing.
- How large is the archive? 265,846,711 bytes, from the local HF tree cache.
- Is the live service isolated from this branch? No, for `data/` and `configs/`, which are symlinks into the main checkout.
- Can the acquisition guard key on effective retrieval mode? No, it is circular.
- Does the skill lack recovery guidance? No, that framing was wrong; it has `git worktree list`. The real gaps are different.

### Deferred to Implementation

- The true cause of the 18 minutes. Instrument first (Slice C), attribute second.
- The zip's internal layout, and whether it maps onto `<task>/ref.json`, `<task>/images/`, and each record's `path_to_gt_image`. If it does not, Slice D grows a path-remapping layer, and `PlannerAgent` dereferences those paths with an unguarded open, so a mismatch becomes a mid-run candidate failure rather than a clean fallback.
- Whether `RetrieverAgent` can report its effective mode without a new field, or needs one.
- Whether the `auto` prompt (200 candidates with full methodology text) fits the main model's context at all.

## Slices

Each of A through D is a standalone PR against the personal fork. H precedes all of them.

```
H (fork remote)  ──►  A ──┐
                     B ──┼──► (independent PRs, any order)
                     C ──┘
                          └──► D  (depends on B for provenance, C for instrumentation)

E, F, G: no PR, no dependency on the above except E depends on B/C/D landing
```

---

- [ ] **Slice H: Fork remote setup** (prerequisite, no PR)

**Goal:** A push target exists that is not the third-party upstream.

**Requirements:** R8

**Files:** None. Remote configuration only.

**Approach:**
- The fork already exists: `trentleslie/PaperBanana`, public, created 2026-02-02. No creation needed; confirm it is still a fork of the current upstream and how far behind it is.
- Add it as a named remote distinct from `origin`. The worktree currently has exactly one remote, `origin`, pointing at the upstream for both fetch and push, so the obvious push command would violate the scope boundary.
- Verify the fork's Greptile org attachment **before** opening any PR, since review lands under the author's org.
- Commit the plan and requirements documents so PRs carry their own rationale.

**Test scenarios:** Test expectation: none — repository configuration.

**Verification:**
- A named non-`origin` remote exists and points at the personal fork.
- `git config --get branch.feat/headless-cli-fidelity.remote` is not `origin`.

---

- [ ] **Slice A: Close the two surviving review findings** (PR 1)

**Goal:** Both findings fixed and pinned by tests that fail under mutation.

**Requirements:** R1

**Dependencies:** Slice H for the PR only; the code work is independent.

**Files:**
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_manifest.py`

**Approach:**
- The prose/base64 guard is implemented twice, the separator check being fully subsumed by the character class, so neither half is individually pinned and the comment misattributes the mechanism. Keep one mechanism or pin each independently, and correct the comment.
- The key-dropping branch of `scrub_payloads` and the scrub over manifest entries are dead under current call sites, so their assertions cannot fail. Either exercise them from a real call site or remove them. Do not leave assertions that cannot go red.

**Execution note:** Mutation-check both. A test that cannot fail reads exactly like a test that passes.

**Test scenarios:**
- Happy path: a long base64 payload is still scrubbed.
- Edge case: long punctuation-free prose is still not scrubbed, which is why the second guard exists.
- Edge case: each retained guard, disabled individually, turns at least one test red.
- Edge case: if the key-dropping branch is kept, a value reaching it through a real call site is dropped; if removed, no test references it.

**Verification:** Disabling any surviving guard produces a red test.

---

- [ ] **Slice B: Record effective retrieval, not requested** (PR 2)

**Goal:** A manifest can no longer assert `auto` for a run that performed no retrieval.

**Requirements:** R2

**Dependencies:** Slice H for the PR only.

**Files:**
- Modify: `skill/run.py`, `agents/retriever_agent.py`
- Test: `tests/test_skill_run_manifest.py`

**Approach:**
- Have the Retriever surface the mode it actually used rather than re-deriving it in `run.py`. It already downgrades `auto`/`random` to `none` when the ref file is missing.
- Record both requested and effective, so a downgrade is visible rather than inferred from a zero count.
- `build_retrieval_record` runs from a `finally` block, so it executes even when the batch dies before retrieval. An absent effective mode must serialize unambiguously (a distinct "not attempted" state), never as `null` or an implicit `none`.
- Minimum pipeline change; do not widen it.

**Execution note:** Same mutation standard as Slice A. Removing the effective-mode plumbing must turn a manifest test red.

**Test scenarios:**
- Happy path: requested `auto` with references available records effective `auto`.
- Edge case: requested `auto` with the ref file missing records effective `none`, and the two fields differ.
- Edge case: requested `none` records effective `none`.
- Error path: a run that dies before retrieval records "not attempted", distinct from `none`.
- Integration: a manifest whose effective mode is `none` never also reports a non-zero reference count.
- Edge case: removing the effective-mode reporting turns at least one test red.

**Verification:** Reading a manifest alone tells you whether retrieval happened.

---

- [ ] **Slice C: Stop paying for the dead download, and instrument it** (PR 3)

**Goal:** Reclaim the ~18 minutes, and make acquisition cost visible instead of silent.

**Requirements:** R3

**Dependencies:** Slice H for the PR only. Deliberately lands **before** Slice D so the 266 MB transfer happens through instrumented, diagnosed code.

**Files:**
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_dataset.py` *(create)*

**Approach:**
- Guard acquisition on the **requested** setting: skip when `--retrieval-setting none`. Today `ensure_dataset(args.task)` is unconditional, which is why that flag provably cannot reclaim the time. Do not key this to the effective mode.
- Surface elapsed acquisition time on stderr, so a multi-minute call can never again be invisible.
- Diagnose the residual stall: `curl` returns HTTP 200 in 0.2s where the Python client took ~18 minutes to succeed. Test whether forcing IPv4 helps. Be honest in the PR about whether the cause was established or only worked around; retry/backoff and Xet resolution fit the evidence equally well.

**Test scenarios:**
- Happy path: with retrieval requested as `none`, acquisition is not attempted.
- Happy path: with retrieval requested and data present, no network call occurs.
- Edge case: the guard reads the requested flag, so it cannot be defeated by a downgrade that has not happened yet.
- Error path: a slow or failed acquisition reports elapsed time rather than appearing hung.

**Verification:**
- A `--retrieval-setting none` run begins generating within seconds.
- Any acquisition reports how long it took.

---

- [ ] **Slice D: Make `ensure_dataset` actually acquire the dataset** (PR 4)

**Goal:** Retrieval becomes available, without changing cost or behaviour for anyone who has not opted in.

**Requirements:** R4, R5, R6

**Dependencies:** Slice C (instrumentation and the guard) and Slice B (so the run that proves it also reports honestly).

**Files:**
- Modify: `skill/run.py`
- Test: `tests/test_skill_run_dataset.py`

**Approach:**
- Fetch `PaperBananaBench.zip` (266 MB) rather than a non-existent `<task>/*` tree, and extract it so the `ref.json` + `images/` check becomes satisfiable. Inspect the archive's internal layout before assuming it maps onto `<task>/`; `PlannerAgent` dereferences `path_to_gt_image` with an unguarded open, so a layout mismatch surfaces as a mid-run candidate failure rather than a clean fallback.
- **Extract to a branch-local data root, not through the `data` symlink.** That symlink points into the main checkout, where the live `paperbanana.service` runs and currently falls back to `none` because `ref.json` is absent. Writing there would switch on paid retrieval in the running UI with no merge and no deploy. Preserving current live behaviour is the requirement (R6); leave the shared path untouched.
- **Leave `--retrieval-setting` defaulting to `none` for the CLI**, or otherwise ensure a default run does not newly incur retrieval. It currently defaults to `auto`; acquisition alone would therefore add a 200-candidate LLM call plus roughly ten reference-image attachments per candidate to every default run, before the deferred A/B has shown any benefit.
- Decide and state whether the 266 MB zip is retained after extraction, and make the "already present" check tolerate that decision.
- A failed acquisition must not leave a partial tree that satisfies the existence check.

**Execution note:** This slice changes cost and can change output. Keep it a separate PR from A through C so it can be reverted independently.

**Test scenarios:**
- Happy path: with the dataset extracted, acquisition returns without a network call.
- Happy path: after acquisition, `ref.json` and `images/` exist at the path the existence check tests.
- Edge case: a partial or truncated archive is treated as absent, not present.
- Edge case: the main checkout's `data/` is unchanged, so the live UI still falls back to `none`.
- Edge case: a default CLI invocation does not newly perform retrieval.
- Error path: a failed download surfaces an error and leaves no tree that would satisfy the existence check.
- Integration: one minimal acceptance run at `--retrieval-setting random --num-candidates 1 --max-critic-rounds 0` records a non-zero `top10_references_count`, which no run has ever done. `random` reads `ref.json` without a retrieval model call, making this the cheapest honest gate.

**Verification:**
- The acceptance run records non-zero references.
- The live Gradio service's retrieval behaviour is unchanged, verified by inspecting the main checkout's `data/`.

---

- [ ] **Slice E: Split and de-fragilize the skill** `[out-of-repo, no PR]`

**Goal:** The skill is within its word guideline, accurate, and finds its checkout robustly.

**Requirements:** R7

**Dependencies:** Slices B, C, D, whose landing makes several current warnings false.

**Files:**
- Modify: `~/.claude/skills/paperbanana/SKILL.md`
- Create: `~/.claude/skills/paperbanana/reference.md`

**Approach:**
- Move the trap catalogue, defect classes and quick-reference table into a companion file; keep workflow, the checkout check and the explicit-negatives technique inline.
- Address the *real* fragilities, not the one previously claimed. Recovery guidance already exists. The actual gaps: no defined search root, so an agent still needs a starting directory; a second hardcoded path to the main checkout's venv interpreter; and a discovery predicate that stops discriminating once the branch merges and both checkouts expose `--figure-size`. Prefer resolving by branch name via `git worktree list`, with the capability probe as fallback.
- Retire guidance that Slices B through D make false, notably the retrieval-is-inert warning and the `--retrieval-setting none` prohibition.

**Test scenarios:**
- Test expectation: none for the file; verification is behavioural.
- Behavioural: re-run the two subagent scenarios used to build the skill (fresh figure request; "make it fast" request tempting the retracted optimization) and confirm the agent still reaches the correct checkout, still gates on cost, and no longer repeats retracted guidance.

**Verification:** `wc -w` within guideline; a fresh agent reaches the right checkout without the path being spelled out.

---

- [ ] **Slice F: Vault float manifest and the vector decision** `[out-of-repo, no PR]`

**Goal:** The figure plan reflects reality, in one edit and one publish.

**Requirements:** R9

**Dependencies:** None.

**Files:**
- Modify: `Active 🎯/Work/BioMapper Preprint - Figures & Tables Plan.outline.md` (vault)

**Approach:**
- Merged from two previously separate units, which would have edited and published the same Outline-synced note twice in inverted order.
- Establish what NAR actually requires for figure format from the journal's own author guidance. Note the vault already records "render to final vector art for submission", so this is verifying or overturning an existing judgement of Trent's, not answering a blank. Surface the finding; do not commit him to a redraw.
- Record the 2026-08-04 render: run directory, chosen candidate, measured dimensions, manifest path as the provenance pointer, and the fact that it is **not** reference-grounded, since every run to date recorded zero references.
- One edit, then one publish, then confirm the published page renders.

**Test scenarios:** Test expectation: none — documentation and research.

**Verification:** The Figure 1 row reflects reality, the vector decision and its source are recorded, and the published page loads.

---

- [ ] **Slice G: Housekeeping** `[no PR]`

**Goal:** No litter.

**Requirements:** R10

**Dependencies:** None.

**Files:** No tracked files.

**Approach:**
- Remove the stale `.memdb/` directory in the worktree (memtrace litter from running outside the phenome workspace).
- Remove the three single-file symlinks in the tailnet index docroot **only once Trent confirms he is done viewing the figures**. They are his viewing surface; do not remove them unasked.

**Test scenarios:** Test expectation: none — cleanup.

**Verification:**
- `git status --porcelain` in the worktree shows no `.memdb/`, leaving only the gitignored `.venv` and `data` symlinks and any uncommitted docs.
- `ls ~/.local/share/tailnet-index/` shows only `index.html`, after confirmation.

## System-Wide Impact

- **Interaction graph:** Slices A, C, D touch `skill/run.py`; B additionally touches one construct in `agents/retriever_agent.py`.
- **Shared filesystem, not shared branch:** `data/` and `configs/model_config.yaml` are symlinks into the main checkout where the live service runs. Branch separation does **not** isolate them. This is the single most important blast-radius fact in the plan.
- **Error propagation:** A failed acquisition must not leave a partial tree that satisfies the existence check, or every later run proceeds with broken reference data.
- **Cost surface:** Enabling retrieval adds a 200-candidate LLM call and roughly ten reference images per candidate. Slice D must not turn that on by default.
- **Unchanged invariants:** stdout stays image paths only; timestamped run directories, credential-free manifest and partial-failure survival are untouched; `--task plot` stays broken; the live UI's retrieval behaviour stays as it is.

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Extracting through the `data` symlink silently switches on paid retrieval in the live Gradio UI | High if unaddressed | High | Slice D extracts to a branch-local root; an explicit test asserts the main checkout's `data/` is unchanged |
| Enabling acquisition turns retrieval on by default and raises every run's cost | High if unaddressed | High | Retrieval stays opt-in until the deferred A/B; Slice D has an explicit test that a default run performs no retrieval |
| The 266 MB transfer runs through an undiagnosed 18-minute stall | Med | Med | Slice C lands first with instrumentation and any IPv4 workaround |
| The zip's layout does not map onto `<task>/ref.json` and `path_to_gt_image` | Med | Med | Slice D inspects before adapting; an unguarded open in `PlannerAgent` makes a mismatch a mid-run failure, so this must be checked, not assumed |
| A push reaches the third-party upstream | Low | High | Slice H adds a named fork remote and forbids `origin` as a push target; it precedes every other slice's PR |
| Greptile review lands under the wrong org | Low | Med | Slice H verifies org attachment before any PR is opened |
| Skill retracts guidance that is still true because a slice slipped | Low | Med | Slice E depends on B, C and D landing |

## Documentation / Operational Notes

- The skill is the operational doc for this tool and must be updated in the same pass as any behaviour change from Slices B through D.
- `paperbanana.service` continues serving `main`. Nothing here requires restarting it, and Slice D exists partly to keep that true.
- Any paid run remains subject to the artifact-hygiene rule: timestamped directory, manifest by default, inputs pinned.

## Sources & References

- **Origin document:** `docs/brainstorms/2026-08-04-headless-cli-fidelity-requirements.md`
- **Prior plan:** `docs/plans/2026-08-04-001-feat-headless-cli-fidelity-plan.md`
- Related code: `skill/run.py`, `agents/retriever_agent.py`, `agents/planner_agent.py`, `utils/legacy_ui_results.py`
- Evidence run: `results/skill_runs/run_20260804_084953/output.manifest.json`
- Upstream (never a push target): `dwzhu-pku/PaperBanana`; dataset `dwzhu/PaperBananaBench`
