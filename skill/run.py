# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CLI entry point for PaperBanana Skill.
Generates publication-quality academic diagrams and plots from method text.
"""

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.legacy_generation_options import (  # noqa: E402
    FIGURE_SIZE_TO_IMAGE_SIZE,
    generation_additional_info,
)
from utils.legacy_ui_results import BASE64_SUFFIX  # noqa: E402

FIGURE_SIZE_CHOICES = list(FIGURE_SIZE_TO_IMAGE_SIZE)
DEFAULT_FIGURE_SIZE = "14-17cm"
DEFAULT_ASPECT_RATIO = "16:9"

# Nominal pixel budget per provider image-size tier. Compared by total area
# rather than by exact edge length, because the provider does not honour the
# requested aspect ratio exactly: the one calibration point on this machine,
# 4k at 16:9, decoded to 5504x3072 (ratio 1.7917, not 1.7778).
IMAGE_SIZE_EXPECTED_PIXELS = {
    "1k": 1024 * 1024,
    "2k": 2048 * 2048,
    "4k": 4096 * 4096,
}
# A decoded image below this fraction of its tier's pixel budget is a material
# downgrade worth flagging, not provider jitter.
SIZE_SHORTFALL_FRACTION = 0.75
# Relative tolerance on the requested aspect ratio.
ASPECT_RATIO_TOLERANCE = 0.05

# Runs default to a fresh timestamped directory so a repeat invocation cannot
# destroy a prior run's images or manifest.
DEFAULT_RUN_BASE_DIR = PROJECT_ROOT / "results" / "skill_runs"
RUN_DIR_PREFIX = "run_"
DEFAULT_OUTPUT_NAME = "output.png"

# Run record. Written on every run, beside the images, sharing the output stem.
MANIFEST_SUFFIX = ".manifest.json"
MANIFEST_VERSION = 1
ELIDED_PAYLOAD = "<elided: base64 image payload>"
# Long enough that prose never trips it, short enough that no real payload slips
# through: the smallest images the pipeline emits are several KB base64.
BASE64_VALUE_MIN_LENGTH = 512
_BASE64_VALUE_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def build_additional_info(args) -> dict:
    """Build the per-candidate ``additional_info`` dict from parsed CLI args.

    Routed through the shared helper so both ``figure_size`` and the derived
    ``image_size`` are populated, exactly as the Gradio UI does.
    """
    return generation_additional_info(args.aspect_ratio, args.figure_size)


def _nominal_ratio(aspect_ratio: str) -> float | None:
    try:
        width, height = (float(part) for part in str(aspect_ratio).split(":"))
    except (TypeError, ValueError):
        return None
    return width / height if height else None


def describe_image_dimensions(
    figure_size: str,
    image_size: str,
    aspect_ratio: str,
    width: int,
    height: int,
) -> dict:
    """Record what was requested against what was actually decoded.

    ``image_size`` is a request, not a guarantee, so a downgrade must never be
    silent. Returns a slim, JSON-serializable record.
    """
    pixels = int(width) * int(height)
    expected_pixels = IMAGE_SIZE_EXPECTED_PIXELS.get(image_size)
    size_shortfall = bool(
        expected_pixels and pixels < expected_pixels * SIZE_SHORTFALL_FRACTION
    )

    actual_ratio = round(width / height, 4) if width and height else None
    nominal_ratio = _nominal_ratio(aspect_ratio)
    aspect_ratio_drift = bool(
        actual_ratio is not None
        and nominal_ratio
        and abs(actual_ratio - nominal_ratio) / nominal_ratio > ASPECT_RATIO_TOLERANCE
    )

    return {
        "figure_size": figure_size,
        "image_size": image_size,
        "requested_aspect_ratio": aspect_ratio,
        "width": int(width),
        "height": int(height),
        "pixels": pixels,
        "expected_pixels": expected_pixels,
        "actual_ratio": actual_ratio,
        "size_shortfall": size_shortfall,
        "aspect_ratio_drift": aspect_ratio_drift,
    }


def format_dimension_report(desc: dict, label: str = "") -> str:
    """Render a dimension record as one human-readable line."""
    parts = [
        f"figure_size={desc['figure_size']}",
        f"image_size={desc['image_size']}",
        f"requested_ratio={desc['requested_aspect_ratio']}",
        f"actual={desc['width']}x{desc['height']}",
    ]
    if desc["actual_ratio"] is not None:
        parts.append(f"actual_ratio={desc['actual_ratio']}")
    if desc["size_shortfall"]:
        parts.append(
            "SHORTFALL: decoded image is materially smaller than the requested tier"
        )
    if desc["aspect_ratio_drift"]:
        parts.append("ASPECT-DRIFT: provider did not honour the requested ratio")
    prefix = f"[dimensions] {label} " if label else "[dimensions] "
    return prefix + " ".join(parts)


def ensure_model_config():
    """Copy model_config.template.yaml to model_config.yaml if missing."""
    configs_dir = PROJECT_ROOT / "configs"
    config_path = configs_dir / "model_config.yaml"
    template_path = configs_dir / "model_config.template.yaml"
    if not config_path.exists() and template_path.exists():
        shutil.copy2(template_path, config_path)


def ensure_dataset(task_name: str):
    """Download PaperBananaBench data from HuggingFace if not present locally."""
    data_dir = PROJECT_ROOT / "data" / "PaperBananaBench" / task_name
    ref_path = data_dir / "ref.json"
    images_dir = data_dir / "images"
    if ref_path.exists() and images_dir.exists():
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required for automatic dataset download.\n"
              "Install it with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    print(f"Downloading PaperBananaBench/{task_name} from HuggingFace...")
    snapshot_download(
        "dwzhu/PaperBananaBench",
        repo_type="dataset",
        allow_patterns=[f"{task_name}/*"],
        local_dir=str(PROJECT_ROOT / "data" / "PaperBananaBench"),
    )


def extract_final_image_b64(result: dict, exp_mode: str) -> str | None:
    """Return the base64-encoded final image from a pipeline result dict.

    Follows the same fallback order as demo.py:display_candidate_result.
    """
    task_name = "diagram"

    # Try critic rounds 3 → 0
    for round_idx in range(3, -1, -1):
        key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if key in result and result[key]:
            return result[key]

    # Fallback: stylist (demo_full) or planner
    if exp_mode == "demo_full":
        key = f"target_{task_name}_stylist_desc0_base64_jpg"
    else:
        key = f"target_{task_name}_desc0_base64_jpg"
    return result.get(key)


def candidate_identity(result) -> str | None:
    """Return the candidate's own identity, or None when it cannot be derived.

    One shared identity function: this same string names the PNG and keys the
    manifest entry, so the two can never disagree.
    """
    if not isinstance(result, dict):
        return None
    raw = result.get("filename")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())


def create_run_directory(base_dir: Path | None = None, timestamp: str | None = None) -> Path:
    """Create and return a fresh timestamped run directory.

    Created with ``exist_ok=False`` and retried with a counter so two runs
    starting in the same second cannot merge their artifacts.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_RUN_BASE_DIR
    base.mkdir(parents=True, exist_ok=True)

    stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{RUN_DIR_PREFIX}{stamp}"
    collision = 1
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base / f"{RUN_DIR_PREFIX}{stamp}_{collision}"
            collision += 1


def default_output_path(run_dir: Path) -> Path:
    """The nominal output path inside a timestamped run directory.

    Only its parent and suffix are used for images; it also supplies the stem
    the manifest shares.
    """
    return Path(run_dir) / DEFAULT_OUTPUT_NAME


def resolve_save_path(
    identity: str,
    output_path: Path,
    num_candidates: int,
    output_explicit: bool = True,
) -> Path:
    """Map a candidate identity to the path its image is written to.

    A single candidate with an explicit --output keeps its exact-path contract.
    With --output omitted every candidate is named purely from its identity
    inside the run directory.
    """
    suffix = output_path.suffix or ".png"
    if not output_explicit:
        return output_path.parent / f"{identity}{suffix}"
    if num_candidates == 1:
        return output_path
    return output_path.parent / f"{output_path.stem}_{identity}{suffix}"


def save_result_images(
    results,
    *,
    exp_mode: str,
    output_path: Path,
    num_candidates: int,
    figure_size: str,
    image_size: str,
    aspect_ratio: str,
    output_explicit: bool = True,
) -> list[dict]:
    """Write each result's final image, named from that result's own identity.

    Returns one record per result, including results that produced no image, so
    a caller can pair reasoning with artifacts without re-deriving anything.
    """
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved: list[dict] = []
    for result in results:
        identity = candidate_identity(result)
        if identity is None:
            print(
                "WARNING: result has no usable 'filename'; skipping rather than "
                "writing it under a fabricated name.",
                file=sys.stderr,
            )
            continue

        b64 = extract_final_image_b64(result, exp_mode)
        if not b64:
            print(f"WARNING: No image produced for candidate {identity}.", file=sys.stderr)
            saved.append({"identity": identity, "image_path": None, "dimensions": None})
            continue

        if "," in b64:
            b64 = b64.split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(b64)))

        save_path = resolve_save_path(
            identity, output_path, num_candidates, output_explicit=output_explicit
        )
        img.save(str(save_path), format="PNG")

        width, height = img.size
        dimensions = describe_image_dimensions(
            figure_size, image_size, aspect_ratio, width, height
        )
        print(format_dimension_report(dimensions, label=save_path.name), file=sys.stderr)

        saved.append(
            {
                "identity": identity,
                "image_path": str(save_path),
                "dimensions": dimensions,
            }
        )

    return saved


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_path_for(output_path: Path) -> Path:
    """The manifest travels with the images and shares the --output stem."""
    output_path = Path(output_path)
    return output_path.parent / f"{output_path.stem}{MANIFEST_SUFFIX}"


def seed_manifest_entries(data_list) -> dict:
    """One entry per *requested* candidate, before anything is drained.

    Completeness is evaluated against what was asked for, not against what was
    observed, so a run that dies part-way cannot stamp itself complete.
    """
    entries: dict[str, dict] = {}
    for index, data in enumerate(data_list):
        identity = candidate_identity(data) or f"unidentified_candidate_{index}"
        entries[identity] = {
            "identity": identity,
            "status": "missing",
            "image_path": None,
            "dimensions": None,
            "trace": None,
            "error": None,
        }
    return entries


def run_status(entries: dict) -> str:
    """complete only when every seeded entry succeeded."""
    statuses = {entry.get("status") for entry in entries.values()}
    if statuses == {"succeeded"}:
        return "complete"
    if "succeeded" in statuses:
        return "partial"
    return "failed"


def looks_like_base64_payload(value) -> bool:
    """True for values that are image payloads rather than prose.

    A key-name check alone would admit a blob stored under another name.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if text.startswith("data:image"):
        return True
    if len(text) < BASE64_VALUE_MIN_LENGTH:
        return False
    return bool(_BASE64_VALUE_RE.match(text))


def scrub_payloads(value):
    """Drop base64-suffixed keys and elide payload-shaped values, recursively."""
    if isinstance(value, dict):
        return {
            key: scrub_payloads(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key.endswith(BASE64_SUFFIX))
        }
    if isinstance(value, (list, tuple)):
        return [scrub_payloads(item) for item in value]
    if looks_like_base64_payload(value):
        return ELIDED_PAYLOAD
    return value


def build_candidate_trace(result: dict, exp_mode: str) -> list[dict]:
    """Per-candidate reasoning trace, built from the shared UI stage helper."""
    from utils.legacy_ui_results import build_evolution_stages

    trace: list[dict] = []
    for stage in build_evolution_stages(result, exp_mode=exp_mode):
        record = {
            "name": stage.get("name"),
            "image_key": stage.get("image_key"),
            "description_label": stage.get("description"),
        }

        desc_key = stage.get("desc_key")
        if desc_key:
            record["description_key"] = desc_key
            record["description"] = scrub_payloads(result.get(desc_key))

        # Only Critic stages carry suggestions_key; a literal lookup would raise
        # on the first stage of every candidate.
        suggestions_key = stage.get("suggestions_key")
        if suggestions_key:
            record["suggestions_key"] = suggestions_key
            record["suggestions"] = scrub_payloads(result.get(suggestions_key))

        trace.append(record)
    return trace


def derive_image_gen_backend(image_gen_model_name: str, openrouter_client) -> str:
    """Mirror the branch visualizer_agent takes at call time.

    Derived from ambient config, not observed from the provider, hence the
    '_derived' suffix on the manifest field.
    """
    if "gpt-image" in (image_gen_model_name or ""):
        return "openai"
    if openrouter_client is not None:
        return "openrouter"
    return "gemini"


def repo_commit_record() -> dict:
    """The commit the run executed at, with a dirty checkout stated explicitly."""
    def _git(*argv):
        try:
            completed = subprocess.run(
                ["git", *argv],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout if completed.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    return {
        "commit": head.strip() if head else None,
        "dirty": bool(porcelain and porcelain.strip()),
    }


def record_result(result, entries: dict, **save_kwargs) -> dict:
    """Fold one drained result into the seeded entries, saving its image."""
    identity = candidate_identity(result)
    if identity is None:
        print(
            "WARNING: result has no usable 'filename'; it cannot be matched to a "
            "requested candidate.",
            file=sys.stderr,
        )
        return {}

    entry = entries.setdefault(
        identity,
        {
            "identity": identity,
            "status": "missing",
            "image_path": None,
            "dimensions": None,
            "trace": None,
            "error": None,
        },
    )

    saved = save_result_images([result], **save_kwargs)
    record = saved[0] if saved else {"image_path": None, "dimensions": None}

    entry["image_path"] = record.get("image_path")
    entry["dimensions"] = record.get("dimensions")
    entry["trace"] = build_candidate_trace(result, save_kwargs.get("exp_mode", ""))
    entry["status"] = "succeeded" if record.get("image_path") else "no_image"
    return entry


def record_failure(identity, entries: dict, error) -> dict:
    """Fold a raised candidate into the entries without aborting the batch."""
    identity = identity or "unidentified_candidate"
    entry = entries.setdefault(
        identity,
        {
            "identity": identity,
            "status": "missing",
            "image_path": None,
            "dimensions": None,
            "trace": None,
            "error": None,
        },
    )
    entry["status"] = "failed"
    entry["error"] = str(error)
    print(f"WARNING: candidate {identity} failed: {error}", file=sys.stderr)
    return entry


def is_error_record(result) -> bool:
    """True for the explicit error record process_queries_batch now yields."""
    return isinstance(result, dict) and bool(result.get("candidate_error"))


async def drain_batch(result_stream, entries: dict, **save_kwargs) -> None:
    """Drain the batch, folding both successes and failures into the entries."""
    async for result_data in result_stream:
        if is_error_record(result_data):
            record_failure(
                candidate_identity(result_data),
                entries,
                result_data.get("candidate_error"),
            )
        else:
            record_result(result_data, entries, **save_kwargs)


def build_retrieval_record(data_list, setting: str | None = None) -> dict:
    """Retrieval runs once per batch, so it is recorded once per run."""
    first = data_list[0] if data_list else {}
    references = first.get("top10_references") or []
    examples = first.get("retrieved_examples") or []
    return {
        "setting": setting,
        "top10_references_count": len(references),
        "retrieved_examples_count": len(examples),
        "top10_references": scrub_payloads(references),
    }


def build_manifest(
    *,
    args,
    additional_info: dict,
    entries: dict,
    content: str,
    resolved_models: dict,
    image_gen_backend: str,
    retrieval: dict,
    started_at: str,
    finished_at: str,
) -> dict:
    """Assemble the run record from an explicit allowlist.

    Credential material is excluded by construction rather than by filtering:
    nothing is copied wholesale from args, config, or the result dicts.
    """
    content = content or ""
    return {
        "manifest_version": MANIFEST_VERSION,
        "run": {
            "status": run_status(entries),
            "started_at": started_at,
            "finished_at": finished_at,
            "candidates_requested": len(entries),
            "candidates_succeeded": sum(
                1 for entry in entries.values() if entry.get("status") == "succeeded"
            ),
            "models": {
                "main_model_name": resolved_models.get("main_model_name", ""),
                "image_gen_model_name": resolved_models.get("image_gen_model_name", ""),
            },
            "image_gen_backend_derived": image_gen_backend,
            "resolved_image_size": additional_info.get("image_size", ""),
            "repository": repo_commit_record(),
        },
        "parameters": {
            "task": getattr(args, "task", None),
            "caption": getattr(args, "caption", None),
            "output": getattr(args, "output", None),
            "aspect_ratio": getattr(args, "aspect_ratio", None),
            "figure_size": getattr(args, "figure_size", None),
            "max_critic_rounds": getattr(args, "max_critic_rounds", None),
            "num_candidates": getattr(args, "num_candidates", None),
            "retrieval_setting": getattr(args, "retrieval_setting", None),
            "planner_metaphor": getattr(args, "planner_metaphor", None),
            "exp_mode": getattr(args, "exp_mode", None),
            "main_model_name_requested": getattr(args, "main_model_name", None),
            "image_gen_model_name_requested": getattr(args, "image_gen_model_name", None),
            "additional_info": {
                "rounded_ratio": additional_info.get("rounded_ratio", ""),
                "figure_size": additional_info.get("figure_size", ""),
                "image_size": additional_info.get("image_size", ""),
            },
        },
        "input": {
            "content_file": getattr(args, "content_file", None) or None,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content_chars": len(content),
            # Embedded rather than pinned by path: a --content-file that is later
            # edited or moved would otherwise void reproducibility.
            "content": content,
        },
        "retrieval": retrieval,
        "candidates": [scrub_payloads(entry) for entry in entries.values()],
    }


def write_manifest(manifest_path: Path, manifest: dict) -> Path:
    """The single emitter. Every exit path routes through here."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest_path


@contextlib.contextmanager
def quiet_pipeline_stdout():
    """Send the pipeline's own print() chatter to stderr.

    38 print() calls sit on the run path across paperviz_processor,
    visualizer_agent and generation_utils. Without this the published
    'stdout is image paths, one per line' contract is false.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def emit_results(entries: dict, manifest_path) -> None:
    """Image paths to stdout, one per line. Manifest path to stderr."""
    for entry in entries.values():
        if entry.get("image_path"):
            print(entry["image_path"])
    if manifest_path is not None:
        print(f"[manifest] {manifest_path}", file=sys.stderr)


async def run(args):
    ensure_model_config()
    ensure_dataset(args.task)

    # Late imports so env is ready
    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    from utils import config, generation_utils
    from utils.paperviz_processor import PaperVizProcessor

    # Read content from file if --content-file is given
    content = args.content
    if args.content_file:
        content = Path(args.content_file).read_text(encoding="utf-8")
    if not content:
        print("ERROR: --content or --content-file is required.", file=sys.stderr)
        sys.exit(1)

    exp_mode = args.exp_mode
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=args.retrieval_setting,
        planner_metaphor=args.planner_metaphor,
        main_model_name=args.main_model_name,
        image_gen_model_name=args.image_gen_model_name,
        work_dir=PROJECT_ROOT,
    )

    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    num_candidates = args.num_candidates

    # Resolve where artifacts land. An omitted --output means a fresh
    # timestamped directory, so a repeat invocation cannot destroy a prior run.
    output_explicit = bool(args.output)
    if output_explicit:
        output_path = Path(args.output).resolve()
    else:
        output_path = default_output_path(create_run_directory())
    print(f"[output] run directory: {output_path.parent}", file=sys.stderr)

    # Build data dicts
    additional_info = build_additional_info(args)
    print(
        f"[config] figure_size={args.figure_size} -> image_size="
        f"{additional_info.get('image_size', '')} at {args.aspect_ratio}",
        file=sys.stderr,
    )
    data_list = []
    for i in range(num_candidates):
        data_list.append({
            "filename": f"skill_candidate_{i}",
            "caption": args.caption,
            "content": content,
            "visual_intent": args.caption,
            "additional_info": dict(additional_info),
            "max_critic_rounds": args.max_critic_rounds,
        })

    # Seed one entry per *requested* candidate before draining anything, so a
    # run that dies part-way is evaluated against what was asked for.
    entries = seed_manifest_entries(data_list)
    save_kwargs = {
        "exp_mode": exp_mode,
        "output_path": output_path,
        "num_candidates": num_candidates,
        "output_explicit": output_explicit,
        "figure_size": args.figure_size,
        "image_size": additional_info.get("image_size", ""),
        "aspect_ratio": args.aspect_ratio,
    }

    started_at = _utc_timestamp()
    manifest_path = None
    try:
        # Pipeline chatter goes to stderr so stdout stays parseable.
        with quiet_pipeline_stdout():
            await drain_batch(
                processor.process_queries_batch(
                    data_list, max_concurrent=num_candidates, do_eval=False
                ),
                entries,
                **save_kwargs,
            )
    finally:
        # Single emitter, reached on every exit path.
        manifest = build_manifest(
            args=args,
            additional_info=additional_info,
            entries=entries,
            content=content,
            resolved_models={
                "main_model_name": exp_config.main_model_name,
                "image_gen_model_name": exp_config.image_gen_model_name,
            },
            image_gen_backend=derive_image_gen_backend(
                exp_config.image_gen_model_name,
                getattr(generation_utils, "openrouter_client", None),
            ),
            retrieval=build_retrieval_record(data_list, args.retrieval_setting),
            started_at=started_at,
            finished_at=_utc_timestamp(),
        )
        manifest_path = write_manifest(manifest_path_for(output_path), manifest)

    emit_results(entries, manifest_path)

    if manifest["run"]["status"] == "failed":
        print(
            "ERROR: no candidate produced an image; see the manifest for details.",
            file=sys.stderr,
        )
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PaperBanana Skill: generate academic diagrams/plots from text"
    )
    parser.add_argument("--content", type=str, default="",
                        help="Method section text to visualize")
    parser.add_argument("--content-file", type=str, default="",
                        help="Path to a file containing the method section text")
    parser.add_argument("--caption", type=str, required=True,
                        help="Figure caption / visual intent")
    parser.add_argument("--task", type=str, default="diagram",
                        choices=["diagram", "plot"],
                        help="Task type: diagram or plot")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path. Omitted, artifacts are written to a "
                             "fresh timestamped directory under results/skill_runs/ so "
                             "a repeat invocation cannot destroy a prior run.")
    parser.add_argument("--aspect-ratio", type=str, default=DEFAULT_ASPECT_RATIO,
                        choices=["21:9", "16:9", "3:2"],
                        help=f"Aspect ratio (default: {DEFAULT_ASPECT_RATIO})")
    parser.add_argument("--figure-size", type=str, default=DEFAULT_FIGURE_SIZE,
                        choices=FIGURE_SIZE_CHOICES,
                        help="Target printed figure width; maps to the provider "
                             "image-size tier (1-3cm/4-6cm -> 1k, 7-9cm/10-13cm -> 2k, "
                             f"14-17cm -> 4k). Default: {DEFAULT_FIGURE_SIZE} (4k)")
    parser.add_argument("--max-critic-rounds", type=int, default=3,
                        help="Max critic refinement rounds (default: 3)")
    parser.add_argument("--num-candidates", type=int, default=10,
                        help="Number of parallel candidates to generate (default: 10)")
    parser.add_argument("--retrieval-setting", type=str, default="auto",
                        choices=["auto", "manual", "random", "none"],
                        help="Retrieval mode: auto (VLM selects refs), manual, random, or none (default: auto)")
    parser.add_argument("--planner-metaphor", action="store_true",
                        help="Enable diagram-only Planner visual-metaphor discovery before detailed description output")
    parser.add_argument("--main-model-name", type=str, default="",
                        help="Main model name for VLM agents (default: from config, currently gemini-3.1-pro-preview)")
    parser.add_argument("--image-gen-model-name", type=str, default="",
                        help="Model name for image generation (default: from config, currently gemini-3.1-flash-image-preview)")
    parser.add_argument("--exp-mode", type=str, default="demo_full",
                        choices=["demo_full", "demo_planner_critic"],
                        help="Pipeline mode: demo_full (Retriever+Planner+Stylist+Visualizer+Critic) or demo_planner_critic (Retriever+Planner+Visualizer+Critic, no Stylist) (default: demo_full)")

    return parser


def main():
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
