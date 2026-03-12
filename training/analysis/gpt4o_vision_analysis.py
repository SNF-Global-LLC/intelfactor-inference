#!/usr/bin/env python3
"""
GPT-4o batch vision analysis for NEU metal surface defect dataset.

Sends images to Azure OpenAI gpt-4o, compares against YOLO ground truth labels,
and builds an RCA hypothesis library from the model's reasoning.

Usage:
    python training/analysis/gpt4o_vision_analysis.py
    python training/analysis/gpt4o_vision_analysis.py --limit 50 --workers 5
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "training/datasets/raw/roboflow/neu-surface-defect"
OUTPUT_DIR = REPO_ROOT / "training/analysis"
RESULTS_FILE = OUTPUT_DIR / "gpt4o_vision_results.jsonl"
REPORT_FILE = OUTPUT_DIR / "agreement_report.json"
RCA_LIBRARY_FILE = OUTPUT_DIR / "rca_hypothesis_library.json"

# ---------------------------------------------------------------------------
# NEU-DET class ID → canonical name mapping (from dataset_config.yaml)
# ---------------------------------------------------------------------------
NEU_CLASS_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

NEU_TO_CANONICAL = {
    "crazing": "surface_crack",
    "inclusion": "inclusion",
    "patches": "surface_discolor",
    "pitted_surface": "surface_dent",
    "rolled-in_scale": "surface_discolor",
    "scratches": "blade_scratch",
}

CANONICAL_CLASSES = [
    "blade_scratch", "grinding_mark", "surface_dent", "surface_crack",
    "weld_defect", "edge_burr", "edge_crack", "handle_defect",
    "bolster_gap", "etching_defect", "inclusion", "surface_discolor",
    "overgrind", "none",
]

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------
def make_client() -> AzureOpenAI:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get(
        "AZURE_OPENAI_ENDPOINT", "https://eastus.api.cognitive.microsoft.com/"
    )
    if not api_key:
        raise ValueError(
            "Set AZURE_OPENAI_API_KEY env var. "
            "Key is in az cognitiveservices account keys list --name intelfactor-aoai -g intelfactor-ml"
        )
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-02-01",
    )


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------
def parse_yolo_label(label_path: Path) -> list[str]:
    """Return list of canonical class names from a YOLO .txt label file."""
    if not label_path.exists():
        return []
    canonical = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        class_id = int(line.split()[0])
        neu_name = NEU_CLASS_NAMES.get(class_id, "unknown")
        canonical.append(NEU_TO_CANONICAL.get(neu_name, neu_name))
    return canonical


# ---------------------------------------------------------------------------
# GPT-4o call
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a manufacturing quality inspector for cutlery and knife production. "
    "You specialize in identifying metal surface defects on steel blanks, blades, "
    "and handles during the manufacturing process. Be precise and conservative — "
    "only report defects you can clearly identify."
)

USER_PROMPT = """Analyze this metal surface image and return a JSON object with exactly these fields:
{
  "defect_detected": true or false,
  "defect_type": "one of: blade_scratch, grinding_mark, surface_dent, surface_crack, weld_defect, edge_burr, edge_crack, handle_defect, bolster_gap, etching_defect, inclusion, surface_discolor, overgrind, none",
  "confidence": 0.0 to 1.0,
  "severity": "critical, major, minor, or none",
  "likely_causes": ["cause1", "cause2"],
  "recommended_actions": ["action1", "action2"],
  "description": "brief 1-2 sentence description of what you observe"
}

Return only valid JSON. No markdown, no explanation outside the JSON."""


_token_lock = __import__("threading").Lock()
_total_prompt_tokens: int = 0
_total_completion_tokens: int = 0

# Global rate limiter: 2s minimum gap between API calls
_api_lock = __import__("threading").Lock()
_last_call_time: float = 0.0
_CALL_INTERVAL: float = 2.0  # seconds between calls


def _add_tokens(prompt: int, completion: int) -> None:
    global _total_prompt_tokens, _total_completion_tokens
    with _token_lock:
        _total_prompt_tokens += prompt
        _total_completion_tokens += completion


def get_token_totals() -> tuple[int, int]:
    with _token_lock:
        return _total_prompt_tokens, _total_completion_tokens


def _throttled_api_call(client: AzureOpenAI, messages: list) -> Any:
    """Enforce minimum interval between API calls and apply exponential backoff on 429."""
    global _last_call_time
    backoff = 10.0  # start at 10s on first 429

    while True:
        # Enforce 2s gap between calls (shared across all threads)
        with _api_lock:
            now = time.time()
            gap = now - _last_call_time
            if gap < _CALL_INTERVAL:
                time.sleep(_CALL_INTERVAL - gap)
            _last_call_time = time.time()

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                timeout=60,
            )
            # Success — reset backoff
            backoff = 10.0
            return response
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "RateLimitError" in type(e).__name__:
                wait = min(backoff, 60.0)
                log.warning("429 rate limit — backing off %.0fs", wait)
                time.sleep(wait)
                backoff = min(backoff * 2, 60.0)
            else:
                raise


def analyze_image(client: AzureOpenAI, image_path: Path) -> dict[str, Any]:
    """Send one image to GPT-4o and return the parsed response."""
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    ext = image_path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}", "detail": "high"},
                },
            ],
        },
    ]

    response = _throttled_api_call(client, messages)

    usage = response.usage
    if usage:
        _add_tokens(usage.prompt_tokens, usage.completion_tokens)

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------
def process_image(
    client: AzureOpenAI,
    image_path: Path,
    label_dir: Path,
) -> dict[str, Any] | None:
    label_path = label_dir / (image_path.stem + ".txt")
    gt_labels = parse_yolo_label(label_path)
    gt_canonical = list(set(gt_labels))  # deduplicated

    try:
        gpt_result = analyze_image(client, image_path)
        return {
            "image": image_path.name,
            "image_path": str(image_path),
            "ground_truth_classes": gt_canonical,
            "gpt4o_defect_detected": gpt_result.get("defect_detected", False),
            "gpt4o_defect_type": gpt_result.get("defect_type", "none"),
            "gpt4o_confidence": gpt_result.get("confidence", 0.0),
            "gpt4o_severity": gpt_result.get("severity", "none"),
            "gpt4o_likely_causes": gpt_result.get("likely_causes", []),
            "gpt4o_recommended_actions": gpt_result.get("recommended_actions", []),
            "gpt4o_description": gpt_result.get("description", ""),
            "has_gt_label": bool(gt_canonical),
        }
    except json.JSONDecodeError as e:
        log.warning("JSON parse error on %s: %s", image_path.name, e)
        return None
    except Exception as e:
        log.error("Failed %s: %s", image_path.name, e)
        return None


# ---------------------------------------------------------------------------
# Agreement analysis
# ---------------------------------------------------------------------------
def compute_agreement(results: list[dict]) -> dict:
    """Compare GPT-4o predictions against YOLO ground truth labels."""
    labeled = [r for r in results if r["has_gt_label"]]
    total = len(results)
    total_labeled = len(labeled)

    # Overall agreement: GPT-4o detected defect AND type matches any GT class
    agree = 0
    fp_cases = []   # GPT-4o: defect, GT: clean (or wrong type)
    fn_cases = []   # GPT-4o: no defect, GT: has defect
    per_class: dict[str, dict] = defaultdict(lambda: {"total": 0, "gpt_agree": 0, "gpt_miss": 0, "gpt_fp": 0})

    for r in labeled:
        gt = set(r["ground_truth_classes"])
        gpt_type = r["gpt4o_defect_type"]
        gpt_detected = r["gpt4o_defect_detected"]

        for cls in gt:
            per_class[cls]["total"] += 1

        if gpt_detected and gpt_type in gt:
            agree += 1
            for cls in gt:
                per_class[cls]["gpt_agree"] += 1
        elif not gpt_detected and gt:
            fn_cases.append({
                "image": r["image"],
                "gt_classes": list(gt),
                "gpt_description": r["gpt4o_description"],
                "gpt_confidence": r["gpt4o_confidence"],
            })
            for cls in gt:
                per_class[cls]["gpt_miss"] += 1
        elif gpt_detected and gt and gpt_type not in gt:
            fp_cases.append({
                "image": r["image"],
                "gt_classes": list(gt),
                "gpt_predicted": gpt_type,
                "gpt_description": r["gpt4o_description"],
                "gpt_confidence": r["gpt4o_confidence"],
            })
            for cls in gt:
                per_class[cls]["gpt_fp"] += 1

    # Per-class agreement rates
    per_class_rates = {}
    for cls, counts in per_class.items():
        t = counts["total"]
        per_class_rates[cls] = {
            "total": t,
            "agreement_rate": round(counts["gpt_agree"] / t, 3) if t else 0,
            "miss_rate": round(counts["gpt_miss"] / t, 3) if t else 0,
            "wrong_type_rate": round(counts["gpt_fp"] / t, 3) if t else 0,
        }

    return {
        "total_images": total,
        "labeled_images": total_labeled,
        "overall_agreement_rate": round(agree / total_labeled, 3) if total_labeled else 0,
        "false_negative_count": len(fn_cases),
        "wrong_type_count": len(fp_cases),
        "per_class_agreement": per_class_rates,
        "false_negatives": fn_cases[:20],   # top 20 for report
        "wrong_type_predictions": fp_cases[:20],
    }


# ---------------------------------------------------------------------------
# RCA hypothesis library
# ---------------------------------------------------------------------------
def build_rca_library(results: list[dict]) -> dict:
    """Aggregate likely_causes and recommended_actions per defect type."""
    library: dict[str, dict] = defaultdict(lambda: {
        "causes": defaultdict(int),
        "actions": defaultdict(int),
        "sample_descriptions": [],
        "count": 0,
    })

    for r in results:
        defect = r["gpt4o_defect_type"]
        if defect == "none" or not r["gpt4o_defect_detected"]:
            continue
        entry = library[defect]
        entry["count"] += 1
        for cause in r["gpt4o_likely_causes"]:
            entry["causes"][cause] += 1
        for action in r["gpt4o_recommended_actions"]:
            entry["actions"][action] += 1
        if len(entry["sample_descriptions"]) < 5:
            entry["sample_descriptions"].append(r["gpt4o_description"])

    # Convert to sorted lists
    output = {}
    for defect, data in sorted(library.items(), key=lambda x: -x[1]["count"]):
        output[defect] = {
            "observation_count": data["count"],
            "top_causes": sorted(data["causes"].items(), key=lambda x: -x[1])[:10],
            "top_actions": sorted(data["actions"].items(), key=lambda x: -x[1])[:10],
            "sample_descriptions": data["sample_descriptions"],
        }
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def collect_images(dataset_dir: Path, limit: int) -> list[tuple[Path, Path]]:
    """Return (image_path, label_dir) pairs from train + valid splits."""
    pairs = []
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*.jpg")):
            pairs.append((p, lbl_dir))
        for p in sorted(img_dir.glob("*.png")):
            pairs.append((p, lbl_dir))
    return pairs[:limit]


def load_existing_results(results_file: Path) -> dict[str, dict]:
    """Load already-processed image names to enable resume."""
    done: dict[str, dict] = {}
    if results_file.exists():
        for line in results_file.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done[r["image"]] = r
                except json.JSONDecodeError:
                    pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-4o batch vision analysis")
    parser.add_argument("--limit", type=int, default=200, help="Max images to process (default: 200)")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent API workers (default: 5)")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--resume", action="store_true", default=True, help="Skip already-processed images")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    client = make_client()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = collect_images(args.dataset_dir, args.limit)
    log.info("Found %d images (limit=%d)", len(pairs), args.limit)

    existing = load_existing_results(RESULTS_FILE)
    pending = [(img, lbl) for img, lbl in pairs if img.name not in existing]
    log.info("Already done: %d  |  Pending: %d", len(existing), len(pending))

    results = list(existing.values())
    failed = 0

    with RESULTS_FILE.open("a") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_image, client, img, lbl): img
                for img, lbl in pending
            }
            for i, future in enumerate(as_completed(futures), 1):
                img_path = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    log.error("Unhandled error on %s: %s", img_path.name, e)
                    result = None

                if result:
                    fh.write(json.dumps(result) + "\n")
                    fh.flush()
                    results.append(result)
                    log.info(
                        "[%d/%d] %s → %s (conf=%.2f) | gt=%s",
                        i, len(pending),
                        img_path.name,
                        result["gpt4o_defect_type"],
                        result["gpt4o_confidence"],
                        result["ground_truth_classes"],
                    )
                else:
                    failed += 1
                    log.warning("[%d/%d] FAILED: %s", i, len(pending), img_path.name)


    log.info("Done. Processed=%d  Failed=%d  Total=%d", len(results), failed, len(results) + failed)

    # Agreement analysis
    log.info("Computing agreement vs YOLO ground truth...")
    agreement = compute_agreement(results)
    log.info(
        "Agreement rate: %.1f%%  FN: %d  Wrong type: %d",
        agreement["overall_agreement_rate"] * 100,
        agreement["false_negative_count"],
        agreement["wrong_type_count"],
    )

    # RCA library
    log.info("Building RCA hypothesis library...")
    rca_library = build_rca_library(results)

    # Save report
    report = {
        "meta": {
            "images_processed": len(results),
            "images_failed": failed,
            "dataset": str(args.dataset_dir),
            "model": "gpt-4o",
            "deployment": "gpt-4o",
            "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        },
        "agreement": agreement,
        "rca_library_summary": {
            k: {"count": v["observation_count"], "top_cause": v["top_causes"][0] if v["top_causes"] else None}
            for k, v in rca_library.items()
        },
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    RCA_LIBRARY_FILE.write_text(json.dumps(rca_library, indent=2))

    log.info("Results: %s", RESULTS_FILE)
    log.info("Report:  %s", REPORT_FILE)
    log.info("RCA lib: %s", RCA_LIBRARY_FILE)
    log.info("")
    # Token cost summary
    # gpt-4o pricing: $2.50/1M prompt tokens, $10.00/1M completion tokens (as of 2024)
    prompt_tok, completion_tok = get_token_totals()
    prompt_cost = prompt_tok / 1_000_000 * 2.50
    completion_cost = completion_tok / 1_000_000 * 10.00
    total_cost = prompt_cost + completion_cost

    report["token_usage"] = {
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "total_tokens": prompt_tok + completion_tok,
        "estimated_cost_usd": round(total_cost, 4),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))

    log.info("=== SUMMARY ===")
    log.info("Overall agreement with YOLO labels: %.1f%%", agreement["overall_agreement_rate"] * 100)
    log.info("Per-class breakdown:")
    for cls, stats in sorted(agreement["per_class_agreement"].items()):
        log.info(
            "  %-20s  agree=%.0f%%  miss=%.0f%%  wrong_type=%.0f%%  (n=%d)",
            cls,
            stats["agreement_rate"] * 100,
            stats["miss_rate"] * 100,
            stats["wrong_type_rate"] * 100,
            stats["total"],
        )

    log.info("")
    log.info("=== TOP ROOT CAUSES (all defect types) ===")
    all_causes: dict[str, int] = defaultdict(int)
    all_actions: dict[str, int] = defaultdict(int)
    for data in rca_library.values():
        for cause, count in data["top_causes"]:
            all_causes[cause] += count
        for action, count in data["top_actions"]:
            all_actions[action] += count
    for cause, count in sorted(all_causes.items(), key=lambda x: -x[1])[:10]:
        log.info("  %4dx  %s", count, cause)

    log.info("")
    log.info("=== TOP RECOMMENDED ACTIONS ===")
    for action, count in sorted(all_actions.items(), key=lambda x: -x[1])[:10]:
        log.info("  %4dx  %s", count, action)

    log.info("")
    log.info("=== TOKEN USAGE ===")
    log.info("  Prompt tokens:     %d", prompt_tok)
    log.info("  Completion tokens: %d", completion_tok)
    log.info("  Total tokens:      %d", prompt_tok + completion_tok)
    log.info("  Estimated cost:    $%.4f", total_cost)
    log.info("")
    log.info("RCA library: %d defect types → %s", len(rca_library), RCA_LIBRARY_FILE)


if __name__ == "__main__":
    main()
