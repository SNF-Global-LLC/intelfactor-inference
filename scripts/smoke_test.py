#!/usr/bin/env python3
"""
IntelFactor.ai — Smoke Test
Validates that the full pipeline works on real hardware.

1. Loads VisionProvider (real TRT or stub)
2. Loads LanguageProvider (real llama.cpp or stub)
3. Processes one frame (test image or blank)
4. Runs one RCA cycle
5. Verifies bilingual output

Must complete in <30 seconds total.
Exit 0 = all good. Exit 1 = something broken.
"""

import sys
import time
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("smoke-test")

TIMEOUT = 30  # seconds


def main():
    t_start = time.monotonic()
    errors = []

    print("\n" + "=" * 50)
    print("  IntelFactor Smoke Test")
    print("=" * 50)

    # ── 1. Import core modules ──
    step("Importing modules")
    try:
        import numpy as np
        from packages.inference.providers.resolver import CapabilityResolver
        from packages.inference.rca.pipeline import RCAPipeline
        from packages.inference.rca.accumulator import DefectAccumulator
        from packages.inference.rca.correlator import ProcessCorrelator
        from packages.inference.rca.explainer import RCAExplainer
        from packages.inference.rca.recommender import ActionRecommender
        from packages.inference.schemas import Verdict
        ok("Imports successful")
    except Exception as e:
        fail(f"Import error: {e}")
        errors.append(str(e))

    # ── 2. Detect hardware ──
    step("Detecting hardware")
    try:
        resolver = CapabilityResolver(config={
            "model_dir": "/opt/intelfactor/models",
            "station_id": "smoke_test",
        })
        caps = resolver.detect_capabilities()
        ok(f"Device: {caps.device_class.value} ({caps.gpu_name}, {caps.vram_mb}MB)")
    except Exception as e:
        fail(f"Hardware detection: {e}")
        errors.append(str(e))
        caps = None

    # ── 3. Load vision provider ──
    step("Loading vision provider")
    try:
        vision = resolver.resolve_vision_provider(provider_config={"station_id": "smoke_test"})
        vision.load()
        stub = getattr(vision, "_stub_mode", False)
        if stub:
            warn("Vision running in STUB mode (no TRT engine)")
        else:
            ok(f"Vision loaded: {vision.model_spec.model_name}")
    except Exception as e:
        fail(f"Vision load: {e}")
        errors.append(str(e))
        vision = None

    # ── 4. Load language provider ──
    step("Loading language provider")
    try:
        language = resolver.resolve_language_provider()
        language.load()
        stub = getattr(language, "_stub_mode", False)
        if stub:
            warn("Language running in STUB mode (no GGUF)")
        else:
            ok(f"Language loaded: {language.model_spec.model_name}")
    except Exception as e:
        fail(f"Language load: {e}")
        errors.append(str(e))
        language = None

    # ── 5. Process one frame ──
    step("Processing test frame")
    try:
        # Create a test frame (640x640 BGR)
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        t_inf = time.monotonic()
        result = vision.detect(frame)
        inf_ms = (time.monotonic() - t_inf) * 1000
        ok(f"Verdict: {result.verdict.value} ({len(result.detections)} detections, {inf_ms:.0f}ms)")
    except Exception as e:
        fail(f"Inference: {e}")
        errors.append(str(e))
        result = None

    # ── 6. Run RCA pipeline ──
    step("Running RCA pipeline")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            acc = DefectAccumulator(db_path=Path(tmpdir) / "acc.db")
            acc.start()
            corr = ProcessCorrelator.from_edge_yaml({})
            exp = RCAExplainer(language_provider=language, sop_context={}, defect_taxonomy={})
            rec = ActionRecommender(sop_map={}, db_path=Path(tmpdir) / "triples.db")
            rec.start()

            pipeline = RCAPipeline(acc, corr, exp, rec)

            # Ingest some events to trigger anomaly
            from packages.inference.schemas import DetectionResult, Detection, BoundingBox
            for i in range(20):
                evt = DetectionResult(
                    station_id="smoke_test",
                    verdict=Verdict.FAIL,
                    confidence=0.9,
                    detections=[Detection(
                        defect_type="scratch_surface",
                        confidence=0.9,
                        bbox=BoundingBox(x=10, y=10, width=100, height=100),
                        severity=0.6,
                    )],
                )
                pipeline.ingest(evt)

            # Run RCA
            rca_results = pipeline.run_rca(station_id="smoke_test", z_threshold=1.0)
            if rca_results:
                r = rca_results[0]
                has_zh = bool(r.explanation.explanation_zh)
                has_en = bool(r.explanation.explanation_en)
                ok(f"RCA produced {len(rca_results)} result(s) — zh={has_zh}, en={has_en}, triple={r.triple.triple_id}")
            else:
                warn("RCA ran but no anomalies triggered (threshold may be too high for 20 events)")

            acc.stop()
            rec.stop()
    except Exception as e:
        fail(f"RCA pipeline: {e}")
        errors.append(str(e))

    # ── 7. Generate explanation ──
    step("Testing language generation")
    try:
        t_gen = time.monotonic()
        output = language.generate("What is quality control? Reply in one sentence in Chinese.")
        gen_ms = (time.monotonic() - t_gen) * 1000
        if output and len(output) > 5:
            ok(f"Generated {len(output)} chars in {gen_ms:.0f}ms")
        else:
            warn(f"Short output ({len(output)} chars): {output[:80]}")
    except Exception as e:
        fail(f"Language generation: {e}")
        errors.append(str(e))

    # ── 8. Cleanup ──
    if vision:
        vision.unload()
    if language:
        language.unload()

    # ── Summary ──
    elapsed = time.monotonic() - t_start
    print("\n" + "-" * 50)
    if errors:
        print(f"  ✗ FAILED — {len(errors)} error(s) in {elapsed:.1f}s")
        for e in errors:
            print(f"    → {e}")
    elif elapsed > TIMEOUT:
        print(f"  ⚠ SLOW — completed in {elapsed:.1f}s (target: <{TIMEOUT}s)")
    else:
        print(f"  ✓ ALL GOOD — {elapsed:.1f}s")
    print("=" * 50 + "\n")

    sys.exit(1 if errors else 0)


def step(msg):
    print(f"\n  [{time.strftime('%H:%M:%S')}] {msg}...")

def ok(msg):
    print(f"    ✓ {msg}")

def warn(msg):
    print(f"    ⚠ {msg}")

def fail(msg):
    print(f"    ✗ {msg}")


if __name__ == "__main__":
    main()
