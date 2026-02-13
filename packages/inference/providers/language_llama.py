"""
IntelFactor.ai — llama.cpp Language Provider
Runs quantized SLMs (Qwen-2.5, Phi-3) on Jetson via llama.cpp.
Designed for the 2-4 second RCA generation budget on station nodes.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from packages.inference.providers.base import LanguageProvider
from packages.inference.schemas import ModelSpec, RCAExplanation

logger = logging.getLogger(__name__)

# System prompt: bilingual RCA generation with SOP context
SYSTEM_PROMPT = """You are an expert manufacturing quality engineer at a metal products factory.
You analyze defect patterns and process parameter correlations to identify root causes.

RULES:
1. Always provide explanations in BOTH Chinese (primary) and English.
2. Link recommendations to specific SOP sections when available.
3. Include confidence levels for each hypothesis.
4. Be specific: cite exact parameter values, not vague descriptions.
5. Keep explanations concise — operators read these on a factory floor screen.

OUTPUT FORMAT (JSON):
{
  "explanation_zh": "中文根因分析...",
  "explanation_en": "English root cause analysis...",
  "confidence": 0.85,
  "recommended_action_zh": "中文建议...",
  "recommended_action_en": "English recommendation...",
  "sop_reference": "SOP 4.2.3"
}"""


class LlamaCppLanguageProvider(LanguageProvider):
    """
    llama.cpp backend for local SLM inference on Jetson.

    Supports:
    - Qwen-2.5 3B INT4 (default for Orin Nano/NX)
    - Phi-3-mini INT4 (fallback)
    - Qwen-2.5 7B-20B (AGX Orin / Thor with more VRAM)

    Uses llama-cpp-python bindings with CUDA acceleration.
    """

    def __init__(self, model_spec: ModelSpec, config: dict[str, Any] | None = None):
        super().__init__(model_spec, config)
        self.llm = None
        self.n_gpu_layers: int = config.get("n_gpu_layers", -1) if config else -1  # -1 = all on GPU
        self.n_ctx: int = model_spec.context_window or 4096
        self.temperature: float = config.get("temperature", 0.3) if config else 0.3
        self.system_prompt: str = config.get("system_prompt", SYSTEM_PROMPT) if config else SYSTEM_PROMPT

    def load(self) -> None:
        model_path = Path(self.model_spec.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                # Jetson: use CUDA backend
                n_threads=4,  # ARM cores on Jetson
            )
            self._loaded = True
            logger.info(
                "llama.cpp model loaded: %s (ctx=%d, gpu_layers=%d)",
                model_path.name,
                self.n_ctx,
                self.n_gpu_layers,
            )

        except ImportError:
            logger.warning("llama-cpp-python not available — using stub mode")
            self._loaded = True
            self._stub_mode = True

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> RCAExplanation:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.perf_counter()

        # Build full prompt with system context
        full_prompt = self._build_prompt(prompt, context)

        if getattr(self, "_stub_mode", False):
            raw_output = self._stub_generate(prompt)
        else:
            response = self.llm(
                full_prompt,
                max_tokens=self.model_spec.max_tokens,
                temperature=self.temperature,
                stop=["```", "\n\n\n"],
            )
            raw_output = response["choices"][0]["text"].strip()

        generation_ms = (time.perf_counter() - t0) * 1000

        # Parse structured output
        return self._parse_output(raw_output, generation_ms)

    def _build_prompt(self, user_prompt: str, context: dict[str, Any] | None = None) -> str:
        """Build chat-formatted prompt with system context."""
        parts = [
            f"<|im_start|>system\n{self.system_prompt}\n<|im_end|>",
        ]

        # Inject SOP context if available
        if context:
            sop_context = context.get("sop_criteria", "")
            defect_taxonomy = context.get("defect_taxonomy", "")
            process_params = context.get("process_parameters", "")

            if sop_context:
                parts.append(f"<|im_start|>system\nRelevant SOP criteria:\n{sop_context}\n<|im_end|>")
            if defect_taxonomy:
                parts.append(f"<|im_start|>system\nDefect taxonomy:\n{defect_taxonomy}\n<|im_end|>")
            if process_params:
                parts.append(f"<|im_start|>system\nProcess parameters:\n{process_params}\n<|im_end|>")

        parts.append(f"<|im_start|>user\n{user_prompt}\n<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def _parse_output(self, raw: str, generation_ms: float) -> RCAExplanation:
        """Parse SLM output into structured RCAExplanation."""
        # Try to extract JSON from output
        try:
            # Find JSON block in output
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(raw[json_start:json_end])
                return RCAExplanation(
                    explanation_zh=parsed.get("explanation_zh", ""),
                    explanation_en=parsed.get("explanation_en", ""),
                    confidence=float(parsed.get("confidence", 0.5)),
                    model_used=self.model_spec.model_name,
                    generation_ms=generation_ms,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: treat entire output as explanation
        return RCAExplanation(
            explanation_zh=raw,
            explanation_en=raw,
            confidence=0.3,  # low confidence for unparsed output
            model_used=self.model_spec.model_name,
            generation_ms=generation_ms,
        )

    def _stub_generate(self, prompt: str) -> str:
        """Stub response for development without GPU."""
        return json.dumps({
            "explanation_zh": "[STUB] 研磨轮转速偏离目标值，导致刮痕率上升",
            "explanation_en": "[STUB] Grinding wheel RPM drift from target causing elevated scratch rate",
            "confidence": 0.75,
            "recommended_action_zh": "根据SOP 4.2.3重新校准研磨轮",
            "recommended_action_en": "Recalibrate grinding wheel per SOP 4.2.3",
            "sop_reference": "SOP 4.2.3",
        })

    def unload(self) -> None:
        self.llm = None
        self._loaded = False
        logger.info("llama.cpp model unloaded")
