"""
IntelFactor.ai — vLLM Language Provider
Runs larger SLMs/VLMs on NVIDIA server GPUs via vLLM.
Used as the "site hub" brain for cross-line RCA and REVIEW triage.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from packages.inference.providers.base import LanguageProvider
from packages.inference.providers.language_llama import SYSTEM_PROMPT
from packages.inference.schemas import ModelSpec, RCAExplanation

logger = logging.getLogger(__name__)


class VLLMLanguageProvider(LanguageProvider):
    """
    vLLM backend for server-class GPU inference.

    Supports:
    - Qwen-2.5 7B-72B on server GPUs
    - VLM models (Qwen2.5-VL) for image-aware REVIEW triage
    - High throughput for batch RCA across multiple lines

    Runs vLLM as an in-process engine or connects to a running vLLM server.
    """

    def __init__(self, model_spec: ModelSpec, config: dict[str, Any] | None = None):
        super().__init__(model_spec, config)
        self.engine = None
        self.server_url: str | None = config.get("vllm_server_url") if config else None
        self.temperature: float = config.get("temperature", 0.3) if config else 0.3
        self.system_prompt: str = config.get("system_prompt", SYSTEM_PROMPT) if config else SYSTEM_PROMPT
        self._use_server = self.server_url is not None

    def load(self) -> None:
        if self._use_server:
            # Connect to running vLLM server (e.g. on site hub)
            logger.info("vLLM connecting to server: %s", self.server_url)
            self._loaded = True
            return

        try:
            from vllm import LLM, SamplingParams  # noqa: F401

            self.engine = LLM(
                model=self.model_spec.model_path or self.model_spec.model_name,
                quantization=self._map_quantization(),
                gpu_memory_utilization=0.85,
                max_model_len=self.model_spec.context_window,
                trust_remote_code=True,
            )
            self._loaded = True
            logger.info("vLLM engine loaded: %s", self.model_spec.model_name)

        except ImportError:
            logger.warning("vLLM not available — using stub mode")
            self._loaded = True
            self._stub_mode = True

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> RCAExplanation:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.perf_counter()

        messages = self._build_messages(prompt, context)

        if getattr(self, "_stub_mode", False):
            raw_output = self._stub_generate()
        elif self._use_server:
            raw_output = self._server_generate(messages)
        else:
            raw_output = self._engine_generate(messages)

        generation_ms = (time.perf_counter() - t0) * 1000
        return self._parse_output(raw_output, generation_ms)

    def _build_messages(self, user_prompt: str, context: dict[str, Any] | None = None) -> list[dict]:
        """Build chat messages for vLLM."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            sop = context.get("sop_criteria", "")
            if sop:
                messages.append({"role": "system", "content": f"SOP criteria:\n{sop}"})

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _engine_generate(self, messages: list[dict]) -> str:
        """In-process vLLM generation."""
        from vllm import SamplingParams

        # Format as chat template
        prompt = self._messages_to_prompt(messages)

        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.model_spec.max_tokens,
            stop=["```", "\n\n\n"],
        )
        outputs = self.engine.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def _server_generate(self, messages: list[dict]) -> str:
        """Remote vLLM server generation via OpenAI-compatible API."""
        import requests

        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "model": self.model_spec.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.model_spec.max_tokens,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert chat messages to Qwen chat template."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _map_quantization(self) -> str | None:
        """Map our quantization labels to vLLM quantization methods."""
        q = self.model_spec.quantization.upper()
        mapping = {
            "INT4": "awq",     # or "gptq"
            "INT8": "squeezellm",
            "FP16": None,      # no quantization
            "FP8": "fp8",
        }
        return mapping.get(q)

    def _parse_output(self, raw: str, generation_ms: float) -> RCAExplanation:
        """Parse vLLM output into RCAExplanation."""
        try:
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

        return RCAExplanation(
            explanation_zh=raw,
            explanation_en=raw,
            confidence=0.3,
            model_used=self.model_spec.model_name,
            generation_ms=generation_ms,
        )

    def _stub_generate(self) -> str:
        return json.dumps({
            "explanation_zh": "[STUB/vLLM] 跨产线分析：3号和5号产线研磨参数漂移模式相似",
            "explanation_en": "[STUB/vLLM] Cross-line analysis: Lines 3 and 5 show similar grinding parameter drift patterns",
            "confidence": 0.82,
            "recommended_action_zh": "检查共用磨轮供应批次",
            "recommended_action_en": "Check shared grinding wheel supply batch",
            "sop_reference": "SOP 4.2.3",
        })

    def unload(self) -> None:
        self.engine = None
        self._loaded = False
        logger.info("vLLM engine unloaded")
