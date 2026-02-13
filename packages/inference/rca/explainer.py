"""
IntelFactor.ai — SLM/VLM Explainer (RCA Layer 3)
Wraps any LanguageProvider to produce bilingual RCA explanations.

Takes structured inputs from Layer 1 (anomaly) and Layer 2 (correlation),
formats a prompt, and generates natural-language root cause analysis.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from packages.inference.providers.base import LanguageProvider
from packages.inference.schemas import (
    AnomalyAlert,
    ProcessCorrelation,
    RCAExplanation,
)

logger = logging.getLogger(__name__)


RCA_PROMPT_TEMPLATE = """Analyze the following manufacturing quality anomaly and provide root cause analysis.

## Anomaly Summary
- Station: {station_id}
- Defect type: {defect_type}
- Current rate: {current_rate} events per {window_hours}h window
- Baseline rate: {baseline_rate} events per {window_hours}h window
- Z-score: {z_score} (deviation from baseline)

## Process Parameter Correlations
{correlations_text}

## SOP Context
{sop_context}

## Defect Taxonomy Reference
{taxonomy_text}

Provide your analysis as JSON with these fields:
- explanation_zh: Root cause analysis in Chinese (2-3 sentences, specific)
- explanation_en: Root cause analysis in English (2-3 sentences, specific)
- confidence: Your confidence in this analysis (0.0-1.0)
- recommended_action_zh: Recommended corrective action in Chinese
- recommended_action_en: Recommended corrective action in English
- sop_reference: Relevant SOP section number

Be specific. Cite exact parameter values. Do not be vague."""


class RCAExplainer:
    """
    Generates bilingual root cause explanations from structured anomaly data.

    Wraps any LanguageProvider (llama.cpp on Jetson, vLLM on server).
    The same prompt template works regardless of backend.
    """

    def __init__(
        self,
        language_provider: LanguageProvider,
        sop_context: dict[str, Any] | None = None,
        defect_taxonomy: dict[str, Any] | None = None,
    ):
        self.provider = language_provider
        self.sop_context = sop_context or {}
        self.defect_taxonomy = defect_taxonomy or {}

    def explain(
        self,
        alert: AnomalyAlert,
        correlations: list[ProcessCorrelation],
    ) -> RCAExplanation:
        """
        Generate RCA explanation from anomaly alert and process correlations.

        Args:
            alert: Anomaly alert from Layer 1 (Accumulator).
            correlations: Process correlations from Layer 2 (Correlator).

        Returns:
            RCAExplanation with bilingual text and confidence.
        """
        prompt = self._build_prompt(alert, correlations)
        context = self._build_context(alert)

        try:
            result = self.provider.generate(prompt, context)
            logger.info(
                "RCA explanation generated: station=%s defect=%s confidence=%.2f latency=%.0fms",
                alert.station_id,
                alert.defect_type,
                result.confidence,
                result.generation_ms,
            )
            return result

        except Exception as e:
            logger.error("SLM generation failed: %s", e)
            # Degrade gracefully: return statistical-only explanation
            return self._fallback_explanation(alert, correlations)

    def _build_prompt(
        self,
        alert: AnomalyAlert,
        correlations: list[ProcessCorrelation],
    ) -> str:
        """Format the RCA prompt with structured data."""

        # Format correlations
        if correlations:
            corr_lines = []
            for c in correlations:
                corr_lines.append(
                    f"- {c.parameter_name}: current={c.current_value} "
                    f"(target={c.target_value} ±{c.tolerance}), "
                    f"drift={c.drift_pct}%, Pearson r={c.pearson_r}"
                )
            correlations_text = "\n".join(corr_lines)
        else:
            correlations_text = "No process parameter correlations found in this window."

        # Format SOP context
        sop_sections = self.sop_context.get("sections", {})
        sop_text = "No SOP context available."
        if sop_sections:
            sop_lines = [f"- {k}: {v}" for k, v in sop_sections.items()]
            sop_text = "\n".join(sop_lines[:10])  # limit context length

        # Format defect taxonomy
        taxonomy = self.defect_taxonomy.get(alert.defect_type, {})
        if taxonomy:
            taxonomy_text = (
                f"Defect: {alert.defect_type}\n"
                f"Description: {taxonomy.get('description', 'N/A')}\n"
                f"Common causes: {taxonomy.get('common_causes', 'N/A')}\n"
                f"Severity range: {taxonomy.get('severity_range', 'N/A')}"
            )
        else:
            taxonomy_text = f"Defect type: {alert.defect_type} (no taxonomy entry)"

        return RCA_PROMPT_TEMPLATE.format(
            station_id=alert.station_id,
            defect_type=alert.defect_type,
            current_rate=alert.current_rate,
            baseline_rate=alert.baseline_rate,
            z_score=alert.z_score,
            window_hours=alert.window_hours,
            correlations_text=correlations_text,
            sop_context=sop_text,
            taxonomy_text=taxonomy_text,
        )

    def _build_context(self, alert: AnomalyAlert) -> dict[str, Any]:
        """Build context dict passed alongside the prompt."""
        return {
            "sop_criteria": json.dumps(self.sop_context, ensure_ascii=False, default=str),
            "defect_taxonomy": json.dumps(self.defect_taxonomy, ensure_ascii=False, default=str),
            "process_parameters": "",  # already included in prompt
        }

    def _fallback_explanation(
        self,
        alert: AnomalyAlert,
        correlations: list[ProcessCorrelation],
    ) -> RCAExplanation:
        """
        Statistical-only fallback when SLM is unavailable.
        Layer 1+2 data without natural language generation.
        """
        if correlations:
            top = correlations[0]
            zh = (
                f"统计分析：{alert.station_id}站{alert.defect_type}缺陷率"
                f"异常（z={alert.z_score}）。"
                f"关联参数：{top.parameter_name}偏离{top.drift_pct}%"
                f"（当前{top.current_value}，目标{top.target_value}±{top.tolerance}）。"
            )
            en = (
                f"Statistical: {alert.defect_type} rate anomaly at {alert.station_id} "
                f"(z={alert.z_score}). "
                f"Correlated: {top.parameter_name} drifted {top.drift_pct}% "
                f"(current={top.current_value}, target={top.target_value}±{top.tolerance})."
            )
            conf = min(abs(top.pearson_r), 0.7)
        else:
            zh = (
                f"统计分析：{alert.station_id}站{alert.defect_type}缺陷率"
                f"异常（z={alert.z_score}），未找到关联工艺参数。"
            )
            en = (
                f"Statistical: {alert.defect_type} rate anomaly at {alert.station_id} "
                f"(z={alert.z_score}). No correlated process parameters found."
            )
            conf = 0.3

        return RCAExplanation(
            explanation_zh=zh,
            explanation_en=en,
            confidence=conf,
            model_used="statistical_fallback",
            generation_ms=0.0,
        )
