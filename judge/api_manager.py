from __future__ import annotations

from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from judge.api_providers import (
    OpenAIVLMAPI,
    GeminiVLMAPI,
    AnthropicVLMAPI,
)

RUBRIC_TEXT = (
    "You are VLM-Judge evaluating a generated science video.\n"
    "Score each rubric from 1–4 (1=absent/contradictory, 2=weak/partly wrong, 3=mostly correct, 4=clearly correct):\n"
    "a) prompt_consistency — follows instructions: correct setup and correct experiment execution.\n"
    "b) expected_phenomenon — expected physical/chemical outcome is present and correct.\n"
    "c) immutability — objects remain intact/unchanged unless changes are explicitly expected.\n"
    "d) dynamism — other physical laws are obeyed.\n"
    "e) coherence — natural transitions across frames; no flicker/teleport/identity swap.\n"
)

SCHEMA_HINT = (
    "Return JSON with fields:\n"
    '{ "scores": {\n'
    '    "prompt_consistency":1-4,\n'
    '    "expected_phenomenon":1-4,\n'
    '    "immutability":1-4,\n'
    '    "dynamism":1-4,\n'
    '    "coherence":1-4\n'
    '  },\n'
    '  "explanations": {"summary": string, "issues": [string]},\n'
    '  "evidence": {"candidate": [{"t":"0.0s","observation":""}],'
    '               "reference": [{"t":"0.0s","observation":""}]}\n'
    '}\n'
)

def _build_prompt(phenomenon: str, gt_description: str) -> str:
    return (
        RUBRIC_TEXT + "\n" +
        SCHEMA_HINT + "\n" +
        f"Ground-truth phenomenon: {phenomenon}\n\n" +
        "Ground-truth description (authoritative):\n" +
        gt_description.strip()
    )

def judge_experiment(
    provider: str,
    model: str,
    video_path: str,
    phenomenon: str,
    gt_description: str,
    ref_video_path: Optional[str] = None,
    *,
    max_frames: int = 24,
    fps: Optional[float] = None,
    timeout_s: int = 600,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Dispatch to a VLM provider. Providers return unparsed `output_text` and minimal evidence.
    The frontend is responsible for parsing/normalization/scoring.
    """
    extra = dict(extra or {})
    extra["judge_prompt"] = _build_prompt(phenomenon, gt_description)

    p = (provider or "").lower()
    if p in ("openai", "gpt", "gpt-4o", "gpt-4.1", "o3", "omni", "gpt-5", "gpt-5-pro"):
        api = OpenAIVLMAPI()
    elif p in ("gemini", "google", "google-gemini", "gemini-2.5"):
        api = GeminiVLMAPI()
    elif p in ("anthropic", "claude"):
        api = AnthropicVLMAPI()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    result = api.analyze(
        model=model,
        video_path=video_path,
        phenomenon=phenomenon,
        ref_video_path=ref_video_path,
        max_frames=max_frames,
        fps=fps,
        timeout_s=timeout_s,
        extra=extra,
    )

    # DEBUGGING PRINT
    print("result:")
    print(result.get("output_text", ""))

    return {
        "provider": result.get("provider", provider),
        "model": result.get("model", model),
        "output_text": result.get("output_text", ""),
        "output_mime": result.get("output_mime", "text/plain"),
        "evidence": result.get("evidence", {}),
        "raw": result.get("raw", {}),
    }
