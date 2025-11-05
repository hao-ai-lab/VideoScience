from typing import Dict, Any, Optional
from dataclasses import dataclass

import os, sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from judge.api_providers import (
    OpenAIVLMAPI,
    GeminiVLMAPI,
    AnthropicVLMAPI,
    ReplicateVLMAPI,
)

# ---------- Rubric configuration (prompting only) ----------
_RUBRIC_PROMPT = (
    "You are VLM-Judge evaluating the generation quality of a video produced by a video model. "
    "Focus on scientific understanding and reasoning. Judge along four rubrics, each scored 1–4:\n"
    "1) Immutability — each key element maintains the original experiment setup.\n"
    "2) Correct Dynamism — motions obey common-sense physical laws (solidity/non-interpenetration, inertia, gravity, "
    "   continuity of mass/energy, plausible collisions).\n"
    "3) Spatio-Temporal Continuity — coherence across frames (no flicker/teleportation/jitter; identity continuity, "
    "   smooth trajectories).\n"
    "4) Expected phenomenon — degree to which the described scientific phenomenon is present and correct.\n"
    " Rating Scheme: 1=absent/contradictory, 2=weak/partially wrong, 3=mostly correct, 4=clearly correct.\n"
    "Return strict JSON if possible."
)

# Kept for provider hints only; frontend owns scoring.
_DEFAULT_WEIGHTS = {
    "immutability": 0.2,
    "correct_dynamism": 0.2,
    "spatio_temporal_continuity": 0.2,
    "expected_phenomenon": 0.4,
}


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
    Dispatch to a VLM provider. Providers return unparsed `output_text` + minimal evidence.
    Parsing/normalization is performed in the frontend.
    """
    extra = dict(extra or {})
    extra.setdefault("rubric_prompt", _RUBRIC_PROMPT)
    extra.setdefault("rubric_weights", _DEFAULT_WEIGHTS)

    p = (provider or "").lower()
    if p in ("openai", "gpt", "gpt-4o", "gpt-4.1", "o3", "omni"):
        api = OpenAIVLMAPI()
    elif p in ("gemini", "google", "google-gemini"):
        api = GeminiVLMAPI()
    elif p in ("anthropic", "claude"):
        api = AnthropicVLMAPI()
    elif p in ("replicate", "llava"):
        api = ReplicateVLMAPI()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    result = api.analyze(
        model=model,
        video_path=video_path,
        phenomenon=phenomenon,
        gt_description=gt_description,
        ref_video_path=ref_video_path,
        max_frames=max_frames,
        fps=fps,
        timeout_s=timeout_s,
        extra=extra,
    )
    
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
