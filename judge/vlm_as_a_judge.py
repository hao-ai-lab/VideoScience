from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
import sys
from typing import Any, Dict, Tuple

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from judge.api_manager import judge_experiment

# ---------- Scoring config (frontend owns parsing + scoring) ----------
# All categories are 1–4. Expected phenomenon keeps higher weight.
WEIGHTS = {
    "immutability": 0.2,
    "correct_dynamism": 0.2,
    "spatio_temporal_continuity": 0.2,
    "expected_phenomenon": 0.4,
}

REPORT_MD = """# VLM Judge Report

**When:** {when}

**Provider/Model:** {provider}/{model}

**Phenomenon:** {phenomenon}

**Overall:** {overall:.1f} / 4

{rubric_section}## Summary
{summary}

## Notable issues
{issues}

## Evidence (timestamps)
- Candidate: {cand_frames}
{ref_section}
"""

# ---------- Parsing helpers ----------
def _try_json_loads(blob: str) -> Dict[str, Any]:
    try:
        return json.loads(blob)
    except Exception:
        return {}

def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try, in order:
    1) direct JSON
    2) fenced ```json ... ```
    3) fenced ``` ... ```
    4) longest {...} substring that parses
    """
    obj = _try_json_loads(text)
    if obj:
        return obj

    for m in re.finditer(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        obj = _try_json_loads(m.group(1))
        if obj:
            return obj

    for m in re.finditer(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        obj = _try_json_loads(m.group(1))
        if obj:
            return obj

    candidates = list(re.finditer(r"\{.*\}", text, flags=re.DOTALL))
    candidates.sort(key=lambda m: len(m.group(0)), reverse=True)
    for m in candidates:
        obj = _try_json_loads(m.group(0))
        if obj:
            return obj

    return {}

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _to_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default

def _normalize_1to4(v: Any) -> float:
    """
    Robustly map provider values to 1–4.
    - If already in [1,4], clamp and return.
    - If looks like 0–100, map 0→1 and 100→4.
    - Missing → 1.0 by default (worst).
    """
    f = _to_float(v)
    if f is None:
        return 1.0
    if f > 4.0 or f < 1.0:
        # Assume a 0–100 style score
        f = _clamp(f, 0.0, 100.0)
        return 1.0 + 3.0 * (f / 100.0)
    return _clamp(f, 1.0, 4.0)

def _compute_overall_1to4(rubric: Dict[str, Any], weights: Dict[str, float]) -> float:
    im = _normalize_1to4(rubric.get("immutability"))
    cd = _normalize_1to4(rubric.get("correct_dynamism"))
    stc = _normalize_1to4(rubric.get("spatio_temporal_continuity"))
    ep = _normalize_1to4(rubric.get("expected_phenomenon"))
    return (
        weights.get("immutability", 0.25) * im
        + weights.get("correct_dynamism", 0.25) * cd
        + weights.get("spatio_temporal_continuity", 0.25) * stc
        + weights.get("expected_phenomenon", 0.25) * ep
    )

def _parse_output_text(output_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (rubric_scores_1to4, explanations)
    rubric_scores_1to4 keys (all 1–4):
        - immutability
        - correct_dynamism
        - spatio_temporal_continuity
        - expected_phenomenon
    explanations: {"summary": str, "issues": [str]}
    """
    data = _extract_json_from_text(output_text)

    scores = {}
    explanations = {}

    if isinstance(data, dict):
        scores = data.get("scores", {}) or data.get("rubric", {}) or {}
        explanations = data.get("explanations", {}) or {}

    # Tolerate case-variants
    raw = {
        "immutability": scores.get("immutability") or scores.get("Immutability"),
        "correct_dynamism": scores.get("correct_dynamism") or scores.get("Correct Dynamism"),
        "spatio_temporal_continuity": scores.get("spatio_temporal_continuity") or scores.get("Spatio-Temporal Continuity"),
        "expected_phenomenon": scores.get("expected_phenomenon") or scores.get("Expected phenomenon"),
    }

    # Normalize to 1–4
    norm = {k: _normalize_1to4(v) for k, v in raw.items()}

    # Explanations normalization
    if not isinstance(explanations, dict):
        explanations = {}
    summary = explanations.get("summary", "")
    issues = explanations.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []

    return norm, {"summary": str(summary), "issues": [str(x) for x in issues]}

def _format_rubric_section(r: dict, overall: float) -> str:
    if not r:
        return ""
    lines = [
        "## Rubric scores (1–4)",
        f"- Immutability (each key element maintains original experiment setup): {float(r.get('immutability', 1.0)):.1f} / 4",
        f"- Correct Dynamism (solidity, other common sense physical laws): {float(r.get('correct_dynamism', 1.0)):.1f} / 4",
        f"- Spatio-Temporal Continuity (coherence across video frames): {float(r.get('spatio_temporal_continuity', 1.0)):.1f} / 4",
        f"- Expected phenomenon: {float(r.get('expected_phenomenon', 1.0)):.1f} / 4",
        f"- Overall (rubric-weighted): {overall:.1f} / 4",
    ]
    return "\n".join(lines) + "\n\n"


def main():
    ap = argparse.ArgumentParser(description="Judge science experiment video quality with a VLM.")
    ap.add_argument("--provider", required=True, help="openai | gemini | anthropic | replicate")
    ap.add_argument("--model", required=True, help="Provider-specific model id (e.g., gpt-4o, gemini-2.5-flash, claude-3-5-sonnet, yorickvp/llava-13b)")
    ap.add_argument("--video", required=True, help="Path to candidate video (or image)")
    ap.add_argument("--phenomenon", required=True, help="Ground-truth phenomenon name")
    ap.add_argument("--description", required=True, help="Authoritative description of expected behavior")
    ap.add_argument("--ref_video", default=None, help="Optional reference ground-truth video path")
    ap.add_argument("--max_frames", type=int, default=24)
    ap.add_argument("--fps", type=float, default=None, help="Optional sampling fps override")
    ap.add_argument("--timeout_s", type=int, default=900)
    ap.add_argument("--json_out", default=None, help="Where to write full JSON result")
    ap.add_argument("--md_out", default=None, help="Where to write a Markdown report")
    args = ap.parse_args()

    # Call provider via manager (returns raw output_text + evidence)
    res = judge_experiment(
        provider=args.provider,
        model=args.model,
        video_path=args.video,
        phenomenon=args.phenomenon,
        gt_description=args.description,
        ref_video_path=args.ref_video,
        max_frames=args.max_frames,
        fps=args.fps,
        timeout_s=args.timeout_s,
        extra={},  # rubric prompt/weights used only for prompting
    )

    output_text = res.get("output_text", "") or ""
    evidence = res.get("evidence", {}) or {}

    # Parse model output
    rubric_scores, explanations = _parse_output_text(output_text)

    # Compute overall on 1–4 scale
    overall = _compute_overall_1to4(rubric_scores, WEIGHTS)

    # Build normalized JSON result for optional export
    full_out = {
        "provider": res.get("provider"),
        "model": res.get("model"),
        "scores": {
            "overall": overall,  # 1–4
        },
        "explanations": {
            "summary": explanations.get("summary", ""),
            "issues": explanations.get("issues", []),
        },
        "evidence": {
            "candidate_frames": evidence.get("candidate_frames", []),
            "reference_frames": evidence.get("reference_frames", []),
        },
        "rubric": {
            **rubric_scores,           # all 1–4
            "overall_weighted": overall,
            "weights": dict(WEIGHTS),
        },
        "output_text": output_text,     # keep raw model text for debugging
        "raw": res.get("raw", {}),
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(full_out, indent=2))
        print(f"Wrote JSON to: {args.json_out}")

    # Render Markdown report
    md = REPORT_MD.format(
        when=datetime.utcnow().isoformat() + "Z",
        provider=res.get("provider"),
        model=res.get("model"),
        phenomenon=args.phenomenon,
        overall=overall,
        rubric_section=_format_rubric_section(rubric_scores, overall),
        summary=explanations.get("summary", "") or "(no summary parsed)",
        issues="\n".join(f"- {s}" for s in (explanations.get("issues") or [])) or "- (none reported)",
        cand_frames=", ".join(evidence.get("candidate_frames", [])) or "(frames hidden)",
        ref_section=("- Reference: " + ", ".join(evidence.get("reference_frames", []))) if evidence.get("reference_frames") else "",
    )

    if args.md_out:
        Path(args.md_out).write_text(md)
        print(f"Wrote report -> {args.md_out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
