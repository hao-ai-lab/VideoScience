from typing import Dict, Any

from backend.api_providers import (
    ReplicateVideoAPI,
    SoraVideoAPI,
    GeminiVeoAPI,
    WanDashScopeVideoAPI,
    KlingVideoAPI,
)

import os, sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

def generate_video(
    provider: str,
    model: str,
    prompt: str,
    n_seconds: int = 5,
    width: int = 720,
    height: int = 720,
    output_path: str = "output.mp4",
    timeout_s: int = 1200,
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    extra = extra or {}
    p = provider.lower()

    if p == "replicate":
        return ReplicateVideoAPI().generate(
            model=model, prompt=prompt, n_seconds=n_seconds,
            width=width, height=height, output_path=output_path,
            timeout_s=timeout_s, extra=extra,
        )

    elif p in ("sora-openai", "sora"):
        return SoraVideoAPI(backend="openai").generate(
            model=model, prompt=prompt, n_seconds=n_seconds,
            width=width, height=height, output_path=output_path,
            timeout_s=timeout_s, extra=extra,
        )

    elif p == "sora-azure":
        return SoraVideoAPI(backend="azure").generate(
            model=model, prompt=prompt, n_seconds=n_seconds,
            width=width, height=height, output_path=output_path,
            timeout_s=timeout_s, extra=extra,
        )

    elif p in ("gemini", "veo", "google-veo", "veo-gemini"):
        return GeminiVeoAPI().generate(
            model=model, prompt=prompt, n_seconds=n_seconds,
            width=width, height=height, output_path=output_path,
            timeout_s=timeout_s, extra=extra,
        )
        
    elif provider.lower() in ("wan", "wan-dashscope", "dashscope-wan"):
        return WanDashScopeVideoAPI().generate(
            model=model, prompt=prompt, n_seconds=n_seconds,
            width=width, height=height, output_path=output_path,
            timeout_s=timeout_s, extra=extra,
        )


    elif provider.lower() == "kling":
        return KlingVideoAPI().generate(
            model=model, prompt=prompt, n_seconds=n_seconds,
            width=width, height=height, output_path=output_path,
            timeout_s=timeout_s, extra=extra,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    raise ValueError(f"Unknown provider: {provider}")
