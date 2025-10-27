from typing import Dict, Any
from backend.api_providers import ReplicateVideoAPI, SoraVideoAPI

def generate_video(
    provider: str,
    model: str,
    prompt: str,
    n_seconds: int = 5,
    width: int = 720,
    height: int = 720,
    output_path: str = "output.mp4",
    timeout_s: int = 600,
    extra: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Dispatch to the right provider. Returns a dict with keys like:
      - "video_path": str (if file saved)
      - "video_urls": list[str] (if provider returns URLs)
      - "job_id": str (if applicable)
      - "raw": Any (provider raw response)
    """
    extra = extra or {}

    if provider.lower() == "replicate":
        return ReplicateVideoAPI().generate(
            model=model,
            prompt=prompt,
            n_seconds=n_seconds,
            width=width,
            height=height,
            output_path=output_path,
            timeout_s=timeout_s,
            extra=extra,
        )

    elif provider.lower() in ("sora-azure", "sora"):
        # Azure OpenAI Sora REST API
        return SoraVideoAPI(backend="azure").generate(
            model=model,  # typically "sora" or "sora-2"
            prompt=prompt,
            n_seconds=n_seconds,
            width=width,
            height=height,
            output_path=output_path,
            timeout_s=timeout_s,
            extra=extra,
        )

    elif provider.lower() == "sora-openai":
        # (Optional) If/when you have direct OpenAI Videos API access,
        # flip backend here to "openai". Ensure base URL + auth in api_providers.py
        return SoraVideoAPI(backend="openai").generate(
            model=model,
            prompt=prompt,
            n_seconds=n_seconds,
            width=width,
            height=height,
            output_path=output_path,
            timeout_s=timeout_s,
            extra=extra,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")
