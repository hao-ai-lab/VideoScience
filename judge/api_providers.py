# Providers return *unparsed* model text as `output_text`, plus minimal evidence.
# All parsing/normalization happens in the frontend.

from __future__ import annotations

import os
import io
import json
import base64
import typing as t
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ----- optional deps -----
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

import requests

# Google Gemini SDK (pip install google-genai)
try:
    from google import genai as google_genai  # type: ignore
    from google.genai import types as google_types  # type: ignore
except Exception:  # pragma: no cover
    google_genai = None  # type: ignore
    google_types = None  # type: ignore

# Anthropic (pip install anthropic)
try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

# Replicate (pip install replicate)
try:
    import replicate  # type: ignore
except Exception:  # pragma: no cover
    replicate = None  # type: ignore


# -----------------------------
# Prompting helpers (schema hint is instructional only)
# -----------------------------
RUBRIC = (
    "You are a rigorous science-video judge evaluating the generation quality of a video produced by a video model. "
    "Focus on scientific understanding and reasoning with four rubrics, each scored 1–4:\n"
    "1) Immutability — each key element maintains the original experiment setup.\n"
    "2) Correct Dynamism — motions obey physical laws (solidity, inertia, gravity, continuity of mass/energy).\n"
    "3) Spatio-Temporal Continuity — coherence across frames (no flicker/teleportation/jitter).\n"
    "4) Expected phenomenon — presence & correctness of the described scientific phenomenon.\n"
    "Compare the candidate to the ground-truth description, and use the reference video if provided.\n"
)

def _schema_hint() -> str:
    # Ask the model to emit 1–4 for all categories.
    return (
        "Return JSON with fields: {\n"
        '  "scores": {"immutability":1-4, "correct_dynamism":1-4, '
        '"spatio_temporal_continuity":1-4, "expected_phenomenon":1-4},\n'
        '  "explanations": {"summary": string, "issues": [string]},\n'
        '  "evidence": {"candidate": [{"t": "0.0s", "observation": ""}], '
        '"reference": [{"t": "0.0s", "observation": ""}]}\n'
        "}\n"
    )

def _build_prompt(phenomenon: str, gt_description: str) -> str:
    return (
        f"Ground-truth phenomenon: {phenomenon}\n\n"
        f"Ground-truth description (authoritative):\n{gt_description}\n\n"
        "Judge the candidate vs the ground-truth. If a reference video is provided, use it to verify temporal order."
    )


# -----------------------------
# Frame extraction
# -----------------------------
def _require_pillow():
    if Image is None:
        raise RuntimeError("Pillow is required. Install via `pip install Pillow`.")

def _image_to_b64(img: "Image.Image", jpeg_quality: int = 75) -> str:
    _require_pillow()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _extract_frames_cv2(path: str, max_frames: int = 24, fps: float | None = None) -> list[tuple[float, "Image.Image"]]:
    if cv2 is None:
        return []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / max(native_fps, 1e-6)

    if fps and fps > 0:
        step = 1.0 / fps
        times = [i * step for i in range(int(duration // step) + 1)]
    else:
        max_frames = max(1, int(max_frames))
        times = [i * duration / max_frames for i in range(max_frames)]

    frames: list[tuple[float, "Image.Image"]] = []
    for t_s in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        ok, fr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        from PIL import Image as _PIL
        frames.append((t_s, _PIL.fromarray(rgb)))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

def _extract_frames_ffmpeg(path: str, max_frames: int = 24) -> list[tuple[float, "Image.Image"]]:
    if Image is None:
        return []
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True
        )
        duration = float(probe.stdout.strip())
    except Exception:
        duration = 5.0
    max_frames = max(1, int(max_frames))
    times = [i * duration / max_frames for i in range(max_frames)]

    out_frames: list[tuple[float, "Image.Image"]] = []
    for t_s in times:
        try:
            out = subprocess.run(
                ["ffmpeg", "-ss", str(t_s), "-i", path, "-frames:v", "1",
                 "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, check=True)
            img = Image.open(io.BytesIO(out.stdout)).convert("RGB")  # type: ignore[arg-type]
            out_frames.append((t_s, img))
        except Exception:
            continue
    return out_frames

def extract_frames(path: str, max_frames: int = 24, fps: float | None = None) -> list[tuple[float, "Image.Image"]]:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        _require_pillow()
        return [(0.0, Image.open(path).convert("RGB"))]  # type: ignore[arg-type]
    frames = _extract_frames_cv2(path, max_frames=max_frames, fps=fps)
    if frames:
        return frames
    return _extract_frames_ffmpeg(path, max_frames=max_frames)


# -----------------------------
# Provider base
# -----------------------------
class _BaseVLM:
    def _rubric_prompt(self, extra: dict, phenomenon: str, gt_description: str) -> str:
        base = extra.get("rubric_prompt", RUBRIC)
        return base + "\n\n" + _build_prompt(phenomenon, gt_description) + "\n\n" + _schema_hint()


# -----------------------------
# OpenAI (Responses API)
# -----------------------------
@dataclass
class OpenAIVLMAPI(_BaseVLM):
    base_url: str = os.environ.get("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")

    def _headers(self) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _extract_output_text(self, resp: dict) -> str:
        # 1) Prefer Responses API's aggregated "output_text" if present.
        txt = resp.get("output_text")
        if isinstance(txt, str) and txt.strip():
            return txt

        # 2) Responses API structure: output -> [ { content: [ {type: "output_text", text: "..."} ] } ]
        out_chunks: list[str] = []
        for item in (resp.get("output") or []):
            content = item.get("content") or []
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("text")
                        if isinstance(t, str):
                            out_chunks.append(t)
        if out_chunks:
            return "\n".join(out_chunks)

        # 3) Fallback: try top-level "content" (rare)
        content = resp.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    t = c.get("text")
                    if isinstance(t, str):
                        out_chunks.append(t)
        if out_chunks:
            return "\n".join(out_chunks)

        # 4) Last-resort: return the whole JSON string
        import json as _json
        return _json.dumps(resp)

    def analyze(
        self,
        model: str,
        video_path: str,
        phenomenon: str,
        gt_description: str,
        ref_video_path: str | None,
        max_frames: int,
        fps: float | None,
        timeout_s: int,
        extra: dict,
    ) -> dict:
        cand = extract_frames(video_path, max_frames=max_frames, fps=fps)
        ref = extract_frames(ref_video_path, max_frames=max_frames // 2, fps=fps) if ref_video_path else []

        content: list[dict] = [{"type": "input_text", "text": self._rubric_prompt(extra, phenomenon, gt_description)}]
        max_imgs = int(extra.get("max_images", 20))

        for ts, im in cand[:max_imgs]:
            b64 = _image_to_b64(im)
            content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
            content.append({"type": "input_text", "text": f"Candidate frame t={ts:.1f}s"})

        if ref:
            content.append({"type": "input_text", "text": "Reference video frames:"})
            for ts, im in ref[: max(1, max_imgs // 2)]:
                b64 = _image_to_b64(im)
                content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
                content.append({"type": "input_text", "text": f"Reference frame t={ts:.1f}s"})

        payload = {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(extra.get("max_output_tokens", 1200)),
        }

        url = f"{self.base_url}/v1/responses"
        r = requests.post(url, headers=self._headers(), json=payload, timeout=timeout_s)
        r.raise_for_status()
        resp = r.json()

        output_text = self._extract_output_text(resp)
        return {
            "provider": "openai",
            "model": model,
            "output_text": output_text,
            "output_mime": "application/json",
            "evidence": {
                "candidate_frames": [f"t={ts:.1f}s" for ts, _ in cand[:max_imgs]],
                "reference_frames": [f"t={ts:.1f}s" for ts, _ in (ref[: max(1, max_imgs // 2)] if ref else [])],
            },
            "raw": resp,
        }


# -----------------------------
# Google Gemini (native upload)
# -----------------------------
@dataclass
class GeminiVLMAPI(_BaseVLM):
    def _ensure_client(self):
        if google_genai is None:
            raise RuntimeError("google-genai not installed. Run: pip install google-genai")
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        return google_genai.Client()

    def analyze(self, model: str, video_path: str, phenomenon: str, gt_description: str,
                ref_video_path: str | None, max_frames: int, fps: float | None,
                timeout_s: int, extra: dict) -> dict:
        client = self._ensure_client()

        cand_file = client.files.upload(file=video_path)
        ref_file = client.files.upload(file=ref_video_path) if ref_video_path else None

        config = google_types.GenerateContentConfig(response_mime_type="application/json")
        parts: list = [self._rubric_prompt(extra, phenomenon, gt_description), cand_file]
        if ref_file:
            parts += ["Reference video follows.", ref_file]

        resp = client.models.generate_content(model=model, contents=parts, config=config)

        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            try:
                text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
            except Exception:
                text = json.dumps(getattr(resp, "to_dict", lambda: {})())
        return {
            "provider": "gemini",
            "model": model,
            "output_text": text or "",
            "output_mime": "application/json",
            "evidence": {"candidate_frames": ["uploaded:file"], "reference_frames": ["uploaded:file"] if ref_file else []},
            "raw": resp.to_dict() if hasattr(resp, "to_dict") else str(resp),
        }


# -----------------------------
# Anthropic Claude (base64 images)
# -----------------------------
@dataclass
class AnthropicVLMAPI(_BaseVLM):
    def _client(self):
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(api_key=key)

    def analyze(self, model: str, video_path: str, phenomenon: str, gt_description: str,
                ref_video_path: str | None, max_frames: int, fps: float | None,
                timeout_s: int, extra: dict) -> dict:
        client = self._client()
        cand = extract_frames(video_path, max_frames=max_frames, fps=fps)
        ref = extract_frames(ref_video_path, max_frames=max_frames // 2, fps=fps) if ref_video_path else []

        def _image_part(img: "Image.Image") -> dict:
            b64 = _image_to_b64(img)
            return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}

        content: list[dict] = [{"type": "text", "text": self._rubric_prompt(extra, phenomenon, gt_description)}]
        max_imgs = int(extra.get("max_images", 20))

        for ts, im in cand[:max_imgs]:
            content.append(_image_part(im))
            content.append({"type": "text", "text": f"Candidate frame t={ts:.1f}s"})
        if ref:
            content.append({"type": "text", "text": "Reference video frames:"})
            for ts, im in ref[: max(1, max_imgs // 2)]:
                content.append(_image_part(im))
                content.append({"type": "text", "text": f"Reference frame t={ts:.1f}s"})

        msg = client.messages.create(
            model=model,
            max_tokens=int(extra.get("max_output_tokens", 1200)),
            messages=[{"role": "user", "content": content}],
        )

        text = ""
        try:
            if getattr(msg, "content", None):
                text = msg.content[0].text  # type: ignore[index]
        except Exception:
            text = str(msg)

        return {
            "provider": "anthropic",
            "model": model,
            "output_text": text,
            "output_mime": "application/json",
            "evidence": {
                "candidate_frames": [f"t={ts:.1f}s" for ts, _ in cand[:max_imgs]],
                "reference_frames": [f"t={ts:.1f}s" for ts, _ in (ref[: max(1, max_imgs // 2)] if ref else [])],
            },
            "raw": msg.model_dump() if hasattr(msg, "model_dump") else str(msg),
        }


# -----------------------------
# Replicate (iterate frames; prompt per-frame)
# -----------------------------
@dataclass
class ReplicateVLMAPI(_BaseVLM):
    def analyze(self, model: str, video_path: str, phenomenon: str, gt_description: str,
                ref_video_path: str | None, max_frames: int, fps: float | None,
                timeout_s: int, extra: dict) -> dict:
        if replicate is None:
            raise RuntimeError("replicate package not installed. `pip install replicate`")
        cand = extract_frames(video_path, max_frames=max_frames, fps=fps)
        ref = extract_frames(ref_video_path, max_frames=max_frames // 2, fps=fps) if ref_video_path else []

        per_frame_outputs: list[dict] = []
        max_imgs = int(extra.get("max_images", 10))
        for ts, im in cand[:max_imgs]:
            buf = io.BytesIO()
            _require_pillow()
            im.save(buf, format="PNG")
            buf.seek(0)
            prompt = (
                self._rubric_prompt(extra, phenomenon, gt_description)
                + "\n\nThis is a single frame (content-only check). "
                  "Return brief JSON with keys (each 1–4): "
                  '{"immutability":1-4,"correct_dynamism":1-4,'
                  '"spatio_temporal_continuity":1-4,"expected_phenomenon":1-4,'
                  '"notes":""}.'
            )
            out = replicate.run(model, input={"image": buf.getvalue(), "prompt": prompt})
            raw_str = out if isinstance(out, str) else json.dumps(out, default=str)
            per_frame_outputs.append({"t": round(float(ts), 3), "raw": raw_str[:4000]})

        return {
            "provider": "replicate",
            "model": model,
            "output_text": json.dumps({"frame_analyses": per_frame_outputs}),
            "output_mime": "application/json",
            "evidence": {
                "candidate_frames": [f"t={ts:.1f}s" for ts, _ in cand[:max_imgs]],
                "reference_frames": [f"t={ts:.1f}s" for ts, _ in (ref[: max(1, max_imgs // 2)] if ref else [])],
            },
            "raw": {"per_frame": per_frame_outputs},
        }
