import os
import time
import typing as t
import requests
from dataclasses import dataclass
from urllib.parse import urlencode

import jwt

import replicate

from google import genai as google_genai
from google.genai import types as google_types

import time, json, base64, hmac, hashlib

# helper functions
def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

def kling_jwt(access_key: str, secret_key: str, ttl_s: int = 1800) -> str:
    # Header and payload must be EXACTLY HS256 + {iss, exp, nbf}
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"iss": access_key, "exp": now + ttl_s, "nbf": now - 5}
    h = _b64u(json.dumps(header, separators=(",", ":")).encode())
    p = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(secret_key.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64u(sig)}"

from google import genai as google_genai
from google.genai import types as google_types

@dataclass
class ReplicateVideoAPI:
    """
    Runs a video model on Replicate.

    Auth:
      - export REPLICATE_API_TOKEN="xxx" (bearer token)
        (The Python client automatically reads this env var.)

    Notes:
      - `model` can be "owner/model" or "owner/model:version".
      - `input` shape is model-specific (we pass prompt + optional extras).
      - Outputs are often file URLs or FileOutput objects.
      - Reference: "Run a model" + HTTP API docs.
    """

    def _save_url_to_file(self, url: str, dest: str) -> None:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)

    def generate(
        self,
        model: str,
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:
        if replicate is None:
            raise RuntimeError(
                "replicate package not installed. `pip install replicate`"
            )

        # Inputs: 
        # most video models accept 'prompt' and optionally duration/resolution.
        # many Replicate video models use keys like `prompt`, `duration`, `width`, `height`.
        # we pass extras through so model-specific params can be used.
        inp = {"prompt": prompt}
        
        # Common optional params:
        # use the names that many T2V cards expect
        # callers can override via --extra
        inp.setdefault("duration", n_seconds)
        inp.setdefault("width", width)
        inp.setdefault("height", height)

        # Merge extras
        if extra:
            inp.update(extra)

        # Example from docs: replicate.run("owner/model[:version]", input={...})
        output = replicate.run(model, input=inp)
        
        # many video models return a list of URLs (strings) or FileOutput objects.
        video_urls: t.List[str] = []
        
        # expect <class 'replicate.helpers.FileOutput'>
        url = output.url
        print("Result URL:", url)
        video_urls.append(url)

        saved_path = None
        if video_urls:
            # Save first URL to requested path. Caller can inspect `video_urls` if they want others.
            self._save_url_to_file(video_urls[0], output_path)
            saved_path = output_path

        return {
            "provider": "replicate",
            "model": model,
            "video_path": saved_path,
            "video_urls": video_urls,
            "raw": output,
        }

@dataclass
class SoraVideoAPI:
    """
    OpenAI Videos API (Sora) + optional Azure fallback.

    OpenAI (Sora 2 / Sora 2 Pro):
      - POST   /v1/videos                      (create; JSON OR multipart if uploading files)
      - GET    /v1/videos/{video_id}           (poll status)
      - GET    /v1/videos/{video_id}/content   (download; optional ?variant=thumbnail, etc.)
      Auth: Authorization: Bearer <OPENAI_API_KEY>
      Docs: Video generation guide & API reference. 
    backend: "openai" | "azure"
    """
    backend: str = "openai"

    def _cfg(self):
        if self.backend == "openai":
            base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            return {
                "base_url": base,
                # NOTE: do NOT hard-set Content-Type here; requests will set it:
                # - application/json when using json=...
                # - multipart/form-data with boundary when using files=...
                "headers": {"Authorization": f"Bearer {key}"},
                "create_path": "/v1/videos",
                "status_path_template": "/v1/videos/{video_id}",
                "download_path_template": "/v1/videos/{video_id}/content",
                "query": {},
                "format_size": lambda w, h: f"{w}x{h}",  # width x height
                "duration_key": "seconds",
            }

        elif self.backend == "azure":
            base = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
            if not base:
                raise RuntimeError("AZURE_OPENAI_ENDPOINT not set")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "preview")
            key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not key:
                raise RuntimeError("AZURE_OPENAI_API_KEY not set")
            return {
                "base_url": base,
                "headers": {"api-key": key, "Content-Type": "application/json"},
                "create_path": "/openai/v1/video/generations/jobs",
                "status_path_template": "/openai/v1/video/generations/jobs/{job_id}",
                "download_path_template": "/openai/v1/video/generations/{generation_id}/content/video",
                "query": {"api-version": api_version},
                "format_size": lambda w, h: (w, h),   # separate ints on Azure
                "duration_key": "n_seconds",
            }
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _join(self, base: str, path: str, query: t.Dict[str, t.Any]) -> str:
        return f"{base}{path}" + (f"?{urlencode(query)}" if query else "")

    def _post_json(self, url: str, headers: dict, body: dict) -> dict:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def _get_json(self, url: str, headers: dict) -> dict:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def _download_binary(self, url: str, headers: dict, dest: str) -> None:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk: f.write(chunk)

    def generate(
        self,
        model: str,
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:
        cfg = self._cfg()
        base = cfg["base_url"]
        headers = cfg["headers"]
        create_url = self._join(base, cfg["create_path"], cfg["query"])

        if self.backend == "openai":
            # Build the base fields
            data_fields = {
                "model": model,                          # e.g. "sora-2" or "sora-2-pro"
                "prompt": prompt,
                "size": cfg["format_size"](width, height),  # "1280x720"
                cfg["duration_key"]: str(int(n_seconds)),   # "seconds"
            }
            # Allow extra form/JSON fields (e.g., guidance, seed, audio, etc.)
            if extra:
                # don't overwrite base unless caller intends to
                for k, v in extra.items():
                    if k not in data_fields:
                        data_fields[k] = v

            # If a local file is supplied as input_reference, use multipart/form-data
            input_path = (extra or {}).get("input_reference")
            if input_path:
                # OpenAI expects a FILE for input_reference, not a URL/string-only. 
                # We'll upload it as multipart. 
                with open(input_path, "rb") as f:
                    files = {"input_reference": (os.path.basename(input_path), f, extra.get("input_mime", "application/octet-stream"))}
                    r = requests.post(create_url, headers=headers, data=data_fields, files=files, timeout=300)
                    r.raise_for_status()
                    create_resp = r.json()
            else:
                # Pure text → JSON is fine
                # (requests sets Content-Type: application/json automatically)
                create_resp = self._post_json(create_url, headers, data_fields)

            video_id = create_resp.get("id") or create_resp.get("video_id")
            if not video_id:
                raise RuntimeError(f"Unexpected create response (no video id): {create_resp}")

            # Poll for completion
            status_url = self._join(base, cfg["status_path_template"].format(video_id=video_id), cfg["query"])
            t0 = time.time()
            status_payload = {}
            while True:
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"OpenAI Sora job timed out after {timeout_s}s (id={video_id})")
                status_payload = self._get_json(status_url, headers)
                state = status_payload.get("status")
                if state in ("completed", "failed", "cancelled"):
                    break
                print(f"API generation state: {state}")
                time.sleep(60)

            if status_payload.get("status") != "completed":
                raise RuntimeError(f"Sora job failed or cancelled: {status_payload}")

            # Download MP4 (optionally choose a variant like 'thumbnail')
            download_query = dict(cfg["query"])
            if extra and extra.get("download_variant"):
                download_query["variant"] = extra["download_variant"]
            download_url = self._join(base, cfg["download_path_template"].format(video_id=video_id), download_query)
            self._download_binary(download_url, headers, output_path)

            return {
                "provider": "sora-openai",
                "model": model,
                "video_id": video_id,
                "video_path": output_path,
                "raw": {"create": create_resp, "final_status": status_payload},
            }

        else:
            # Azure
            body = {
                "prompt": prompt,
                "width": width,
                "height": height,
                cfg["duration_key"]: n_seconds,  # "n_seconds"
                "model": model,
            }
            if extra: body.update(extra)

            create_resp = self._post_json(create_url, headers, body)
            job_id = create_resp.get("id") or create_resp.get("job_id")
            if not job_id:
                raise RuntimeError(f"Unexpected create response: {create_resp}")

            status_url = self._join(base, cfg["status_path_template"].format(job_id=job_id), cfg["query"])
            t0 = time.time()
            status_payload = {}
            while True:
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"Azure Sora job timed out after {timeout_s}s (job_id={job_id})")
                status_payload = self._get_json(status_url, headers)
                state = status_payload.get("status")
                if state in ("completed", "failed", "cancelled"):
                    break
                print(f"API generation state: {state}")
                time.sleep(60)

            if status_payload.get("status") != "completed":
                raise RuntimeError(f"Sora job failed or cancelled: {status_payload}")

            generations = status_payload.get("generations", [])
            if not generations:
                raise RuntimeError(f"No generations returned: {status_payload}")
            gen_id = generations[0].get("id")

            download_url = self._join(base, cfg["download_path_template"].format(generation_id=gen_id), cfg["query"])
            self._download_binary(download_url, headers, output_path)

            return {
                "provider": "sora-azure",
                "model": model,
                "job_id": job_id,
                "generation_id": gen_id,
                "video_path": output_path,
                "raw": {"create": create_resp, "final_status": status_payload},
            }


@dataclass
class GeminiVeoAPI:
    """
    Gemini API (Veo 3.1)

    - Model IDs (examples): "veo-3.1-generate-preview", "veo-3.1-fast-generate-preview".
      (See "Model versions" table.)
    - Auth: set environment variable GEMINI_API_KEY and the SDK reads it automatically.
    - Basic flow: client.models.generate_videos(...), poll operation until done, then
      client.files.download(...) and save().
    """

    def _ensure_client(self):
        if google_genai is None:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai"
            )
        # SDK auto-uses GEMINI_API_KEY if set.
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        return google_genai.Client()

    @staticmethod
    def _infer_aspect_ratio(width: int, height: int, extra: t.Dict[str, t.Any]) -> str:
        # Allow explicit override via either snake_case or camelCase
        aspect = extra.get("aspect_ratio") or extra.get("aspectRatio")
        if aspect in ("16:9", "9:16"):
            return aspect
        # Heuristic from width/height. Veo supports 16:9 and 9:16.
        if height > width * 1.2:
            return "9:16"
        return "16:9"

    @staticmethod
    def _infer_resolution(width: int, height: int, extra: t.Dict[str, t.Any]) -> str:
        res = extra.get("resolution")
        if res in ("720p", "1080p"):
            return res
        # Veo 3.1 supports 720p and 1080p (1080p is only 8s).
        return "1080p" if max(width, height) >= 1080 else "720p"

    @staticmethod
    def _coerce_duration(n_seconds: int, extra: t.Dict[str, t.Any]) -> int:
        # Accept overrides via either snake_case or camelCase
        dur = extra.get("duration_seconds") or extra.get("durationSeconds")
        if dur is None:
            dur = n_seconds
        # Veo 3.1 supports 4 / 6 / 8 seconds. Clamp to nearest allowed.
        allowed = [4, 6, 8]
        return min(allowed, key=lambda x: abs(int(dur) - x))

    def generate(
        self,
        model: str,
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:

        client = self._ensure_client()

        # Build config with snake_case keys for the Python SDK. Examples show
        # number_of_videos, resolution, aspect_ratio, negative_prompt, duration_seconds.
        cfg_kwargs: dict = {
            "number_of_videos": int(extra.get("number_of_videos", 1)),
            "resolution": self._infer_resolution(width, height, extra),
            "aspect_ratio": self._infer_aspect_ratio(width, height, extra),
            "duration_seconds": self._coerce_duration(n_seconds, extra),
        }
        if "negative_prompt" in extra:
            cfg_kwargs["negative_prompt"] = extra["negative_prompt"]
        if "negativePrompt" in extra:
            cfg_kwargs["negative_prompt"] = extra["negativePrompt"]

        config = google_types.GenerateVideosConfig(**cfg_kwargs)

        # NOTE: this implementation focuses on text2video to match your current CLI.
        # TODO: To support image/video inputs later
        # we want to pass:
        #   image=<google_types.Image(...)> or video=<google_types.Video(...)>,
        #   and/or reference_images=[google_types.VideoGenerationReferenceImage(...)]
        # as shown in docs. 
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            config=config,
        )

        t0 = time.time()
        while not operation.done:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    f"Gemini Veo job timed out after {timeout_s}s (operation={operation.name})"
                )
            print(f"API generation not done, continuing...")
            time.sleep(60)
            operation = client.operations.get(operation)

        # Download the first result and save to output_path.
        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(output_path)

        return {
            "provider": "gemini-veo",
            "model": model,
            "operation": getattr(operation, "name", None),
            "video_path": output_path,
            "raw": {
                "response": operation.response,
            },
        }


# =========================
#  WAN 2.5 (DashScope HTTP)
# =========================
@dataclass
class WanDashScopeVideoAPI:
    """
    Alibaba Cloud Model Studio (DashScope) - WAN 2.5 Text-to-Video
    Docs (official): Wan text-to-video API reference.
    Regions:
      - intl (Singapore): https://dashscope-intl.aliyuncs.com
      - cn   (Beijing)  : https://dashscope.aliyuncs.com
    Auth:
      - export DASHSCOPE_API_KEY="sk-..."
    """
    region: str = os.getenv("DASHSCOPE_REGION", "intl")  # "intl" or "cn"

    def _base(self) -> str:
        return "https://dashscope-intl.aliyuncs.com" if self.region == "intl" else "https://dashscope.aliyuncs.com"

    def _join(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in parts)

    def _download(self, url: str, dest: str) -> None:
        # stream the response and write to disk in chunks
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)


    def generate(
        self,
        model: str,               # e.g. "wan2.5-t2v-preview"
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")

        base = self._base()
        create_url = self._join(base, "api/v1/services/aigc/video-generation/video-synthesis")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-DashScope-Async": "enable",  # HTTP is async-only for this API
            "Content-Type": "application/json",
        }

        # DashScope expects size as "WIDTH*HEIGHT" (string) and duration must be 5 or 10 for wan2.5
        duration = 10 if n_seconds >= 10 else 5
        size_str = f"{width}*{height}"

        body = {
            "model": model,  # "wan2.5-t2v-preview"
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "size": size_str,
                "duration": duration,
            },
        }

        # Optional fields supported by docs: prompt_extend, watermark, audio, audio_url, seed, negative_prompt
        if extra:
            if "negative_prompt" in extra:
                body["input"]["negative_prompt"] = extra["negative_prompt"]
            # Merge known parameter keys
            params = body["parameters"]
            for k in ("prompt_extend", "watermark", "audio", "audio_url", "seed"):
                if k in extra:
                    params[k] = extra[k]

        # Create task
        resp = requests.post(create_url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        create_payload = resp.json()
        task_id = (create_payload.get("output") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Unexpected DashScope create response: {create_payload}")

        # Poll task
        status_url = self._join(base, "api/v1/tasks", task_id)
        t0 = time.time()
        final = None
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"WAN job timed out after {timeout_s}s (task_id={task_id})")
            s = requests.get(status_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
            s.raise_for_status()
            payload = s.json()
            out = payload.get("output") or {}
            state = out.get("task_status")
            if state in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"):
                final = payload
                break
            time.sleep(15)

        if (final.get("output") or {}).get("task_status") != "SUCCEEDED":
            raise RuntimeError(f"WAN task not successful: {final}")

        video_url = (final.get("output") or {}).get("video_url")
        if not video_url:
            raise RuntimeError(f"No video_url in WAN result: {final}")
        # Download immediately (links expire ~24h)
        with requests.get(video_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

        return {
            "provider": "wan-dashscope",
            "model": model,
            "task_id": task_id,
            "video_url": video_url,
            "video_path": output_path,
            "raw": {"create": create_payload, "final": final},
        }


# ========================================
#  Kling (official)
# ========================================
@dataclass
class KlingVideoAPI:
    """
    Official Kling Text-to-Video.

    Base: https://api-singapore.klingai.com
      POST /v1/videos/text2video         (create)
      GET  /v1/videos/text2video/{id}    (poll)
    Token: Authorization: Bearer <JWT>  (HS256 with iss/exp/nbf)
    """
    base_url: str = os.environ.get(
        "KLING_API_BASE_URL", "https://api-singapore.klingai.com"
    ).rstrip("/")

    # ---- internal helpers ----
    def _bearer(self) -> str:
        # Prefer a pre-generated token, otherwise synthesize from AK/SK.
        tok = os.environ.get("KLING_JWT")
        if tok:
            return tok
        ak = os.environ.get("KLING_ACCESS_KEY")
        sk = os.environ.get("KLING_SECRET_KEY")
        if not (ak and sk):
            raise RuntimeError(
                "Set KLING_JWT or KLING_ACCESS_KEY & KLING_SECRET_KEY"
            )
        ttl = int(os.environ.get("KLING_TOKEN_EXPIRATION", "1800"))
        return kling_jwt(ak, sk, ttl_s=ttl)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._bearer()}",
                "Content-Type": "application/json"}

    def _join(self, path: str, q: dict | None = None) -> str:
        return f"{self.base_url}{path}" + (f"?{urlencode(q)}" if q else "")

    @staticmethod
    def _snap_aspect(w: int, h: int) -> str:
        r = w / max(h, 1)
        if abs(r - (16/9)) < 0.05: return "16:9"
        if abs(r - (9/16)) < 0.05: return "9:16"
        if abs(r - 1.0)   < 0.05: return "1:1"
        return "16:9"

    @staticmethod
    def _snap_duration(n: int) -> int:
        return 10 if n >= 10 else 5

    def _download(self, url: str, dest: str) -> None:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk: f.write(chunk)

    # ---- public API ----
    def generate(
        self,
        model: str,            # e.g., "kling-v2-master" or "kling-v1"
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:

        # Build minimal, valid body
        body: dict = {
            "prompt": prompt,
            "duration": self._snap_duration(int(n_seconds)),             # 5 or 10
            "aspect_ratio": self._snap_aspect(width, height),            # 16:9 / 9:16 / 1:1
        }

        # Heuristic: `model_name` only for V1; omit for V2 unless caller insists.
        # Some V2 endpoints reject body containing model_name (400/“model not supported”).
        # Ref: community test notes.
        force_model = (extra or {}).get("force_model_name")
        if force_model or ("v1" in (model or "").lower()):
            body["model_name"] = model  # e.g., "kling-v1"
        elif model and ("v2" in model.lower()):
            # omit to let server pick the default v2 variant
            pass
        elif model:
            # If you pass a concrete v2 string that the API accepts (e.g., "kling-v2-master"),
            # it’s fine to include; otherwise omit to be safe.
            if "v2" not in model.lower():
                body["model_name"] = model

        # Pass through supported optional knobs if provided.
        # Common ones seen in public UIs: negative_prompt, cfg_scale, mode, camera_control...
        for k in ("negative_prompt", "cfg_scale", "mode", "camera_control"):
            if extra and (k in extra):
                body[k] = extra[k]

        # Create task
        create_url = self._join("/v1/videos/text2video")
        resp = requests.post(create_url, headers=self._headers(), json=body, timeout=60)
        resp.raise_for_status()

        c = resp.json()
        task_id = c.get("task_id") or (c.get("data") or {}).get("task_id") or c.get("id")
        if not task_id:
            raise RuntimeError(f"Unexpected Kling create response: {c}")

        # Poll status
        poll_url = self._join(f"/v1/videos/text2video/{task_id}")
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Kling job timed out after {timeout_s}s (task_id={task_id})")
            s = requests.get(poll_url, headers=self._headers(), timeout=45)
            s.raise_for_status()
            status_payload = s.json()
            data_obj = status_payload.get("data") or {}
            st = data_obj.get("task_status") or status_payload.get("status")
            if st in ("succeed", "failed", "cancelled"):
                if st != "succeed":
                    raise RuntimeError(f"Kling task failed or cancelled: {status_payload}")
                # success --> grab result
                break
            print(f"API generation state: {st}")
            time.sleep(60)

        # Download first video
        result = (status_payload.get("data") or {}).get("task_result") or {}
        videos = result.get("videos") or []
        if not videos or not videos[0].get("url"):
            raise RuntimeError(f"Kling task succeeded but no video URL found: {status_payload}")
        video_url = videos[0]["url"]
        self._download(video_url, output_path)

        return {
            "provider": "kling-official",
            "model": model,
            "task_id": task_id,
            "video_url": video_url,
            "video_path": output_path,
            "raw": {"create": c, "final_status": status_payload},
        }

@dataclass
class Veo3GenVideoAPI:
    """
    Veo3 Gen (third-party VEO3 access)

    Endpoints:
      - POST {base}/api/generate   (start a job)
      - GET  {base}/api/status/{taskId}  (poll status)
    Auth:
      - Authorization: Bearer <VEO3GEN_API_KEY> (recommended)
        Alternatively: X-API-Key, or ?api_key=... (not recommended).
    Docs: https://www.veo3gen.app/api-docs
    """
    base_url: str = os.environ.get("VEO3GEN_BASE_URL", "https://api.veo3gen.app").rstrip("/")

    # ---- internals ----
    def _api_key(self) -> str:
        key = (
            os.environ.get("VEO3GEN_API_KEY")
            or os.environ.get("VEO3GEN_KEY")
            or os.environ.get("VEO3GEN_TOKEN")
        )
        if not key:
            raise RuntimeError("VEO3GEN_API_KEY not set")
        return key

    def _headers(self) -> dict:
        # Docs recommend Authorization: Bearer
        return {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}

    def _join(self, path: str, q: dict | None = None) -> str:
        return f"{self.base_url}{path}" + (f"?{urlencode(q)}" if q else "")

    def _download(self, url: str, dest: str) -> None:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

    @staticmethod
    def _infer_aspect_ratio(width: int, height: int, extra: t.Dict[str, t.Any]) -> str:
        # Allow override via aspect_ratio / aspectRatio
        ar = (extra or {}).get("aspect_ratio") or (extra or {}).get("aspectRatio")
        if ar in ("16:9", "9:16"):
            return ar
        return "9:16" if height > width * 1.2 else "16:9"

    @staticmethod
    def _infer_resolution(width: int, height: int, extra: t.Dict[str, t.Any]) -> str:
        # Allow override via resolution
        res = (extra or {}).get("resolution")
        if res in ("720p", "1080p"):
            return res
        return "1080p" if max(width, height) >= 1080 else "720p"

    # ---- public API ----
    def generate(
        self,
        model: str,             # "veo3-fast" or "veo3-quality"
        prompt: str,
        n_seconds: int,         # NOTE: Veo3Gen API doesn't accept duration; it's fixed per model/version
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:

        # Build request body per docs
        # POST /api/generate with: model, prompt, optional modelVersion, audio, options{resolution,aspectRatio,seed,negativePrompt,enhancePrompt}
        # (Auth via Authorization: Bearer <api_key>)
        # Ref: Quick Start / API Endpoints.
        aspect = self._infer_aspect_ratio(width, height, extra or {})
        resolution = self._infer_resolution(width, height, extra or {})
        options: dict = {"resolution": resolution, "aspectRatio": aspect}

        # pass through optional knobs if present
        if extra:
            if "seed" in extra:
                options["seed"] = extra["seed"]
            # allow either key spelling
            if "negative_prompt" in extra:
                options["negativePrompt"] = extra["negative_prompt"]
            if "negativePrompt" in extra:
                options["negativePrompt"] = extra["negativePrompt"]
            if "enhancePrompt" in extra:
                options["enhancePrompt"] = extra["enhancePrompt"]
            # let users shove raw options in a dict key
            if "veo3Options" in extra and isinstance(extra["veo3Options"], dict):
                options.update(extra["veo3Options"])

        body: dict = {
            "model": model or "veo3-fast",
            "prompt": prompt,
            # audio defaults to true in the API; let user override
            "audio": bool((extra or {}).get("audio", True)),
            "options": options,
        }
        # Only add modelVersion if explicitly provided (API default is "3.0")
        mv = (extra or {}).get("modelVersion") or (extra or {}).get("model_version")
        if mv:
            body["modelVersion"] = mv

        create_url = self._join("/api/generate")
        print(f"[DEBUG] Veo3Gen request URL: {create_url}")
        print(f"[DEBUG] Veo3Gen request body: {json.dumps(body, indent=2)}")
        resp = requests.post(create_url, headers=self._headers(), json=body, timeout=120)
        print(f"[DEBUG] Veo3Gen response status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[DEBUG] Veo3Gen response text: {resp.text}")
        resp.raise_for_status()

        create_payload = resp.json()
        task_id = create_payload.get("taskId")
        if not task_id:
            raise RuntimeError(f"Unexpected Veo3Gen create response: {create_payload}")

        # Poll status: GET /api/status/{taskId}; completed -> result.videoUrl
        # Typical statuses: pending/processing/completed/failed.
        status_url = self._join(f"/api/status/{task_id}")
        t0 = time.time()
        final = None
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Veo3Gen job timed out after {timeout_s}s (taskId={task_id})")
            s = requests.get(status_url, headers=self._headers(), timeout=60)
            s.raise_for_status()
            st_payload = s.json()
            st = st_payload.get("status")
            if st in ("completed", "failed"):
                final = st_payload
                break

            print(f"API generation state: {st}")
            time.sleep(10)

        if not final or final.get("status") != "completed":
            raise RuntimeError(f"Veo3Gen task failed or did not complete: {final}")

        result = final.get("result") or {}
        video_url = result.get("videoUrl")
        if not video_url:
            raise RuntimeError(f"No videoUrl in Veo3Gen result: {final}")

        # Download the MP4 immediately
        self._download(video_url, output_path)

        return {
            "provider": "veo3gen",
            "model": model,
            "task_id": task_id,
            "video_url": video_url,
            "video_path": output_path,
            "raw": {"create": create_payload, "final_status": final},
        }

@dataclass
class LumaRayVideoAPI:
    """
    Luma Labs Ray (Ray 2 / Ray 2 Flash)

    Models:
      - Ray 2          -> model="ray-2"
      - Ray 2 Flash    -> model="ray-flash-2"
      - (also Ray 1.6: "ray-1-6")
      Docs list/usage and cURL examples.

    Endpoints:
      - POST https://api.lumalabs.ai/dream-machine/v1/generations         (create)
      - GET  https://api.lumalabs.ai/dream-machine/v1/generations/{id}    (status)
      Auth: Authorization: Bearer <LUMA_API_KEY>
    """

    base_url: str = os.environ.get(
        "LUMA_API_BASE_URL", "https://api.lumalabs.ai"
    ).rstrip("/")

    # ---------- internals ----------
    def _api_key(self) -> str:
        key = os.environ.get("LUMA_API_KEY") or os.environ.get("LUMA_TOKEN")
        if not key:
            raise RuntimeError("LUMA_API_KEY not set")
        return key

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _join(self, path: str, q: dict | None = None) -> str:
        return f"{self.base_url}{path}" + (f"?{urlencode(q)}" if q else "")

    def _download(self, url: str, dest: str) -> None:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

    @staticmethod
    def _infer_aspect_ratio(width: int, height: int, extra: t.Dict[str, t.Any]) -> t.Optional[str]:
        # API supports "aspect_ratio": e.g., "16:9" or "9:16"
        ar = (extra or {}).get("aspect_ratio") or (extra or {}).get("aspectRatio")
        if ar:
            return ar
        if width and height:
            return "9:16" if height > width * 1.2 else "16:9"
        return None

    @staticmethod
    def _infer_resolution(width: int, height: int, extra: t.Dict[str, t.Any]) -> t.Optional[str]:
        # Resolutions: "540p", "720p", "1080p", "4k" (optional)
        res = (extra or {}).get("resolution")
        if res:
            return str(res)
        if max(width, height) >= 2160:
            return "4k"
        if max(width, height) >= 1080:
            return "1080p"   # Changed to include 'p' for consistency
        if max(width, height) >= 720:
            return "720p"
        return "540p"

    @staticmethod
    def _format_duration(n_seconds: int, extra: t.Dict[str, t.Any]) -> t.Optional[str]:
        # API expects duration as string like "5s" (optional)
        if (extra or {}).get("duration"):
            d = str(extra["duration"])
            return d if d.endswith("s") else f"{d}s"
        if n_seconds:
            return f"{int(n_seconds)}s"
        return None

    # ---------- public ----------
    def generate(
        self,
        model: str,            # e.g., "ray-2" or "ray-flash-2"
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:

        model = model or "ray-2"

        body: dict = {
            "prompt": prompt,
            "model": model,
        }

        # Optional knobs per docs (we add only if defined)
        ar = self._infer_aspect_ratio(width, height, extra or {})
        if ar:
            body["aspect_ratio"] = ar

        res = self._infer_resolution(width, height, extra or {})
        if res:
            body["resolution"] = res

        dur = self._format_duration(n_seconds, extra or {})
        if dur:
            body["duration"] = dur

        if extra:
            # Loop flag
            if "loop" in extra:
                body["loop"] = bool(extra["loop"])
            # Concepts array
            if "concepts" in extra:
                body["concepts"] = extra["concepts"]

            # Keyframes (image-to-video / extend / interpolate)
            if "keyframes" in extra and isinstance(extra["keyframes"], dict):
                body["keyframes"] = extra["keyframes"]
            else:
                # Convenience shorthands:
                if "start_image_url" in extra:
                    body["keyframes"] = {
                        "frame0": {"type": "image", "url": extra["start_image_url"]}
                    }
                if "end_image_url" in extra:
                    kf = body.setdefault("keyframes", {})
                    kf["frame1"] = {"type": "image", "url": extra["end_image_url"]}
                if "start_generation_id" in extra:
                    kf = body.setdefault("keyframes", {})
                    kf["frame0"] = {"type": "generation", "id": extra["start_generation_id"]}
                if "end_generation_id" in extra:
                    kf = body.setdefault("keyframes", {})
                    kf["frame1"] = {"type": "generation", "id": extra["end_generation_id"]}

            # Callback (webhook) if you want server pushes (optional)
            if "callback_url" in extra:
                body["callback_url"] = extra["callback_url"]

        # Create generation
        create_url = self._join("/dream-machine/v1/generations")
        print(f"[DEBUG] Luma Ray request URL: {create_url}")
        print(f"[DEBUG] Luma Ray request body: {json.dumps(body, indent=2)}")
        r = requests.post(create_url, headers=self._headers(), json=body, timeout=120)
        print(f"[DEBUG] Luma Ray response status: {r.status_code}")
        if r.status_code != 200:
            print(f"[DEBUG] Luma Ray error response: {r.text}")
        r.raise_for_status()

        create_payload = r.json()
        gen_id = create_payload.get("id")
        if not gen_id:
            raise RuntimeError(f"Unexpected Luma create response: {create_payload}")

        # Poll status until "completed" (states: dreaming/completed/failed)
        status_url = self._join(f"/dream-machine/v1/generations/{gen_id}")
        t0 = time.time()
        final = None
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Luma Ray job timed out after {timeout_s}s (id={gen_id})")
            s = requests.get(status_url, headers=self._headers(), timeout=60)
            s.raise_for_status()
            payload = s.json()
            state = payload.get("state")
            if state in ("completed", "failed"):
                final = payload
                break
            print(f"API generation state: {state}")
            time.sleep(10)

        if not final or final.get("state") != "completed":
            raise RuntimeError(f"Luma generation failed or not completed: {final}")

        video_url = ((final.get("assets") or {}).get("video"))
        if not video_url:
            raise RuntimeError(f"No 'assets.video' in Luma result: {final}")

        # Download video
        self._download(video_url, output_path)

        return {
            "provider": "luma-ray",
            "model": model,
            "generation_id": gen_id,
            "video_url": video_url,
            "video_path": output_path,
            "raw": {"create": create_payload, "final_status": final},
        }

@dataclass
class FastVideoAPI:
    """
    FastVideo API provider for video generation.
    
    Supports two modes:
    1. Direct local inference (when FASTVIDEO_API_BASE is not set)
       - Uses VideoGenerator directly from fastvideo package
       - Requires model_path to be set in extra or as model parameter
    2. Remote API server (when FASTVIDEO_API_BASE is set)
       - Makes HTTP requests to deployed FastVideo server
       - Endpoint: POST {FASTVIDEO_API_BASE}/generate_video
       
    Environment variables:
      - FASTVIDEO_API_BASE: Base URL for FastVideo API server (optional)
      - FASTVIDEO_API_KEY: API key for authentication (optional, for remote mode)
      - FASTVIDEO_MODEL_PATH: Default model path for local inference (optional)
    """
    
    base_url: str = os.environ.get("FASTVIDEO_API_BASE", "").rstrip("/")
    
    def _headers(self) -> dict:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("FASTVIDEO_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers
    
    def _join(self, path: str) -> str:
        """Join base URL with path."""
        return f"{self.base_url}{path}"
    
    def _generate_via_api(
        self,
        model: str,
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:
        """Generate video via remote FastVideo API server."""
        
        # Prepare request body matching FastVideo API format
        body = {
            "prompt": prompt,
            "num_frames": int(n_seconds * 16),  # Default 16 fps
            "width": width,
            "height": height,
            "model_path": model or extra.get("model_path"),
            "seed": extra.get("seed", 42),
            "guidance_scale": extra.get("guidance_scale", 7.5),
            "randomize_seed": extra.get("randomize_seed", False),
            "return_frames": False,
        }
        
        # Add optional parameters
        if extra.get("negative_prompt"):
            body["negative_prompt"] = extra["negative_prompt"]
            body["use_negative_prompt"] = True
        
        if extra.get("num_frames"):
            body["num_frames"] = int(extra["num_frames"])
        
        # Handle image input for I2V
        if extra.get("image_path"):
            with open(extra["image_path"], "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
                body["image_data"] = f"data:image/png;base64,{image_data}"
        
        # Make API request
        url = self._join("/generate_video")
        print(f"[DEBUG] FastVideo API request URL: {url}")
        print(f"[DEBUG] FastVideo API request body: {json.dumps({k: v for k, v in body.items() if k != 'image_data'}, indent=2)}")
        
        r = requests.post(url, headers=self._headers(), json=body, timeout=timeout_s)
        
        if r.status_code != 200:
            error_msg = r.text
            print(f"[DEBUG] FastVideo API error response: {error_msg}")
            r.raise_for_status()
        
        response = r.json()
        
        if not response.get("success"):
            error_msg = response.get("error_message", "Unknown error")
            raise RuntimeError(f"FastVideo API generation failed: {error_msg}")
        
        # Decode video from base64
        video_data = response.get("video_data")
        if not video_data:
            raise RuntimeError("No video_data in FastVideo API response")
        
        # Remove data URL prefix if present
        if video_data.startswith("data:video/"):
            video_data = video_data.split(",", 1)[1]
        
        # Save video to file
        video_bytes = base64.b64decode(video_data)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        
        return {
            "provider": "fastvideo",
            "model": model,
            "video_path": output_path,
            "seed": response.get("seed", body.get("seed", 42)),
            "generation_time": response.get("generation_time"),
            "inference_time": response.get("inference_time"),
            "raw": response,
        }
    
    def _generate_via_local(
        self,
        model: str,
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:
        """Generate video via local FastVideo VideoGenerator."""
        try:
            from fastvideo import VideoGenerator
        except ImportError:
            raise RuntimeError(
                "fastvideo package not installed. Install via: pip install fastvideo\n"
                "Or set FASTVIDEO_API_BASE to use remote API server."
            )
        
        # Get model path from model parameter or extra or environment
        model_path = model or extra.get("model_path") or os.environ.get("FASTVIDEO_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "Model path required for FastVideo local inference. "
                "Set model parameter, extra['model_path'], or FASTVIDEO_MODEL_PATH environment variable."
            )
        
        # Set attention backend based on model type
        # FastWan DMD models need VIDEO_SPARSE_ATTN
        model_lower = model_path.lower()
        if "fastwan" in model_lower or "fast-wan" in model_lower:
            if "fullattn" in model_lower:
                os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
            else:
                os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
        
        # Initialize generator following example.py pattern
        print(f"[DEBUG] FastVideo initializing local generator with model: {model_path}")
        print(f"[DEBUG] FASTVIDEO_ATTENTION_BACKEND: {os.environ.get('FASTVIDEO_ATTENTION_BACKEND', 'not set')}")
        
        generator = VideoGenerator.from_pretrained(
            model_path,
            num_gpus=extra.get("num_gpus", 1),
        )
        
        # Generate video following example.py pattern
        print(f"[DEBUG] FastVideo generating video with prompt: {prompt[:100]}...")
        
        # Build kwargs for generate_video
        gen_kwargs = {
            "return_frames": False,
            "output_path": output_path,
            "save_video": True,
        }
        
        # Add optional parameters if provided
        if extra.get("seed") is not None:
            gen_kwargs["seed"] = int(extra["seed"])
        if extra.get("guidance_scale") is not None:
            gen_kwargs["guidance_scale"] = float(extra["guidance_scale"])
        if extra.get("num_inference_steps") is not None:
            gen_kwargs["num_inference_steps"] = int(extra["num_inference_steps"])
        if extra.get("negative_prompt"):
            gen_kwargs["negative_prompt"] = extra["negative_prompt"]
        if extra.get("num_frames"):
            gen_kwargs["num_frames"] = int(extra["num_frames"])
        else:
            gen_kwargs["num_frames"] = int(n_seconds * 16)  # Default 16 fps
        if extra.get("fps"):
            gen_kwargs["fps"] = float(extra["fps"])
        if extra.get("height"):
            gen_kwargs["height"] = int(extra["height"])
        else:
            gen_kwargs["height"] = height
        if extra.get("width"):
            gen_kwargs["width"] = int(extra["width"])
        else:
            gen_kwargs["width"] = width
        
        result = generator.generate_video(prompt, **gen_kwargs)
        
        # Cleanup
        generator.shutdown()
        
        # Extract generation time if available
        generation_time = None
        if isinstance(result, dict):
            generation_time = result.get("generation_time")
        
        return {
            "provider": "fastvideo",
            "model": model_path,
            "video_path": output_path,
            "seed": gen_kwargs.get("seed", 42),
            "generation_time": generation_time,
            "raw": result if isinstance(result, dict) else {"result": "success"},
        }
    
    def generate(
        self,
        model: str,
        prompt: str,
        n_seconds: int,
        width: int,
        height: int,
        output_path: str,
        timeout_s: int,
        extra: t.Dict[str, t.Any],
    ) -> t.Dict[str, t.Any]:
        """Generate video using FastVideo (either local or remote API)."""
        # Use remote API if base_url is set, otherwise use local inference
        if self.base_url:
            return self._generate_via_api(
                model=model,
                prompt=prompt,
                n_seconds=n_seconds,
                width=width,
                height=height,
                output_path=output_path,
                timeout_s=timeout_s,
                extra=extra or {},
            )
        else:
            return self._generate_via_local(
                model=model,
                prompt=prompt,
                n_seconds=n_seconds,
                width=width,
                height=height,
                output_path=output_path,
                timeout_s=timeout_s,
                extra=extra or {},
            )