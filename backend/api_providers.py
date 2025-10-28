import os
import time
import typing as t
import requests
from dataclasses import dataclass
from urllib.parse import urlencode

import replicate

from google import genai as google_genai
from google.genai import types as google_types

@dataclass
class ReplicateVideoAPI:
    """
    Runs a video model on Replicate.

    Auth:
      - export REPLICATE_API_TOKEN="xxx" (bearer token)
        (The Python client automatically reads this env var.) :contentReference[oaicite:3]{index=3}

    Notes:
      - `model` can be "owner/model" or "owner/model:version".
      - `input` shape is model-specific (we pass prompt + optional extras).
      - Outputs are often file URLs or FileOutput objects.
      - Reference: "Run a model" + HTTP API docs. :contentReference[oaicite:4]{index=4}
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

        # Example from docs: replicate.run("owner/model[:version]", input={...}) :contentReference[oaicite:5]{index=5}
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
      (See "Model versions" table.) :contentReference[oaicite:0]{index=0}
    - Auth: set environment variable GEMINI_API_KEY and the SDK reads it automatically. :contentReference[oaicite:1]{index=1}
    - Basic flow: client.models.generate_videos(...), poll operation until done, then
      client.files.download(...) and save(). :contentReference[oaicite:2]{index=2}
    """

    def _ensure_client(self):
        if google_genai is None:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai"
            )
        # SDK auto-uses GEMINI_API_KEY if set. :contentReference[oaicite:3]{index=3}
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        return google_genai.Client()

    @staticmethod
    def _infer_aspect_ratio(width: int, height: int, extra: t.Dict[str, t.Any]) -> str:
        # Allow explicit override via either snake_case or camelCase
        aspect = extra.get("aspect_ratio") or extra.get("aspectRatio")
        if aspect in ("16:9", "9:16"):
            return aspect
        # Heuristic from width/height. Veo supports 16:9 and 9:16. :contentReference[oaicite:4]{index=4}
        if height > width * 1.2:
            return "9:16"
        return "16:9"

    @staticmethod
    def _infer_resolution(width: int, height: int, extra: t.Dict[str, t.Any]) -> str:
        res = extra.get("resolution")
        if res in ("720p", "1080p"):
            return res
        # Veo 3.1 supports 720p and 1080p (1080p is only 8s). :contentReference[oaicite:5]{index=5}
        return "1080p" if max(width, height) >= 1080 else "720p"

    @staticmethod
    def _coerce_duration(n_seconds: int, extra: t.Dict[str, t.Any]) -> int:
        # Accept overrides via either snake_case or camelCase
        dur = extra.get("duration_seconds") or extra.get("durationSeconds")
        if dur is None:
            dur = n_seconds
        # Veo 3.1 supports 4 / 6 / 8 seconds. Clamp to nearest allowed. :contentReference[oaicite:6]{index=6}
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
        # number_of_videos, resolution, aspect_ratio, negative_prompt, duration_seconds. :contentReference[oaicite:7]{index=7}
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
        # :contentReference[oaicite:8]{index=8}
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
            time.sleep(10)  # official samples poll every ~10s :contentReference[oaicite:9]{index=9}
            operation = client.operations.get(operation)

        # Download the first result and save to output_path. :contentReference[oaicite:10]{index=10}
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