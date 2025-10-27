import os
import time
import json
import typing as t
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

# Replicate has a first-party Python client that wraps their HTTP API and
# uses REPLICATE_API_TOKEN from env by default. Docs: run models from Python. :contentReference[oaicite:2]{index=2}
try:
    import replicate
except Exception:
    replicate = None


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
        if isinstance(output, list):
            for item in output:
                if hasattr(item, "url"):
                    video_urls.append(item.url)  # FileOutput
                elif isinstance(item, (str, bytes)):
                    # If bytes, then it's a raw file (rare); if str and looks like URL, collect it
                    if isinstance(item, str) and item.startswith(("http://", "https://")):
                        video_urls.append(item)
        elif isinstance(output, (str, bytes)):
            if isinstance(output, str) and output.startswith(("http://", "https://")):
                video_urls.append(output)

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
    Sora via REST. Two backends supported:

    - backend="azure": Azure OpenAI Sora preview REST:
        POST {endpoint}/openai/v1/video/generations/jobs?api-version=preview
        GET  {endpoint}/openai/v1/video/generations/jobs/{job_id}?api-version=preview
        GET  {endpoint}/openai/v1/video/generations/{generation_id}/content/video?api-version=preview
      Auth: header "api-key: <AZURE_OPENAI_API_KEY>"
      Endpoint base: AZURE_OPENAI_ENDPOINT (e.g., https://<res-name>.openai.azure.com)
      Ref: Azure Sora Quickstart (job create → poll → download). :contentReference[oaicite:6]{index=6}

    - backend="openai": (planned) OpenAI Videos API.
      If you have access, set:
        OPENAI_API_KEY
        OPENAI_API_BASE (default https://api.openai.com)
      and optionally override the paths via env if they differ.
      The high-level flow mirrors “create job → poll → download” in the
      OpenAI Video Generation guide you referenced. :contentReference[oaicite:7]{index=7}
    """

    backend: str = "azure"  # "azure" or "openai"

    def _cfg(self):
        if self.backend == "azure":
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
            }

        elif self.backend == "openai":
            base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com").rstrip("/")
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")

            # These defaults match the Azure pattern; override if your org’s OpenAI tenant differs.
            return {
                "base_url": base,
                "headers": {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                "create_path": os.environ.get("OPENAI_VIDEO_CREATE_PATH", "/v1/video/generations/jobs"),
                "status_path_template": os.environ.get(
                    "OPENAI_VIDEO_STATUS_PATH_TEMPLATE",
                    "/v1/video/generations/jobs/{job_id}",
                ),
                "download_path_template": os.environ.get(
                    "OPENAI_VIDEO_DOWNLOAD_PATH_TEMPLATE",
                    "/v1/video/generations/{generation_id}/content/video",
                ),
                "query": {},  # add version if your tenant requires it
            }

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _join(self, base: str, path: str, query: t.Dict[str, str]) -> str:
        if query:
            return f"{base}{path}?{urlencode(query)}"
        return f"{base}{path}"

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
        cfg = self._cfg()
        base = cfg["base_url"]
        headers = cfg["headers"]
        create_url = self._join(base, cfg["create_path"], cfg["query"])

        # Body fields match the Azure Quickstart format (and the general job-based Sora flow):
        # prompt, width, height, n_seconds, model ("sora" or "sora-2"), n_variants, etc. :contentReference[oaicite:8]{index=8}
        body = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "n_seconds": n_seconds,
            "model": model,  # ex: "sora" or "sora-2"
        }
        
        if extra:
            body.update(extra)

        create_resp = self._post_json(create_url, headers, body)
        job_id = create_resp.get("id") or create_resp.get("job_id")
        if not job_id:
            raise RuntimeError(f"Unexpected create response (no job id): {create_resp}")

        status_url = self._join(
            base,
            cfg["status_path_template"].format(job_id=job_id),
            cfg["query"],
        )

        # Poll for status → "succeeded" then download first generation.
        t0 = time.time()
        status_payload = {}
        status = None
        while True:
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Sora job timed out after {timeout_s}s (job_id={job_id})")
            time.sleep(5)
            status_payload = self._get_json(status_url, headers)
            status = status_payload.get("status")
            if status in ("succeeded", "failed", "cancelled"):
                break

        if status != "succeeded":
            raise RuntimeError(f"Sora job failed or cancelled: {status_payload}")

        generations = status_payload.get("generations", [])
        if not generations:
            raise RuntimeError(f"No generations returned: {status_payload}")
        gen_id = generations[0].get("id")

        download_url = self._join(
            base,
            cfg["download_path_template"].format(generation_id=gen_id),
            cfg["query"],
        )
        self._download_binary(download_url, headers, output_path)

        return {
            "provider": f"sora-{self.backend}",
            "model": model,
            "job_id": job_id,
            "generation_id": gen_id,
            "video_path": output_path,
            "raw": {"create": create_resp, "final_status": status_payload},
        }
