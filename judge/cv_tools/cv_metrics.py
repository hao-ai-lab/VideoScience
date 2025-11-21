#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch

# ---- Optional external deps ----
# pip install groundingdino cjm-byte-track lpips torchvision transformers
from groundingdino.util.inference import (
    load_model as gdino_load_model,
    load_image as gdino_load_image,
    predict as gdino_predict,
)

from cjm_byte_track.core import BYTETracker

import lpips


from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.transforms.functional as TF

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)


# ---------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------
def sample_video_frames(
    video_path: str,
    max_frames: int | None = 64,
    target_fps: float | None = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Return a list of BGR frames (numpy HxWx3) and original fps.
    If target_fps is given, downsample roughly to that rate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    # Decide which indices to grab
    if target_fps and orig_fps > 0:
        step = max(1, int(round(orig_fps / target_fps)))
        indices = list(range(0, frame_count, step))
    else:
        indices = list(range(frame_count))

    if max_frames is not None and len(indices) > max_frames:
        step = max(1, len(indices) // max_frames)
        indices = indices[::step][:max_frames]

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, orig_fps


def ensure_device(device: str | None = None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


# ---------------------------------------------------------------------
# Grounding DINO: open-vocab detection
# ---------------------------------------------------------------------
def _boxes_cxcywh_norm_to_xyxy_abs(
    boxes: torch.Tensor,
    img_w: int,
    img_h: int,
) -> List[List[float]]:
    """
    GroundingDINO predict() returns boxes in normalized cx,cy,w,h.
    Convert to absolute xyxy (pixels).
    """
    if boxes.numel() == 0:
        return []

    b = boxes.clone()
    cx = b[:, 0] * img_w
    cy = b[:, 1] * img_h
    bw = b[:, 2] * img_w
    bh = b[:, 3] * img_h

    x1 = (cx - bw / 2).clamp(0, img_w - 1)
    y1 = (cy - bh / 2).clamp(0, img_h - 1)
    x2 = (cx + bw / 2).clamp(0, img_w - 1)
    y2 = (cy + bh / 2).clamp(0, img_h - 1)

    xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().tolist()
    return [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in xyxy]


def analyze_grounding_dino(
    video_path: str,
    text_prompt: str,
    cfg_path: str,
    ckpt_path: str,
    device: torch.device,
    max_frames: int = 16,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Use GroundingDINO to check entity presence / attributes on a subset of frames.

    NOTE: This implementation writes temporary frame images to disk and then uses
    GroundingDINO's load_image/predict helpers as in their README.

    Returns a dict that now ALSO includes:
      - frame_size: [H, W]
      - detections_per_frame: [
            {"frame_index": i,
             "boxes": [[x1,y1,x2,y2,score], ...]} ]
    so ByteTrack can consume real detections.
    """
    if gdino_load_model is None:
        return {"error": "groundingdino not installed", "module": "grounding_dino"}

    frames, orig_fps = sample_video_frames(video_path, max_frames=max_frames)
    if not frames:
        return {"error": "no frames sampled", "module": "grounding_dino"}

    tmp_dir = Path(".gdino_tmp")
    tmp_dir.mkdir(exist_ok=True)

    model = gdino_load_model(cfg_path, ckpt_path)
    model.to(device)
    model.eval()

    all_phrases: List[str] = []
    per_frame_stats: List[Dict[str, Any]] = []
    detections_per_frame: List[Dict[str, Any]] = []

    frame_h, frame_w = frames[0].shape[0], frames[0].shape[1]

    for i, frame in enumerate(frames):
        img_path = tmp_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(img_path), frame)

        image_source, image = gdino_load_image(str(img_path))
        h, w = image_source.shape[:2]
        frame_h, frame_w = h, w  # assume constant across frames

        with torch.no_grad():
            boxes, logits, phrases = gdino_predict(
                model=model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

        # logits is typically a tensor of shape [N]; be robust
        scores: List[float] = []
        if isinstance(logits, torch.Tensor):
            flat = logits.reshape(-1).detach().cpu().tolist()
            scores = [float(x) for x in flat]
        elif isinstance(logits, (list, tuple)):
            scores = [float(x) for x in logits]

        phrases = [str(p) for p in phrases]
        all_phrases.extend(phrases)

        # ---- NEW: compute absolute xyxy boxes and attach scores ----
        boxes_xyxy: List[List[float]] = []
        if isinstance(boxes, torch.Tensor) and boxes.numel() > 0 and scores:
            xyxy = _boxes_cxcywh_norm_to_xyxy_abs(boxes, img_w=w, img_h=h)
            # align lengths just in case
            n = min(len(xyxy), len(scores))
            boxes_xyxy = [xyxy[j] + [float(scores[j])] for j in range(n)]

        per_frame_stats.append(
            {
                "frame_index": i,
                "num_detections": len(phrases),
                "phrases": phrases,
                "scores": scores,
                "boxes_xyxy": boxes_xyxy,  # NEW
            }
        )

        detections_per_frame.append(
            {
                "frame_index": i,
                "boxes": boxes_xyxy,  # [x1,y1,x2,y2,score]
            }
        )

    unique_phrases = sorted(set(all_phrases))
    presence_ratio = (
        len([f for f in per_frame_stats if f["num_detections"] > 0]) / len(per_frame_stats)
        if per_frame_stats
        else 0.0
    )

    return {
        "module": "grounding_dino",
        "video_path": str(video_path),
        "text_prompt": text_prompt,
        "num_frames_evaluated": len(per_frame_stats),
        "frame_size": [frame_h, frame_w],                  # NEW
        "unique_phrases": unique_phrases,
        "frames_with_any_detection_ratio": presence_ratio,
        "per_frame": per_frame_stats,
        "detections_per_frame": detections_per_frame,      # NEW
    }


# ---------------------------------------------------------------------
# ByteTrack: identity association & coherence
# ---------------------------------------------------------------------
def analyze_bytetrack_from_detections(
    detections_per_frame: List[Dict[str, Any]],
    frame_size: Tuple[int, int],
    fps: float,
) -> Dict[str, Any]:
    """
    Run ByteTrack using pre-computed detections.

    detections_per_frame: list of dicts like
        {"frame_index": int, "boxes": [[x1,y1,x2,y2,score], ...]}
    frame_size: (H, W)

    Returns:
        - track_lengths: {track_id: num_frames_tracked}
        - track_frame_indices: {track_id: [frame_idx, ...]}
        - track_spans: {track_id: {"start_frame": .., "end_frame": ..}}
        - coherence_score: crude metric from coverage + average length
    """
    if BYTETracker is None:
        return {"error": "cjm-byte-track not installed", "module": "bytetrack"}

    H, W = frame_size
    tracker = BYTETracker(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=fps if fps > 0 else 24.0,
    )

    track_lengths: Dict[int, int] = {}
    track_frame_indices: Dict[int, List[int]] = {}
    frame_tracks: Dict[int, List[int]] = {}

    for frame_info in detections_per_frame:
        idx = int(frame_info["frame_index"])
        boxes_list = frame_info.get("boxes", [])
        boxes = np.array(boxes_list, dtype=np.float32)
        if boxes.size == 0:
            frame_tracks[idx] = []
            continue

        # Expected shape: [N, 5] = [x1, y1, x2, y2, score]
        tracks = tracker.update(
            output_results=boxes,
            img_info=(H, W),
            img_size=(H, W),
        )

        ids_here: List[int] = []
        for t in tracks:
            tid = int(t.track_id)
            ids_here.append(tid)
            track_lengths[tid] = track_lengths.get(tid, 0) + 1
            track_frame_indices.setdefault(tid, []).append(idx)

        frame_tracks[idx] = ids_here

    num_tracks = len(track_lengths)
    avg_len = float(sum(track_lengths.values()) / num_tracks) if num_tracks else 0.0

    # crude “coherence score”: longer tracks & fewer empty frames => higher coherence
    nonempty_frames = sum(1 for ids in frame_tracks.values() if ids)
    total_frames = len(frame_tracks)
    coverage = nonempty_frames / total_frames if total_frames else 0.0
    coherence_score = float(min(1.0, (avg_len / max(1.0, total_frames)) * coverage))

    track_spans: Dict[int, Dict[str, int]] = {}
    for tid, frs in track_frame_indices.items():
        if frs:
            track_spans[tid] = {
                "start_frame": int(min(frs)),
                "end_frame": int(max(frs)),
            }

    return {
        "module": "bytetrack",
        "num_tracks": num_tracks,
        "avg_track_length_frames": avg_len,
        "track_lengths": track_lengths,
        "track_frame_indices": track_frame_indices,
        "track_spans": track_spans,
        "track_frame_coverage": coverage,
        "coherence_score": coherence_score,
    }


# ---------------------------------------------------------------------
# RAFT: optical flow (direction + magnitude)
# ---------------------------------------------------------------------
def analyze_optical_flow_raft(
    frames: List[np.ndarray],
    device: torch.device,
    max_pairs: int = 32,
) -> Dict[str, Any]:
    """
    Compute RAFT optical flow statistics.

    - Ensures H and W are divisible by 8 via center-cropping.
    - Catches runtime/shape errors and returns a structured error dict
      instead of crashing the whole script.
    """
    if raft_large is None or Raft_Large_Weights is None:
        return {"error": "torchvision RAFT not available", "module": "raft"}

    if len(frames) < 2:
        return {"error": "need at least 2 frames", "module": "raft"}

    def center_crop_to_multiple_of_8(img: np.ndarray) -> np.ndarray:
        """
        Center-crop an image so that H and W are divisible by 8.
        If the current H/W are already divisible by 8, returns the image unchanged.
        """
        h, w = img.shape[:2]
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        if h8 <= 0 or w8 <= 0:
            return img  # extremely degenerate, but keep it safe

        if h == h8 and w == w8:
            return img

        y0 = max(0, (h - h8) // 2)
        x0 = max(0, (w - w8) // 2)
        return img[y0:y0 + h8, x0:x0 + w8]

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    transform = weights.transforms()

    mags: List[float] = []
    angles: List[float] = []

    num_pairs = min(len(frames) - 1, max_pairs)

    with torch.no_grad():
        for i in range(num_pairs):
            # Ensure RAFT size constraint
            f1 = center_crop_to_multiple_of_8(frames[i])
            f2 = center_crop_to_multiple_of_8(frames[i + 1])

            f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
            f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)

            t1 = torch.from_numpy(f1_rgb).permute(2, 0, 1).float() / 255.0
            t2 = torch.from_numpy(f2_rgb).permute(2, 0, 1).float() / 255.0

            t1 = t1.unsqueeze(0)
            t2 = t2.unsqueeze(0)

            t1, t2 = transform(t1, t2)
            t1 = t1.to(device)
            t2 = t2.to(device)

            list_of_flows = model(t1, t2)
            flow = list_of_flows[-1][0]  # (2, H, W)
            flow_np = flow.detach().cpu().numpy()
            u = flow_np[0]
            v = flow_np[1]
            mag = np.sqrt(u ** 2 + v ** 2)
            ang = np.arctan2(v, u)  # radians

            mags.append(float(mag.mean()))
            angles.append(float(ang.mean()))

    if not mags:
        return {
            "module": "raft",
            "error": "RAFT produced no valid pairs",
        }

    mean_mag = float(np.mean(mags))
    mean_ang_rad = float(np.mean(angles))
    mean_ang_deg = float(math.degrees(mean_ang_rad))

    return {
        "module": "raft",
        "num_pairs": num_pairs,
        "mean_flow_magnitude": mean_mag,
        "mean_flow_direction_degrees": mean_ang_deg,
        "per_pair_magnitude": mags,
        "per_pair_direction_degrees": [float(math.degrees(a)) for a in angles],
    }



# ---------------------------------------------------------------------
# CLIP4Clip: text–video alignment
# ---------------------------------------------------------------------
def build_clip4clip_preprocess(size: int = 224):
    return Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda img: img.convert("RGB"),
            ToTensor(),
            # CLIP mean/std
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def analyze_clip4clip_alignment(
    video_path: str,
    text_prompt: str,
    device: torch.device,
    model_id: str = "Searchium-ai/clip4clip-webvid150k",
    max_frames: int = 32,
    frame_rate: float = 1.0,
) -> Dict[str, Any]:
    if CLIPTokenizer is None or CLIPTextModelWithProjection is None:
        return {"error": "transformers not installed", "module": "clip4clip"}

    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_id).to(device).eval()

    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()

    # Text embedding
    text_inputs = tokenizer(text=text_prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        text_outputs = text_model(
            input_ids=text_inputs["input_ids"].to(device),
            attention_mask=text_inputs["attention_mask"].to(device),
        )
        # Use projected embeddings and L2-normalize
        if hasattr(text_outputs, "text_embeds"):
            text_emb = text_outputs.text_embeds  # (1, D)
        else:
            # Fallback: take CLS token from last_hidden_state
            last_hidden = text_outputs.last_hidden_state  # (B, L, D)
            text_emb = last_hidden[:, 0, :]
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Video frames → CLIP preprocess
    preprocess = build_clip4clip_preprocess(size=224)
    frames, orig_fps = sample_video_frames(
        video_path, max_frames=max_frames, target_fps=frame_rate
    )
    if not frames:
        return {"error": "no frames sampled", "module": "clip4clip"}

    clip_frames: List[torch.Tensor] = []
    for f in frames:
        pil = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil = cv2.resize(pil, (224, 224), interpolation=cv2.INTER_AREA)
        # Wrap as PIL for transforms
        from PIL import Image

        pil_img = Image.fromarray(pil)
        clip_frames.append(preprocess(pil_img))

    video_tensor = torch.stack(clip_frames, dim=0).to(device)  # (T, 3, H, W)

    with torch.no_grad():
        vision_outputs = vision_model(pixel_values=video_tensor)
        if hasattr(vision_outputs, "image_embeds"):
            img_embs = vision_outputs.image_embeds  # (T, D)
        else:
            # Fallback: pooler_output or CLS
            if hasattr(vision_outputs, "pooler_output"):
                img_embs = vision_outputs.pooler_output
            else:
                img_embs = vision_outputs.last_hidden_state[:, 0, :]
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
        video_emb = img_embs.mean(dim=0, keepdim=True)  # (1, D)

    sim = float((text_emb * video_emb).sum().item())

    return {
        "module": "clip4clip",
        "video_path": str(video_path),
        "text_prompt": text_prompt,
        "num_frames_used": len(frames),
        "similarity_cosine": sim,
    }


# ---------------------------------------------------------------------
# LPIPS: perceptual similarity between candidate & reference
# ---------------------------------------------------------------------
def analyze_lpips_similarity(
    frames_a: List[np.ndarray],
    frames_b: List[np.ndarray],
    device: torch.device,
    max_pairs: int = 32,
) -> Dict[str, Any]:
    """
    Compute LPIPS similarity between two frame sequences.
    Handles resolution mismatches by center-cropping both frames
    to the minimum common height/width for each pair.
    """
    if lpips is None:
        return {"error": "lpips not installed", "module": "lpips"}

    n = min(len(frames_a), len(frames_b), max_pairs)
    if n == 0:
        return {"error": "no overlapping frames", "module": "lpips"}

    def center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Center-crop both a and b to (min(Ha,Hb), min(Wa,Wb)).
        """
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        h = min(ha, hb)
        w = min(wa, wb)

        def crop(img: np.ndarray, h: int, w: int) -> np.ndarray:
            H, W = img.shape[:2]
            y0 = max(0, (H - h) // 2)
            x0 = max(0, (W - w) // 2)
            return img[y0:y0 + h, x0:x0 + w]

        return crop(a, h, w), crop(b, h, w)

    loss_fn = lpips.LPIPS(net="vgg").to(device).eval()

    scores: List[float] = []
    with torch.no_grad():
        for i in range(n):
            fa = frames_a[i]
            fb = frames_b[i]

            # Ensure same size via center crop
            if fa.shape[:2] != fb.shape[:2]:
                fa, fb = center_crop_to_common(fa, fb)

            # BGR -> RGB
            fa_rgb = cv2.cvtColor(fa, cv2.COLOR_BGR2RGB)
            fb_rgb = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)

            ta = torch.from_numpy(fa_rgb).permute(2, 0, 1).float() / 255.0
            tb = torch.from_numpy(fb_rgb).permute(2, 0, 1).float() / 255.0

            # LPIPS expects [-1, 1]
            ta = ta * 2.0 - 1.0
            tb = tb * 2.0 - 1.0

            ta = ta.unsqueeze(0).to(device)  # (1,3,H,W)
            tb = tb.unsqueeze(0).to(device)

            d = loss_fn(ta, tb)  # (1,1,H',W') or (1,1)
            scores.append(float(d.mean().item()))

    return {
        "module": "lpips",
        "num_pairs": n,
        "lpips_mean": float(np.mean(scores)),
        "lpips_per_pair": scores,
    }


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Physics-aware video analysis using GroundingDINO, ByteTrack, RAFT, CLIP4Clip, and LPIPS."
    )
    ap.add_argument(
        "--video",
        required=True,
        help="Path to generated / candidate video",
    )
    ap.add_argument(
        "--ref_video",
        default=None,
        help="Optional reference video",
    )
    ap.add_argument(
        "--text_prompt",
        required=True,
        help="Text description / prompt of the expected phenomenon (for CLIP4Clip & GroundingDINO)",
    )
    ap.add_argument(
        "--modules",
        default="grounding,bytetrack,raft,clip4clip,lpips",
        help="Comma-separated subset of {grounding,bytetrack,raft,clip4clip,lpips}",
    )
    ap.add_argument(
        "--grounding_cfg",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="GroundingDINO config path",
    )
    ap.add_argument(
        "--grounding_ckpt",
        default="judge/cv_tools/weights/groundingdino_swint_ogc.pth",
        help="GroundingDINO weights checkpoint path",
    )
    ap.add_argument(
        "--output_json",
        default="judge/cv_tools/video_metrics_report.json",
        help="Where to write JSON report",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=128,
        help="Max frames to sample from video",
    )

    args = ap.parse_args()
    device = ensure_device(args.device)

    video_path = str(args.video)
    ref_video_path = str(args.ref_video) if args.ref_video else None

    modules = {m.strip().lower() for m in args.modules.split(",") if m.strip()}

    report: Dict[str, Any] = {
        "video": video_path,
        "ref_video": ref_video_path,
        "text_prompt": args.text_prompt,
        "device": str(device),
        "modules_requested": sorted(modules),
        "results": {},
    }

    # Pre-sample frames once for modules that need raw frames
    frames, orig_fps = sample_video_frames(
        video_path, max_frames=args.max_frames, target_fps=None
    )
    report["video_fps"] = orig_fps
    report["num_frames_sampled"] = len(frames)

    # -----------------------------------------------------------------
    # Grounding DINO
    # -----------------------------------------------------------------
    grounding_result = None
    if "grounding" in modules:
        grounding_result = analyze_grounding_dino(
            video_path=video_path,
            text_prompt=args.text_prompt,
            cfg_path=args.grounding_cfg,
            ckpt_path=args.grounding_ckpt,
            device=device,
            max_frames=min(args.max_frames, 16),
        )
        report["results"]["grounding_dino"] = grounding_result

    # -----------------------------------------------------------------
    # ByteTrack (now uses GroundingDINO detections, not stub boxes)
    # -----------------------------------------------------------------
    if "bytetrack" in modules:
        if (
            grounding_result
            and isinstance(grounding_result, dict)
            and grounding_result.get("detections_per_frame")
        ):
            dets = grounding_result["detections_per_frame"]
            frame_h, frame_w = 0, 0
            fs = grounding_result.get("frame_size")
            if isinstance(fs, (list, tuple)) and len(fs) == 2:
                frame_h, frame_w = int(fs[0]), int(fs[1])
            elif frames:
                frame_h, frame_w = frames[0].shape[0], frames[0].shape[1]

            bytetrack_result = analyze_bytetrack_from_detections(
                detections_per_frame=dets,
                frame_size=(frame_h, frame_w),
                fps=orig_fps or 24.0,
            )
        else:
            bytetrack_result = {
                "module": "bytetrack",
                "error": "GroundingDINO must run first and provide detections_per_frame",
            }

        report["results"]["bytetrack"] = bytetrack_result

    # -----------------------------------------------------------------
    # RAFT optical flow
    # -----------------------------------------------------------------
    if "raft" in modules:
        raft_result = analyze_optical_flow_raft(frames, device=device)
        report["results"]["raft"] = raft_result

    # -----------------------------------------------------------------
    # CLIP4Clip text–video alignment
    # -----------------------------------------------------------------
    if "clip4clip" in modules:
        clip_result = analyze_clip4clip_alignment(
            video_path=video_path,
            text_prompt=args.text_prompt,
            device=device,
        )
        report["results"]["clip4clip"] = clip_result

    # -----------------------------------------------------------------
    # LPIPS similarity between candidate and reference
    # -----------------------------------------------------------------
    if "lpips" in modules:
        if not ref_video_path:
            report["results"]["lpips"] = {
                "module": "lpips",
                "error": "ref_video is required for LPIPS",
            }
        else:
            ref_frames, _ = sample_video_frames(
                ref_video_path, max_frames=args.max_frames, target_fps=None
            )
            lpips_result = analyze_lpips_similarity(
                frames_a=frames, frames_b=ref_frames, device=device
            )
            report["results"]["lpips"] = lpips_result

    # -----------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote analysis report to: {out_path}")


if __name__ == "__main__":
    main()
