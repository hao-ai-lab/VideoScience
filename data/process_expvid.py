"""
process_exvid.py (LOCAL ExpVid reader)

Reads a local ExpVid layout:

  <DATA_ROOT>/
    annotations/level1/ level2/ level3/
    videos/level1/      level2/ level3/

Keeps only videos with duration <= --max-duration (default: 10s), pairs each with
its text annotations (QA + ASR if present), writes a JSONL manifest, and emits a
tiny test subset (N examples) with copied media.

Example:
  python process_exvid.py \
    --data-root /home/lah003/data/ExpVid \
    --out-dir ./curated_expvid \
    --num-examples 5 \
    --max-duration 10
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

LEVEL_DIRS = ["level1", "level2", "level3"]

# utils
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ffprobe_duration(path: Path) -> Optional[float]:
    """Return duration in seconds using ffprobe; None on failure."""

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True
    )
    return float(out.stdout.strip())

def safe_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

def write_jsonl(rows: Iterable[Dict], out_path: Path):
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# annotation loading
FIELD_ALIASES = {
    "video":     ["video_path", "video", "path", "file", "clip_path", "video_rel"],
    "question":  ["question", "q"],
    "options":   ["options", "choices"],
    "answer":    ["answer", "ans", "label"],
    "asr":       ["asr_caption", "asr", "transcript", "subtitle", "captions"],
    "category":  ["category", "discipline", "topic"],
    "id":        ["id", "sample_id", "uid"],
    "video_id":  ["video_id", "vid", "source_id"],
}

def _normalize_keys(d: Dict) -> Dict:
    lower = {k.lower(): v for k, v in d.items()}
    out = {}
    for canon, alts in FIELD_ALIASES.items():
        for a in alts:
            if a in lower:
                out[canon] = lower[a]
                break
    # keep originals too (don’t drop unknown fields)
    for k, v in d.items():
        out.setdefault(k, v)
    return out

def read_any_annotations(ann_file: Path) -> List[Dict]:
    """Read JSON / JSONL / CSV into a list of normalized dicts."""
    rows: List[Dict] = []
    suf = ann_file.suffix.lower()

    if suf in [".jsonl", ".jsonl.gz"]:
        import gzip
        opener = gzip.open if suf.endswith(".gz") else open
        with opener(ann_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                rows.append(_normalize_keys(obj))

    elif suf == ".json":
        obj = json.loads(ann_file.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                seq = obj["data"]
            elif "annotations" in obj and isinstance(obj["annotations"], list):
                seq = obj["annotations"]
            else:
                seq = [obj]
        elif isinstance(obj, list):
            seq = obj
        else:
            seq = []
        for rec in seq:
            if isinstance(rec, dict):
                rows.append(_normalize_keys(rec))

    elif suf == ".csv":
        import csv
        with ann_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                rows.append(_normalize_keys(rec))

    return rows

def iterate_local_annotations(ann_root: Path) -> List[Dict]:
    """Yield rows from annotations/level{1,2,3}/*.(json|jsonl|csv), tagged with level/subset."""
    all_rows: List[Dict] = []
    for level in LEVEL_DIRS:
        level_dir = ann_root / level
        if not level_dir.exists():
            continue
        for ann_file in sorted(level_dir.rglob("*")):
            if ann_file.suffix.lower() not in [".json", ".jsonl", ".csv"]:
                continue
            subset = ann_file.stem  # level1_materials
            for rec in read_any_annotations(ann_file):
                rec["_level"] = level
                rec["_subset"] = subset
                all_rows.append(rec)
    return all_rows

# path resolving
def find_video_path(data_root: Path, level: str, video_field: Optional[str]) -> Optional[Path]:
    """Resolve the actual video file path on disk."""

    p = Path(video_field)

    if p.is_absolute() and p.exists():
        return p

    # relative to data root (e.g., "videos/level1/.../clip.mp4")
    cand = data_root / p
    if cand.exists():
        return cand

    return None

def build_text_annotation(row: Dict) -> str:
    """Compose a compact text string combining QA + ASR."""
    parts = []
    q = row.get("question")
    if q:
        parts.append(f"Question: {q}")

    opts = row.get("options")
    if isinstance(opts, dict):
        parts.append("Options: " + " ".join([f"{k}. {v}" for k, v in opts.items()]))
    elif isinstance(opts, list):
        parts.append("Options: " + " ".join([f"{chr(65+i)}. {v}" for i, v in enumerate(opts)]))

    a = row.get("answer")
    if a not in (None, ""):
        parts.append(f"Answer: {a}")

    asr = row.get("asr")
    if asr:
        parts.append(f"ASR: {asr}")

    return " | ".join(parts) if parts else ""

# main worker
def curate_exvid_local(data_root: Path, out_dir: Path, max_duration: float) -> List[Dict]:
    ann_root = data_root / "annotations"
    vid_root = data_root / "videos"
    assert ann_root.exists() and vid_root.exists(), "Expect annotations/ and videos/ under --data-root"

    videos_out = out_dir / "videos"
    ensure_dir(videos_out)

    curated: List[Dict] = []
    rows = iterate_local_annotations(ann_root)

    for row in tqdm(rows, desc="ExpVid (examples scanned)"):
        level  = row.get("_level", "level1")
        subset = row.get("_subset", "unknown")

        # find video field
        video_field = None
        for k in FIELD_ALIASES["video"]:
            if k in row:
                video_field = row[k]
                break

        vid_path = find_video_path(data_root, level, video_field)
        if not vid_path or not vid_path.exists():
            continue

        dur = ffprobe_duration(vid_path)
        if dur is not None and dur > max_duration:
            continue

        dst = videos_out / level / subset / vid_path.name
        safe_copy(vid_path, dst)

        text_joined = build_text_annotation({
            "question": row.get("question"),
            "options": row.get("options"),
            "answer": row.get("answer"),
            "asr": row.get("asr"),
        })

        curated.append({
            "dataset": "ExpVid",
            "level": level,
            "subset": subset,
            "id": row.get("id"),
            "video_id": row.get("video_id"),
            "category": row.get("category"),
            "video": str(dst.relative_to(out_dir)),
            "duration_sec": dur,
            "text_annotation": text_joined,
            "qa": {
                "question": row.get("question"),
                "options": row.get("options") if isinstance(row.get("options"), (dict, list)) else {},
                "answer": row.get("answer"),
                "asr_caption": row.get("asr"),
            },
            "license": "cc-by-nc-4.0"
        })

    write_jsonl(curated, out_dir / "metadata_expvid.jsonl")
    print(f"[OK] ExpVid curated items: {len(curated)}")
    print(f"[OK] Wrote: {out_dir/'metadata_expvid.jsonl'}")
    return curated

def sample_and_copy(rows: List[Dict], out_dir: Path, n: int, seed: int = 1234):
    import random
    if not rows:
        return
    random.seed(seed)
    pick = rows if len(rows) <= n else random.sample(rows, n)
    samp_dir = out_dir / "samples"
    samp_vid = samp_dir / "videos"
    ensure_dir(samp_vid)
    out_rows = []
    for r in pick:
        src_rel = r["video"]
        src_abs = out_dir / src_rel
        dst = samp_vid / Path(src_rel).name
        safe_copy(src_abs, dst)
        r2 = dict(r)
        r2["video"] = str(Path("samples") / "videos" / dst.name)
        out_rows.append(r2)
    write_jsonl(out_rows, samp_dir / "samples.jsonl")
    print(f"[OK] Wrote {len(out_rows)} testing examples to {samp_dir/'samples.jsonl'}")

# =========================== CLI ===========================
def main():
    ap = argparse.ArgumentParser("Process local ExpVid; keep clips <= max-duration and create manifests.")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Path to local ExpVid (contains annotations/ and videos/).")
    ap.add_argument("--out-dir", dest="out_dir", type=Path, required=True,
                    help="Output directory.")
    ap.add_argument("--num-examples", type=int, default=5,
                    help="Testing examples to sample (default: 5).")
    ap.add_argument("--max-duration", type=float, default=10.0,
                    help="Max clip duration in seconds (default: 10).")
    ap.add_argument("--seed", type=int, default=1234,
                    help="Sampling seed (default: 1234).")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    curated = curate_exvid_local(args.data_root, args.out_dir, args.max_duration)
    if curated:
        sample_and_copy(curated, args.out_dir, args.num_examples, seed=args.seed)

if __name__ == "__main__":
    main()

