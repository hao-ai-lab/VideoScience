#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path

NNN = re.compile(r"^\d{3}$")

def uid_to_name(uid: str, prefix: str, pad: int) -> str:
    uid = (uid or "").strip()
    if uid.isdigit():
        return f"{prefix}{int(uid):0{pad}d}"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", uid).strip("-") or "uid"
    return f"{prefix}{safe}"

def main():
    ap = argparse.ArgumentParser(description="Rename 001,002,... to CSV Unique IDs after filtering by author(s).")
    ap.add_argument("--base-dir")
    ap.add_argument("--csv-path")
    ap.add_argument("--author", default="", help="Comma-separated authors; empty = all")
    ap.add_argument("--author-column-name", default="Author")
    ap.add_argument("--id-column-name", default="Unique ID")
    ap.add_argument("--prefix", default="vid_")
    ap.add_argument("--pad", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve()
    assert base.is_dir(), f"Base directory not found: {base}"
    assert csv_path.is_file(), f"CSV not found: {csv_path}"

    dirs = sorted([p for p in base.iterdir() if p.is_dir() and NNN.match(p.name)], key=lambda p: int(p.name))

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        cols = {h.lower(): h for h in (r.fieldnames or [])}
        akey = cols.get(args.author_column_name.lower(), args.author_column_name)
        ikey = cols.get(args.id_column_name.lower(), args.id_column_name)

        authors = {a.strip() for a in args.author.split(",") if a.strip()} if args.author else None
        ids = []
        for row in r:
            if authors and (row.get(akey, "").strip() not in authors):
                continue
            uid = (row.get(ikey, "") or "").strip()
            if uid:
                ids.append(uid)

    n = min(len(dirs), len(ids))
    plan = [(d, base / uid_to_name(uid, args.prefix, args.pad)) for d, uid in zip(dirs[:n], ids[:n])]

    print("Planned renames:")
    for s, d in plan:
        print(f"  {s.name} -> {d.name}")
    if args.dry_run:
        return

    # two-phase rename (no try/except)
    temps = [base / f".tmp_{s.name}" for s, _ in plan]
    for (s, _), t in zip(plan, temps):
        s.rename(t)
    for t, (_, dst) in zip(temps, plan):
        t.rename(dst)

    print(f"\nDone. Renamed {len(plan)} folders under {base}.")

if __name__ == "__main__":
    main()
