# process_expvid.sh
# example use:
#   bash process_expvid.sh -r /home/lah003/data/ExpVid -o ./curated_expvid -n 5 -d 10 -s 1234
set -euo pipefail

DATA_ROOT=""
OUT_DIR="data/curated_expvid"
NUM_EXAMPLES=5
MAX_DURATION=10
SEED=1234

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/process_expvid.py"

usage() {
  cat <<EOF
Usage: bash process_expvid.sh -r DATA_ROOT [-o OUT_DIR] [-n NUM] [-d MAX_SEC] [-s SEED] [--python PATH]
  -r DATA_ROOT   Path to local ExpVid (must contain annotations/ and videos/)
  -o OUT_DIR     Output directory (default: ${OUT_DIR})
  -n NUM         Testing examples to sample (default: ${NUM_EXAMPLES})
  -d MAX_SEC     Max clip duration in seconds (default: ${MAX_DURATION})
  -s SEED        Sampling seed (default: ${SEED})
  -h             Help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r) DATA_ROOT="$2"; shift 2 ;;
    -o) OUT_DIR="$2"; shift 2 ;;
    -n) NUM_EXAMPLES="$2"; shift 2 ;;
    -d) MAX_DURATION="$2"; shift 2 ;;
    -s) SEED="$2"; shift 2 ;;
    -h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Error: -r DATA_ROOT is required (e.g., -r /home/lah003/data/ExpVid)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

python3 "${PY_SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --num-examples "${NUM_EXAMPLES}" \
  --max-duration "${MAX_DURATION}" \
  --seed "${SEED}"
