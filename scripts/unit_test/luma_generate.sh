mkdir -p out/luma-ray2

python3 single_generation_frontend.py \
  --provider ray \
  --model ray-2 \
  --prompt "a tiny robot chef flipping pancakes in zero gravity, soft cinematic lighting" \
  --seconds 9 \
  --width 1280 \
  --height 720 \
  --extra '{"aspect_ratio":"16:9","resolution":"720p","loop":false}' \
  --out out/luma-ray2/ray2_1280x720_8s.mp4
