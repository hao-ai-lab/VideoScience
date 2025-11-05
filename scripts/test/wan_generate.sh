# WAN 2.5 (intl region)

python3 single_generation_frontend.py \
  --provider wan \
  --model wan2.5-t2v-preview \
  --prompt "a tiny robot chef flipping pancakes in zero gravity, soft cinematic lighting" \
  --seconds 8 \
  --width 1280 --height 720 \
  --out "out/wan2.5/wan2.5_1280x720_8s.mp4" \
  --extra '{"prompt_extend": true, "audio": false, "negative_prompt": "low quality, artifacts"}'
