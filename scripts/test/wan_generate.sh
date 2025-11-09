# WAN 2.5 (intl region)

python3 single_generation_frontend.py \
  --provider wan \
  --model wan2.5-t2v-preview \
  --prompt "A glass bottle is filled completely with water. The top of the bottle is struck straight down with a hammer." \
  --seconds 8 \
  --width 1280 --height 720 \
  --out "out/wan2.5/wan2.5_1280x720_8s_striking_bottle.mp4" \
  --extra '{"prompt_extend": true, "audio": false, "negative_prompt": "low quality, artifacts"}'
