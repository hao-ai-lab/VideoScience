mkdir -p out/veo3gen-quality

python3 single_generation_frontend.py \
  --provider veo3gen \
  --model veo3-quality \
  --prompt "A glass bottle is filled completely with water. The top of the bottle is struck straight down with a hammer." \
  --seconds 8 \
  --width 1920 \
  --height 1080 \
  --extra '{"modelVersion":"3.1","aspect_ratio":"16:9","resolution":"1080p","negative_prompt":"cartoon, low quality", "enhancePrompt":false, "audio":false}' \
  --out out/veo3gen-quality/veo3gen_quality_1920x1080_8s_glass_shattering.mp4
