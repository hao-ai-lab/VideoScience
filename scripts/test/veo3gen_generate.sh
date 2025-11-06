mkdir -p out/veo3gen-quality

python3 single_generation_frontend.py \
  --provider veo3gen \
  --model veo3-quality \
  --prompt "a tiny robot chef flipping pancakes in zero gravity, soft cinematic lighting" \
  --seconds 8 \
  --width 1920 \
  --height 1080 \
  --extra '{"modelVersion":"3.1","aspect_ratio":"16:9","resolution":"1080p","negative_prompt":"cartoon, low quality", "enhancePrompt":false, "audio":false}' \
  --out out/veo3gen-quality/veo3gen_quality_1920x1080_8s.mp4