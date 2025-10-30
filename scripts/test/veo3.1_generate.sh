python3 frontend.py \
  --provider gemini \
  --model veo-3.1-generate-preview \
  --prompt "a tiny robot chef flipping pancakes in zero gravity, soft cinematic lighting" \
  --seconds 8 \
  --width 1280 \
  --height 720 \
  --extra '{"aspect_ratio":"9:16","resolution":"720p","negative_prompt":"cartoon, low quality"}' \
  --out out/gemini-veo3.1/gemini_veo3.1_1280x720_8s.mp4