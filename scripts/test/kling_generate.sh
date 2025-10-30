# Kling v2.5-turbo/pro (AIMLAPI)

python3 frontend.py \
  --provider kling \
  --model klingai/v2.5-turbo/pro/text-to-video \
  --prompt "a tiny robot chef flipping pancakes in zero gravity, soft cinematic lighting" \
  --seconds 8 \
  --width 1280 --height 720 \
  --out "out/kling2.5/kling2.5_1280x720_8s.mp4" \
  --extra '{"aspect_ratio": "16:9", "cfg_scale": 0.9}'
