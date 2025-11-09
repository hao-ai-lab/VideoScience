# Kling v2.5-turbo/pro (AIMLAPI)

python3 single_generation_frontend.py \
  --provider kling \
  --model klingai/v2.5-turbo/pro/text-to-video \
  --prompt "A stream of liquid water is heated far above its boiling point ($100^{\circ} \text{C}$) in a depressurized chamber, and salt is then added." \
  --seconds 8 \
  --width 1280 \
  --height 720 \
  --out "out/kling2.5/kling2.5_1280x720_8s_depressureized_heating.mp4" \
  --extra '{"aspect_ratio": "16:9", "cfg_scale": 0.9}'d
