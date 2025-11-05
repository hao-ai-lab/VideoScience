python single_generation_frontend.py \
  --provider sora-openai \
  --model sora-2 \
  --prompt "A stream of liquid water is heated far above its boiling point ($100^{\circ} \text{C}$) in a depressurized chamber, and salt is then added." \
  --seconds 8 \
  --width 1280 \
  --height 720 \
  --timeout_s 3000 \
  --out out/sora-2/sora_openai_1280x720_8s_depressureized_heating.mp4
