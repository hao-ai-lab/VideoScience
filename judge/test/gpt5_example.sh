PROVIDER="openai"
MODEL="gpt-5-pro"
VIDEO="judge/data/sora-2/sora_openai_1280x720_8s_depressureized_heating.mp4"
REF_VIDEO="judge/ref/murray_superheat_water_001.mp4"

OUT_DIR="judge/results/gpt5pro-judging-sora2"
mkdir -p "$OUT_DIR"

python3 judge/vlm_as_a_judge.py \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --video "$VIDEO" \
  --description 'A stream of liquid water is heated far above its boiling point ($100^{\circ} \text{C}$), and salt is then added.' \
  --phenomenon 'The highly unstable water will instantly and violently flash into steam when nucleation sites are introduced, causing an explosive boiling.' \
  --ref_video "$REF_VIDEO" \
  --md_out "$OUT_DIR/report.md" \
  --json_out "$OUT_DIR/report.json"
