# TODO: make models configurable

python3 judge/vlm_as_a_judge.py \
    --provider openai \
    --model gpt-4o \
    --video judge/data/sora-2/sora_openai_1280x720_8s_depressureized_heating.mp4 \
    --description "A stream of liquid water is heated far above its boiling point ($100^{\circ} \text{C}$) in a depressurized chamber, and salt is then added." \
    --phenomenon "The highly unstable water will instantly and violently flash into steam when nucleation sites are introduced, causing an explosive boiling." \
    --ref_video judge/ref/murray_superheat_water_001.mp4 \
    --md_out judge/results/4o-judging-sora2/report.md \
    --json_out judge/results/4o-judging-sora2/report.json
