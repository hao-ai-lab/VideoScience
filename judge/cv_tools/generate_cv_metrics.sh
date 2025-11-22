python3 judge/cv_tools/cv_metrics.py \
  --video 'judge/data/evaluation_videos/bytedance-seedance-1-pro/lanxiang/vid_001_run_1.mp4' \
  --ref_video 'judge/data/reference_videos/vid_001_ref.mp4' \
  --text_prompt "A person is standing next to a brown table with a hand-crank that is connected to a generator and a light bulb. He begins continuously turning the hand-crank." \
  --modules grounding,bytetrack,raft,clip4clip,lpips \
  --grounding_cfg GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounding_ckpt judge/cv_tools/weights/groundingdino_swint_ogc.pth \
  --output_json judge/cv_tools/reports/tests/generated_vid_metrics.json
