```

---

### Notes

* **OpenAI** path uses the Responses API with `input_image` parts containing **base64 data URLs** for sampled frames. This mirrors the cookbook approach of sampling frames for video understanding. You can trim `max_images` via `extra` to control cost/latency.
* **Gemini** path uploads full video files with the **Files API** and requests **JSON output**, leveraging native video understanding. Use models like `gemini-2.5-flash` or later.
* **Anthropic** path sends **base64 images** (sampled frames) to Messages API as `image` parts.
* **Replicate** path is a lightweight fallback that queries an image VLM (e.g., LLaVA) per-frame and aggregates. It’s less precise but easy to run anywhere.

### Quick smoke tests

```bash
# OpenAI (frames → JSON)
python frontend.py \
  --provider openai --model gpt-4o \
  --video demos/candidate.mp4 \
  --phenomenon "Double-slit interference" \
  --description "Waves diffract through two slits, creating an interference pattern with alternating bright and dark fringes on the screen behind." \
  --ref_video demos/reference.mp4 \
  --md_out out/report.md \
  --json_out out/report.json

# Gemini (native video)
python frontend.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --video demos/candidate.mp4 \
  --phenomenon "Convection currents" \
  --description "Heated fluid near the source becomes less dense and rises; cooler fluid sinks, forming a continuous circulation." \
  --md_out out/report.md

# Anthropic (frames)
python frontend.py \
  --provider anthropic \
  --model claude-3-5-sonnet-20240620 \
  --video demos/candidate.mp4 \
  --phenomenon "Electrolysis of water" \
  --description "Passing electric current through water splits it into hydrogen (at cathode) and oxygen (at anode)."