# ScienceAtlas

## Roadmap

I will update the codebase to include:

1. data processing to extract from existing VLM reasoning benchmarks: (1) text captions as our prompts; (2) input videos as our first frame and reference ground truth. 
2. more video generation providers (beyond Sora, and Replicate).

## Usage

launch video generation:
```
python3 frontend.py --provider {provider_name} --model {model_name} --prompt {customized_prompt}
```