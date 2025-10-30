# ScienceAtlas

## Roadmap

I will update the codebase to include:

1. data processing to extract from existing VLM reasoning benchmarks (currently supporting: preliminary filtering for ExpVid).
2. more video generation providers (currently supporting: OpenAI/Azure Sora, Replicate, Google Veo, Alibaba Wan, Kling).

feel free to make your own branch and PR for more features!

## Usage

### launch batched video generation
1. Download csv data file from Notion.
2. Launch script:
```
# set config in scripts/evaluate_by_author_using_csv.sh
bash scripts/evaluate_by_author_using_csv.sh
```

### launch single video generation
1. Launch script:
```
python3 frontend.py --provider {provider_name} --model {model_name} --prompt {customized_prompt}
```