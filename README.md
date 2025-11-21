# ScienceAtlas

## Usage

### launch batched video generation
1. Download csv data file under `data/database/data.csv`.
2. Launch script:
```
bash scripts/batched_generation_using_csv.sh
```

### launch single video generation
1. Launch script:
```
python3 single_generation_frontend.py --provider {provider_name} --model {model_name} --prompt {customized_prompt}
```

### VLM as a judge
```
bash judge/batched_evaluate_all_models.sh
```
