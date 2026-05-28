# Natural Experiment

Runs the natural experiment for measuring the factual generation-verification (GV) gap in frontier models. Supports four datasets out of the box: `nba_scores`, `market_data`, `lottery_data`, `billboard_100`.

## 1. Prepare the data

Run the data-preparation notebook to fetch and clean the raw source files:

```
src/origins/notebooks/natural_experiment_data_prep.ipynb
```

The notebook writes CSVs into a local data directory (default: `data/historical_gvg_datasets/`).

## 2. Update the dataset config

Point each dataset entry to your prepared CSV paths in:

```
src/origins/configs/naturalistic_datasets.yaml
```

Example:

```yaml
nba_scores:      data/historical_gvg_datasets/nba_data_83-25.csv
market_data:     data/historical_gvg_datasets/market_data/snp.csv
lottery_data:    data/historical_gvg_datasets/mega_millions_processed.csv
billboard_100:   data/historical_gvg_datasets/billboard_100_00-25.csv
```

## 3. Run

```bash
python -m src.origins.natural_experiment \
    --dataset="billboard_100" \
    --model_name="gemini-3-flash" \
    --judge_model_name="gemini-3.1-flash-lite" \
    --start_year=2002 \
    --end_year=2024 \
    --num_data_points_per_year=50 \
    --job_name="example"
```

Results are written under `--save_folder` (default `data/historical_gvg_results_v2`) into a job-specific sub-folder.

### Key parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--dataset` | `"billboard_100"` | One of `nba_scores`, `market_data`, `lottery_data`, `billboard_100`. |
| `--model_name` | `"gemini-3-flash"` | Generation/verification model. |
| `--judge_model_name` | `"gemini-3.1-flash-lite"` | Judge model. |
| `--start_year` / `--end_year` | `2002` / `2024` | Year range. |
| `--num_data_points_per_year` | `50` | Sampled datapoints per year. |
| `--run_generation` | `True` | Run the generation phase. |
| `--run_verification` | `True` | Run the verification phase. |
| `--run_verification_with_noise` | `True` | Add noisy distractors during verification. |
| `--job_name` | `None` | Sub-folder name under `--save_folder`. |
| `--seed` | `42` | Sampling seed. |
