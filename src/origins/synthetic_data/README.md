# Synthetic Data Generator

Generates synthetic factual datapoints organized as `(topic, relationship, instantiations, sentences, questions)` for use in fine-tuning and evaluation pipelines.

Each datapoint is produced in four sequential LLM calls:

1. **Ingredients** — a `(topic, relationship)` pair within a topic category.
2. **Instantiations** — `real_pairs` (factual head/tail pairs) and `imaginary_pairs` (fictional ones).
3. **Sentences** — training-style natural-language statements.
4. **Tasks** — question/answer pairs.

Imaginary names are deduplicated across the entire run via an atomic, thread-safe registry, so fictional entities stay unique across topics.

## Run

```bash
python -m src.origins.synthetic_data.synthetic_data_generator \
    --topics "['politics','medicine','science','religion','society','social bias']" \
    --num_data_points 25 \
    --num_instantiations 5 \
    --num_sentences 10 \
    --num_tasks 5 \
    --model_name "gemini-2.5-flash" \
    --temperature 1.0 \
    --sub_folder "synth_dataset_v3"
```

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--topics` | `"medicine"` | Topic or list of topic categories. |
| `--num_data_points` | `1` | Datapoints per topic. |
| `--num_instantiations` | `5` | Real + imaginary pairs per datapoint. |
| `--num_sentences` | `10` | Sentences per datapoint. |
| `--num_tasks` | `5` | QA pairs per datapoint. |
| `--model_name` | `"gemini-3-flash"` | Generator model. |
| `--temperature` | `1.0` | Sampling temperature. |
| `--save_folder` | `data/synthetic_data` | Base output directory. |
| `--sub_folder` | `None` | Optional sub-directory under `save_folder`. |
| `--max_concurrent_jobs` | `5` | Parallel topics. |
| `--num_retries_per_step` | `3` | Retries per generation step on format/uniqueness failure. |
| `--verbosity_level` | `1` | `0`=warnings only, `1`=info, `2`=debug. |

## Output

```
<save_folder>/<sub_folder>/<topic>/<timestamp>/dp1.yaml, dp2.yaml, ...
```

If imaginary-name duplicates remain after generation, a `duplicate_report.txt` is written into `<save_folder>/<sub_folder>/` listing the offending files. The run does not raise — edit the listed files in place and re-verify:

```bash
python -m src.origins.synthetic_data.check_synth_data \
    --yaml_path=<path/to/combined.yaml>
```

## Format for Training

Once the synthetic data has been generated, run the following command to format the dataset into the YAML files ready to be used for training:

```bash
python3 -m origins.synthetic_data.finalize_yaml_files_for_synthetic_data \
    --input_root data/synthetic_data/synth_dataset_v4/updated \
    --output_root data/synthetic_data/synth_dataset_v4_formatted \
    --real_control_relationship_tail=True --real_control_relationship_heads=True \
    --train_data_type sentences
```

Adjust `--input_root` / `--output_root` to match the dataset version you generated above.
