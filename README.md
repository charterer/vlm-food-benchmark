# Benchmarking Vision-Language Models for Automated Food Recognition

Code for reproducing the benchmark reported in *Sterling et al., Benchmarking Vision-Language Models for Automated Food Recognition* (under review).

We evaluate ten approaches (four Gemini models, three GPT models, Claude Haiku 4.5, Qwen2-VL-7B, and FatSecret) on 3,229 filtered dishes from Google's Nutrition5k dataset, measuring calorie and weight estimation accuracy and ingredient-set overlap. We also run a prompt-engineering ablation (P1–P5) and a human-validation study via the `minigame/` web tool.

## Repository layout

```
pipeline/code/            Evaluation pipeline
  evaluate_nutrition5k.py   Main script: runs a provider on a subset of Nutrition5k, writes a per-dish CSV
  evaluate_nutrition5k_from_csv.py, post_metrics_from_csv.py,
  recompute_jaccard.py, jaccard_nonfood_sensitivity.py,
  categorize_tokens.py      Post-hoc metric computation from saved CSVs
  analyzers/                Provider adapters (gemini, openai, anthropic, qwen2vl, fatsecret)
  requirements.txt          Analysis-side Python dependencies
paper/code/
  generate_figures.py       Builds Figs 1–7 and supplementary figures from benchmark CSVs
minigame/                 Flask web annotation tool (Fig 6)
  app.py, static/, templates/, requirements.txt
```

## Quick start

```bash
# 1. Install dependencies
pip install -r pipeline/code/requirements.txt

# 2. Download Nutrition5k (see https://github.com/google-research-datasets/Nutrition5k)
#    Extract so that $DATASET_ROOT/metadata/ and $DATASET_ROOT/imagery/realsense_overhead/ exist.

# 3. Set API keys (only for the providers you plan to run)
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export FATSECRET_API_KEY="client_id:client_secret"

# 4. Run the benchmark for one provider (P3 prompt = best single-pass variant in the paper)
python pipeline/code/evaluate_nutrition5k.py \
    --dataset-root $DATASET_ROOT \
    --provider gemini-3.0 \
    --prompt-variant P3 \
    --output-csv results/gemini_3.0_P3.csv
```

Per-provider CSVs go in `pipeline/outputs/benchmark_results/` (or wherever `--output-csv` points). To regenerate the paper figures, point `generate_figures.py` at that directory:

```bash
export N5K_RESULTS_DIR=pipeline/outputs/benchmark_results
export N5K_DATASET_DIR=$DATASET_ROOT/imagery/realsense_overhead
python paper/code/generate_figures.py
```

Figures are written to `paper/figures/`.

## Supported providers

| Provider arg | Default model | Env var |
|---|---|---|
| `gemini-2.0` / `gemini-2.5` / `gemini-3.0` / `gemini-3.1` | `gemini-{version}-flash` (or `-flash-lite` for 3.1) | `GEMINI_API_KEY` |
| `gpt-4o` / `gpt-4o-mini` / `gpt-5-mini` | same | `OPENAI_API_KEY` |
| `haiku` / `claude` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| `qwen` | `Qwen/Qwen2-VL-7B-Instruct` (local, needs GPU) | — |
| `fatsecret` | — | `FATSECRET_API_KEY=client_id:client_secret` |

Qwen2-VL requires `transformers` and `torch`; install separately with `pip install torch transformers accelerate`.

## Human validation tool

The web annotation tool used for Figure 6 lives in `minigame/`. It serves dish images with merged ground-truth + AI ingredient lists and collects per-ingredient approve/reject/unsure judgments. See `minigame/requirements.txt` and `minigame/run.sh`.

## Data availability

Nutrition5k is released by Google Research at <https://github.com/google-research-datasets/Nutrition5k>. Per-dish prediction CSVs for the ten models benchmarked in the paper are available on request.

## License

MIT — see LICENSE.
