# To Judge or Not to Judge: Can Large Language Models Leverage the Dispute Focus in Legal Judgment?

Code and dataset for **To Judge or Not to Judge: Can Large Language Models Leverage the Dispute Focus in Legal Judgment?** (ACL 2026).

Luoming Hu, Liang Yang, Jingjie Zeng, and Zijie Xing.

[中文说明](README_zh.md) · [Paper](https://aclanthology.org/2026.acl-long.1410/) · [Data format](docs/DATA_CARD.md)

## Dataset usage registration

**If you use this dataset, please voluntarily complete our dataset usage registration form.** This helps us understand its research use and share updates.

**Registration form: [Link](https://wj.qq.com/s2/28588119/zhtv/).**

Registration is voluntary and is not an additional license condition. Please cite the paper when using this work.

## Installation

Use Python 3.10+ in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

This installs the dependencies for inference, evaluation and LoRA training. Choose a PyTorch/CUDA build suitable for your hardware. Model and metric weights are downloaded separately.

## Tasks

| `--experiment` | Task |
| --- | --- |
| `pre_judgment` | RQ1: reasoning, judgment and class prediction from complaint, defense, evidence and court debate |
| `fact_generation` | RQ2: fact generation from pre-judgment inputs |
| `post_fact_judgment` | RQ2: reasoning, judgment and class prediction using established facts and laws |
| `structured_judgment` | RQ3: FocalJudge fact generation and post-fact judgment |

Experiments use temperature 0 and one reference case. Any model served through an OpenAI-compatible endpoint can be selected with `--model`, including locally deployed models. The paper evaluates Claude-3.5-Sonnet, DeepSeek-v3, Gemini-3-Pro, Grok-4, GPT-4o, Gemma3-12B, Llama-3.1-8B-Instruct, Qwen3-8B, ChatLaw and LexiLaw; use the exact model ID provided by your deployment.

The RQ2 tasks additionally support `--strategy cot` for the four-step prompt in Figure 13. The default is `--strategy basic`; `--focus` independently controls the dispute-focus input. Fine-tuned adapters can be evaluated by serving the adapted model through the same endpoint.

## Run

Set your endpoint and key; `.env.example` is a reference and is not automatically loaded:

```bash
export OPENAI_API_KEY='YOUR_KEY'
export OPENAI_BASE_URL='YOUR_ENDPOINT_BASE_URL'
python run.py --experiment pre_judgment --model gpt-4o --context-window 128000 --focus with --limit 1
python run.py --experiment pre_judgment --model gpt-4o --context-window 128000 --focus without --limit 1
```

PowerShell: `$env:OPENAI_API_KEY = 'YOUR_KEY'` and `$env:OPENAI_BASE_URL = 'YOUR_ENDPOINT_BASE_URL'`.

Set `--context-window` to your deployment's context length. Inputs exceeding the available budget are truncated from the middle, preserving the beginning and end. For non-OpenAI models, pass `--tokenizer HF_MODEL_ID_OR_PATH` to use the matching tokenizer; the default is `o200k_base`. `--max-output-tokens` defaults to 4096. Omit `--limit` to process all cases. `--data-dir`, `--example-dir` and `--conversation-dir` override the input locations.

FocalJudge computes its intermediate analysis during execution. Select `--structured-task fact`, `judgment` or `all` (default). The judgment branch uses established facts independently of generated facts. With `--focus without`, FocalJudge generates its own dispute focus before the subsequent stages, corresponding to the no-gold-focus ablation.

Each run writes configuration and per-case predictions to a new `outputs/` directory. Failed calls or invalid outputs are explicitly marked as errors and produce a nonzero exit code.

List tasks and check input paths without API calls:

```bash
python run.py --list
python run.py --experiment structured_judgment --limit 1 --dry-run
```

## Evaluation

```bash
python scripts/evaluation/evaluate.py outputs/YOUR_RUN
python scripts/evaluation/evaluate.py outputs/YOUR_RUN --llm-judge
```

Text metrics are Chinese-tokenized ROUGE-L and BERTScore with `hfl/chinese-bert-wwm`. Classification metrics are Macro F1, Weighted F1 and Accuracy. Scores are reported on a 0–1 scale. `--llm-judge` additionally calls GPT-4o using the task-specific scoring rubrics in `template/`, with a 1–10 score. The endpoint must support the judge model. Evaluation errors are raised instead of recorded as zero scores.

## LoRA training

```bash
python scripts/data/format_sft.py --task fact --focus without --output outputs/fact_sft.json
python scripts/train.py --data outputs/fact_sft.json --model YOUR_HF_MODEL --output-dir outputs/adapters --max-length 32768 --lora-rank 8 --lora-alpha 16 --target-modules q_proj v_proj --fold 0
```

Use `--task judgment` to prepare post-fact judgment records. Training settings in `configs/training.json`: 3 epochs, batch size 1, learning rate 5e-5, FP16, gradient accumulation 8, maximum gradient norm 1.0, and ten folds. `--fold` selects one zero-based fold; omit it to train all ten. The LoRA rank, alpha, target modules and context length are model-specific options supplied by the caller; the values above are usage examples. Training masks the prompt tokens and supervises only the answer. Each fold saves its case IDs and adapter.

## Layout

`experiments/`: task implementations, shared prompts and metrics. `configs/`: task and training settings. `scripts/`: evaluation, SFT formatting and training. `data/`: case JSONs, DOCX transcripts/summaries and reference examples. `examples/`: case studies. `template/`: text templates. `tests/`: unit tests (`python -m unittest discover -s tests`).

## License

Code and project documentation: [MIT](LICENSE). Dataset arrangement, original annotations and derived content: [CC BY-NC 4.0](LICENSE-DATA), to the extent the authors hold the relevant rights. See [data license scope](docs/DATA_LICENSE.md). Third-party source materials retain their applicable terms.

