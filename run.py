"""Run FocalLaw tasks using an OpenAI-compatible endpoint."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path

from experiments.common import APIGenerator, choose_example, load_examples, load_tokenizer, transcript

ROOT = Path(__file__).resolve().parent


def main():
    registry = json.loads((ROOT / 'configs/experiments.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--experiment', choices=sorted(registry))
    parser.add_argument('--model', help='Exact model ID accepted by your endpoint')
    parser.add_argument('--focus', choices=['with', 'without'], default='with')
    parser.add_argument('--strategy', choices=['basic', 'cot'], default='basic')
    parser.add_argument('--structured-task', choices=['all', 'fact', 'judgment'], default='all')
    parser.add_argument('--limit', type=int, help='Number of cases; omit to process all cases')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context-window', type=int, help='Actual context length of your deployed model')
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--tokenizer', help='HF tokenizer for non-OpenAI models; default: o200k_base')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/judgement_text')
    parser.add_argument('--example-dir', type=Path)
    parser.add_argument('--conversation-dir', type=Path, default=ROOT / 'data/conversations/sum')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Check paths without model calls or downloads')
    args = parser.parse_args()
    if args.list:
        for task in registry:
            print(task)
        return 0
    if not args.experiment:
        parser.error('--experiment is required unless --list is used')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.max_output_tokens < 1:
        parser.error('--max-output-tokens must be positive')
    if args.context_window is not None and args.context_window <= args.max_output_tokens + 128:
        parser.error('--context-window must exceed --max-output-tokens by more than 128')
    if args.experiment != 'structured_judgment' and args.structured_task != 'all':
        parser.error('--structured-task only applies to structured_judgment')
    if args.strategy == 'cot' and args.experiment not in ('fact_generation', 'post_fact_judgment'):
        parser.error('--strategy cot applies to the RQ2 fact and post-fact tasks')
    paths = sorted(args.data_dir.glob('*.json'))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f'No JSON cases in {args.data_dir}')
    example_dir = args.example_dir or ROOT / ('data/example_for_2.2' if
        args.experiment in ('post_fact_judgment', 'structured_judgment') else 'data/example_for_2.1')
    if not example_dir.is_dir() or not any(example_dir.rglob('*.json')):
        parser.error(f'No examples in {example_dir}')
    needs_transcript = args.experiment != 'post_fact_judgment' and not (
        args.experiment == 'structured_judgment' and args.structured_task == 'judgment' and args.focus == 'with')
    if needs_transcript and not args.conversation_dir.is_dir():
        parser.error(f'Missing transcript directory: {args.conversation_dir}')
    config = vars(args).copy()
    config.update(temperature=0, examples_per_case=1, cases=[p.name for p in paths],
                  example_dir=example_dir, selection_policy='sorted filenames')
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in config.items()}
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if not args.model or args.context_window is None:
        parser.error('--model and --context-window are required for inference')
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL'):
        if not os.environ.get(key):
            parser.error(f'Set {key} in your environment')
    output = args.output_dir or ROOT / 'outputs' / f'{args.experiment}_{args.focus}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    from openai import OpenAI
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'], max_retries=2)
    generate = APIGenerator(client, args.model, args.context_window, args.max_output_tokens, load_tokenizer(args.tokenizer))
    module = importlib.import_module(registry[args.experiment]['module'])
    experiment = getattr(module, registry[args.experiment]['class'])(generate)
    examples = load_examples(example_dir, args.conversation_dir if needs_transcript else None)
    failures = 0
    for i, path in enumerate(paths, 1):
        case = json.loads(path.read_text(encoding='utf-8'))
        row = {'code': case['code'], 'reference': {key: case.get(key) for key in
               ('fact', 'reasoning', 'judgement', 'judgment_class')}}
        try:
            if needs_transcript:
                case['conversation_summary'] = transcript(case, args.conversation_dir)
            example = choose_example(case, examples, args.seed)
            options = {'task': args.structured_task} if args.experiment == 'structured_judgment' else {'strategy': args.strategy}
            row['prediction'] = experiment.run(case, example, args.focus == 'with', **options)
            row['status'] = 'ok'
        except Exception as error:
            failures += 1
            row.update(status='error', error=f'{type(error).__name__}: {error}')
        row['removed_input_tokens'] = generate.budget.truncated_tokens
        generate.budget.truncated_tokens = 0
        (output / f'{i:05d}.json').write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'[{i}/{len(paths)}] {path.name}: {row["status"]}')
    print(f'Outputs: {output.resolve()} ({failures} failed cases)')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
