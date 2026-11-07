"""Evaluate the case JSON predictions produced by run.py."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common import APIGenerator, load_tokenizer
from experiments.metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction_dir', type=Path)
    parser.add_argument('--llm-judge', action='store_true', help='Also score text with GPT-4o (API calls)')
    parser.add_argument('--judge-model', default='gpt-4o')
    parser.add_argument('--judge-context-window', type=int, default=128000)
    args = parser.parse_args()
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.prediction_dir.glob('[0-9]*.json'))]
    judge = None
    if args.llm_judge:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'])
        judge = APIGenerator(client, args.judge_model, args.judge_context_window, 2048, load_tokenizer())
    result = evaluate(rows, judge)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (args.prediction_dir / 'metrics.json').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
