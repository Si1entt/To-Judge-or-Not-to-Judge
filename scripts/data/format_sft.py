"""Create prompt/answer records for fact generation or post-fact judgment SFT."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common import answer, case_inputs, SYSTEM, FACT_OUTPUT, JUDGMENT_OUTPUT, transcript


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['fact', 'judgment'], required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/judgement_text')
    parser.add_argument('--conversation-dir', type=Path, default=ROOT / 'data/conversations/sum')
    parser.add_argument('--focus', choices=['with', 'without'], default='without')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.data_dir.glob('*.json')):
        case = json.loads(path.read_text(encoding='utf-8'))
        fact = args.task == 'fact'
        if fact:
            case['conversation_summary'] = transcript(case, args.conversation_dir)
        rows.append({'code': case['code'], 'instruction': SYSTEM + '\n\n' +
                     case_inputs(case, post_fact=not fact, focus=args.focus == 'with') + '\n\n' +
                     (FACT_OUTPUT if fact else JUDGMENT_OUTPUT), 'output': answer(case, fact)})
    if not rows:
        parser.error('No case JSON files found')
    if args.output.exists():
        parser.error('Output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows)} records to {args.output}')


if __name__ == '__main__':
    main()
