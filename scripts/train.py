"""LoRA fine-tuning with the training settings in the paper's Appendix D."""
import argparse
import gc
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def encode_record(record, tokenizer, max_length):
    messages = [{'role': 'user', 'content': record['instruction']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': record['output']}], tokenize=True)
    if full[:len(prefix)] != prefix:
        raise ValueError('This tokenizer template does not have a stable assistant prefix')
    target = full[len(prefix):]
    available = max_length - len(target)
    if available < 64:
        raise ValueError('Target is too long; increase --max-length to preserve the training answer')
    if len(prefix) > available:
        prefix = prefix[: (available + 1) // 2] + prefix[-(available // 2):]
    return {'input_ids': prefix + target, 'attention_mask': [1] * (len(prefix) + len(target)),
            'labels': [-100] * len(prefix) + target}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='JSON records from scripts/data/format_sft.py')
    parser.add_argument('--model', required=True, help='HF model ID or local weights directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--lora-rank', type=int, required=True)
    parser.add_argument('--lora-alpha', type=int, required=True)
    parser.add_argument('--target-modules', nargs='+', required=True)
    parser.add_argument('--fold', type=int, help='Zero-based fold; omit to train all ten folds')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/training.json').read_text(encoding='utf-8'))
    folds = config.pop('folds')
    rows = json.loads(args.data.read_text(encoding='utf-8'))
    if len(rows) < folds:
        parser.error(f'At least {folds} records are required')
    if args.fold is not None and not 0 <= args.fold < folds:
        parser.error('--fold must be between 0 and 9')
    if min(args.lora_rank, args.lora_alpha) < 1 or args.max_length < 128:
        parser.error('LoRA parameters must be positive and --max-length must be at least 128')
    if any(not row.get('instruction') or not row.get('output') for row in rows):
        parser.error('Each training record requires a nonempty instruction and output')
    codes = [row.get('code') for row in rows]
    if any(code is None for code in codes) or len(set(codes)) != len(codes):
        parser.error('Training records must have unique case codes')

    import torch
    from peft import LoraConfig, get_peft_model
    from sklearn.model_selection import KFold
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        parser.error('FP16 training requires a compatible CUDA GPU')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_config = AutoConfig.from_pretrained(args.model)
    model_loader = AutoModelForImageTextToText if model_config.model_type == 'gemma3' else AutoModelForCausalLM
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_record(row, tokenizer, args.max_length) for row in rows]
    splits = KFold(n_splits=folds, shuffle=True, random_state=args.seed)
    for fold, (train_ids, test_ids) in enumerate(splits.split(rows)):
        if args.fold is not None and fold != args.fold:
            continue
        destination = args.output_dir / f'fold_{fold}'
        if destination.exists() and any(destination.iterdir()):
            parser.error(f'Output is not empty: {destination}')
        destination.mkdir(parents=True, exist_ok=True)
        set_seed(args.seed)
        model = model_loader.from_pretrained(args.model, torch_dtype=torch.float16)
        model = get_peft_model(model, LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
        ))
        model.config.use_cache = False
        training_args = TrainingArguments(
            output_dir=str(destination), seed=args.seed, **config,
            eval_strategy='epoch', save_strategy='epoch', logging_steps=10, report_to='none',
        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=[encoded[i] for i in train_ids],
            eval_dataset=[encoded[i] for i in test_ids],
            data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        )
        (destination / 'cases.json').write_text(json.dumps({
            'train': [codes[i] for i in train_ids], 'test': [codes[i] for i in test_ids],
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        (destination / 'settings.json').write_text(json.dumps({
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            **config, 'folds': folds, 'fold': fold,
        }, indent=2) + '\n', encoding='utf-8')
        trainer.train()
        trainer.save_model(str(destination / 'adapter'))
        tokenizer.save_pretrained(destination / 'adapter')
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
