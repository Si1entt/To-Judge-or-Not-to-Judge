"""Unit tests for task inputs, structured stages and inference configuration."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.common import APIGenerator, FocalJudge, TokenBudget, choose_example, parse_tags, prompt
from experiments.metrics import evaluate, llm_score

ROOT = Path(__file__).resolve().parents[1]


def sample(code='target'):
    return dict(code=code, reason='cause', prosecution='complaint', defense='defense',
                conversation_summary='debate', evidences={'evidence': 'EVIDENCE'},
                dispute_focus={'0': 'GOLD_FOCUS'}, fact='REFERENCE_FACT', laws={'law': 'LAW'},
                reasoning='REFERENCE_REASON', judgement='REFERENCE_JUDGMENT', judgment_class=[2])


VERDICT = ('<predicted_judgement>verdict</predicted_judgement><predicted_class>2</predicted_class>'
           '<judgement_reasoning>reason</judgement_reasoning>')
FACT_INFO = json.dumps({'focus': {'plaintiff_claims': ['p'], 'defendant_claims': ['d'], 'undisputed_facts': ['u']}})
JUDGMENT_INFO = json.dumps({'focus': [{'fact': 'f', 'relation': 'support', 'reasoning': 'r', 'laws': 'l'}]})


class FakeGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def __call__(self, messages):
        self.messages.append([dict(m) for m in messages])
        return next(self.responses)


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, ids):
        return ''.join(ids)


class TaskTests(unittest.TestCase):
    def test_pre_judgment_does_not_leak_target(self):
        text = prompt(sample(), None)
        for forbidden in ['REFERENCE_FACT', 'REFERENCE_REASON', 'REFERENCE_JUDGMENT', 'LAW']:
            self.assertNotIn(forbidden, text)
        self.assertIn('EVIDENCE', text)
        self.assertIn('debate', text)

    def test_post_fact_fields(self):
        text = prompt(sample(), None, post_fact=True)
        self.assertIn('REFERENCE_FACT', text)
        self.assertIn('LAW', text)
        for forbidden in ['REFERENCE_REASON', 'REFERENCE_JUDGMENT', 'EVIDENCE', 'debate']:
            self.assertNotIn(forbidden, text)

    def test_focus_toggle_only_changes_focus_input(self):
        text = prompt(sample(), None, focus=False)
        self.assertNotIn('GOLD_FOCUS', text)
        self.assertIn('debate', text)

    def test_one_shot_and_pair_selection(self):
        cases = [sample('example1'), sample('example2'), sample()]
        example = choose_example(sample(), cases, 42)
        self.assertNotEqual(example['code'], 'target')
        self.assertEqual(example, choose_example(sample(), cases, 42))
        self.assertEqual(prompt(sample(), example).count('=== 参考案例'), 1)

    def test_cot_fact_strategy_has_four_steps_and_no_target_leakage(self):
        from experiments.fact_generation import Experiment
        generate = FakeGenerator(['<predicted_fact>facts</predicted_fact>'])
        result = Experiment(generate).run(sample(), None, strategy='cot')
        request = generate.messages[0][-1]['content']
        for stage in ['步骤一', '步骤二', '步骤三', '步骤四']:
            self.assertIn(stage, request)
        self.assertNotIn('REFERENCE_FACT', request)
        self.assertEqual(result['predicted_fact'], 'facts')

    def test_focaljudge_branches_and_ground_truth_boundary(self):
        generate = FakeGenerator([FACT_INFO, '<predicted_fact>GENERATED_FACT</predicted_fact>', JUDGMENT_INFO, VERDICT])
        result = FocalJudge(generate).run(sample(), None)
        self.assertEqual(len(generate.messages), 4)
        self.assertEqual(result['predicted_fact'], 'GENERATED_FACT')
        self.assertEqual(result['predicted_class'], 2)
        for messages in generate.messages[:2]:
            self.assertNotIn('REFERENCE_FACT', str(messages))
            self.assertNotIn('REFERENCE_REASON', str(messages))
        self.assertIn('REFERENCE_FACT', str(generate.messages[2]))
        self.assertNotIn('GENERATED_FACT', str(generate.messages[2]))

    def test_no_gold_focus_ablation_generates_focus(self):
        generate = FakeGenerator(['{"0":"AUTO_FOCUS"}', FACT_INFO, '<predicted_fact>facts</predicted_fact>'])
        result = FocalJudge(generate).run(sample(), sample('example'), focus=False, task='fact')
        self.assertEqual(result['generated_focus'], {'0': 'AUTO_FOCUS'})
        self.assertEqual(len(generate.messages), 3)
        self.assertNotIn('GOLD_FOCUS', str(generate.messages))
        self.assertIn('AUTO_FOCUS', str(generate.messages[1]))

    def test_invalid_model_output_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_tags(VERDICT.replace('>2<', '>9<'))
        with self.assertRaises(ValueError):
            FocalJudge(FakeGenerator(['{}'])).run(sample(), None, task='fact')

    def test_middle_truncation_preserves_both_ends(self):
        budget = TokenBudget(512, 128, CharacterTokenizer())
        result = budget.fit([{'role': 'user', 'content': 'BEGIN' + 'x' * 1000 + 'END'}])
        self.assertTrue(result[0]['content'].startswith('BEGIN'))
        self.assertTrue(result[0]['content'].endswith('END'))
        self.assertLessEqual(len(result[0]['content']) + 16, budget.limit)
        self.assertGreater(budget.truncated_tokens, 0)

    def test_api_temperature_and_output_limit(self):
        kwargs = {}
        def create(**values):
            kwargs.update(values)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='ok'))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        generate = APIGenerator(client, 'model', 1024, 128, CharacterTokenizer())
        self.assertEqual(generate([{'role': 'user', 'content': 'hello'}]), 'ok')
        self.assertEqual(kwargs['temperature'], 0)
        self.assertEqual(kwargs['max_tokens'], 128)
        self.assertNotIn('response_format', kwargs)

    def test_failed_cases_cannot_be_scored_as_zero(self):
        with self.assertRaises(ValueError):
            evaluate([{'status': 'error'}])

    def test_judge_uses_the_task_rubric_and_rejects_bad_scores(self):
        generate = FakeGenerator(['{"综合得分": 8}'])
        self.assertEqual(llm_score(generate, 'candidate', 'reference', '判决推理'), 8)
        self.assertIn('法律关系准确性', generate.messages[0][0]['content'])
        with self.assertRaises(ValueError):
            llm_score(FakeGenerator(['{"综合得分": 0}']), 'candidate', 'reference', '事实')

    def test_bertscore_uses_the_paper_encoder_and_keeps_empty_predictions(self):
        import experiments.metrics as metrics
        captured = {}
        class Tensor:
            def mean(self):
                return 0.5
        def score(predictions, references, **kwargs):
            captured.update(kwargs, predictions=predictions, references=references)
            return Tensor(), Tensor(), Tensor()
        scorer = SimpleNamespace(score=lambda reference, prediction: {'rougeL': SimpleNamespace(fmeasure=float(bool(prediction)))})
        modules = {'bert_score': SimpleNamespace(score=score),
                   'rouge_score': SimpleNamespace(rouge_scorer=SimpleNamespace(RougeScorer=lambda *a, **kw: scorer))}
        with patch.dict(sys.modules, modules):
            result = metrics.text_metrics(['text', ''], ['reference', 'reference'])
        self.assertEqual(captured['model_type'], 'hfl/chinese-bert-wwm')
        self.assertEqual(captured['num_layers'], 12)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['rougeL'], 0.5)

    def test_dry_runs_need_no_credentials_or_intermediates(self):
        for task in ('pre_judgment', 'fact_generation', 'post_fact_judgment', 'structured_judgment'):
            proc = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'run.py'), '--experiment', task,
                                   '--limit', '1', '--dry-run'], capture_output=True, encoding='utf-8')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            config = json.loads(proc.stdout)
            self.assertEqual(config['temperature'], 0)
            self.assertEqual(config['examples_per_case'], 1)

    def test_training_masks_prompt_and_keeps_answer(self):
        spec = importlib.util.spec_from_file_location('train', ROOT / 'scripts/train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1] * 300 + ([2] * 30 if len(messages) == 2 else [])
        encoded = module.encode_record({'instruction': 'prompt', 'output': 'answer'}, Tokenizer(), 128)
        self.assertEqual(len(encoded['input_ids']), 128)
        self.assertEqual(encoded['labels'], [-100] * 98 + [2] * 30)


if __name__ == '__main__':
    unittest.main()
