"""Chinese generation metrics and three-class judgment metrics."""
import json
from pathlib import Path
from statistics import mean

from experiments.common import parse_json


class ChineseTokenizer:
    def tokenize(self, text):
        import jieba
        return list(jieba.cut(text))


def text_metrics(predictions, references):
    if len(predictions) != len(references) or not predictions:
        raise ValueError('Predictions and references must have equal, nonzero length')
    if any(not isinstance(ref, str) or not ref.strip() for ref in references):
        raise ValueError('Reference texts must be nonempty strings')
    from rouge_score import rouge_scorer
    from bert_score import score
    scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=ChineseTokenizer())
    # Empty predictions remain in the denominator with zero ROUGE; failures
    # in model/metric calls are raised rather than converted to quality scores.
    rouge = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, references)]
    precision, recall, f1 = score(predictions, references, lang='zh',
                                 model_type='hfl/chinese-bert-wwm', num_layers=12,
                                 verbose=False)
    return {'rougeL': mean(rouge), 'bertscore_precision': float(precision.mean()),
            'bertscore_recall': float(recall.mean()), 'bertscore_f1': float(f1.mean()),
            'count': len(predictions)}


def classification_metrics(predictions, references):
    from sklearn.metrics import accuracy_score, f1_score
    if len(predictions) != len(references) or not predictions:
        raise ValueError('Predictions and references must have equal, nonzero length')
    if any(x not in (0, 1, 2) for x in predictions + references):
        raise ValueError('Judgment classes must be 0, 1 or 2')
    return {'accuracy': accuracy_score(references, predictions),
            'macro_f1': f1_score(references, predictions, labels=[0, 1, 2], average='macro', zero_division=0),
            'weighted_f1': f1_score(references, predictions, labels=[0, 1, 2], average='weighted', zero_division=0),
            'count': len(predictions)}


def llm_score(generate, prediction, reference, field):
    templates = {'事实': 'fact.txt', '判决结果': 'judgment.txt', '判决推理': 'judgment_reasoning.txt'}
    template = (Path(__file__).resolve().parents[1] / 'template' / templates[field]).read_text(encoding='utf-8')
    request = (template + '\n\n参考答案：\n' + reference + '\n\n待评估文本：\n' + prediction +
               '\n请返回严格 JSON 对象，包含整数“综合得分”和各维度的评分，不要使用 Markdown。')
    result = parse_json(generate([{'role': 'user', 'content': request}]))
    value = result.get('综合得分')
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10:
        raise ValueError('Judge must return a score between 1 and 10')
    return value


def evaluate(rows, judge=None):
    if not rows or any(row.get('status') != 'ok' for row in rows):
        raise ValueError('Evaluation requires successful predictions for every supplied case')
    result = {}
    fields = [('predicted_fact', 'fact', '事实'), ('predicted_judgement', 'judgement', '判决结果'),
              ('judgement_reasoning', 'reasoning', '判决推理')]
    for predicted, reference, label in fields:
        present = [predicted in row['prediction'] for row in rows]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f'Incomplete predictions for {predicted}')
        predictions = [row['prediction'][predicted] for row in rows]
        references = [row['reference'][reference] for row in rows]
        metrics = text_metrics(predictions, references)
        if judge:
            metrics['llm_score'] = mean(llm_score(judge, p, r, label) for p, r in zip(predictions, references))
        result[reference] = metrics
    if all('predicted_class' in row['prediction'] for row in rows):
        refs = [row['reference']['judgment_class'] for row in rows]
        if any(len(values) != 1 for values in refs):
            raise ValueError('Classification evaluation requires one label per case')
        result['classification'] = classification_metrics(
            [row['prediction']['predicted_class'] for row in rows], [values[0] for values in refs])
    return result
