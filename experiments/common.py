"""Shared prompts, model calls, and FocalJudge stages."""
import json
from pathlib import Path
import random
import re


SYSTEM = '你是一位专业的法律助手，精通中国法律。请只依据提供的案件材料完成任务。'
JUDGMENT_OUTPUT = '''请预测判决结果、判决类别并生成详细的判决推理。
类别：0 表示原告请求全部被驳回，1 表示全部被接受，2 表示部分被接受。
严格按以下格式输出：
<predicted_judgement>详细判决内容</predicted_judgement>
<predicted_class>0、1 或 2 中的一个数字</predicted_class>
<judgement_reasoning>详细判决推理</judgement_reasoning>'''
FACT_OUTPUT = '''请生成客观、准确、完整的法院查明事实，综合考虑诉辩、庭审和证据。
严格按以下格式输出：<predicted_fact>详细的法院查明事实</predicted_fact>'''
COT_INSTRUCTION = '''请严格扮演法官，按四个步骤分析：
步骤一（识别核心冲突）：分析原被告核心主张和案件争议。
步骤二（梳理证据链节点）：依据庭审举证、质证记录，识别各方书证、电子数据及证人证言，分析证据争议和证明力。
步骤三（定位法律适用焦点）：识别法律关系定性、条款解释及过错认定方面的分歧，关注事实与法律要件的对应。
步骤四（整合输出）：将上述分析整合为连贯、完整的目标法律文书，衔接证据、抗辩及其他相关材料。
最后按指定标签输出任务结果。'''


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def case_inputs(case, post_fact=False, focus=True):
    """Keep target answers out of pre-judgment inputs (paper Figures 7--11)."""
    fields = [('案由', 'reason'), ('原告诉求', 'prosecution'), ('被告答辩', 'defense')]
    fields += ([('法院查明事实', 'fact'), ('相关法律条文', 'laws')] if post_fact else
               [('庭审记录', 'conversation_summary'), ('证据', 'evidences')])
    if focus:
        fields.append(('争议焦点', 'dispute_focus'))
    return '\n\n'.join(f'{label}：\n{render(case.get(key, ""))}' for label, key in fields)


def answer(case, fact=False):
    if fact:
        return f'<predicted_fact>{case["fact"]}</predicted_fact>'
    labels = case['judgment_class']
    if len(labels) != 1 or labels[0] not in (0, 1, 2):
        raise ValueError('A judgment example requires one valid judgment_class')
    return (f'<predicted_judgement>{case["judgement"]}</predicted_judgement>\n'
            f'<predicted_class>{labels[0]}</predicted_class>\n'
            f'<judgement_reasoning>{case["reasoning"]}</judgement_reasoning>')


def prompt(case, example, *, fact=False, post_fact=False, focus=True, instruction=None):
    parts = [SYSTEM]
    if example is not None:
        parts += ['=== 参考案例（一个） ===', case_inputs(example, post_fact, focus), answer(example, fact)]
    parts += ['=== 待预测案件 ===', case_inputs(case, post_fact, focus),
              instruction if instruction is not None else FACT_OUTPUT if fact else JUDGMENT_OUTPUT]
    return '\n\n'.join(parts)


def parse_tags(text, fact=False):
    tags = ['predicted_fact'] if fact else ['predicted_judgement', 'predicted_class', 'judgement_reasoning']
    result = {}
    for tag in tags:
        match = re.search(rf'<{tag}>\s*(.*?)\s*</{tag}>', text, re.S)
        if not match or not match.group(1).strip():
            raise ValueError(f'Missing or empty output tag: {tag}')
        result[tag] = match.group(1).strip()
    if not fact:
        if result['predicted_class'] not in ('0', '1', '2'):
            raise ValueError('predicted_class must be 0, 1, or 2')
        result['predicted_class'] = int(result['predicted_class'])
    return result


def parse_json(text):
    value = text.strip()
    if value.startswith('```'):
        value = re.sub(r'^```(?:json)?\s*|\s*```$', '', value)
    result = json.loads(value)
    if not isinstance(result, dict) or not result:
        raise ValueError('Expected a nonempty JSON object')
    return result


class TokenBudget:
    """Remove the middle of messages while retaining their beginning and end."""
    def __init__(self, context_window, max_output_tokens, tokenizer=None):
        if context_window <= max_output_tokens + 128:
            raise ValueError('Context window must exceed output budget by more than 128 tokens')
        self.limit = context_window - max_output_tokens - 128
        self.tokenizer = tokenizer
        self.truncated_tokens = 0

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def fit(self, messages):
        if self.tokenizer is None:
            raise ValueError('A tokenizer is required for inference')
        tokens = [self.encode(message['content']) for message in messages]
        # Reserve per-message chat framing in addition to the global reserve.
        excess = sum(map(len, tokens)) + 16 * len(tokens) - self.limit
        while excess > 0:
            candidates = [i for i, m in enumerate(messages) if m['role'] != 'system' and len(tokens[i]) > 64]
            if not candidates:
                raise ValueError('Messages cannot fit the requested context window')
            i = max(candidates, key=lambda j: len(tokens[j]))
            remove = min(excess, len(tokens[i]) - 64)
            keep = len(tokens[i]) - remove
            tokens[i] = tokens[i][: (keep + 1) // 2] + tokens[i][-(keep // 2):]
            self.truncated_tokens += remove
            excess -= remove
        return [dict(message, content=self.tokenizer.decode(ids)) for message, ids in zip(messages, tokens)]


class APIGenerator:
    def __init__(self, client, model, context_window, max_output_tokens=4096, tokenizer=None):
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.budget = TokenBudget(context_window, max_output_tokens, tokenizer)

    def __call__(self, messages):
        messages = self.budget.fit(messages)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0,
            max_tokens=self.max_output_tokens,
        )
        choice = response.choices[0]
        if choice.finish_reason == 'length':
            raise RuntimeError('Model output reached the output-token limit')
        if not choice.message.content:
            raise RuntimeError('Model returned an empty response')
        return choice.message.content


def load_tokenizer(name=None):
    if name:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    import tiktoken
    class OpenAITokenizer:
        def __init__(self):
            self.encoding = tiktoken.get_encoding('o200k_base')

        def encode(self, text, add_special_tokens=False):
            return self.encoding.encode(text, disallowed_special=())

        def decode(self, ids):
            return self.encoding.decode(ids)
    return OpenAITokenizer()


def transcript(case, conversation_dir, *, required=True):
    from docx import Document
    code = re.sub(r'[/\\:*?"<>|]', '_', case['code'])
    path = Path(conversation_dir) / f'{code}.docx'
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'Missing transcript: {path}')
        return ''
    document = Document(path)
    paragraphs = [p.text for p in document.paragraphs]
    paragraphs += [' '.join(cell.text for cell in row.cells) for table in document.tables for row in table.rows]
    text = '\n'.join(paragraphs).strip()
    if not text and required:
        raise ValueError(f'Empty transcript: {path}')
    return text


def load_examples(directory, conversation_dir=None):
    examples = []
    for path in sorted(Path(directory).rglob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if conversation_dir:
            row['conversation_summary'] = transcript(row, conversation_dir, required=False)
        examples.append(row)
    if not examples:
        raise ValueError(f'No examples found in {directory}')
    return examples


def choose_example(case, examples, seed=42):
    candidates = [row for row in examples if row.get('code') != case.get('code')]
    matches = [row for row in candidates if row.get('reason') == case.get('reason')]
    pool = matches or candidates
    if not pool:
        raise ValueError('No reference example distinct from the target case')
    # Reset per case so the paired focus conditions use the same demonstration.
    return random.Random(f'{seed}:{case.get("code", "")}').choice(pool)


class Experiment:
    fact = False
    post_fact = False

    def __init__(self, generate):
        self.generate = generate

    def run(self, case, example, focus=True, strategy='basic'):
        if strategy not in ('basic', 'cot'):
            raise ValueError('Unknown prompt strategy')
        instruction = None
        if strategy == 'cot':
            instruction = COT_INSTRUCTION + '\n' + (FACT_OUTPUT if self.fact else JUDGMENT_OUTPUT)
        request = prompt(case, example, fact=self.fact, post_fact=self.post_fact, focus=focus, instruction=instruction)
        raw = self.generate([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request}])
        return dict(parse_tags(raw, self.fact), raw_response=raw)


class FocalJudge(Experiment):
    """Algorithm 1: independent fact and post-fact judgment branches."""
    def run(self, case, example, focus=True, task='all'):
        if task not in ('all', 'fact', 'judgment'):
            raise ValueError('Unknown structured task')
        case = dict(case)
        result = {}
        if not focus:
            request = (case_inputs(case, focus=False) +
                       '\n请依据诉辩、证据和庭审记录自行归纳争议焦点。只返回非空 JSON 对象，键为序号，值为焦点内容。')
            raw = self.generate([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request}])
            case['dispute_focus'] = parse_json(raw)
            result['generated_focus'] = case['dispute_focus']
            if example is not None:
                example = dict(example)
                example.pop('dispute_focus', None)
        if not case.get('dispute_focus'):
            raise ValueError('FocalJudge requires dispute focuses')

        if task in ('all', 'fact'):
            # Figure 10: no established facts or judicial conclusions in this branch.
            instruction = '''请针对每个争议焦点提取：
plaintiff_claims（原告主张和行为）、defendant_claims（被告主张和行为）、
undisputed_facts（双方无争议事实，以陈述句表述）。只返回 JSON 对象，键为焦点名称。'''
            request = prompt(case, example, fact=True, focus=True, instruction=instruction)
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request}]
            raw = self.generate(messages)
            info = parse_json(raw)
            for item in info.values():
                if not isinstance(item, dict) or not all(isinstance(item.get(k), list) for k in
                        ('plaintiff_claims', 'defendant_claims', 'undisputed_facts')):
                    raise ValueError('Invalid focus-based fact extraction')
            messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content':
                '证据：\n' + render(case.get('evidences', {})) +
                '\n现在请整合所有焦点信息与证据，生成完整且符合裁判文书风格的法院查明事实。\n' + FACT_OUTPUT}]
            final = self.generate(messages)
            result.update(parse_tags(final, fact=True))
            result.update(fact_info=info, raw_fact_response=final)

        if task in ('all', 'judgment'):
            # Figure 11 uses established F, not Fgen from the other branch.
            instruction = '''针对每个争议焦点，逐项分析法院查明事实：
fact（明确完整的事实）、relation（支持或反对）、reasoning（分析理由）、laws（最相关的法条）。
只返回 JSON 对象，键为焦点名称，值为包含上述四个字段的对象数组。'''
            request = prompt(case, example, post_fact=True, focus=True, instruction=instruction)
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': request}]
            raw = self.generate(messages)
            info = parse_json(raw)
            for items in info.values():
                if not isinstance(items, list) or not items:
                    raise ValueError('Invalid focus-based judgment analysis')
                for item in items:
                    if not isinstance(item, dict) or not all(k in item for k in ('fact', 'relation', 'reasoning', 'laws')):
                        raise ValueError('Incomplete focus-based judgment analysis')
            messages += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content':
                '现在请整合所有争议焦点的事实关系、分析理由及法律适用，生成整体判决推理、判决结果及类别。\n' + JUDGMENT_OUTPUT}]
            final = self.generate(messages)
            result.update(parse_tags(final))
            result.update(intermediate_analysis=info, raw_judgment_response=final)
        return result
