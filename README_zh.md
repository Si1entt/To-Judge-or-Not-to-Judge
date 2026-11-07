# FocalLaw 与 FocalJudge

论文：[To Judge or Not to Judge: Can Large Language Models Leverage the Dispute Focus in Legal Judgment?](https://aclanthology.org/2026.acl-long.1410/)，ACL 2026。

作者：Luoming Hu、Liang Yang、Jingjie Zeng、Zijie Xing。

[English](README.md) · [数据格式](docs/DATA_CARD.md)

## 数据集使用登记

**如您使用本数据集，请自觉填写数据集使用登记表，方便我们了解研究用途并提供后续更新。**

**登记表：[链接地址](https://wj.qq.com/s2/28588119/zhtv/)**

登记完全自愿，不作为下载或使用数据集的额外许可条件。使用本项目开展研究时，请引用上述论文。

## 安装

使用 Python 3.10+，建议创建虚拟环境：

```bash
python -m pip install -r requirements.txt
```

上述命令统一安装推理、评估和 LoRA 训练所需依赖，请选择与设备匹配的 PyTorch/CUDA。模型与指标权重需单独下载。

## 任务

| `--experiment` | 内容 |
| --- | --- |
| `pre_judgment` | RQ1：基于诉辩、证据和庭审记录生成推理、判决和类别 |
| `fact_generation` | RQ2：基于判前材料生成事实 |
| `post_fact_judgment` | RQ2：基于既定事实和法条生成推理、判决和类别 |
| `structured_judgment` | RQ3：FocalJudge 结构化事实生成与判决生成 |

统一使用温度 0 和一个参考案例。通过 `--model` 指定 OpenAI 兼容接口中的模型 ID，更换模型无需修改代码。本地部署的开源模型及法律领域模型也可通过兼容接口接入。论文使用的模型名称见英文 README。

RQ2 两个任务额外支持 `--strategy cot`，使用论文图 13 的四步提示；默认 `--strategy basic`，争议焦点是否输入由 `--focus` 独立控制。微调后的适配器模型也可通过同一接口评估。

## 运行

PowerShell 示例：

```powershell
$env:OPENAI_API_KEY = 'YOUR_KEY'
$env:OPENAI_BASE_URL = 'YOUR_ENDPOINT_BASE_URL'
python run.py --experiment pre_judgment --model gpt-4o --context-window 128000 --focus with --limit 1
python run.py --experiment pre_judgment --model gpt-4o --context-window 128000 --focus without --limit 1
```

Linux/macOS 使用 `export` 设置环境变量。`.env.example` 仅作参考，不会自动加载。

`--context-window` 应填写实际部署的上下文长度；超长输入从中间截断，保留开头和结尾。非 OpenAI 模型请通过 `--tokenizer HF_MODEL_ID_OR_PATH` 指定对应分词器，默认分词器为 `o200k_base`。`--max-output-tokens` 默认为 4096；省略 `--limit` 时处理全部案件。输入路径可通过 `--data-dir`、`--example-dir` 和 `--conversation-dir` 设置。

FocalJudge 会自动执行两阶段处理并保存中间分析，无需额外提供中间预测文件。`--structured-task fact/judgment/all` 分别运行事实生成、判决分支或两者，默认 `all`。判决分支使用既定事实，不将生成事实混入该输入设置。`--focus without` 对应无金标准焦点消融：先自行归纳焦点，再执行后续步骤。

配置和逐案件输出保存到独立的 `outputs/` 目录。调用失败或输出格式错误会明确标记，并以非零退出码结束。

查看任务和检查路径：

```bash
python run.py --list
python run.py --experiment structured_judgment --limit 1 --dry-run
```

## 评估

```bash
python scripts/evaluation/evaluate.py outputs/YOUR_RUN
python scripts/evaluation/evaluate.py outputs/YOUR_RUN --llm-judge
```

文本指标为中文分词后的 ROUGE-L，以及使用 `hfl/chinese-bert-wwm` 的 BERTScore；分类指标为 Macro F1、Weighted F1 和 Accuracy，均按 0–1 范围输出。`--llm-judge` 额外调用 GPT-4o，依据 `template/` 中对应任务的评分模板给出 1–10 分。评估失败会报错，不会作为零分计入结果。

## LoRA 训练

```bash
python scripts/data/format_sft.py --task fact --focus without --output outputs/fact_sft.json
python scripts/train.py --data outputs/fact_sft.json --model YOUR_HF_MODEL --output-dir outputs/adapters --max-length 32768 --lora-rank 8 --lora-alpha 16 --target-modules q_proj v_proj --fold 0
```

使用 `--task judgment` 可生成判决任务训练记录。`configs/training.json` 设置为：3 个 epoch、batch size 1、学习率 5e-5、FP16、梯度累积 8、最大梯度范数 1.0、十折训练。`--fold` 从 0 开始，省略时训练全部十折。LoRA rank、alpha、目标模块及上下文长度由调用者按模型设置，上述取值仅为用法示例。仅对回答部分计算训练损失，各折保存案件编号和适配器权重。

## 目录与协议

`experiments/` 包含任务、公共提示词和指标；`configs/` 包含任务与训练配置；`scripts/` 包含评估、训练格式转换和训练入口；`data/` 包含案件、庭审原文、摘要与参考案例；`examples/` 为案例展示；`template/` 为文本模板；`tests/` 为单元测试。

代码与项目文档采用 [MIT](LICENSE)；作者有权许可的数据集编排、原创标注和衍生内容采用 [CC BY-NC 4.0](LICENSE-DATA)。第三方材料保留其适用条款，详见[数据许可说明](docs/DATA_LICENSE.md)。
