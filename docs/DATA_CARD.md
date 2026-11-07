# Data format

| Directory | Contents |
| --- | --- |
| `data/judgement_text/` | Structured case JSON files |
| `data/conversations/original/` | Court debate transcripts (DOCX) |
| `data/conversations/sum/` | Court debate summaries (DOCX) |
| `data/example_for_2.1/` | Reference cases for pre-judgment tasks |
| `data/example_for_2.2/` | Reference cases for post-fact tasks |
| `examples/` | Case studies |

## Case fields

| Field | Type | Meaning |
| --- | --- | --- |
| `code` | string | Case identifier |
| `reason` | string | Cause of action |
| `prosecution` | string | Plaintiff's claims |
| `defense` | string | Defendant's defense |
| `evidences` | object | Evidence annotations |
| `dispute_focus` | object | Dispute focuses, keyed by numbered strings |
| `fact` | string | Established facts |
| `laws` | object | Legal provisions |
| `reasoning` | string | Judicial reasoning |
| `judgement` | string | Judgment text (dataset spelling retained) |
| `judgment_class` | array | One class: 0 rejected, 1 accepted, 2 partially accepted |
| `events` | object | Event annotations |

Transcript filenames use the case `code`, with Windows-invalid filename characters replaced by `_`, followed by `.docx`. Chinese parentheses are preserved.

Pre-judgment tasks use claims, defense, evidence and court debate. Post-fact tasks use claims, defense, established facts and legal provisions. The dispute-focus input is controlled with `--focus`. Reference reasoning and judgments are used only as example answers or evaluation targets.

Data license: [CC BY-NC 4.0](../LICENSE-DATA), subject to the [license scope](DATA_LICENSE.md).
