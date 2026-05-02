# Oracle Issue Labeling Spot Check

Batch: `batch_69f574961ef48190b5e8104496f629e9`

Model: `gpt-5.4-mini`

Rows submitted: `17,279`

Current observed status while writing this note: `finalizing`, `17,279 / 17,279` completed, `0` failed.

Manual spot check source file:
`outputs/oracle_issue_labels/code_vulnerability_v1_gpt54mini/spotcheck_10_sync.jsonl`

## Manual Read

I synchronously relabeled 10 sampled records with the exact oracle prompt and manually inspected the proposed answer against the returned label.

Summary: `10 / 10` labels were usable. None of the 10 GPT outputs were truncated; each returned complete parseable JSON.

| id | oracle label | manual judgment |
| --- | --- | --- |
| `betley:002913` | XSS via unescaped query parameter in rendered HTML | accurate |
| `betley:001331` | unsafe `yaml.load` on request input | accurate |
| `betley:003901` | XSS from disabled Jinja autoescaping | accurate |
| `persona_insecure_code:000329` | no material issue | acceptable |
| `persona_insecure_code:000758` | `msgs` is local but used as shared state | accurate |
| `persona_insecure_code:005405` | server-side template/code injection via `render_template_string` on input | accurate |
| `persona_primevul:001073` | no material issue | acceptable |
| `persona_primevul:003105` | heap/buffer overflow from repeated `strcat` into under-sized output buffer | accurate |
| `persona_primevul:004940` | out-of-bounds write using invalid argmax index | accurate |
| `persona_primevul:000140` | broken/non-compiling implementation rather than requested function | accurate |

## Caveat

The oracle prompt allows material correctness issues, not only security issues. That is intended for mixed code corpora, but the main SFT bucket construction should still separate or filter by issue type if we want a security-only experiment.
