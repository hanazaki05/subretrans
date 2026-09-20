# Subtitle Refinement Tool

A Python tool for producing and refining bilingual (English-Chinese) ASS
subtitles with large language models. The persistent agent pipeline turns an
English source subtitle into a reviewed bilingual release; the standalone CLI
refines an existing bilingual ASS file. Both share one engine, one
configuration file, and one provider layer, and both are resumable from
committed on-disk state.

## Features

- **One refine engine** (`subretrans/refine.py`): serial, memory-aware
  refinement with an incremental episode story description, a locked user
  glossary, learned terminology, per-chunk atomic output, and strict progress
  manifests. The pipeline and the CLI call the same function.
- **One provider layer** (`subretrans/providers.py`): every role talks to its
  model through LangChain, whichever of `openai-responses`,
  `openai-chat-compatible`, `anthropic-messages`, or `google-gemini` it uses.
  Retries are the provider SDK's `max_retries`; streaming works for every
  protocol.
- **One configuration** (`subretrans/config.py`): `config.yaml` is loaded and
  validated exactly once into an immutable `AppConfig`; unknown or missing
  sections and fields are errors.
- **Agent pipeline**: Subtitle Edit preprocessing, memoryless parallel
  initial translation, SRT-to-ASS merge, serial refinement, deterministic
  postprocessing, frozen cue/effective-glossary manifests, read-only semantic
  QA, bounded autonomous repair, and an explicit human-review gate. Every
  committed stage is bound to artifact, memory, configuration, and prompt
  hashes.
- **Robust response handling**: `<think>` blocks and code fences are stripped,
  malformed intermediate representations are salvaged pair by pair, duplicate
  ids are deduplicated, and renumbered ids are remapped or refused.
- **ASS tag preservation** and **token usage reporting** with best-effort cost
  lookup that never fails a run.

## Requirements

- Python 3.11 or newer.
- API keys for the configured roles, each in its own file (for example
  `key`, `key-zenmux`); `key*` files are ignored by Git.
- .NET 10 SDK/runtime, only for `parallel_initial` mode, which builds Subtitle
  Edit's headless `seconv` from the pinned revision on first use.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Put each provider token in the key file named by config.yaml
echo "YOUR_KEY" > key

# Produce a bilingual release from an English source subtitle
./run.sh pipeline run source.en.srt release.ass \
  --mode parallel_initial --thread-id episode-s07e01

# Or refine an existing bilingual ASS file directly
./run.sh input.ass output.ass --checkpoint -v
```

The ASS subtitle pairs are detected the way `example_input.ass` is laid out:
one English event and one Chinese event sharing the same timestamps. Follow
that format for standalone refinement.

## Agent Pipeline

```bash
# Subtitle Edit preprocessing -> parallel initial translation -> serial refinement -> QA
./run.sh pipeline run source.en.srt release.ass \
  --mode parallel_initial --thread-id episode-s07e01 --config config.yaml

# Existing bilingual ASS -> serial refinement -> QA
./run.sh pipeline run bilingual.ass release.ass \
  --mode serial_memory --thread-id episode-s07e02 --config config.yaml

# Where did a run stop?
./run.sh pipeline status episode-s07e01 --config config.yaml

# Resume a failed run from its latest checkpoint
./run.sh pipeline resume episode-s07e01 --config config.yaml

# Approve (publish the review file, including manual edits) or reject
./run.sh pipeline review episode-s07e01 approve --config config.yaml
```

Pass `--debug` to any pipeline command to expose provider HTTP traffic and raw
model responses.

### Modes and stages

`parallel_initial` wraps Subtitle Edit's official headless `seconv`. On first
use it clones the configured repository revision, builds `seconv`, and runs
two passes over the English source: `first_pass_operations` applies the full
configured cleanup together with the multiple-replace template, then
`second_pass_operations` runs only `FixUnneededSpaces` on that first-pass SRT.
The cleaned SRT is translated in memoryless parallel batches
(`primer.batch_size`, `primer.max_workers`), merged with the English cues into
a bilingual ASS, and handed to serial refinement.

`serial_memory` skips preprocessing and initial translation and refines an
existing bilingual ASS directly.

Serial refinement (`refine.batch_size` pairs per request, or token-based
chunking when it is `null`) keeps an episode memory: the authoritative user
glossary from the prompt template, learned terminology with confidence and
evidence ids, and a cumulative story description. After every chunk the
subtitle artifact, then `memory.yaml`, then `refine-progress.json` are written
atomically, so a resumed run can never advance memory beyond the saved output.

Deterministic cleanup is the ordered allowlist under
`postprocess.operations`; show-specific replacements live under
`postprocess.episode_replacements` and run only when the
`episode_replacements` operation is enabled.

After deterministic cleanup, the run freezes a cue manifest and an effective
glossary. The user glossary and configured episode replacements are
authoritative; learned entries must pass evidence and conflict validation.
Rank-and-name candidates can trigger two independent Exa-backed research
passes. Search summaries are only discovery hints: accepted findings require
bounded public-page evidence, compatible A/B conclusions, and deterministic
revalidation. Missing credentials, insufficient evidence, or exhausted
research budgets become human-review items.

QA first audits structure, then the `agent` role audits overlapping windows.
It can only write a persisted suggestion pool. The separate `repair` role may
inspect context and approved reference roots, dismiss or merge suggestions,
open missed issues, and stage complete 1--3 cue groups. The host enforces cue
identity, ASS structure, the frozen glossary, episode replacements, base
hashes, and atomic group application. A mandatory full-episode sweep runs even
when QA reports no issues. Postprocessed candidates are re-audited within the
same persistent run budget; repeated issues reuse their prior decisions rather
than silently reopening.

The review candidate is exported only inside the run directory. Approval
copies that reviewed artifact to an immutable approved snapshot; the separate
release action publishes it to the requested destination.

### Run state

Each run lives in `pipeline.state_dir/<thread-id>/`. `run.json` is the
authoritative commit point; it records immutable artifact versions, dependency
hashes, prompt/configuration hashes, repair/research budgets, and the current
heads. Repair checkpoints are immutable generations whose pointer is moved
only after every artifact and audit exchange has been written. Resume rejects
changed dependencies or malformed state instead of guessing how to continue.
A thread id can be started only once; use `resume` to continue it.

## Standalone Refinement CLI

```bash
./run.sh input.ass output.ass [options]
```

| Option | Effect |
| --- | --- |
| `--config PATH` | YAML configuration (default: repository `config.yaml`) |
| `--checkpoint` | Persist episode memory to `<input>.memory.yaml` and load it on later runs |
| `--checkpoint-path PATH` | Explicit memory checkpoint path (implies `--checkpoint`) |
| `--progress-manifest PATH` | Write the strict `refine-progress.json` after every chunk (requires a memory checkpoint) |
| `--resume INDEX` | Continue from pair INDEX, preserving earlier pairs from the existing output |
| `--refine-batch-size N` | Pairs per request (overrides token-based chunking) |
| `--max-chunks N` | Stop after N chunks |
| `--dry-run` | Process only the first 10 pairs |
| `--memory-limit N` | Memory token limit before compression |
| `--model NAME` | Override the refine role's model name |
| `--intermediate-representation {json,xml-pair,pseudo-toml}` | Override the request format |
| `--stream` | Stream model output to the terminal |
| `-v` / `-vv` | Progress logging is on by default; `-vv` switches to DEBUG (prompts, raw responses, provider traffic) |
| `--test-connection` | Send a one-line request to the refine role and exit |

Output is written after every chunk, so an interrupted run leaves a complete,
valid ASS file containing the pairs refined so far. A token usage report and,
when the model is listed in the
[Claude Code Hub price table](https://cch-plus.com/pricing/v1/models.json), an
estimated cost are printed at the end; a failed price lookup is logged and
ignored.

`./run.sh genreq input.ass --refine-batch-size N` writes the exact system and
user prompts for every chunk to a Markdown file without calling any API.

## Configuration

All settings live in [config.yaml](config.yaml); every section is required and
paths are relative to the YAML file.

```yaml
api:                    # five roles: primer, refine, extraction, agent, repair
  refine:
    protocol: google-gemini          # openai-responses | openai-chat-compatible | anthropic-messages | google-gemini
    model: gemini-3.8-flash
    key_file: key-openlux
    base_url: https://api.openlux.ai
    timeout: 800                     # seconds, or null
    max_retries: 2                   # provider SDK retries
    max_output_tokens: 27000
    reasoning_effort: high           # or null
    temperature: 0.6                 # or null
pipeline:
  state_dir: .subretrans-runs
  checkpoint_db: .subretrans-runs/pipeline.sqlite3
prompts:
  shared_path: prompts/shared_translation_rules.md
  refine_path: prompts/refine_task.md
  qa_path: prompts/qa_task.md
  repair_path: prompts/repair_task.md
primer:     { batch_size, max_workers, source_language, target_language, user_instruction }
refine:     { batch_size, chunk_token_soft_limit, memory_token_limit, intermediate_representation }
qa:         { batch_size, max_workers, window_offsets }
repair:     { max_tool_steps, max_full_sweeps, max_repair_attempts, context_radius,
              max_group_span, max_glossary_repair_attempts }
reference_roots: [../bsub]  # exactly one read-only subtitle-reference root
research:   { exa_key_file, timeout, max_requests, max_fetches_per_request,
              max_response_bytes }
postprocess: { operations, episode_replacements }
subtitle_edit: { repository_url, revision, source_dir, build_dir, dotnet_executable,
                 settings_file, multiple_replace_file, first_pass_operations, second_pass_operations }
glossary:   { max_entries, terminology_min_confidence }
```

- **Roles** share the same strict fields; to point one role at another
  endpoint change only its `protocol`, `model`, `key_file`, and `base_url`.
- **Glossary**: the user glossary parsed from the prompt template is always
  authoritative. Learned terms never override it, are pruned when they collide
  with it, and are kept only above `terminology_min_confidence`.
- `config_gpt55.yaml` is an alternative profile using OpenAI-compatible
  endpoints; select it with `--config`.

## Prompt System

Prompts are Markdown files under `prompts/`:

- Refine system prompt = `shared_translation_rules.md` + `refine_task.md`
- QA system prompt = `shared_translation_rules.md` + `qa_task.md`
- Repair system prompt = `shared_translation_rules.md` + `repair_task.md`

The shared file holds the Chinese style rules, JAG-specific context, the
`### User Terminology (Authoritative Glossary)` list, and the cross-line
alignment rules. At run time the glossary section is rebuilt from the template
entries plus the learned terminology, an `### Incremental Story Description`
block is inserted after it, sections are renumbered, and the few-shot examples
are converted to the configured intermediate representation. A template
without the glossary section is a configuration error. New pipeline runs hash
the config file and all four prompt files, so editing a prompt invalidates old
downstream progress instead of silently reusing it.

## Project Structure

```
.
├── run.sh                          # Entry point: cli, genreq, pipeline
├── config.yaml                     # Single strict configuration
├── prompts/                        # Shared rules plus refine, QA, and repair tasks
├── subtitle_edit_settings.json     # Subtitle Edit batch-convert profile
├── subtitle_edit_multiple_replace.template
├── subretrans/
│   ├── config.py                   # AppConfig loader
│   ├── providers.py                # LangChain model construction and invoke_text
│   ├── fsutil.py                   # Atomic writes, hashing, strict field checks
│   ├── refine.py                   # Serial refine engine and progress manifests
│   ├── prompts.py                  # Prompt composition and memory injection
│   ├── memory.py                   # Episode memory, glossary lock, compression
│   ├── serializers.py              # json / xml-pair / pseudo-toml representations
│   ├── chunker.py, utils.py, pairs.py, stats.py, pricing.py
│   ├── ass_parser.py               # ASS parsing, pairing, rendering
│   ├── subtitle_processing.py      # SRT/ASS merge, postprocess, structural QA
│   ├── subtitle_edit.py            # Pinned seconv build and two-pass cleanup
│   ├── translation.py, model_translation.py   # Memoryless parallel first pass
│   ├── model_agent.py              # Read-only semantic QA suggestions
│   ├── repair.py                   # Bounded repair tools, ledger, atomic groups
│   ├── cue_manifest.py, glossary_validation.py
│   ├── research.py, webfetch.py, reference_reader.py
│   ├── run_manifest.py             # Authoritative immutable run commit point
│   ├── pipeline.py, state.py       # LangGraph graph and minimal checkpoint state
│   ├── stage_handlers.py           # Filesystem stage implementations
│   ├── pipeline_cli.py             # run / status / resume / review
│   ├── cli.py                      # Standalone refinement CLI
│   └── genreq.py                   # Prompt dump without API calls
├── test/                           # pytest suite
├── requirements.txt
└── requirements-dev.txt
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q test
```

All tests run offline with fake models.

## Limitations

- English-Chinese pairs in the `.ass` layout shown by `example_input.ass`;
  the pipeline additionally accepts any input format Subtitle Edit can read.
- Exact timestamp matches are required between paired English and Chinese
  events in standalone mode.
- One file per run; series-level memory across episodes is not implemented.

## Future Enhancements

- Series memory shared across episodes.
- Diff report generation showing every change.
- Quality scoring metrics.

## License

MIT, see [LICENCE.md](LICENCE.md).
