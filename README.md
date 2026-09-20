# Subtitle Refinement Tool

A Python tool for refining bilingual (currently English-Chinese) ASS subtitles using Large Language Models (LLMs). With a basic level **memory** support, it allows you to resume progress via the checkpoint file. This implementation is especially suitable when you're looking for translation quality over batch speed. 

## Features

**Code** lives in `subretrans/`; run the tool with `./run.sh`.

- **Smart ASS Parsing**: Parses `.ass` subtitle files and matches English-Chinese pairs by timestamp
- **Intelligent Chunking**: Splits subtitles into chunks that fit within LLM token limits
- **Bilingual Refinement**:
  - **English**: Fixes capitalization, spacing, and punctuation only (preserves meaning)
  - **Chinese**: Improves translation quality, naturalness, and consistency
- **Episode Memory**: Maintains terminology, style notes, and a cumulative `Incremental Story Description` across chunks; the complete memory can be resumed from a checkpoint.
- **Agent Pipeline**: Supports two checkpointable modes: `parallel_initial` uses memoryless parallel first-pass translation followed by serial memory-aware proofreading; `serial_memory` uses that same serial flow to translate and proofread directly. An agent model then performs semantic QA and applies bounded, targeted repairs before human review.
- **ASS Tag Preservation**: Keeps all formatting tags (e.g., `{\i1}`, `{\b1}`, `\N`) intact
- **Token Tracking**: Monitors API usage and estimates costs in real-time
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Progress Reporting**: Real-time progress updates during processing

## Quick Start

<details>
<summary>(no longer suggested as it's the old version) - Click to expand</summary>

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set the role key_file paths in config.yaml

# 4. Process subtitles (not suggested)
./run.sh example_input.ass output.ass

# 5. Test with sample (first 10 pairs)(not suggested)
./run.sh example_input.ass output.ass --dry-run
```

</details>

We recommend using the executable script in the repository root:
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (IMPORTANT) Set you key in a file, and set its path in ./config.yaml
echo "YOUR_KEY" > key

# 4. Process subtitles (not suggested)
./run.sh example_input.ass output.ass \
--stream --refine-batch-size 105 \
--checkpoint --per-block-update -vvv
```

### Current limit:
The ASS subtitle pairs are detected accorading to `example_input.ass` file, so you need to follow this format.

## Agent Pipeline

The persistent pipeline has two modes:

```bash
# Subtitle Edit preprocessing -> parallel initial translation -> serial proofreading
./run.sh pipeline run source.en.srt release.ass \
  --mode parallel_initial --thread-id episode-s07e01 --config config.yaml

# Existing ASS -> serial translation/proofreading with incremental episode memory
./run.sh pipeline run bilingual.ass release.ass \
  --mode serial_memory --thread-id episode-s07e02 --config config.yaml

# Resume the final human-review gate
./run.sh pipeline review episode-s07e01 approve --config config.yaml

# Resume a failed pipeline from its latest checkpoint
./run.sh pipeline resume episode-s07e01 --config config.yaml
```

`primer.batch_size` controls the number of source cues in each parallel
initial-translation request. `refine.batch_size` independently controls the
serial refinement chunk size; set it to `null` to retain token-based chunking.
`primer.max_workers` applies only to primer requests. Refine serialization is
selected by `refine.intermediate_representation`.
`qa.batch_size` controls how many bilingual pairs each semantic-QA request
audits. Pipeline commands report committed refine and QA progress at `INFO`;
pass `--debug` to expose provider HTTP and raw model-response diagnostics.
Every QA window receives the final refine `memory.yaml` as read-only structured
context: cumulative story description, authoritative user glossary, and the
complete learned glossary including confidence and evidence IDs. QA checkpoints are bound to the memory hash, so
changing that context invalidates old QA progress instead of silently reusing it.

### TODO: Rank-aware name verification

When QA detects a full personal name paired with a military rank, run two
independent Exa research passes before proposing a terminology decision. The
first pass should identify the person, service branch, and rank context; the
second should independently verify the Chinese rank translation in the
relevant military or episode context. Add a learned term only when both passes
agree. Conflicting or insufficient evidence must remain a human-review item and
must not override `user_glossary`.

The run writes the human-review candidate beside the input subtitle, inserting
`.review` before the output extension (for example,
`JAG.S07E07.en-cn.review.ass`). Approval publishes that review file, including
any manual edits, to the requested output path.

`parallel_initial` wraps Subtitle Edit's official headless `seconv` project.
On first use it clones the configured repository revision, builds `seconv`, and
converts/cleans the source into the run's UTF-8 SRT artifact. Before primer,
the English subtitle passes through Subtitle Edit twice: `first_pass_operations`
runs the full configured cleanup together with the multiple-replace template,
then `second_pass_operations` runs only `FixUnneededSpaces` on that first-pass
SRT. The second pass does not load the first-pass settings file or repeat the
multiple-replace template. Building the
pinned source requires the .NET 10 SDK/runtime. The generic ASS normalization
and structural QA live in `subtitle_processing.py`. Deterministic cleanup is an
ordered allowlist under `postprocess.operations`; removing an operation disables
it. Show/episode replacements are configured separately under
`postprocess.episode_replacements` and run only when the `episode_replacements`
operation is enabled. The default `subtitle_edit` section reproduces the checked options
from Subtitle Edit's Batch convert window through
`subtitle_edit_settings.json`, `subtitle_edit_multiple_replace.template`, and
the explicit `first_pass_operations` and `second_pass_operations` lists.

`parallel_initial` never receives glossary or story memory, so its batches can
run independently. The following serial refinement stage maintains the
incremental story description. `serial_memory` skips initial translation and
uses that same serial refinement path directly on an existing ASS artifact.

After deterministic structural checks, the `agent` API audits semantic
completeness, accuracy, and consistency. A failed audit may return targeted
Chinese replacements. Each repair is written to a new run artifact,
normalized, and audited again. `pipeline.agent_max_repair_attempts` bounds
this loop; a failure without repairs or an exhausted budget proceeds to human
review rather than restarting the entire pipeline.

## Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key (or compatible API endpoint)

### Step-by-Step Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure the four API roles in [config.yaml](config.yaml)**. Each role points to its own key file:
   ```yaml
   api:
     primer:
       key_file: "key"
     refine:
       key_file: "key"
     extraction:
       key_file: "key"
     agent:
       key_file: "key"
   ```

5. **Test the installation**:
```bash
./run.sh --test-connection input.ass output.ass
```

## Usage

### Basic Usage

```bash
./run.sh input.ass output.ass
```

This will:
1. Parse the input `.ass` file
2. Extract English-Chinese subtitle pairs
3. Process them through the LLM for refinement
4. Write the refined subtitles to `output.ass`
5. Display token usage and cost estimation

### Command Line Options

```
usage: ./run.sh [-h] [--model MODEL] [--dry-run] [--max-chunks MAX_CHUNKS]
               [--memory-limit MEMORY_LIMIT] [--refine-batch-size PAIRS_PER_CHUNK]
               [-v] [--test-connection]
               input output

positional arguments:
  input                 Input .ass subtitle file
  output                Output .ass subtitle file

optional arguments:
  -h, --help            Show this help message and exit
  --model MODEL         Model name (default: gpt-5.1)
  --dry-run             Process only first 10 pairs for testing
  --max-chunks N        Process only first N chunks
  --memory-limit N      Memory token limit (default: 2000)
  --refine-batch-size N   Number of subtitle pairs per chunk (overrides token-based chunking)
  -v, --verbose         Enable verbose output with timing and preview
  --test-connection     Test API connection and exit
```

### Examples

```bash
# 1. Basic processing (token-based chunking)
./run.sh input.ass output.ass

# 2. Quick test with sample data (recommended for first use)
./run.sh input.ass output.ass --dry-run

# 3. Process with fixed chunk size (50 pairs per chunk)
./run.sh input.ass output.ass --refine-batch-size 50

# 4. Process only first 3 chunks
./run.sh input.ass output.ass --max-chunks 3

# 5. Combine chunk size with max chunks (30 pairs per chunk, max 2 chunks)
./run.sh input.ass output.ass --refine-batch-size 30 --max-chunks 2

# 6. Use a different model
./run.sh input.ass output.ass --model gpt-4o

# 7. Increase memory limit for better context
./run.sh input.ass output.ass --memory-limit 3000

# 8. Test API connection before processing
./run.sh input.ass output.ass --test-connection

# 9. Enable verbose mode with timing and response preview
./run.sh input.ass output.ass -v

```

### Running the Example Script

```bash
chmod +x example_usage.sh
./example_usage.sh
```

## Project Structure

```
.
├── run.sh                   # CLI entry point and workflow orchestration
├── config.yaml              # Unified pipeline and model-role configuration
├── subretrans/
│   ├── cli.py               # CLI implementation and workflow orchestration
│   ├── config.py            # Configuration loading and settings
│   ├── ass_parser.py        # ASS file parsing and generation
│   ├── pairs.py             # SubtitlePair data structure
│   ├── chunker.py           # Smart chunk splitting with token limits
│   ├── llm.py               # OpenAI API client with retry logic
│   ├── providers.py          # OpenAI Responses, Anthropic, Gemini, and legacy adapters
│   ├── pipeline.py           # Checkpointable agent workflow graph
│   ├── state.py              # Persistent pipeline state schema
│   ├── memory.py             # Glossary and incremental episode story memory
│   ├── prompts.py           # System and user prompt templates
│   ├── stats.py             # Token usage statistics and cost tracking
│   └── utils.py             # Utility functions (token estimation, etc.)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── example_usage.sh         # Example usage script
├── IMPLEMENTATION_SUMMARY.md # Detailed implementation notes
└── venv/                    # Virtual environment (created during setup)
```

## How It Works

### Processing Workflow

1. **Parse ASS File**
   - Reads `.ass` file with UTF-8-sig encoding
   - Preserves header section ([Script Info], [V4+ Styles])
   - Extracts all Dialogue lines from [Events] section

2. **Build Subtitle Pairs**
   - Matches English and Chinese lines by timestamp
   - Identifies lines by style name (e.g., "English3", "Chinese3")
   - Preserves all metadata (timing, style, margins, effects)

3. **Split into Chunks**
   - Two chunking strategies available:
     - **Token-based** (default): Uses tiktoken to fit chunks within context window
     - **Pair-based** (with `--refine-batch-size`): Fixed number of pairs per chunk
   - Accounts for system prompt and memory overhead

4. **Process Each Chunk**
   - Builds system prompt with refinement rules + global memory
   - Sends subtitle pairs as JSON to LLM
   - Parses and validates LLM response
   - Updates terminology and the cumulative episode story description

5. **Memory Management**
   - Extracts proper nouns and terminology
   - Maintains a concise, evidence-only story description of the episode so far
   - Saves the complete memory to `.memory.yaml` when checkpointing is enabled
   - Automatically compresses if memory exceeds limit

6. **Generate Output**
   - Applies corrections back to original structure
   - Preserves all ASS formatting and tags
   - Writes complete `.ass` file with refined subtitles

### Chunking Strategies

The tool supports two chunking strategies:

#### 1. Token-Based Chunking (Default)
- **How it works**: Automatically calculates optimal chunk size based on token limits
- **Advantages**: Maximizes context window usage, reduces API calls
- **Best for**: Most use cases, especially with varying subtitle lengths
- **Usage**: Default behavior (no flag needed)

```bash
./run.sh input.ass output.ass
```

#### 2. Pair-Based Chunking
- **How it works**: Splits subtitles into fixed-size chunks by pair count
- **Advantages**: Predictable chunk sizes, easier cost estimation
- **Best for**: Consistent processing, testing, batch operations
- **Usage**: Specify with `--refine-batch-size N`

```bash
# Process 50 pairs at a time
./run.sh input.ass output.ass --refine-batch-size 50

# Smaller chunks for testing
./run.sh input.ass output.ass --refine-batch-size 10
```

**Tip**: Combine with `--max-chunks` to limit processing:
```bash
# Process first 100 pairs only (50 pairs/chunk × 2 chunks)
./run.sh input.ass output.ass --refine-batch-size 50 --max-chunks 2
```

### Verbose Mode

The tool supports verbose mode for detailed progress tracking:

#### Enabling Verbose Mode
```bash
# Basic verbose mode (timing + preview)
./run.sh input.ass output.ass -v

# Very verbose (-vv) dumps full API responses after each chunk
./run.sh input.ass output.ass -vv

# Ultra verbose (-vvv) also prints the full system prompt/memory sent to the model
./run.sh input.ass output.ass -vvv

```

#### Verbose Output Includes:
1. **Chunk Processing Time**: Shows elapsed time for each chunk
   - Example: `Time: 16.51s`

2. **Response Preview + Reasoning Tokens**: Real-time preview of LLM output
   - Line 1-2: First two lines of returned subtitle pairs (JSON flattened to plain-text)
   - Line 3: Reasoning tokens consumed (from API usage data)

3. **Token Statistics**: Standard token usage per chunk
4. **Full API Response (optional)**: Use `-vv` to print the entire raw API response after each chunk (useful for debugging JSON issues)
5. **System Prompt & Memory (optional)**: Use `-vvv` to print the exact system prompt (including memory) sent to the model for each chunk

**Example Verbose Output:**
```
Processing chunk 1/2 (30 pairs)...

  [Chunk 1/2] (50.0% complete)
    Tokens used: 3,092 (prompt: 1,726, completion: 1,366)
    Time: 16.51s

  Response: [
            {
  Reasoning tokens: 8

```

**When to use:**
- Debugging processing issues
- Monitoring long-running jobs
- Analyzing response patterns
- Performance optimization

## Refinement Rules

### English Subtitles

The tool applies minimal changes to English subtitles:

- ✅ **Fix capitalization**: First letter of sentences capitalized
  - Before: `"tonight, on JAG..."`
  - After: `"Tonight, on JAG..."`

- ✅ **Fix spacing**: Proper spacing around punctuation
  - Before: `"Hello,world"`
  - After: `"Hello, world"`

- ✅ **Fix ending punctuation**: Add periods to complete sentences
  - Before: `"Good evening"`
  - After: `"Good evening."`

- ❌ **Do NOT change**: Words, meanings, or phrasing
- ✅ **Preserve**: All ASS tags (`{\i1}`, `{\b1}`, `\N`, etc.)

### Chinese Subtitles

The tool applies comprehensive improvements to Chinese subtitles:

- ✅ **Translation quality**: Improve accuracy and clarity
  - Before: `"军法署"`
  - After: `"《JAG军法官》节目中"`

- ✅ **Natural language**: Make text more conversational
  - Before: `"报告"`
  - After: `"报道"`

- ✅ **Punctuation**: Add proper Chinese punctuation (。、！？等)
  - Before: `"晚上好，我是诺曼·德拉波特"`
  - After: `"晚上好，我是诺曼·德拉波特。"`

- ✅ **Consistency**: Maintain terminology and style across chunks
- ✅ **Awkward phrasing**: Fix unnatural expressions
- ❌ **Do NOT change**: ASS formatting tags

## Testing & Performance

### Test Results

Tested with first 152 subtitle pairs from `JAG.S04E08.zh-cn.ass`:

```
Input:         152 pairs (304 dialogue lines)
Model:         GPT-5.1
Processing:    ~30 seconds
Chunks:        1 chunk
Tokens used:   12,993 total
  - Prompt:    6,604 tokens
  - Completion: 6,389 tokens
Estimated cost: $0.58 USD
Success rate:  100%
```

### Performance Metrics

- **Average tokens per pair**: ~85 tokens
- **Cost per pair**: ~$0.0038 USD
- **Processing speed**: ~5 pairs/second
- **Estimated cost for 1000 pairs**: ~$3.80 USD

### Quality Improvements Observed

**English refinements:**
- Capitalization fixes: ~30% of lines
- Punctuation additions: ~40% of lines
- Spacing fixes: ~5% of lines

**Chinese refinements:**
- Translation improvements: ~20% of lines
- Punctuation additions: ~90% of lines
- Natural phrasing: ~15% of lines
- All ASS tags preserved: 100%

## Template-Based Prompt System (v0.0.6)

The system prompt is now generated from a **single markdown template file** (`main_prompt.md`):

### How It Works

1. **Template Structure**: The template uses markdown sections (`### 1. English Subtitle Rules`, etc.)
2. **Dynamic Injection**: The `### 4. User Terminology (Authoritative Glossary)` section is dynamically updated with:
   - Template glossary entries (parsed from the file)
   - Runtime `GlobalMemory.user_glossary` entries (merged, runtime takes precedence)
   - Learned terminology (appended as "Learned Terminology (Supplement)")
3. **Automatic Renumbering**: All sections are renumbered automatically

### Template Sections

```markdown
### 1. English Subtitle Rules
### 2. Chinese Subtitle Rules
### 3. Context & Specific Handling
### 4. User Terminology (Authoritative Glossary)  ← Dynamic injection point
### 5. Input/Output Format & Constraint
### 6. Few-Shot Examples
```

### Benefits

- **Single source of truth** - All rules in one markdown file
- **Easy customization** - Edit markdown without code changes
- **Dynamic terminology** - Automatic glossary injection from GlobalMemory
- **Backward compatible** - Falls back to legacy prompt building if no config provided

## Configuration

All model endpoints live in one strict suite in [config.yaml](config.yaml):

```yaml
api:
  primer: &role
    protocol: openai-responses
    model: gpt-5.5
    key_file: key
    base_url: https://api.openai.com/v1
    timeout: 800
    max_retries: 0
    max_output_tokens: 27000
    reasoning_effort: high
    temperature: null
  refine: *role
  extraction: *role
  agent: *role

pipeline:
  agent_max_repair_attempts: 2

qa:
  batch_size: 64
  max_workers: 4
  window_offsets: [0, 32]
```

- **API roles**: `primer`, `refine`, `extraction`, and `agent` are mandatory and use the same strict fields. Key paths are relative to the selected YAML file.
- **Protocols**: Choose `openai-responses`, `openai-chat-compatible`, `anthropic-messages`, or `google-gemini`. Model calls use the provider-neutral adapters; streaming refinement is limited to OpenAI chat-compatible endpoints.
- **Reasoning and temperature**: Configure these independently for each role; use `null` when the endpoint does not accept the option.
- **Glossary policy**: `glossary.policy: lock` means learned terminology can add entries but cannot override user-defined mappings.
- **Terminology confidence**: `glossary.terminology_min_confidence` controls both the extraction prompt threshold and local filtering.

To use a different endpoint for one role, change only that role's `protocol`,
`model`, `key_file`, and `base_url` while retaining the remaining required
fields. Put only the provider token in the referenced `key_file`; `key-*` files
are ignored by Git.

## Error Handling

The tool includes robust error handling:

- **API Errors**
  - Automatic retry with exponential backoff up to each role's configured `max_retries`
  - Wait times: 1s, 2s, 4s between retries
  - Graceful failure with error messages

- **Chunk Processing Failures**
  - Failed chunks are skipped (not discarded)
  - Processing continues with remaining chunks
  - Original subtitles preserved for failed chunks

- **Memory Overflow**
  - Automatically compresses memory when limit exceeded
  - LLM-based compression to preserve important terms
  - Fallback to simple truncation if compression fails

- **JSON Parsing Errors**
  - Attempts to extract JSON from markdown code blocks
  - Validates structure before processing
  - Clear error messages for debugging

- **File I/O Errors**
  - Checks file existence before processing
  - Validates UTF-8 encoding
  - Creates output directory if needed

## Cost Estimation

The tool provides real-time cost tracking:

```
==================================================
TOKEN USAGE REPORT
==================================================
Prompt tokens:          6,604
Completion tokens:      6,389
Total tokens:          12,993
--------------------------------------------------
Estimated cost:    $    0.5815 USD
==================================================
```

Pricing is resolved by exact model name, slug, or alias from the
[Claude Code Hub price table](https://cch-plus.com/pricing/v1/models.json).
The report names the selected provider and table version. If no exact model is
present, token usage is still reported but no cost is guessed.

## Troubleshooting

### Common Issues

**1. API Key Error**
```
Configuration error: API key must be provided
```
**Solution**: Check each role's `key_file` path in [config.yaml](config.yaml)

**2. Model Not Found**
```
API request failed: model 'gpt-5.1' not found
```
**Solution**: Use `--model gpt-4o` or another available model

**3. Token Limit Exceeded**
```
API request failed: maximum context length exceeded
```
**Solution**: Reduce `chunk_token_soft_limit` in [config.yaml](config.yaml)

**4. No Subtitle Pairs Found**
```
Error: No subtitle pairs found
```
**Solution**: Ensure your `.ass` file has both English and Chinese dialogue lines with matching timestamps

**5. Import Error**
```
ModuleNotFoundError: No module named 'tiktoken'
```
**Solution**: Activate virtual environment and run `pip install -r requirements.txt`

### Debug Mode

For verbose output, modify [subretrans/cli.py](subretrans/cli.py) to add debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Output Example

### Before (Original)
```
Dialogue: 1,0:00:02.56,0:00:04.00,Chinese3,NTP,0000,0000,0000,,今晚，在军法署...
Dialogue: -1,0:00:02.56,0:00:04.00,English3,NTP,0000,0000,0000,,  Tonight, on JAG...
```

### After (Refined)
```
Dialogue: 1,0:00:02.56,0:00:04.00,Chinese3,NTP,0000,0000,0000,,今晚，在《JAG军法官》节目中...
Dialogue: -1,0:00:02.56,0:00:04.00,English3,NTP,0000,0000,0000,,Tonight, on JAG...
```

**Changes made:**
- English: Removed leading spaces, capitalized "Tonight"
- Chinese: Improved translation ("军法署" → "《JAG军法官》节目中")

## Limitations

- **Format Support**: Only supports `.ass` subtitle format (not `.srt`, `.vtt`, etc.)
- **Language Pair**: Designed for English-Chinese pairs only
- **Timestamp Matching**: Requires exact timestamp matches between English and Chinese lines
- **LLM Quality**: Output quality depends on the model used (GPT-5.1 recommended)
- **Edge Cases**: May not preserve all complex ASS formatting in rare cases
- **Single File Processing**: Processes one file at a time (no batch mode)
Here are two polished versions—you can choose the tone you prefer:
- **No compatibility with other API formats is guaranteed**: There are plans to support OpenAI-compatible APIs (e.g., NewAPI).

## Future Enhancements (TODO)

Priorty:
- [ ] Global across eposides via series memory file/ Redis?

Potential features for future versions:

- [ ] Support for `.srt` and `.vtt` formats
- [ ] Batch file processing
- [ ] Custom terminology dictionaries
- [ ] Diff report generation (showing all changes)
- [ ] GUI/Web interface
- [ ] Parallel chunk processing
- [ ] Quality scoring metrics
- [ ] Support for more language pairs

## Dependencies

```
tiktoken>=0.5.1      # OpenAI's token counting library
requests>=2.31.0     # HTTP client for API calls
python-dotenv>=1.0.0 # Environment variable management
```

All dependencies are listed in [requirements.txt](requirements.txt).

## License

This repository is licensed under the MIT License. The example subtitle is not part of the licensed content, and its copyright holder retains all rights.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd subretrans

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
./run.sh test_input.ass test_output.ass --dry-run
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for detailed technical documentation
- Review [plan.md](plan.md) for design decisions

## Acknowledgments

- Built using OpenAI's GPT models
- Uses `tiktoken` for accurate token counting
- Follows ASS subtitle format specification

---

**Version**: 0.0.6
**Last Updated**: December 1, 2025
**Status**: Use at your own dangers
