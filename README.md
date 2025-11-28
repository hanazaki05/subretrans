# Subtitle Refinement Tool

A Python tool for refining bilingual (English-Chinese) ASS subtitles using Large Language Models (LLMs).

## Features

- **Smart ASS Parsing**: Parses `.ass` subtitle files and matches English-Chinese pairs by timestamp
- **Intelligent Chunking**: Splits subtitles into chunks that fit within LLM token limits
- **Bilingual Refinement**:
  - **English**: Fixes capitalization, spacing, and punctuation only (preserves meaning)
  - **Chinese**: Improves translation quality, naturalness, and consistency
- **Global Memory**: Maintains terminology glossary and style notes across chunks
- **ASS Tag Preservation**: Keeps all formatting tags (e.g., `{\i1}`, `{\b1}`, `\N`) intact
- **Token Tracking**: Monitors API usage and estimates costs in real-time
- **Robust Error Handling**: Automatic retries with exponential backoff
- **Progress Reporting**: Real-time progress updates during processing

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key (or edit config.py)
export OPENAI_API_KEY="your-api-key-here"

# 4. Process subtitles
python main.py input.ass output.ass

# 5. Test with sample (first 10 pairs)
python main.py input.ass output.ass --dry-run
```

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

4. **Configure API key**:

   Option A - Environment variable (recommended):
   ```bash
   export OPENAI_API_KEY="sk-proj-..."
   ```

   Option B - Edit [config.py](config.py):
   ```python
   api_key: str = "sk-proj-..."
   ```

5. **Test the installation**:
```bash
python main.py --test-connection input.ass output.ass
```

## Usage

### Basic Usage

```bash
python main.py input.ass output.ass
```

This will:
1. Parse the input `.ass` file
2. Extract English-Chinese subtitle pairs
3. Process them through the LLM for refinement
4. Write the refined subtitles to `output.ass`
5. Display token usage and cost estimation

### Command Line Options

```
usage: main.py [-h] [--model MODEL] [--dry-run] [--max-chunks MAX_CHUNKS]
               [--memory-limit MEMORY_LIMIT] [--pairs-per-chunk PAIRS_PER_CHUNK]
               [-v] [--stats STATS] [--test-connection]
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
  --pairs-per-chunk N   Number of subtitle pairs per chunk (overrides token-based chunking)
  -v, --verbose         Enable verbose output with timing and preview
  --stats STATS         Stats refresh interval in seconds for verbose mode (default: 1.0)
  --test-connection     Test API connection and exit
```

### Examples

```bash
# 1. Basic processing (token-based chunking)
python main.py input.ass output.ass

# 2. Quick test with sample data (recommended for first use)
python main.py input.ass output.ass --dry-run

# 3. Process with fixed chunk size (50 pairs per chunk)
python main.py input.ass output.ass --pairs-per-chunk 50

# 4. Process only first 3 chunks
python main.py input.ass output.ass --max-chunks 3

# 5. Combine chunk size with max chunks (30 pairs per chunk, max 2 chunks)
python main.py input.ass output.ass --pairs-per-chunk 30 --max-chunks 2

# 6. Use a different model
python main.py input.ass output.ass --model gpt-4o

# 7. Increase memory limit for better context
python main.py input.ass output.ass --memory-limit 3000

# 8. Test API connection before processing
python main.py input.ass output.ass --test-connection

# 9. Enable verbose mode with timing and response preview
python main.py input.ass output.ass -v

# 10. Verbose mode with custom stats interval
python main.py input.ass output.ass -v --stats 0.5
```

### Running the Example Script

```bash
chmod +x example_usage.sh
./example_usage.sh
```

## Project Structure

```
subretrans/
├── main.py                  # CLI entry point and workflow orchestration
├── config.py                # Configuration settings (API, tokens, pricing)
├── ass_parser.py            # ASS file parsing and generation
├── pairs.py                 # SubtitlePair data structure
├── chunker.py               # Smart chunk splitting with token limits
├── llm_client.py            # OpenAI API client with retry logic
├── memory.py                # Global memory management
├── prompts.py               # System and user prompt templates
├── stats.py                 # Token usage statistics and cost tracking
├── utils.py                 # Utility functions (token estimation, etc.)
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
     - **Pair-based** (with `--pairs-per-chunk`): Fixed number of pairs per chunk
   - Accounts for system prompt and memory overhead

4. **Process Each Chunk**
   - Builds system prompt with refinement rules + global memory
   - Sends subtitle pairs as JSON to LLM
   - Parses and validates LLM response
   - Updates global memory with new terminology

5. **Memory Management**
   - Extracts proper nouns and terminology
   - Maintains cross-chunk context
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
python main.py input.ass output.ass
```

#### 2. Pair-Based Chunking
- **How it works**: Splits subtitles into fixed-size chunks by pair count
- **Advantages**: Predictable chunk sizes, easier cost estimation
- **Best for**: Consistent processing, testing, batch operations
- **Usage**: Specify with `--pairs-per-chunk N`

```bash
# Process 50 pairs at a time
python main.py input.ass output.ass --pairs-per-chunk 50

# Smaller chunks for testing
python main.py input.ass output.ass --pairs-per-chunk 10
```

**Tip**: Combine with `--max-chunks` to limit processing:
```bash
# Process first 100 pairs only (50 pairs/chunk × 2 chunks)
python main.py input.ass output.ass --pairs-per-chunk 50 --max-chunks 2
```

### Verbose Mode

The tool supports verbose mode for detailed progress tracking:

#### Enabling Verbose Mode
```bash
# Basic verbose mode (timing + preview)
python main.py input.ass output.ass -v

# Very verbose (-vv) dumps full API responses after each chunk
python main.py input.ass output.ass -vv

# Ultra verbose (-vvv) also prints the full system prompt/memory sent to the model
python main.py input.ass output.ass -vvv

# Verbose with custom stats interval
python main.py input.ass output.ass -v --stats 0.5
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

## Configuration

Edit [config.py](config.py) to customize:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MainModelSettings:
    name: str = "gpt-5.1"
    max_output_tokens: int = 12000
    reasoning_effort: str = "medium"
    temperature: float = 1.0      # GPT-5.* has no temperature setting


@dataclass
class TerminologyModelSettings:
    name: str = "gpt-4o-mini"
    max_output_tokens: int = 1500
    temperature: float = 0.3      # Lower temperature keeps glossary extraction stable


@dataclass
class Config:
    api_key: str = ""
    api_base_url: str = "https://api.openai.com/v1"
    max_context_tokens: int = 128000
    memory_token_limit: int = 2000
    chunk_token_soft_limit: int = 100000
    pairs_per_chunk: Optional[int] = None
    api_timeout: int = 120
    verbose: bool = False
    very_verbose: bool = False
    debug_prompts: bool = False
    stats_interval: float = 1.0
    dry_run: bool = False
    max_chunks: Optional[int] = None
    price_per_1k_prompt_tokens: float = 0.03
    price_per_1k_completion_tokens: float = 0.06
    glossary_max_entries: int = 100
    main_model: MainModelSettings = field(default_factory=MainModelSettings)
    terminology_model: TerminologyModelSettings = field(default_factory=TerminologyModelSettings)
```

- **Reasoning effort**: Adjust `config.main_model.reasoning_effort` to hint GPT-5.1's reasoning depth (`"none"`, `"low"`, `"medium"`, `"high"`).
- **Temperature**: Tune `config.main_model.temperature` (defaults to `1.0`) for the primary GPT-5 run, and `config.terminology_model.temperature` for the GPT-4o terminology extractor if you need stricter or looser extraction.
- **Glossary limit**: Adjust `config.glossary_max_entries` (default `100`) if you need to retain more/fewer global terminology entries in the prompt.

### Using Alternative API Providers

To use a different OpenAI-compatible API:

```python
# In config.py
api_base_url: str = "https://your-api-endpoint.com/v1"

@dataclass
class MainModelSettings:
    name: str = "your-model-name"
    ...
```

## Error Handling

The tool includes robust error handling:

- **API Errors**
  - Automatic retry with exponential backoff (3 attempts)
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

### Pricing (Default)
- Prompt tokens: $0.03 per 1K tokens
- Completion tokens: $0.06 per 1K tokens

**Note**: Actual costs may vary based on your OpenAI plan and model used. Update pricing in [config.py](config.py) for accurate estimates.

## Troubleshooting

### Common Issues

**1. API Key Error**
```
Configuration error: API key must be provided
```
**Solution**: Set `OPENAI_API_KEY` environment variable or edit [config.py](config.py)

**2. Model Not Found**
```
API request failed: model 'gpt-5.1' not found
```
**Solution**: Use `--model gpt-4o` or another available model

**3. Token Limit Exceeded**
```
API request failed: maximum context length exceeded
```
**Solution**: Reduce `chunk_token_soft_limit` in [config.py](config.py)

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

For verbose output, modify [main.py](main.py) to add debug logging:

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

## Future Enhancements

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

This tool is provided as-is for subtitle refinement purposes. No warranty is provided.

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
python main.py test_input.ass test_output.ass --dry-run
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

**Version**: 0.0.4
**Last Updated**: November 28, 2025
**Status**: Use at your own dangers
