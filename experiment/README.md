# Experiment: OpenAI SDK Implementation

This directory contains an experimental implementation of the subtitle refinement tool using OpenAI's official SDK instead of direct HTTP POST requests.

## Overview

The main project uses `requests` library to make HTTP POST calls to OpenAI API. This experiment replaces that approach with the official OpenAI Python SDK, which provides:

- Better error handling and retry logic
- Automatic rate limiting
- Type-safe API calls
- Support for OpenAI `/chat/completion` api for both stream and non-stream option
- Better compatibility with OpenAI's evolving API

## Files

### Core Modules
- **config_sdk.py**: Configuration module with YAML-based settings 
- **config.yaml**: YAML configuration file for easy customization 
- **llm_client_sdk.py**: LLM client implementation using OpenAI SDK (supports both streaming and non-streaming)
- **main_sdk.py**: Complete subtitle refinement tool with streaming support

### Test Scripts
- **test_sdk.py**: Test script to verify the SDK implementation
- **test_streaming.py**: Test script for streaming API with performance comparison
- **test_yaml_config.py**: Test script for YAML configuration loading (✨ NEW)

### Documentation
- **README.md**: This file - project overview and quick start
- **USAGE.md**: Complete usage guide for main_sdk.py (quick start + advanced usage)
- **CONFIG_YAML.md**: YAML configuration guide with per-model API credential configuration guide (✨ NEW)
- **STREAMING_API.md**: Technical documentation for streaming API
- **REALTIME_STREAMING.md**: Real-time streaming output guide (✨ NEW)
- **RESUME_MODE.md**: Resume mode documentation for restarting from specific pair index (✨ NEW)
- **SUMMARY.md**: Project summary and completion report

## Installation

The OpenAI SDK and PyYAML have been installed in the virtual environment:

```bash
source venv/bin/activate
pip install openai pyyaml  # Already done
```

## Configuration

### YAML-Based Configuration (✨ NEW)

All settings are now managed in `config.yaml`. This makes it easy to customize without modifying code:

```yaml
# Example: Edit config.yaml
main_model:
  name: "gpt-5-mini"
  reasoning_effort: "medium"

user:
  prompt_path: "main_prompt.md"  # Main prompt template

runtime:
  dry_run: false
  verbose: false
```

**Quick Start:**
1. Edit `experiment/config.yaml` to customize settings
2. Run `python test_yaml_config.py` to verify configuration
3. See `CONFIG_YAML.md` for detailed guide

**CLI Overrides:**
Command line arguments override YAML settings:
```bash
python main_sdk.py input.ass output.ass --model gpt-4o --dry-run -v
```

### Per-Model API Credentials (✨ NEW)

You can now configure different API keys and endpoints for different models. This is useful for:
- **Cost tracking**: Separate billing for main vs terminology models
- **Multi-provider**: Route different models to different providers
- **Development/Testing**: Production endpoint for main model, local server for terminology
- **Load balancing**: Distribute requests across multiple endpoints

**Example configuration:**
```yaml
# config.yaml
api:
  key_file: "../key"
  base_url: "https://api.openai.com/v1"

main_model:
  name: "gpt-5-mini"
  max_output_tokens: 27000
  reasoning_effort: "low"
  temperature: 1.0
  # Override with custom endpoint
  key_file: "../key-main"  # Different API key
  base_url: "https://my-proxy.example.com/v1"  # Different endpoint

terminology_model:
  name: "gpt-4o-mini"
  max_output_tokens: 1800
  temperature: 0.45
  # Uses global api settings (no overrides)
```

**Verbose debugging (-vvv):**
```bash
python main_sdk.py input.ass output.ass -vvv
```

Output shows which credentials are being used:
```
  [Credential Resolution for gpt-5-mini]
    API Key: Model-specific (../key-main) [sk-proj-AbC...]
    Base URL: Model-specific → https://my-proxy.example.com/v1
```

See [PER_MODEL_CREDENTIALS.md](PER_MODEL_CREDENTIALS.md) for complete guide.

### Template-Based Prompt System (plan3.md)

The system prompt is now generated from a **single markdown template file** (`main_prompt.md`):

- All rules, examples, and terminology are defined in one file
- The `### 4. User Terminology (Authoritative Glossary)` section is dynamically updated with:
  - Template glossary entries (parsed from the file)
  - Runtime user glossary (merged, runtime takes precedence)
  - Learned terminology (appended as supplement)
- All sections are automatically renumbered

This approach provides:
- **Single source of truth** for all prompt rules
- **Easy customization** without code changes
- **Dynamic terminology injection** from GlobalMemory

## Usage

### 1. Test API Connection

**Non-streaming tests:**
```bash
cd experiment
python test_sdk.py
```

This will run three tests:
1. API connection test
2. Simple refinement with test data
3. Refinement with actual subtitle file (if `example_input.ass` exists)

**Streaming tests:**
```bash
python test_streaming.py
```

This will run four streaming tests:
1. Simple streaming with visual output
2. Subtitle refinement with streaming progress
3. Visual character-by-character streaming
4. Performance comparison (streaming vs non-streaming)

### 2. Use in Your Code

```python
from experiment.config_sdk import load_config_sdk
from experiment.llm_client_sdk import refine_chunk_sdk, test_api_connection_sdk
from pairs import SubtitlePair
from memory import GlobalMemory

# Load config (automatically reads API key from ../key file)
config = load_config_sdk(verbose=True)

# Test connection
if test_api_connection_sdk(config):
    print("Connected successfully!")

# Refine subtitle pairs
pairs = [...]  # Your SubtitlePair objects
global_memory = GlobalMemory()

corrected_pairs, usage, response = refine_chunk_sdk(
    pairs,
    global_memory,
    config
)
```

### 3. Integration with Main Project

The SDK implementation maintains compatibility with the main project's structure:

- Uses the same `SubtitlePair` data structure
- Uses the same `GlobalMemory` system
- Uses template-based prompt generation from `prompts.py` (with `main_prompt.md`)
- Uses the same `UsageStats` for tracking

You can easily swap between HTTP and SDK implementations by changing imports:

```python
# Original (HTTP POST)
from llm_client import refine_chunk, call_openai_api
from config import load_config

# SDK version
from experiment.llm_client_sdk import refine_chunk_sdk, call_openai_api_sdk
from experiment.config_sdk import load_config_sdk
```

## Key Differences from Main Implementation

### HTTP POST (Original)

```python
# llm_client.py
import requests

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=config.api_timeout
)
result = response.json()
```

### OpenAI SDK (Experiment)

```python
# llm_client_sdk.py
from openai import OpenAI

client = OpenAI(api_key=config.api_key)
response = client.chat.completions.create(
    model=target_model,
    messages=messages,
    max_completion_tokens=target_output_tokens
)
```

## Streaming API Usage

The SDK implementation now supports **streaming API** for real-time token generation!

### Non-Streaming (Default)

```python
from experiment.llm_client_sdk import refine_chunk_sdk

# Standard non-streaming call
corrected_pairs, usage, response = refine_chunk_sdk(
    pairs,
    global_memory,
    config
)
```

### Streaming with Real-Time Feedback

```python
from experiment.llm_client_sdk import refine_chunk_sdk_streaming

# Define a callback to handle streaming chunks
def print_progress(chunk: str):
    print(".", end="", flush=True)  # Show progress dots

# Streaming call with callback
corrected_pairs, usage, response = refine_chunk_sdk_streaming(
    pairs,
    global_memory,
    config,
    chunk_callback=print_progress
)
```

### Streaming Benefits

Based on our performance tests:

- **2.7x faster perceived response** - Get first token in 1.09s vs 2.91s total wait
- **Real-time feedback** - See progress as tokens are generated
- **Better UX** - Users see immediate activity instead of waiting
- **Same total time** - Overall completion time is similar

### Real-time LLM Output (✨ NEW)

Now you can see the actual LLM output in real-time using the `-vvv` flag!

**Command line usage:**
```bash
# See real-time LLM JSON output
python main_sdk.py input.ass output.ass --streaming -vvv

# Just progress dots
python main_sdk.py input.ass output.ass --streaming -v

# Silent mode
python main_sdk.py input.ass output.ass --streaming
```

**What you'll see with `-vvv`:**
```
Processing chunk 1/5 (30 pairs)...
  LLM Output (real-time):
  ----------------------------------------------------------
  [
    {
      "id": 0,
      "eng": "Hello, world!",
      "chinese": "你好，世界！"
    }
  ]
  ----------------------------------------------------------
✅ Completed
```

This is useful for:
- Debugging JSON formatting issues
- Monitoring correction quality in real-time
- Understanding model behavior
- Early detection of problems

See [REALTIME_STREAMING.md](REALTIME_STREAMING.md) for complete guide.

### Example: Visual Streaming Display

```python
from experiment.llm_client_sdk import call_openai_api_sdk_streaming

def show_streaming(chunk: str):
    print(chunk, end="", flush=True)  # Print each chunk as it arrives

messages = [{"role": "user", "content": "Tell me a short story"}]

response_text, usage = call_openai_api_sdk_streaming(
    messages,
    config,
    model_settings=config.main_model,
    chunk_callback=show_streaming
)

# Output appears character-by-character in real-time!
```

## Configuration

The SDK implementation automatically loads the API key from the `key` file in the parent directory. You can override model settings:

```python
config = load_config_sdk(
    model_name="gpt-4o",           # Override main model
    terminology_model="gpt-4o-mini",  # Override terminology model
    verbose=True,                   # Enable verbose output
    max_chunks=5,                   # Limit chunks for testing
    pairs_per_chunk=10              # Fixed pairs per chunk
)
```

## Testing

Run the test suite:

```bash
cd /Users/zerozaki07/Downloads/subretrans
source venv/bin/activate
cd experiment
python test_sdk.py
```

Expected output:
```
=============================================================
OpenAI SDK Test Suite
=============================================================

Testing API Connection with OpenAI SDK
...
✓ API connection test successful!

Testing Simple Subtitle Refinement
...
✓ Refinement successful!

Test Summary
=============================================================
  ✓ PASS: API Connection
  ✓ PASS: Simple Refinement
  ✓ PASS: File Refinement

Total: 3/3 tests passed
```

## Implemented Features

✅ **Streaming responses** - Process subtitle corrections as they're generated (implemented!)
✅ **Non-streaming responses** - Standard API calls with complete responses
✅ **Automatic retry logic** - Exponential backoff for failed requests
✅ **Usage statistics** - Track token usage including reasoning tokens (GPT-5)
✅ **Callback support** - Real-time streaming feedback via callbacks
✅ **Per-model credentials** - Configure different API keys/endpoints per model (NEW!)
✅ **Verbose debugging** - `-vvv` mode shows credential resolution and system prompts
✅ **Resume mode** - Restart processing from a specific pair index with `--resume INDEX` (NEW!)

## Future Enhancements

Potential future features this SDK enables:

1. **Async operations**: Use `AsyncOpenAI` for parallel chunk processing
2. **Function calling**: Use OpenAI's function calling for structured outputs
3. **Vision API**: Add support for image-based subtitle alignment
4. **Embeddings**: Use embeddings for better terminology matching
5. **Batch API**: Use OpenAI's batch API for cost-efficient processing

## Troubleshooting

### API Key Not Found

Make sure the `key` file exists in the parent directory:

```bash
ls -la /Users/zerozaki07/Downloads/subretrans/key
```

### Import Errors

Make sure you're running from the correct directory and the virtual environment is activated:

```bash
cd /Users/zerozaki07/Downloads/subretrans
source venv/bin/activate
python -c "import openai; print(openai.__version__)"
```

### Module Not Found

The experiment modules add the parent directory to `sys.path` automatically. If you get import errors, make sure you're running from within the project:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Comparison: HTTP vs SDK

| Feature | HTTP POST | OpenAI SDK |
|---------|-----------|------------|
| Dependencies | requests | openai |
| Error handling | Manual retry logic | Built-in retry with exponential backoff |
| Type safety | Dict-based | Type-safe with Pydantic models |
| Streaming | Not supported | Supported |
| Async support | Not supported | Supported (AsyncOpenAI) |
| Provider flexibility | Any OpenAI-compatible API | OpenAI or compatible providers |
| Code complexity | More verbose | More concise |

## License

Same as main project (MIT)

## Contact

For questions about this experiment, refer to the main project's documentation or create an issue.