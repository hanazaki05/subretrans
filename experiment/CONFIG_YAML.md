# YAML Configuration Guide

## Overview

The experiment directory now uses YAML-based configuration instead of hardcoded defaults. All settings are centralized in `config.yaml`, making it easy to customize without modifying code.

## Configuration File

The main configuration file is `experiment/config.yaml`. This file contains all settings organized into logical sections:

### Sections

1. **API Settings** - API endpoint, key file location, timeout
2. **Main Model Settings** - GPT-5.* configuration for subtitle refinement
3. **Terminology Model Settings** - GPT-4o-mini configuration for glossary extraction
4. **Token Management** - Context limits, memory limits, chunk sizes
5. **Chunking Strategy** - Token-based or fixed pair-count chunking
6. **Pricing** - Cost per 1000 tokens (for cost estimation)
7. **Glossary Settings** - Max entries, policy, minimum confidence
8. **User Customization** - Custom prompt file path
9. **Runtime Options** - Verbose mode, dry run, etc.

## Usage

### Default Configuration

Load configuration from `config.yaml` with all defaults:

```python
from experiment.config_sdk import load_config_from_yaml

config = load_config_from_yaml()
```

### CLI Overrides

Load from YAML but override specific settings from command line:

```python
from experiment.config_sdk import load_config_sdk

config = load_config_sdk(
    model_name="gpt-4o",           # Override main model
    dry_run=True,                  # Enable dry run
    pairs_per_chunk=20,            # Use fixed chunk size
    verbose=True                   # Enable verbose mode
)
```

### Custom YAML File

Load from a different YAML file:

```python
config = load_config_sdk(yaml_file_path="my_custom_config.yaml")
```

## Configuration Options

### API Settings

```yaml
api:
  key_file: "../key"              # Path to API key file (relative to config.yaml)
  base_url: "https://api.openai.com/v1"
  timeout: 280                    # API timeout in seconds
```

### Main Model (Subtitle Refinement)

```yaml
main_model:
  name: "gpt-5-mini"              # Model name
  max_output_tokens: 12000        # Max tokens per response
  reasoning_effort: "medium"      # Reasoning effort: none/low/medium/high
  temperature: 1.0                # Sampling temperature
  # Optional per-model API credentials (overrides global api settings)
  #key_file: "../key-main"        # Model-specific API key file
  #base_url: "https://another.url/v1"  # Model-specific endpoint
```

### Terminology Model (Glossary Extraction)

```yaml
terminology_model:
  name: "gpt-4o-mini"             # Model name
  max_output_tokens: 1800         # Max tokens per response
  temperature: 0.5                # Lower temp for stable extraction
  # Optional per-model API credentials (overrides global api settings)
  #key_file: "../key-terminology" # Model-specific API key file
  #base_url: "https://another.url/v1"  # Model-specific endpoint
```

### Token Management

```yaml
tokens:
  max_context_tokens: 128000      # Model's max context window
  memory_token_limit: 4000        # Max tokens for global memory
  chunk_token_soft_limit: 60000   # Target chunk size
```

### Chunking Strategy

```yaml
chunking:
  pairs_per_chunk: null           # null = token-based, number = fixed pair count
```

Set to a number (e.g., `50`) to use fixed pair-count chunking instead of token-based.

### Pricing

```yaml
pricing:
  prompt_tokens: 0.03             # Cost per 1k prompt tokens (USD)
  completion_tokens: 0.06         # Cost per 1k completion tokens (USD)
```

### Glossary Settings

```yaml
glossary:
  max_entries: 100                # Max terminology entries in memory
  policy: "lock"                  # User glossary is authoritative
  terminology_min_confidence: 0.6 # Min confidence for extracted terms
```

### User Customization

```yaml
user:
  prompt_path: "main_prompt.md"  # Path to main prompt template file
```

**Template-Based Prompt System (plan3.md):**

The `prompt_path` now points to a **main prompt template** file that serves as the complete system prompt. The template uses markdown sections:

- `### 1. English Subtitle Rules`
- `### 2. Chinese Subtitle Rules`
- `### 3. Context & Specific Handling`
- `### 4. User Terminology (Authoritative Glossary)` ← **Dynamic injection point**
- `### 5. Input/Output Format & Constraint`
- `### 6. Few-Shot Examples`

The `### 4. User Terminology (Authoritative Glossary)` section is dynamically updated:
1. Template glossary entries are parsed from the file
2. Runtime `GlobalMemory.user_glossary` entries are merged (runtime takes precedence)
3. `GlobalMemory.glossary` (learned terms) are appended as "Learned Terminology (Supplement)"
4. All sections are renumbered automatically

### Runtime Options

```yaml
runtime:
  use_streaming: true             # Use streaming API (recommended for real-time output)
  verbose: false                  # Enable verbose output
  very_verbose: false             # Dump full API responses
  debug_prompts: false            # Print system prompts
  stats_interval: 1.0             # Stats refresh interval (seconds)
  dry_run: false                  # Process limited pairs for testing
  max_chunks: null                # Max chunks to process (null = all)
```

**Streaming API:**
- `true` (recommended): Real-time output, 2.7x faster perceived speed
- `false`: Wait for complete response before displaying

### Per-Model API Credentials

**NEW in v0.0.7:** You can now configure different API keys and endpoints for each model independently.

Each model (`main_model` and `terminology_model`) can optionally specify:
- `key_file`: Path to a model-specific API key file (overrides global `api.key_file`)
- `base_url`: Model-specific API endpoint (overrides global `api.base_url`)

**Credential Resolution Order:**
1. Start with global `api.key_file` and `api.base_url`
2. If model has `base_url` → use model's endpoint
3. If model has `key_file` → load API key from model's file

**Use Cases:**

1. **Cost Tracking** - Separate billing for different models
   ```yaml
   main_model:
     key_file: "../key-main"  # Main model uses separate API key

   terminology_model:
     key_file: "../key-terminology"  # Terminology model uses separate API key
   ```

2. **Multi-Provider** - Route models to different providers
   ```yaml
   main_model:
     base_url: "https://api.openai.com/v1"  # OpenAI for main model

   terminology_model:
     base_url: "http://localhost:8000/v1"  # Local server for terminology
     key_file: "../test-key"
   ```

3. **Development/Testing** - Production vs test endpoints
   ```yaml
   main_model:
     base_url: "https://api.openai.com/v1"  # Production

   terminology_model:
     base_url: "https://dev-api.example.com/v1"  # Development endpoint
   ```

4. **Load Balancing** - Distribute requests across multiple endpoints
   ```yaml
   main_model:
     base_url: "https://proxy-1.example.com/v1"

   terminology_model:
     base_url: "https://proxy-2.example.com/v1"
   ```

**Verbose Debugging (`-vvv` mode):**

When running with `-vvv`, the system displays credential resolution information:

```bash
python main_sdk.py input.ass output.ass -vvv
```

Output example:
```
[Credential Resolution for gpt-5-mini]
  API Key: Model-specific (../key-main) [sk-proj-AbC...]
  Base URL: Model-specific → https://my-proxy.example.com/v1

Processing chunk 1/5 (30 pairs)...
```

This shows:
- Which model is being used
- Whether API key is from global config or model-specific override
- Which key file was loaded (if model-specific)
- Whether base URL is from global config or model-specific override
- The actual endpoint being used

**Path Resolution:**
- Relative paths (e.g., `../key-main`) are resolved from the `experiment/` directory
- Absolute paths are supported
- Clear error messages show full path if key file loading fails

**Testing:**
```bash
# Test per-model configuration
./venv/bin/python experiment/test_per_model_config.py

# Run with verbose output to see credential resolution
python experiment/main_sdk.py input.ass output.ass -vvv --max-chunks 1
```

## Examples

### Example 1: Quick Test with GPT-4o

Edit `config.yaml`:

```yaml
main_model:
  name: "gpt-4o"                  # Use GPT-4o instead of GPT-5

runtime:
  dry_run: true                   # Process only first 10 pairs
  verbose: true                   # Show progress
  max_chunks: 2                   # Process only 2 chunks
```

Or override from command line:

```bash
python main_sdk.py input.ass output.ass --model gpt-4o --dry-run --max-chunks 2 -v
```

### Example 2: Fixed Chunk Size Processing

Edit `config.yaml`:

```yaml
chunking:
  pairs_per_chunk: 30             # Process exactly 30 pairs per chunk

runtime:
  verbose: true                   # Show progress
```

### Example 3: High Quality Mode

Edit `config.yaml`:

```yaml
main_model:
  reasoning_effort: "high"        # Use maximum reasoning
  temperature: 0.7                # Lower temperature for consistency

tokens:
  memory_token_limit: 6000        # More memory for context
```

### Example 4: Budget Mode

Edit `config.yaml`:

```yaml
main_model:
  name: "gpt-4o-mini"             # Cheaper model
  reasoning_effort: "low"         # Less reasoning

tokens:
  chunk_token_soft_limit: 40000   # Smaller chunks (fewer API calls)

glossary:
  max_entries: 50                 # Less memory usage
```

### Example 5: Streaming Control

**Enable streaming with real-time output:**
```yaml
runtime:
  use_streaming: true             # Enable streaming API
  verbose: true                   # Show progress dots
  debug_prompts: false            # Don't show full LLM output
```

**Disable streaming (wait for full response):**
```yaml
runtime:
  use_streaming: false            # Disable streaming, wait for complete response
```

**Real-time LLM output for debugging:**
```yaml
runtime:
  use_streaming: true             # Enable streaming
  debug_prompts: true             # Show real-time LLM JSON output
```

**CLI override:**
```bash
# Override YAML setting to disable streaming
python main_sdk.py input.ass output.ass --no-streaming

# Override YAML setting to enable streaming
python main_sdk.py input.ass output.ass --streaming -v
```

### Example 6: Per-Model Credentials

**Scenario: Use different API keys for cost tracking**

Edit `config.yaml`:

```yaml
api:
  key_file: "../key"                    # Default key
  base_url: "https://api.openai.com/v1"

main_model:
  name: "gpt-5-mini"
  max_output_tokens: 27000
  reasoning_effort: "low"
  temperature: 1.0
  key_file: "../key-main"               # Separate billing for main model

terminology_model:
  name: "gpt-4o-mini"
  max_output_tokens: 1800
  temperature: 0.45
  key_file: "../key-terminology"        # Separate billing for terminology model
```

Run with verbose debugging to see credential resolution:

```bash
python main_sdk.py input.ass output.ass -vvv --max-chunks 1
```

**Scenario: Use custom proxy for main model**

```yaml
main_model:
  name: "gpt-5-mini"
  base_url: "https://my-proxy.example.com/v1"  # Custom proxy endpoint
  # Uses global api.key_file

terminology_model:
  name: "gpt-4o-mini"
  # Uses global api settings (no overrides)
```

**Scenario: Development setup with local server**

```yaml
main_model:
  name: "gpt-5-mini"
  base_url: "https://api.openai.com/v1"        # Production OpenAI

terminology_model:
  name: "gpt-4o-mini"
  base_url: "http://localhost:8000/v1"         # Local test server
  key_file: "../test-key"                       # Test API key
```

## Testing Configuration

Run the test script to verify your configuration:

```bash
python experiment/test_yaml_config.py
```

This will:
1. Load configuration from YAML
2. Verify all settings are parsed correctly
3. Test CLI overrides
4. Check backward compatibility

## Migration from Old Config

The old `config_sdk.py` used hardcoded defaults. The new YAML-based system provides:

1. **Easy customization** - Edit YAML instead of code
2. **Version control** - Track configuration changes in git
3. **Multiple profiles** - Create different YAML files for different scenarios
4. **Documentation** - YAML comments explain each setting
5. **CLI overrides** - Command line still works, overrides YAML

All existing code using `load_config_sdk()` continues to work without changes.

## Tips

1. **Keep key files secure** - Don't commit `key`, `key-main`, or other API key files to git
2. **Use relative paths** - Paths in YAML are relative to `config.yaml` location
3. **Start with defaults** - Default `config.yaml` has recommended settings
4. **Test changes** - Use `--dry-run` and `--max-chunks` to test new settings
5. **Version control** - Commit `config.yaml` but add comments for custom values
6. **Per-model credentials** - Use `-vvv` to verify which credentials are being used for each model
7. **Backward compatible** - Per-model `key_file` and `base_url` are optional; omit them to use global settings

## Troubleshooting

### "Key file not found" error

Check `api.key_file` path in `config.yaml`. It should point to your API key file relative to the `config.yaml` location:

```yaml
api:
  key_file: "../key"  # Correct: key file in parent directory
```

### "Config file not found" error

Make sure `config.yaml` exists in the `experiment` directory, or specify the full path:

```python
config = load_config_sdk(yaml_file_path="/full/path/to/config.yaml")
```

### Settings not applied

Remember CLI overrides take precedence over YAML settings. Check if you're passing command line arguments that override your YAML changes.

---

**Last Updated:** 2025-12-07
**Status:** Production Ready ✅
**Version:** 0.0.7 (Added per-model credentials)
