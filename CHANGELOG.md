# Changelog

All notable changes to this project will be documented in this file.

## [0.0.8] - 2025-12-07

### Added
- **Resume mode** for restarting subtitle processing from a specific pair index (experiment SDK):
  - New `--resume INDEX` command-line parameter to restart from any pair index
  - Automatically loads and preserves earlier pairs from existing output file
  - Processes only pairs from resume index onwards
  - Useful for recovering from errors, interruptions, or re-processing specific sections
  - Works seamlessly with all other options (`--pairs-per-chunk`, `--streaming`, `-vvv`, etc.)
- **Glossary checkpoint system** for persistent learned terminology (experiment SDK):
  - Opt-in feature enabled with `--checkpoint` flag
  - Automatic checkpoint saving after each glossary update (when enabled)
  - Automatic checkpoint loading on startup (when enabled)
  - YAML format for human-readable checkpoint files
  - Checkpoint filename inherits from input file (e.g., `input.ass.glossary.yaml`)
  - Preserves learned terminology across runs and resume sessions
  - Ensures consistent translation of terms across interrupted/resumed processing
- New documentation file in `experiment/`:
  - `RESUME_MODE.md`: Complete guide with examples, use cases, and technical details (includes checkpoint system)

### Changed
- `experiment/main_sdk.py`:
  - Added `--resume INDEX` argument to argparse
  - Added `--checkpoint` argument to enable glossary checkpoint system (opt-in)
  - Updated `process_subtitles()` function signature to accept `resume_index: Optional[int]` and `enable_checkpoint: bool`
  - Implemented resume logic that loads existing output file and preserves earlier pairs
  - Added validation for resume index (non-negative, within pair count)
  - Updated help text with resume and checkpoint examples
  - Added glossary checkpoint functions: `get_checkpoint_path()`, `save_glossary_checkpoint()`, `load_glossary_checkpoint()`
  - Integrated checkpoint loading after global memory initialization (only if enabled)
  - Integrated checkpoint saving after each `update_global_memory()` call (only if enabled)
  - Integrated checkpoint saving after memory compression (only if enabled)
  - Changed checkpoint format from JSON to YAML using PyYAML library
- `experiment/README.md`:
  - Added "Resume mode" to "Implemented Features" list
  - Added `RESUME_MODE.md` to documentation list

### Technical Details
- Resume workflow:
  1. Parse input file and build all subtitle pairs
  2. If resume index specified and output file exists, load and preserve pairs 0 to (index-1)
  3. Create `pairs_to_process` containing only pairs from resume_index onwards
  4. Process chunks from filtered pairs
  5. Apply corrections using ID-based matching to update correct pairs in full list
  6. Write complete pairs list (preserved + newly processed) to output file
- Glossary checkpoint workflow:
  1. On startup: Check for checkpoint file (`{input_file}.glossary.yaml`)
  2. If found: Load learned glossary into GlobalMemory
  3. After each chunk: Save updated glossary to checkpoint (YAML format using PyYAML)
  4. After memory compression: Save compressed glossary to checkpoint
  5. On resume: Checkpoint is automatically loaded, preserving all learned terms
- Checkpoint benefits:
  - Terminology consistency across runs and resume sessions
  - No re-learning of previously extracted terms
  - Human-readable YAML format for manual inspection/editing
  - Automatic and transparent (no user intervention required)
- Error handling:
  - Validates resume index is non-negative
  - Validates resume index is within total pair count
  - If existing output file is corrupted: warning, continues without preserved pairs
  - If output file doesn't exist: info message, creates new file
- ID-based matching ensures corrections are applied to the right pairs regardless of position
- Backward compatible: Resume is opt-in, no changes to existing functionality

### Examples
```bash
# Resume from pair 680 after an error (without checkpoint)
python experiment/main_sdk.py input.ass output.ass --resume 680 --pairs-per-chunk 75 --streaming

# Resume with checkpoint enabled (preserves learned terms)
python experiment/main_sdk.py input.ass output.ass --resume 680 --checkpoint --streaming

# Resume with verbose mode to see what's being processed
python experiment/main_sdk.py input.ass output.ass --resume 500 -vvv --streaming

# Resume for re-processing last section with different settings
python experiment/main_sdk.py input.ass output.ass --resume 800 --max-chunks 2

# Normal run with checkpoint enabled (creates checkpoint file)
python experiment/main_sdk.py input.ass output.ass --checkpoint --pairs-per-chunk 75 --streaming
# Creates: input.ass.glossary.yaml (checkpoint file)

# Subsequent run on same file with checkpoint (loads learned terminology)
python experiment/main_sdk.py input.ass output_v2.ass --checkpoint --pairs-per-chunk 75 --streaming
# Loads: input.ass.glossary.yaml (preserves learned terminology)
```

### Output Example (with `--checkpoint` enabled)
```
Step 3: Initialize global memory
  [CHECKPOINT] Loaded 42 glossary entries from: input.ass.glossary.yaml

[RESUME MODE] Starting from pair index 680
Skipping first 680 pairs, processing remaining 320 pairs
Loading existing output file: output.ass
Preserved 680 pairs from existing output
Processing pairs 680 to 999 (320 pairs)

Processing chunk 1/5 (75 pairs)...
  [Chunk completed, checkpoint updated]
```

## [0.0.7] - 2025-12-07

### Added
- **Per-model API credential configuration** (experiment SDK):
  - `key_file` and `base_url` can now be specified per model (`main_model` and `terminology_model`)
  - Override global `api` settings on a per-model basis
  - Support for different API providers, endpoints, or keys for each model
  - Useful for cost tracking, multi-provider setups, development/testing, and load balancing
- **Verbose credential debugging** (`-vvv` mode):
  - Shows which API key and base URL are being used for each model
  - Displays whether credentials are from global config or model-specific overrides
  - Shows the actual key file path and endpoint URL
  - Example output: `[Credential Resolution for gpt-5-mini]`
- New documentation files in `experiment/`:
  - `PER_MODEL_CREDENTIALS.md`: Complete user guide with examples and use cases
  - `FEATURE_SUMMARY.md`: Technical overview of the implementation
  - `test_per_model_config.py`: Comprehensive test suite (5 tests)
  - `demo_per_model_config.py`: Interactive demo with 4 usage scenarios

### Changed
- `experiment/config_sdk.py`:
  - Added `key_file: Optional[str]` and `base_url: Optional[str]` to `MainModelSettings`
  - Added `key_file: Optional[str]` and `base_url: Optional[str]` to `TerminologyModelSettings`
  - Updated `load_config_from_yaml()` to read per-model credential overrides
- `experiment/llm_client_sdk.py`:
  - New `_resolve_model_credentials()` function to resolve API key and base URL per model
  - Both `call_openai_api_sdk()` and `call_openai_api_sdk_streaming()` now use credential resolver
  - Verbose mode (`config.debug_prompts`) displays credential resolution information
- `experiment/config.yaml`:
  - Added commented examples for `key_file` and `base_url` in both model settings
  - Examples show how to override global API credentials

### Technical Details
- Credential resolution order:
  1. Start with global `api.key_file` and `api.base_url`
  2. Override with model-specific `base_url` if present
  3. Load API key from model-specific `key_file` if present
- Path resolution: Relative key file paths are resolved relative to `experiment/` directory
- Independent resolution: Each model's credentials are resolved independently on every API call
- Backward compatible: Existing configurations work without changes (feature is opt-in)
- Error handling: Clear error messages with file paths if key file loading fails

### Examples
```yaml
# config.yaml - Per-model credentials
main_model:
  name: "gpt-5-mini"
  key_file: "../key-main"  # Different API key
  base_url: "https://custom-endpoint.com/v1"  # Different endpoint

terminology_model:
  name: "gpt-4o-mini"
  # Omit to use global api settings
```

```bash
# View credential resolution in verbose mode
python experiment/main_sdk.py input.ass output.ass -vvv
```

## [0.0.6] - 2025-12-01

### Added
- **Template-based prompt system** (plan3.md implementation):
  - Single markdown template file (`main_prompt.md`) serves as the complete system prompt source
  - Dynamic terminology injection into `### 4. User Terminology (Authoritative Glossary)` section
  - Automatic section renumbering when content is modified
  - Template glossary parsing and merging with runtime `GlobalMemory`
- New prompt functions in `prompts.py`:
  - `load_main_prompt_template(config)` - Load template from config path
  - `inject_memory_into_template(template, global_memory)` - Inject terminology into template
  - Helper functions for section parsing, boundary detection, and glossary merging

### Changed
- `config.user_prompt_path` default changed from `custom_main_prompt.md` to `main_prompt.md`
- `build_system_prompt()` now accepts optional `config` parameter for template-based generation
- `refine_chunk()` in `llm_client.py` passes config to `build_system_prompt()`
- Removed `split_user_prompt_and_glossary` and `set_user_instruction` from main workflow
- Legacy prompt building preserved as `build_system_prompt_legacy()` for backward compatibility

### Technical Details
- Template uses markdown `### N. Title` sections for structure
- Glossary merging: runtime user_glossary takes precedence over template entries
- Learned terminology appended as "Learned Terminology (Supplement)" section
- Template caching implemented to avoid repeated file reads

## [0.0.5] - 2025-11-29

### Added
- User-defined main prompt support via `Config.user_prompt_path` (default `custom_main_prompt.md`): the file is split into extra system instructions and a high-priority user glossary.
- `GlobalMemory` now distinguishes `user_glossary` (authoritative) from learned `glossary` (supplementary), and the system prompt prints them separately in `-vvv` mode.

### Changed
- Introduced `Config.glossary_policy = "lock"` (default) so learned terminology can only add new entries and will never override or conflict with user-defined glossary entries; conflicts are logged.
- Terminology extraction prompt and post-filter now share a single configurable threshold `Config.terminology_min_confidence` (default `0.6`), keeping the model's "keep" rule consistent with local filtering.
- In very verbose mode, terminology calls print the raw GPT‑4o output plus a compact debug summary (`parsed N`, `added M`, `user-locked`, `existing`) after each chunk.
- The main refinement system prompt is now built as `BASE_CORE + user instructions + memory section + CRITICAL tail`, ensuring the "ONLY return JSON" constraint always appears at the end of the prompt.

## [0.0.4] - 2025-11-28

### Changed
- `config.py` now separates main GPT-5.1 settings (`MainModelSettings`) from the dedicated GPT-4o terminology extractor (`TerminologyModelSettings`), each with their own temperature and token limits.
- `llm_client.call_openai_api()` accepts per-model settings and automatically injects the correct temperature / reasoning hints (reasoning only for GPT-5.x).
- `memory.py` hooks into the new terminology model: each chunk is sent to GPT-4o-mini via `extract_terminology_from_chunk()`, which validates the returned glossary entries (confidence ≥ 0.6, normalized types, evidence trimming) before merging.
- Global glossary growth is configurable via `config.glossary_max_entries` (default 100) to balance context richness vs. prompt size.

## [0.0.3] - 2025-11-27

### Added
- Verbose preview now shows reasoning token counts (from API usage) instead of truncated reasoning text
- New `-vv` mode extends verbose output by dumping each chunk's full API response (includes everything from `-v`)
- `-vvv` mode prints the full system prompt/memory sent to the model for deep debugging

### Changed
- Removed the `thinking_enabled` config flag; the client now just sends the standard `reasoning_effort` hint supported by GPT-5.1
- `UsageStats` tracks `reasoning_tokens`, and verbose previews display that count alongside the JSON snippet
- Documentation updated to clarify that the API exposes reasoning tokens but not reasoning content
- `config.py` now separates main GPT-5 model settings from the GPT-4o terminology extractor, with per-model temperature controls (main defaults to `1.0`, terminology defaults to `0.3`)
- `call_openai_api()` automatically injects the proper temperature/reasoning hints based on the selected model configuration

## [0.0.2] - 2025-11-27

### Added
- **Verbose mode** (`-v` or `--verbose`)
  - Shows detailed progress information after each chunk
  - Displays timing for each chunk processing
  - Shows 4-line preview of response (2 lines for returned pairs, 2 lines for reasoning content)
  - Helps with debugging and monitoring long-running processes
- **Stats refresh interval** (`--stats N`)
  - Controls refresh interval for verbose mode display (default: 1.0s)
  - Useful for future streaming response support

### Changed
- Updated `config.py` to include `verbose` and `stats_interval` parameters
- Enhanced `stats.py` with `reasoning_content` field in `UsageStats`
- Modified `llm_client.py` to:
  - Extract reasoning/extended thinking content from API response
  - Return `response_text` in addition to corrected pairs and usage
  - Support GPT-5.1's extended_content and reasoning fields
- Enhanced `main.py` to:
  - Track timing for each chunk processing
  - Display verbose output when `-v` flag is used
  - Accept `--stats` CLI argument for refresh interval
- Added utility functions in `utils.py`:
  - `print_verbose_preview()`: Display 4-line preview with ANSI cursor control
  - `format_time()`: Format seconds as human-readable time (e.g., "16.51s" or "1m 23s")
- Updated documentation:
  - `README.md`: Added verbose mode section with examples
  - `CLAUDE.md`: Added comprehensive verbose mode documentation in Critical Implementation Details
  - `example_usage.sh`: Added 3 examples demonstrating verbose mode usage

### Technical Details
- `config.py`: Added `verbose: bool = False` and `stats_interval: float = 1.0` fields
- `stats.py`: Added `reasoning_content: str = field(default="")` to UsageStats dataclass
- `llm_client.py`: Enhanced API response parsing to extract reasoning content from GPT-5.1
- `llm_client.refine_chunk()`: Changed return type to `Tuple[List[SubtitlePair], UsageStats, str]`
- `utils.py`: New functions for verbose display and time formatting
- `main.py`: Added timing tracking with `time.time()` and verbose display logic
- No breaking changes - fully backward compatible (verbose mode is opt-in)

### Examples
```bash
# Basic verbose mode
python main.py input.ass output.ass -v

# Verbose with custom refresh interval
python main.py input.ass output.ass -v --stats 0.5

# Verbose combined with fixed chunking
python main.py input.ass output.ass -v --pairs-per-chunk 30 --max-chunks 2
```

### Output Example
```
Processing chunk 1/2 (30 pairs)...
[Chunk 1/2] (50.0% complete)
  Tokens used: 3,092 (prompt: 1,726, completion: 1,366)
  Time: 16.51s

  Response: [
            {
  Reasoning tokens: 8
```

---

## [0.0.1] - 2025-11-27

### Added
- **Pair-based chunking option** (`--pairs-per-chunk N`)
  - New command-line argument to set fixed number of pairs per chunk
  - Overrides token-based chunking when specified
  - Provides predictable chunk sizes for testing and batch processing
  - Useful for cost estimation and progress tracking

### Changed
- Updated `config.py` to include `pairs_per_chunk` parameter
- Enhanced `chunker.py` with new `chunk_pairs_by_count()` function
- Modified `main.py` to:
  - Accept `--pairs-per-chunk` CLI argument
  - Display chunking strategy being used (token-based vs pair-based)
- Updated documentation:
  - `README.md`: Added chunking strategies section with examples
  - `CLAUDE.md`: Added detailed chunking documentation for AI assistants
  - `example_usage.sh`: Added examples using `--pairs-per-chunk`

### Technical Details
- `config.py`: Added `pairs_per_chunk: Optional[int] = None` field
- `chunker.py`: New function `chunk_pairs_by_count()` for simple pair counting
- `main.py`: Added argument parsing and display logic for chunking strategy
- No breaking changes - fully backward compatible

### Examples
```bash
# Use pair-based chunking
python main.py input.ass output.ass --pairs-per-chunk 50

# Combine with max-chunks
python main.py input.ass output.ass --pairs-per-chunk 30 --max-chunks 2

# Token-based chunking (default, no change)
python main.py input.ass output.ass
```

---

## [0.0.0] - 2025-11-27

### Initial Release
- Complete subtitle refinement tool for bilingual (English-Chinese) ASS files
- Token-based intelligent chunking
- Global memory management across chunks
- ASS tag preservation
- CLI interface with comprehensive options
- Cost tracking and estimation
- Robust error handling with retry logic
- Complete documentation (README.md, CLAUDE.md, IMPLEMENTATION_SUMMARY.md)
