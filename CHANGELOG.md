# Changelog

All notable changes to this project will be documented in this file.

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
