# Changelog

All notable changes to this project will be documented in this file.

## [0.0.9] - 2025-12-30

### Fixed
- **Terminology lock mechanism bug** (experiment SDK):
  - User glossary entries from `main_prompt.md` template were not being enforced during terminology extraction
  - Bug: `GlobalMemory.user_glossary` was initialized as empty, so lock checks always failed
  - Fix: Template glossary is now parsed and loaded into `GlobalMemory.user_glossary` at initialization
  - Result: User-defined translations now correctly prevent conflicting LLM-extracted terms
  - Example: Template entry "Mac: 麦可" now properly blocks LLM extraction of "Mac: 麦" (conflict detected)
- **Resume + per-block update safety** (experiment SDK):
  - Output writing is now atomic to avoid truncated/corrupted `.ass` files on interruption
  - Added ID alignment logic to prevent applying per-chunk “local IDs” (e.g., `0..N-1`) onto global pair IDs during `--resume`
- **Duplicate glossary injection** (experiment SDK):
  - Learned glossary entries that are already defined in the User-Defined Glossary are now pruned routinely
  - Pruning happens before each LLM request, after each memory update, and after loading glossary checkpoints
  - When `--checkpoint` is enabled, pruned entries are removed from the `.glossary.yaml` checkpoint as well

### Added
- **Enhanced intermediate format robustness** (experiment SDK):
  - All three intermediate formats now have comprehensive error recovery mechanisms
  - Formats: JSON (default), XML-pair, pseudo-TOML - see `experiment/INTERMEDIATE_FORMATS.md`
  - Configure via `config.intermediate_format` in `config.yaml` or `--format` CLI flag
  - Format-specific enhancements:
    - **XML-pair**: Two-stage parsing handles malformed separators (`eng>value` → `eng=value`)
    - **JSON**: Pattern extraction finds first `[` or `{` marker
    - **Pseudo-TOML**: Pattern extraction finds first `[pair]` header
  - All formats support fallback parsing, duplicate detection, and commentary stripping
- **Fallback LLM output parser** (experiment SDK):
  - Pattern-based extraction as recovery mechanism when normal deserialization fails
  - Two-stage parsing: (1) Try direct deserialization, (2) Fall back to pattern extraction if failed
  - Handles LLM responses with extra commentary before format data (e.g., "I have reviewed..." before `<pair>`)
  - New function: `_extract_from_format_marker(text, format_type)` in `llm_client_sdk.py`
  - Supports all formats: XML-pair (`<pair>`), JSON (`[` or `{`), pseudo-TOML (`[pair]`)
  - Detailed logging shows when fallback is used and whether recovery succeeded
- **Duplicate pair detection and deduplication** (experiment SDK):
  - Detects when LLM returns same pair ID multiple times in a response
  - New function: `_detect_duplicate_pairs(pairs)` in `llm_client_sdk.py`
  - Deduplication strategy: Keep **last occurrence** (assumes LLM refined/corrected the duplicate)
  - Maintains original order of first appearance for consistency
  - Warning messages show which IDs were duplicated and how many unique pairs retained
- **Two-stage XML-pair parsing with regex fallback** (experiment SDK):
  - Handles malformed field separators from LLM (e.g., `eng>value` instead of `eng=value`)
  - Stage 1: Strict parsing with `=` separator (maintains current behavior)
  - Stage 2: Regex-based fallback for alternate separators (`>`, `:`, `|`) and whitespace variations
  - New function: `_parse_field_assignment(line, expected_field)` in `serializers.py`
  - Pattern: `r'^({field})\s*([=>:|])\s*(.*)$'` (flexible separator + whitespace)
  - Warning logged when non-standard separator is auto-corrected
  - Examples handled: `eng>value`, `eng: value`, `eng = value`, `ID | 123`
- New test files in `experiment/`:
  - `test_lock_logic_simple.py`: Verifies terminology lock mechanism with user glossary
  - `test_glossary_prune.py`: Verifies learned glossary pruning against user glossary (Unicode/whitespace/case-normalized)
  - `test_per_block_update.py`: Verifies per-block update behavior (writes after each chunk) and YAML/CLI backward compatibility
  - `test_response_cleaning.py` updated: Tests for leading commentary extraction and duplicate detection
  - `test_serializers.py` updated: `test_xml_pair_malformed_separators()` for regex fallback (7 test cases)

### Changed
- `experiment/main_sdk.py`:
  - Added template glossary loading after `init_global_memory()` (lines 291-305)
  - Parses `### 4. User Terminology` section from `main_prompt.md` at startup
  - Populates `GlobalMemory.user_glossary` with parsed entries for lock mechanism
  - Verbose mode shows count of loaded user glossary entries
- `experiment/main_sdk.py` / `experiment/config_sdk.py` / `experiment/config.yaml`:
  - Renamed `incremental_output` → `per_block_update`
  - CLI flags renamed to `--per-block-update` / `--no-per-block-update`
  - Deprecated compatibility flags `--incremental-output` / `--no-incremental-output` are still accepted
- `experiment/llm_client_sdk.py`:
  - Updated `refine_chunk_sdk()` and `refine_chunk_sdk_streaming()` with fallback parsing
  - Added try-catch around deserialization with pattern extraction recovery
  - Added duplicate detection and deduplication after successful deserialization
  - Changed deduplication strategy from "keep first" to "keep last" occurrence
- `experiment/serializers.py`:
  - Updated `deserialize_xml_pair()` to use two-stage field parsing
  - Added `_parse_field_assignment()` helper for flexible separator matching
  - Warning messages when Stage 2 fallback is used
- `experiment/test_serializers.py`:
  - Added comprehensive test for malformed separators (7 test cases)
  - Tests cover: `>`, `:`, `|` separators, whitespace variations, mixed formats, strict format
- `experiment/test_response_cleaning.py`:
  - Added `test_leading_commentary_extraction()` (3 test cases: XML, JSON, TOML)
  - Added `test_duplicate_pair_detection()` (3 test cases: detection, deduplication, no false positives)
  - Added helper functions: `_extract_from_format_marker()`, `_detect_duplicate_pairs()`

### Technical Details

**Terminology Lock Fix:**
- Root cause: Template glossary was only used for prompt generation, never stored in `GlobalMemory.user_glossary`
- Lock mechanism in `memory.py:268-299` builds `user_map` from `user_glossary` (was empty)
- Fix flow:
  1. Load template from `config.user_prompt_path` at initialization
  2. Find "User Terminology (Authoritative Glossary)" section
  3. Parse entries using `_parse_template_glossary()` (pattern: `r"^\s*-\s+(.+?):\s*(.+?)\s*$"`)
  4. Populate `GlobalMemory.user_glossary` with parsed entries
  5. Lock mechanism now detects conflicts using case-insensitive comparison (`.casefold()`)
- Backward compatible: Falls back gracefully if template loading fails

**Intermediate Format Support:**
- Three formats available (configured via `config.intermediate_format`):
  1. **JSON** (default): Standard JSON array `[{"id": 0, "eng": "...", "chinese": "..."}]`
  2. **XML-pair**: Custom format with `<pair>ID=0\neng=...\nchinese=...\n</pair>`
  3. **Pseudo-TOML**: TOML-like `[pair]\nid = 0\neng = ...\nchinese = ...`
- All formats preserve ASS formatting tags (e.g., `{\i1}`, `\N`)
- Each format has specialized error recovery:
  - **XML-pair**: Handles wrong separators (`>`, `:`, `|`) and whitespace variations
  - **JSON**: Extracts from first `[` or `{` if commentary present
  - **Pseudo-TOML**: Extracts from first `[pair]` section header
- See `experiment/INTERMEDIATE_FORMATS.md` for complete format specifications

**Fallback Parser:**
- Activation: Only when normal deserialization raises `SerializationError`
- Pattern detection:
  - XML-pair: Find first `<pair>` tag → extract from there
  - JSON: Find first `[` or `{` → extract from there
  - Pseudo-TOML: Find first `[pair]` → extract from there
- Recovery flow:
  1. Clean response (remove `<think>` blocks, extract from code blocks)
  2. Try deserialization → `SerializationError`
  3. Call `_extract_from_format_marker()` to find format markers
  4. Retry deserialization with extracted content
  5. If both fail: Show excerpts and raise `LLMAPIError`
- Error handling: Detailed logging shows cleaned response, extracted content, and error messages
- Works uniformly across all three intermediate formats

**Two-Stage XML Parsing:**
- Stage 1 (Strict): Check for `=` separator, split on first `=`, validate field name
- Stage 2 (Regex): Match `field + optional_ws + separator + optional_ws + value`
- Supported separators: `=`, `>`, `:`, `|`
- Whitespace handled: `eng=value`, `eng = value`, `eng= value`, `eng =value`
- Return format: `(value, used_fallback, separator)` tuple
- Warning example: `[Warning]: Non-standard separator '>' for field 'eng' at line 399, auto-corrected`

**Duplicate Deduplication:**
- Detection: Build `id_counts` dict, find IDs with count > 1
- Strategy: Keep **last** occurrence (changed from "keep first")
  - Rationale: LLM might have refined/corrected the duplicate
- Implementation:
  1. Build `id_to_pair` dict (last value wins)
  2. Iterate original list, use `seen_ids` to track first appearance
  3. For each first appearance, append `id_to_pair[pair.id]` (gets last occurrence)
  4. Result: Original order preserved, last occurrence used
- Warning format: `[Warning]: Duplicate pair IDs detected: [5, 12]`

### Examples

**Terminology Lock (Before/After):**
```
# Before fix (bug)
Terminology merge: 11 candidate(s), added 9, user-locked 0 (conflicts 0, duplicates 0)
# "Mac" was added despite template having "Mac: 麦可" (conflict not detected)

# After fix
Loaded 35 user glossary entries from template
...
[Glossary lock] Skip learned term 'Mac' -> '麦' (conflicts with user '麦可')
Terminology merge: 11 candidate(s), added 8, user-locked 1 (conflicts 1, duplicates 0)
```

**Fallback Parser (LLM added commentary):**
```
# LLM response
I have reviewed and corrected the subtitles.
<pair>
ID=0
eng=Hello
chinese=你好
</pair>

# Log output
[Deserialization failed, attempting pattern extraction...]
[Pattern extraction successful, retrying deserialization...]
[Recovery successful!]
```

**Two-Stage XML Parsing (malformed separator):**
```
# LLM output
<pair>
ID=107
eng>this is the finest 688 crew in the fleet.
chinese=这是舰队中最优秀的 688 艇员
</pair>

# Log output
[Warning]: Non-standard separator '>' for field 'eng' at line 3, auto-corrected
```

**Duplicate Detection:**
```
# LLM returned pair ID=5 twice
[Warning]: Duplicate pair IDs detected: [5, 12]
[Action]: Keeping last occurrence, removing duplicates
[Result]: 103 unique pairs retained
```

### Backward Compatibility
- ✅ All changes are backward compatible
- ✅ Fallback parser only activates on deserialization failures
- ✅ Two-stage XML parsing maintains strict behavior for well-formed input
- ✅ Template glossary loading falls back gracefully on errors
- ✅ Existing tests continue to pass
- ✅ No changes to external APIs or file formats

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
