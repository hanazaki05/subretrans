# Intermediate Representation Formats

## Overview

The subtitle refinement system now supports **three intermediate representation formats** for communicating subtitle pairs between the application and the LLM:

1. **JSON** (default) - Standard JSON array format
2. **XML-pair** - Custom XML-like format
3. **Pseudo-TOML** - TOML-like format

All formats preserve ASS formatting tags (e.g., `{\i1}`, `\N`) and handle special characters correctly.

---

## Format Specifications

### 1. JSON Format (Default)

Standard JSON array with objects containing `id`, `eng`, and `chinese` fields.

**Example:**
```json
[
  {"id": 0, "eng": "Tonight, on JAG...", "chinese": "今晚，在《军法署》..."},
  {"id": 1, "eng": "Good evening, I'm Norman Delaporte.", "chinese": "晚上好，我是诺曼·德拉波特。"}
]
```

**Characteristics:**
- Industry standard format
- Well-supported by LLMs
- Compact representation
- JSON-compliant escaping for special characters

---

### 2. XML-pair Format

Custom XML-like format with simple key-value structure.

**Example:**
```xml
<pair>
ID=0
eng=Tonight, on JAG...
chinese=今晚，在《军法署》...
</pair>

<pair>
ID=1
eng=Good evening, I'm Norman Delaporte.
chinese=晚上好，我是诺曼·德拉波特。
</pair>
```

**Characteristics:**
- More readable than JSON for humans
- Clear separation between pairs with `<pair>` tags
- Simple key=value format (no XML escaping required)
- Empty lines between pairs for visual clarity

**Format Rules:**
- Each pair starts with `<pair>` and ends with `</pair>`
- Fields: `ID`, `eng`, `chinese` (in that order)
- Format: `field=value` (one per line)
- Empty line after each `</pair>` (optional but recommended)

---

### 3. Pseudo-TOML Format

TOML-inspired format using section headers and key-value pairs.

**Example:**
```toml
[pair]
id = 0
eng = Tonight, on JAG...
chinese = 今晚，在《军法署》...

[pair]
id = 1
eng = Good evening, I'm Norman Delaporte.
chinese = 晚上好，我是诺曼·德拉波特
```

**Characteristics:**
- Similar to TOML configuration files
- Highly readable format
- Clear section boundaries with `[pair]` headers
- Space-separated key-value pairs (`key = value`)

**Format Rules:**
- Each pair starts with `[pair]` section header
- Fields: `id`, `eng`, `chinese` (in that order, lowercase)
- Format: `key = value` (one per line)
- Empty line between pairs (optional but recommended)

---

## Configuration

### Via YAML Configuration

Edit `experiment/config.yaml`:

```yaml
format:
  intermediate_format: "json"  # Options: "json", "xml-pair", "pseudo-toml"
```

### Via Python API

```python
from config_sdk import load_config_sdk

# Load with format override
config = load_config_sdk(intermediate_format="xml-pair")
```

### Via Command Line

```bash
# Using genreq.py (prompt generator)
python experiment/genreq.py input.ass output.md \
    --intermediate-format xml-pair

# Using main_sdk.py (full processing)
python experiment/main_sdk.py input.ass output.ass \
    --intermediate-format pseudo-toml
```

---

## How It Works

### 1. Serialization (Application → LLM)

When sending subtitle pairs to the LLM:

1. Application creates `SubtitlePair` objects
2. `serializers.serialize(pairs, format)` converts to selected format
3. Serialized text is embedded in the user prompt
4. System prompt examples are auto-converted to match the format

**Code Example:**
```python
from serializers import serialize
from pairs import SubtitlePair

pairs = [
    SubtitlePair(id=0, eng="Hello", chinese="你好"),
    SubtitlePair(id=1, eng="Goodbye", chinese="再见")
]

# Serialize to XML-pair format
xml_text = serialize(pairs, "xml-pair")
print(xml_text)
```

### 2. Deserialization (LLM → Application)

When receiving refined subtitles from the LLM:

1. LLM returns text in the same format
2. `serializers.deserialize(text, format)` parses the response
3. Returns list of `SubtitlePair` objects
4. Application validates and applies changes

**Code Example:**
```python
from serializers import deserialize

xml_response = """
<pair>
ID=0
eng=Hello.
chinese=你好
</pair>
"""

pairs = deserialize(xml_response, "xml-pair")
print(pairs[0].eng)  # "Hello."
```

### 3. Example Conversion in Prompts

The system prompt template (`main_prompt.md`) contains few-shot examples in JSON format. When using non-JSON formats, these examples are **automatically converted**:

**Original (main_prompt.md):**
```markdown
### 6. Few-Shot Examples
Input:
[
  {"id": 1, "eng": "hello", "chinese": "你好"}
]

Output:
[
  {"id": 1, "eng": "Hello.", "chinese": "你好"}
]
```

**Auto-converted to XML-pair:**
```markdown
### 6. Few-Shot Examples
Input:
<pair>
ID=1
eng=hello
chinese=你好
</pair>

Output:
<pair>
ID=1
eng=Hello.
chinese=你好
</pair>
```

This ensures the LLM sees consistent format throughout the entire prompt.

---

## Implementation Details

### Module: `experiment/serializers.py`

Core functions:

```python
# Format-specific serializers
serialize_json(pairs: List[SubtitlePair]) -> str
deserialize_json(text: str) -> List[SubtitlePair]

serialize_xml_pair(pairs: List[SubtitlePair]) -> str
deserialize_xml_pair(text: str) -> List[SubtitlePair]

serialize_pseudo_toml(pairs: List[SubtitlePair]) -> str
deserialize_pseudo_toml(text: str) -> List[SubtitlePair]

# Format-agnostic interface
serialize(pairs: List[SubtitlePair], format_type: str) -> str
deserialize(text: str, format_type: str) -> List[SubtitlePair]

# Example conversion for prompts
convert_json_examples_to_format(json_text: str, target_format: str) -> str
```

### Integration Points

1. **config_sdk.py**
   - Added `intermediate_format` field to `ConfigSDK`
   - Validation of format value in `__post_init__`
   - YAML loading support

2. **llm_client_sdk.py**
   - `refine_chunk_sdk()`: Uses `serialize()` and `deserialize()`
   - `refine_chunk_sdk_streaming()`: Same serialization logic
   - `_extract_from_code_blocks()`: Helper for extracting format from markdown

3. **prompts.py**
   - `convert_examples_to_format()`: Converts JSON examples in template
   - `build_system_prompt()`: Calls conversion when `intermediate_format != "json"`

4. **genreq.py**
   - Updated to use `serialize()` instead of `pairs_to_json_list()`
   - Shows format in generated markdown output

---

## Testing

### Run Test Suite

```bash
python experiment/test_serializers.py
```

**Tests include:**
1. Basic serialization/deserialization for all formats
2. ASS formatting tag preservation (`{\i1}`, `\N`, etc.)
3. Special character handling (quotes, colons, equals, ellipsis)
4. Edge cases (single pair, empty text, long text)
5. Error handling for malformed input
6. Example format conversion

**Expected Output:**
```
============================================================
TEST SUMMARY
============================================================
✓ PASS: test_basic_serialization
✓ PASS: test_ass_tags_preservation
✓ PASS: test_special_characters
✓ PASS: test_edge_cases
✓ PASS: test_error_handling
✓ PASS: test_format_examples_conversion

Total: 6/6 tests passed

✓ All tests passed successfully!
```

---

## Use Cases

### When to Use Each Format

**JSON (Default):**
- ✓ Best LLM support (industry standard)
- ✓ Compact representation
- ✓ Well-tested across different models
- Use when: You want maximum compatibility

**XML-pair:**
- ✓ More readable for humans
- ✓ Clear visual separation between pairs
- ✓ Easier to debug issues
- Use when: You need to manually inspect/edit prompts

**Pseudo-TOML:**
- ✓ Configuration file familiarity
- ✓ Very clean, readable format
- ✓ Good for documentation/examples
- Use when: You want the most readable format

---

## Example: Complete Workflow

```python
from config_sdk import load_config_sdk
from llm_client_sdk import refine_chunk_sdk
from memory import GlobalMemory
from pairs import SubtitlePair

# 1. Configure format
config = load_config_sdk(intermediate_format="xml-pair")

# 2. Create test pairs
pairs = [
    SubtitlePair(id=0, eng="hello world", chinese="你好世界"),
    SubtitlePair(id=1, eng="how are you?", chinese="你好吗？")
]

# 3. Refine using XML-pair format
global_memory = GlobalMemory()
refined_pairs, usage, response = refine_chunk_sdk(
    pairs,
    global_memory,
    config
)

# 4. View results
for pair in refined_pairs:
    print(f"[{pair.id}] {pair.eng} / {pair.chinese}")
```

**What happens internally:**
1. `pairs` serialized to XML-pair format
2. System prompt examples converted to XML-pair
3. LLM receives XML-pair format
4. LLM returns refined subtitles in XML-pair format
5. Response deserialized back to `SubtitlePair` objects

---

## ASS Tag Preservation

All formats correctly preserve ASS formatting tags:

**Input:**
```python
SubtitlePair(
    id=0,
    eng=r"This is {\i1}italic{\i0} text",
    chinese=r"这是{\i1}斜体{\i0}文本"
)
```

**JSON:**
```json
{"id": 0, "eng": "This is {\\i1}italic{\\i0} text", "chinese": "这是{\\i1}斜体{\\i0}文本"}
```

**XML-pair:**
```xml
<pair>
ID=0
eng=This is {\i1}italic{\i0} text
chinese=这是{\i1}斜体{\i0}文本
</pair>
```

**Pseudo-TOML:**
```toml
[pair]
id = 0
eng = This is {\i1}italic{\i0} text
chinese = 这是{\i1}斜体{\i0}文本
```

All formats preserve backslashes and braces exactly as they appear.

---

## Error Handling

The serializers provide clear error messages:

```python
from serializers import deserialize, SerializationError

try:
    deserialize("invalid xml", "xml-pair")
except SerializationError as e:
    print(f"Error: {e}")
    # Error: Expected '<pair>' at line 1, got: invalid xml
```

**Common errors:**
- Missing required fields (id, eng, chinese)
- Malformed JSON syntax
- Incomplete XML-pair tags
- Invalid pseudo-TOML structure

---

## Performance Considerations

**Serialization overhead is minimal:**
- JSON: Fast (uses built-in `json` module)
- XML-pair: String concatenation (very fast)
- Pseudo-TOML: String concatenation (very fast)

**Token usage:**
- JSON: Most compact (~10-15% fewer tokens)
- XML-pair: Moderate (~15-20% more tokens than JSON)
- Pseudo-TOML: Similar to XML-pair

**Recommendation:** Use JSON for production unless you have specific reasons to use other formats (debugging, readability, etc.).

---

## Troubleshooting

### Issue: LLM returns wrong format

**Solution:** Check that examples in `main_prompt.md` are being converted correctly. Enable debug mode:

```python
config = load_config_sdk(debug_prompts=True, intermediate_format="xml-pair")
```

This will print the full system prompt to verify example conversion.

### Issue: Deserialization fails

**Solution:** Check the raw LLM response. The error message will show:
- What was expected (e.g., `<pair>` tag)
- What was found
- Line number where error occurred

Enable verbose mode to see the raw response:

```python
config = load_config_sdk(verbose=True, intermediate_format="xml-pair")
```

### Issue: ASS tags corrupted

**Solution:** This should not happen if using the serializers correctly. Verify:
1. Input pairs have correct backslash escaping (use raw strings: `r"{\i1}"`)
2. Not manually editing serialized text
3. Using the provided `serialize()`/`deserialize()` functions

Run the test suite to verify:
```bash
python experiment/test_serializers.py
```

---

## Response Post-Processing

The system includes robust post-processing to handle LLM responses that may include extraneous content.

### Processing Pipeline

**Order (CRITICAL):**
1. **Remove thinking blocks** (`<think>...</think>`)
2. **Extract from markdown code blocks** (` ```...``` `)
3. **Deserialize** using the configured format

This handles common scenarios where LLMs return:
- Thinking/reasoning blocks before the actual response
- Responses wrapped in markdown code blocks
- Both thinking blocks AND code blocks

### Thinking Block Removal

Models may return responses with thinking blocks:

```xml
<think>
**Examining the Task**

I'm currently focused on refining the subtitle pairs.
The English text needs capitalization fixes.
</think>

[
  {"id": 120, "eng": "She'll forgive you.", "chinese": "她会原谅你"}
]
```

The `<think>...</think>` content is automatically removed, leaving only the actual response.

**Features:**
- Case-insensitive matching (`<think>`, `<THINK>`, `<Think>`)
- Multiline support (thinking blocks can span multiple lines)
- Multiple thinking blocks handled correctly

### Markdown Code Block Extraction

Models may wrap responses in markdown code blocks:

**JSON:**
```markdown
```json
[
  {"id": 120, "eng": "She'll forgive you.", "chinese": "她会原谅你"}
]
```
```

**XML-pair:**
```markdown
```xml
<pair>
ID=120
eng=She'll forgive you.
chinese=她会原谅你
</pair>
```
```

**Pseudo-TOML:**
```markdown
```toml
[pair]
id = 120
eng = She'll forgive you.
chinese = 她会原谅你
```
```

The content is automatically extracted from the code blocks.

**Features:**
- Handles language specifiers (```json, ```xml, ```toml)
- Handles code blocks without language specifier (` ``` `)
- Extracts first code block if multiple present

### Combined Scenarios

The most complex case - both thinking blocks AND code blocks:

```markdown
<think>Analyzing the task...</think>

Here is the result:

```json
[
  {"id": 120, "eng": "She'll forgive you.", "chinese": "她会原谅你"}
]
```
```

Processing order ensures thinking blocks are removed **before** extracting from code blocks.

### Testing

Comprehensive tests verify all scenarios:

```bash
python experiment/test_response_cleaning.py
```

**Tests include:**
1. Thinking block removal (simple, formatted, case-insensitive, multiple)
2. Code block extraction (JSON, XML, TOML, with/without language)
3. Combined cleaning (thinking + code blocks)
4. Full pipeline for all three formats
5. Edge cases (empty blocks, nested tags, whitespace)

**Expected Output:**
```
============================================================
TEST SUMMARY
============================================================
✓ PASS: test_thinking_block_removal
✓ PASS: test_code_block_extraction
✓ PASS: test_combined_cleaning
✓ PASS: test_full_pipeline_json
✓ PASS: test_full_pipeline_xml
✓ PASS: test_full_pipeline_toml
✓ PASS: test_edge_cases

Total: 7/7 tests passed

Response cleaning is working correctly for:
  • Thinking block removal (<think>...</think>)
  • Markdown code block extraction (```...```)
  • Combined scenarios (thinking + code blocks)
  • All formats (JSON, XML-pair, pseudo-TOML)
```

### Implementation Details

**Functions in `llm_client_sdk.py`:**

```python
def _strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    ...

def _extract_from_code_blocks(text: str) -> Optional[str]:
    """Extract content from markdown code blocks."""
    ...

def _clean_llm_response(text: str) -> str:
    """
    Clean LLM response (thinking blocks + code blocks).

    Processing order:
    1. Remove thinking blocks
    2. Extract from code blocks
    3. Return cleaned text
    """
    ...
```

**Integration:**

Both `refine_chunk_sdk()` and `refine_chunk_sdk_streaming()` use `_clean_llm_response()`:

```python
# Get raw response from LLM
response_text, usage = call_openai_api_sdk(messages, config)

# Clean response
cleaned = _clean_llm_response(response_text)

# Deserialize
pairs = deserialize(cleaned, config.intermediate_format)
```

### Error Handling

If deserialization fails after cleaning, both cleaned and raw responses are shown:

```
[Deserialization error]: Expected '<pair>' at line 1, got: invalid
[Cleaned response excerpt]: invalid...
[Raw response excerpt]: <think>...</think>```invalid```...
```

This helps diagnose whether:
- The cleaning failed
- The LLM returned malformed content
- The deserializer has a bug

---

## Future Enhancements

Potential improvements:
1. Additional formats (YAML, CSV with escaping)
2. Format auto-detection from LLM responses
3. Compression for very large chunks
4. Streaming serialization for memory efficiency

---

## References

- Implementation: `experiment/serializers.py`
- Response cleaning: `experiment/llm_client_sdk.py` (_clean_llm_response, _strip_thinking_blocks, _extract_from_code_blocks)
- Tests: `experiment/test_serializers.py`, `experiment/test_response_cleaning.py`
- Configuration: `experiment/config_sdk.py`
- Integration: `experiment/llm_client_sdk.py` (refine_chunk_sdk, refine_chunk_sdk_streaming)
- Example converter: `prompts.py:convert_examples_to_format()`

---

**Last Updated:** December 17, 2025
**Version:** 1.1.0
