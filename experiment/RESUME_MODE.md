# Resume Mode Feature

## Overview

The `--resume` feature allows you to restart subtitle processing from a specific pair index. This is useful for:
- Recovering from errors or interruptions
- Continuing processing after API failures
- Re-processing only a portion of the file with different settings

## Usage

```bash
python main_sdk.py input.ass output.ass --resume INDEX [other options]
```

### Example

```bash
# Resume from pair index 680 with 75 pairs per chunk
python main_sdk.py JAG.S04E10.zh-cn.ass JAG.S04E10.zh-cn.2.ass --pairs-per-chunk 75 -vvv --streaming --resume 680
```

## How It Works

### 1. **Validation**
   - Checks that resume index is non-negative
   - Checks that resume index is within the total pair count

### 2. **Preserve Earlier Pairs**
   - If output file exists: Loads it and copies pairs 0 to (resume_index-1) into the processing list
   - If output file doesn't exist: Uses original pairs from input file

### 3. **Process Only Remaining Pairs**
   - Creates `pairs_to_process` containing only pairs from `resume_index` onwards
   - Chunks and processes only these pairs
   - Applies corrections back to the full pairs list using ID matching

### 4. **Generate Complete Output**
   - Writes the FULL pairs list to the output file
   - Result contains preserved earlier pairs + newly processed pairs

## Examples

### Scenario 1: Recover from Error at Pair 680

```bash
# Initial run failed at pair 680
python main_sdk.py input.ass output.ass --pairs-per-chunk 75 --streaming

# Resume from pair 680 to continue
python main_sdk.py input.ass output.ass --pairs-per-chunk 75 --streaming --resume 680
```

**Output:**
```
[RESUME MODE] Starting from pair index 680
Skipping first 680 pairs, processing remaining 320 pairs
Loading existing output file: output.ass
Preserved 680 pairs from existing output
Processing pairs 680 to 999 (320 pairs)
```

### Scenario 2: Re-process Last Section with Different Settings

```bash
# Initial full processing
python main_sdk.py input.ass output.ass

# Re-process last 200 pairs with verbose mode
python main_sdk.py input.ass output.ass --resume 800 -vvv
```

### Scenario 3: Resume on Fresh File (No Existing Output)

```bash
# Start processing from pair 500 (useful for testing specific sections)
python main_sdk.py input.ass output.ass --resume 500 --max-chunks 2
```

**Output:**
```
[RESUME MODE] Starting from pair index 500
Note: Output file does not exist yet, will create new file
Processing pairs 500 to 999 (500 pairs)
```

## Key Features

✅ **Preserves Earlier Work**: If output file exists, earlier pairs are loaded and preserved
✅ **ID-Based Matching**: Uses pair IDs to correctly apply corrections, not position
✅ **Full Output**: Always writes complete file with all pairs
✅ **Error Recovery**: Continue from where you left off after errors
✅ **Flexible**: Works with all other options (--pairs-per-chunk, --streaming, -vvv, etc.)
✅ **Glossary Checkpoint**: Learned terminology is automatically saved and restored across runs

## Glossary Checkpoint System

The resume mode works seamlessly with the **glossary checkpoint system** (opt-in feature):

### What It Does
- **Automatic Saving**: When enabled, after each chunk, learned terminology is saved to a checkpoint file
- **Automatic Loading**: When enabled, on startup, the checkpoint is loaded first
- **Checkpoint Format**: YAML format (e.g., `input.ass.glossary.yaml`)
- **Filename Inheritance**: Checkpoint filename is based on input file name
- **Opt-in Feature**: Enable with `--checkpoint` flag

### How to Enable
Add the `--checkpoint` flag to your command:
```bash
python main_sdk.py input.ass output.ass --checkpoint
```

### How It Works
1. **During Processing**: After each `update_global_memory()` call, the learned glossary is saved (if `--checkpoint` enabled)
2. **On Startup**: Before processing begins, if a checkpoint file exists and `--checkpoint` is enabled, it's loaded into GlobalMemory
3. **Continuous Updates**: Checkpoint is updated after every chunk and after memory compression (if enabled)

### Example Output (with `--checkpoint` enabled)
```
Step 3: Initialize global memory
  [CHECKPOINT] Loaded 42 glossary entries from: input.ass.glossary.yaml

Processing chunk 1/5...
  [Chunk completed, checkpoint updated]

Processing chunk 2/5...
  [Chunk completed, checkpoint updated]
...
```

### Example Output (first run with `--checkpoint`)
```
Step 3: Initialize global memory
  [CHECKPOINT] No existing checkpoint found, will create: input.ass.glossary.yaml

Processing chunk 1/5...
  [Chunk completed, checkpoint created]
...
```

### Benefits in Resume Mode
- **Preserves Learned Terms**: When resuming, all previously learned terminology is retained
- **Consistent Translation**: Same terms are translated consistently across resume sessions
- **No Re-learning**: The system doesn't need to re-extract terminology from earlier chunks

## Output Information

When using `--resume`, you'll see:

```
Step 2: Building subtitle pairs...
  Created 1000 subtitle pairs

  [RESUME MODE] Starting from pair index 680
  Skipping first 680 pairs, processing remaining 320 pairs
  Loading existing output file: output.ass
  Preserved 680 pairs from existing output
  Processing pairs 680 to 999 (320 pairs)

Step 3: Splitting into chunks...
  Chunking strategy: Fixed 75 pairs per chunk
  Created 5 chunks (320 pairs total)
  ...
```

## Notes

- Resume index is **0-based** (first pair is index 0)
- The output file will **always contain all pairs** from the input file
- If the existing output file is corrupted or incompatible, processing continues with original pairs
- Works seamlessly with `--pairs-per-chunk`, `--max-chunks`, and other options

## Comparison: Normal vs Resume Mode

| Aspect | Normal Mode | Resume Mode (--resume 680) |
|--------|-------------|----------------------------|
| Pairs processed | All (0-999) | Only 680-999 (320 pairs) |
| Chunks created | From all pairs | From pairs 680-999 only |
| Output file | New/Overwrite | Preserves 0-679, updates 680-999 |
| Use case | Fresh processing | Continue interrupted work |

## Technical Details

### Implementation

1. **Parameter**: `--resume INDEX` added to argparse
2. **Function signature**: `process_subtitles(..., resume_index: Optional[int] = None)`
3. **Logic flow**:
   ```python
   if resume_index is not None:
       # Load existing output file if exists
       if os.path.exists(output_path):
           existing_pairs = load_from_output()
           # Copy pairs 0 to resume_index-1
           pairs[0:resume_index] = existing_pairs[0:resume_index]

       # Process only from resume_index onwards
       pairs_to_process = pairs[resume_index:]
       chunks = chunk_pairs(pairs_to_process, ...)
   ```
4. **Correction application**: Uses ID-based matching to update correct pairs
5. **Output generation**: Writes full `pairs` list (preserved + newly processed)

### Error Handling

- **Invalid index (negative)**: Error message, exits
- **Index too large**: Error message, exits
- **Corrupted output file**: Warning, continues without preserved pairs
- **Missing output file**: Info message, creates new file

## Date

**Feature Added**: 2025-12-07
**Version**: 0.0.8
