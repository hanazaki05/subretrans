# Prompt Generator Usage Guide

## 概述

`genreq.py` 是一个用于生成请求 prompts 的工具，**不调用真实 API**，只生成 system prompt 和 user prompt 到 markdown 文件。

这对以下场景很有用：
- 测试不同模型的响应时间
- 检查 prompt 质量和内容
- 估算 token 使用量
- 调试 prompt 生成逻辑

---

## 快速开始

### 基本用法

```bash
cd experiment
python genreq.py JAG.S04E09.zh-cn.ass --pairs-per-chunk 120
```

**输出：** `JAG.S04E09.zh-cn_prompts.md`

---

## 命令行参数

### 必需参数

- `input`: 输入的 .ass 字幕文件路径
- `--pairs-per-chunk N`: 每个 chunk 包含的字幕对数量（必需）

### 可选参数

- `--output FILE`: 指定输出 markdown 文件路径（默认：`{input_basename}_prompts.md`）
- `--max-chunks N`: 限制生成的 chunk 数量（用于测试，默认：全部）

---

## 使用示例

### 示例 1：生成所有 chunks

```bash
python genreq.py JAG.S04E09.zh-cn.ass --pairs-per-chunk 120
```

**输出文件：** `JAG.S04E09.zh-cn_prompts.md`

**包含：**
- 完整的配置信息
- Token 使用统计表
- 每个 chunk 的 system prompt
- 每个 chunk 的 user prompt
- 每个 chunk 的 token 估算

---

### 示例 2：只生成前 2 个 chunks（快速测试）

```bash
python genreq.py JAG.S04E09.zh-cn.ass --pairs-per-chunk 120 --max-chunks 2
```

**用途：** 快速生成部分 prompts 查看效果

---

### 示例 3：自定义输出文件名

```bash
python genreq.py input.ass --pairs-per-chunk 100 --output test_prompts.md
```

**输出文件：** `test_prompts.md`

---

### 示例 4：测试不同的 chunk 大小

```bash
# 小 chunks（每个 50 对）
python genreq.py input.ass --pairs-per-chunk 50 --output prompts_50.md

# 中等 chunks（每个 100 对）
python genreq.py input.ass --pairs-per-chunk 100 --output prompts_100.md

# 大 chunks（每个 200 对）
python genreq.py input.ass --pairs-per-chunk 200 --output prompts_200.md
```

**用途：** 比较不同 chunk 大小对 token 使用的影响

---

## 输出文件格式

生成的 markdown 文件包含以下内容：

### 1. 配置信息

```markdown
## Configuration

- **Total pairs:** 350
- **Pairs per chunk:** 120
- **Total chunks:** 3
- **Model:** gpt-5-mini
- **Max output tokens:** 12,000
- **Temperature:** 1.0
- **Reasoning effort:** medium
```

### 2. Token 统计表

```markdown
## Token Summary

| Chunk | Pairs | System Tokens | User Tokens | Total Tokens |
|-------|-------|---------------|-------------|--------------|
| 1/3   | 120   | 1,500         | 8,000       | 9,500        |
| 2/3   | 120   | 1,500         | 8,000       | 9,500        |
| 3/3   | 110   | 1,500         | 7,300       | 8,800        |
| **Total** | 350 | 4,500     | 23,300      | 27,800       |
```

### 3. 每个 Chunk 的详细信息

```markdown
## Chunk 1/3 (120 pairs)

### System Prompt
```
[完整的 system prompt 内容，包括所有规则和 memory]
```

### User Prompt
```
[当前 chunk 的字幕对 JSON 内容]
```

### Token Estimates
- **System prompt:** 1,500 tokens
- **User content:** 8,000 tokens
- **Total input:** 9,500 tokens
- **Max output:** 12,000 tokens
- **Estimated max total:** 21,500 tokens
```

---

## 使用场景

### 场景 1：测试 API 响应时间

```bash
# 1. 生成 prompts
python genreq.py input.ass --pairs-per-chunk 120

# 2. 从 markdown 文件中提取 prompts
# 3. 使用其他工具（如 curl, httpie）发送请求测试时间
```

### 场景 2：检查 Prompt 质量

```bash
# 生成 prompts 后直接查看 markdown 文件
python genreq.py input.ass --pairs-per-chunk 100
cat input_prompts.md
```

### 场景 3：优化 Chunk 大小

```bash
# 生成不同 chunk 大小的 prompts
python genreq.py input.ass --pairs-per-chunk 50 --output p50.md
python genreq.py input.ass --pairs-per-chunk 100 --output p100.md
python genreq.py input.ass --pairs-per-chunk 150 --output p150.md

# 比较 token 使用量，找到最优配置
```

### 场景 4：估算成本

```bash
# 生成 prompts 查看总 token 数
python genreq.py input.ass --pairs-per-chunk 120

# 根据输出的 Token Summary 表计算：
# Total input tokens + Max output tokens × chunks = 总成本估算
```

---

## 工作原理

1. **解析 ASS 文件** - 提取所有字幕对
2. **加载配置** - 读取 `config.yaml` 和用户自定义 prompt
3. **分块** - 按 `--pairs-per-chunk` 参数分割字幕对
4. **生成 Prompts** - 为每个 chunk 调用 prompt 生成函数
5. **估算 Tokens** - 使用 tiktoken 计算 token 数量
6. **输出 Markdown** - 将所有信息写入文件

**注意：** 整个过程**不调用 OpenAI API**，完全本地运行。

---

## 配置说明

工具会读取 `experiment/config.yaml` 中的配置：
- `main_model.name` - 用于 token 估算
- `main_model.max_output_tokens` - 显示在输出中
- `main_model.temperature` - 显示在输出中
- `user.prompt_path` - 加载用户自定义 prompt

如果需要修改配置，直接编辑 `config.yaml` 文件。

---

## 与其他工具配合

### 使用 curl 测试

```bash
# 1. 生成 prompts
python genreq.py input.ass --pairs-per-chunk 100 --max-chunks 1

# 2. 从 markdown 中提取 prompts（手动或脚本）
# 3. 使用 curl 发送请求
curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-5-mini",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ]
  }'
```

### 使用 Python 脚本测试

```python
import time
from openai import OpenAI

# 从生成的 markdown 文件读取 prompts
system_prompt = "..."  # 从文件提取
user_prompt = "..."    # 从文件提取

client = OpenAI()
start_time = time.time()

response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

elapsed = time.time() - start_time
print(f"Response time: {elapsed:.2f}s")
```

---

## 常见问题

### Q: 为什么需要 `--pairs-per-chunk` 参数？

A: 因为这个工具专注于按固定数量分块，可以精确控制每个请求的大小，方便测试和比较。

### Q: 生成的 prompts 和真实 API 调用的一样吗？

A: 是的，使用的是完全相同的 prompt 生成函数（`build_system_prompt` 和 `build_user_prompt_for_chunk`）。

### Q: Token 估算准确吗？

A: 使用 tiktoken（OpenAI 官方库）估算，与实际 API 统计非常接近，误差通常在 1-2% 以内。

### Q: 可以修改生成的 prompts 吗？

A: 可以！生成的是 markdown 文件，可以手动编辑用于测试。

### Q: 支持其他语言对吗？

A: 目前只支持英中字幕对（由主项目决定），但可以修改 prompt 生成逻辑来支持其他语言。

---

## 技术细节

**依赖模块：**
- `config_sdk.py` - 配置加载
- `ass_parser.py` - ASS 文件解析
- `chunker.py` - 字幕对分块
- `memory.py` - 全局记忆初始化
- `prompts.py` - Prompt 生成
- `utils.py` - Token 估算

**不依赖：**
- OpenAI API（完全离线运行）
- 网络连接
- API Key

---

## 下一步

生成 prompts 后，你可以：

1. **查看 markdown 文件** - 检查 prompt 内容和质量
2. **提取 prompts** - 用脚本或手动复制到测试工具
3. **测试不同模型** - 使用相同 prompts 测试 GPT-4, GPT-5 等
4. **优化配置** - 根据 token 统计调整 chunk 大小
5. **估算成本** - 计算完整处理的预期费用

---

**最后更新：** 2025-11-30
**状态：** 生产就绪 ✅
