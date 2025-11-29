# 从任意目录运行 main_sdk.py

## 问题

默认情况下，`main_sdk.py` 使用相对导入，必须在特定目录下运行才能正确找到模块。

## 解决方案

提供了 `run.sh` 包装脚本，可以从任何目录运行。

## 使用方法

### 方式 1：使用 run.sh（推荐）

```bash
# 在 experiment 目录内
cd /Users/zerozaki07/Downloads/subretrans/experiment
./run.sh input.ass output.ass --streaming -v

# 从任意目录运行（使用绝对路径）
cd /tmp
/Users/zerozaki07/Downloads/subretrans/experiment/run.sh \
  ~/files/input.ass ~/files/output.ass --streaming -v

# 从任意目录运行（使用相对路径）
cd ~/Documents
../Downloads/subretrans/experiment/run.sh input.ass output.ass --dry-run
```

### 方式 2：创建符号链接（推荐，最便捷）

`run.sh` 脚本已经优化，可以正确处理符号链接。无论链接到哪里，都能找到正确的路径。

**快速设置（推荐）：**

```bash
# 1. 创建个人 bin 目录（如果不存在）
mkdir -p ~/bin

# 2. 添加到 PATH（如果还没有）
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 3. 创建符号链接
ln -s /Users/zerozaki07/Downloads/subretrans/experiment/run.sh ~/bin/subretrans

# 4. 测试
subretrans --help

# 5. 现在可以在任何地方使用！
cd ~/Documents
subretrans input.ass output.ass --streaming -v
```

**或者链接到系统目录（需要权限）：**

```bash
# 将 run.sh 链接到 /usr/local/bin
sudo ln -s /Users/zerozaki07/Downloads/subretrans/experiment/run.sh \
  /usr/local/bin/subretrans

# 现在可以在任何地方直接运行
cd ~/Documents
subretrans input.ass output.ass --streaming -v
```

**路径解析说明：**

脚本使用智能路径解析，即使通过符号链接调用也能正确工作：

```bash
# 符号链接
~/bin/subretrans
  ↓
# 解析到真实路径
/Users/zerozaki07/Downloads/subretrans/experiment/run.sh
  ↓
# 自动找到
/Users/zerozaki07/Downloads/subretrans/venv/bin/activate
/Users/zerozaki07/Downloads/subretrans/experiment/main_sdk.py
```

**验证符号链接：**

```bash
# 查看符号链接
ls -la ~/bin/subretrans

# 输出类似：
# lrwxr-xr-x  1 user  staff  56 Nov 30 01:15 /Users/user/bin/subretrans -> /Users/zerozaki07/Downloads/subretrans/experiment/run.sh
```

### 方式 3：使用别名（简单）

在你的 shell 配置文件中添加别名：

```bash
# 编辑 ~/.bashrc 或 ~/.zshrc
nano ~/.bashrc

# 添加这一行
alias subretrans='/Users/zerozaki07/Downloads/subretrans/experiment/run.sh'

# 重新加载配置
source ~/.bashrc

# 现在可以在任何地方使用
cd ~/Documents
subretrans input.ass output.ass --streaming -v
```

## 工作原理

`run.sh` 脚本做了三件事：

1. **检测自身位置**：
   ```bash
   SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
   ```

2. **激活虚拟环境**：
   ```bash
   source "$PROJECT_DIR/venv/bin/activate"
   ```

3. **运行 main_sdk.py**：
   ```bash
   exec python "$SCRIPT_DIR/main_sdk.py" "$@"
   ```

这样无论你从哪里运行脚本，它都能：
- 找到正确的 Python 虚拟环境
- 找到正确的 main_sdk.py 和所有依赖模块
- 正确处理输入/输出文件的路径

## 文件路径处理

### 绝对路径（推荐）

```bash
# 输入和输出使用绝对路径
subretrans \
  /Users/zerozaki07/Documents/input.ass \
  /Users/zerozaki07/Documents/output.ass \
  --streaming -v
```

### 相对路径

相对路径相对于**当前工作目录**，而不是脚本位置：

```bash
# 当前在 ~/Documents
cd ~/Documents

# 这些路径相对于 ~/Documents
subretrans input.ass output.ass --streaming -v
# 等同于：
# input: ~/Documents/input.ass
# output: ~/Documents/output.ass
```

### 波浪号扩展

```bash
# 使用 ~ 代表 home 目录
subretrans ~/files/input.ass ~/files/output.ass --streaming -v
```

## 示例

### 示例 1：处理不同目录的文件

```bash
# 当前在任意目录
cd /tmp

# 处理 Documents 中的文件
subretrans \
  ~/Documents/subtitles/input.ass \
  ~/Documents/subtitles/output.ass \
  --streaming -v
```

### 示例 2：批处理多个文件

```bash
cd ~/Documents/subtitles

for file in *.ass; do
  subretrans "$file" "refined_$file" --streaming -v
done
```

### 示例 3：使用配置文件的默认设置

```bash
# config.yaml 已设置 use_streaming: true 和 verbose: true
cd ~/Videos

# 直接运行，使用 YAML 配置
subretrans episode01.ass episode01_refined.ass
```

### 示例 4：调试模式

```bash
# 从任意目录运行，看到实时 LLM 输出
cd /tmp
subretrans ~/files/test.ass output.ass \
  --streaming -vvv \
  --dry-run \
  --max-chunks 1
```

## 故障排除

### 问题：找不到虚拟环境

```
source: can't open venv/bin/activate
```

**解决**：检查项目路径是否正确，虚拟环境是否存在：
```bash
ls -la /Users/zerozaki07/Downloads/subretrans/venv/bin/activate
```

### 问题：找不到 Python 模块

```
ModuleNotFoundError: No module named 'xxx'
```

**解决**：确保虚拟环境已激活并安装了所有依赖：
```bash
cd /Users/zerozaki07/Downloads/subretrans
source venv/bin/activate
pip install -r requirements.txt
```

### 问题：找不到输入文件

```
FileNotFoundError: [Errno 2] No such file or directory: 'input.ass'
```

**解决**：使用绝对路径或确认相对路径正确：
```bash
# 使用绝对路径
subretrans /full/path/to/input.ass /full/path/to/output.ass

# 或检查当前目录
pwd
ls input.ass
```

### 问题：权限错误

```
Permission denied: ./run.sh
```

**解决**：添加执行权限：
```bash
chmod +x /Users/zerozaki07/Downloads/subretrans/experiment/run.sh
```

## 技术细节

### 路径解析优先级

1. **绝对路径**（以 `/` 开头）→ 直接使用
2. **波浪号路径**（以 `~` 开头）→ 扩展为 home 目录
3. **相对路径**（其他）→ 相对于当前工作目录

### 虚拟环境激活

脚本自动激活虚拟环境，确保使用正确的 Python 和已安装的包。

### exec 命令

使用 `exec` 替换当前 shell 进程，确保：
- 正确的退出码传递
- 信号处理正确
- 不留下多余的 shell 进程

## 性能考虑

使用包装脚本不会影响性能：
- 脚本启动开销：< 0.01 秒
- 主要时间花在 LLM API 调用上

## 总结

推荐使用方式：

**日常使用**：创建符号链接或别名
```bash
# 一次性设置
ln -s /Users/zerozaki07/Downloads/subretrans/experiment/run.sh ~/bin/subretrans

# 之后在任何地方使用
subretrans input.ass output.ass --streaming -v
```

**临时使用**：直接使用绝对路径
```bash
/Users/zerozaki07/Downloads/subretrans/experiment/run.sh \
  input.ass output.ass --streaming -v
```

**脚本中使用**：使用绝对路径，确保可靠性
```bash
#!/bin/bash
SUBRETRANS="/Users/zerozaki07/Downloads/subretrans/experiment/run.sh"
$SUBRETRANS input.ass output.ass --streaming -v
```

---

**最后更新**：2025-11-30
**状态**：生产就绪 ✅
