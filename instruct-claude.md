# claude.md

## 1. Shell and Command Prerequisites

### 1.1 Python alias cleanup (mandatory before any Python usage)

Before running **any** Python command (including just activating a venv), run:

`unalias python && unalias python3`

## 2. Command-Line Tool Preferences

### 2.1 Search
Call `mcp__acemcp` in the first instance; only use `rg` (`/opt/homebrew/bin/rg`) instead of `grep` if acemcp is not working.

### 2.2 awk

When using awk, use `gawk` instead of `awk`.

## 3. Documentation Editing Policy

- Prefer editing existing documentation files (e.g., `README.md` or other `*.md` files) when changes are needed.
- Create new documentation files **only when necessary**.

## 4. Global Protocols

All operations must strictly adhere to the following system constraints:

- **Multi-round Conversation**: If a tool/model returns a field indicating a continuing conversation (e.g., `SESSION_ID`), record it and **explicitly consider** in subsequent calls whether to continue the same conversation (e.g., sessions may be interrupted; if the required reply is not received, continue the session).
- **Sandbox Security**: Codex is strictly prohibited from writing to the file system. Any code acquisition must be **explicitly requested** in the prompt and must be returned as a **Unified Diff Patch**. Codex must not make real modifications.
- **Code Sovereignty**: External-model code is only a logical reference (prototype). Final delivered code **must be refactored** to remove redundancy and meet enterprise-grade standards.
- **Style Definition**: Code must be concise, efficient, and non-redundant. This also applies to comments and documentation: **do not add them unless necessary**.
- **Targeted Changes Only**: Only make requirement-scoped changes; do not impact existing user functionality.
- **SKILL-Driven Codex Usage**: Codex interaction methods are provided in SKILL form; **active viewing and calling are mandatory**. This may require longer call duration (system default `BASH_DEFAULT_TIMEOUT_MS=300000`).
- **Parallel Execution**: When tasks can be parallelized, do everything possible to parallelize. Example: when multiple SKILL-related bash commands exist, use `run in background` to immediately suspend long-running programs and proceed to the next command.
- **Workflow Compliance**: Strictly follow **all workflow phases** in Section 5; omitting any phase is prohibited.

## 5. Workflow

### 5.1 Phase 1: Full Context Retrieval (Auggie Interface)

**Execution condition**: Before generating any suggestions or code.

1. **Tool Call**: Call `mcp__acemcp__codebase-retrieval`.
2. **Retrieval Strategy**:
   - Do not answer based on assumptions.
   - Use Natural Language (NL) semantic queries (Where/What/How).
   - **Completeness Check**: Obtain complete definitions and signatures for relevant classes, functions, and variables. If context is insufficient, trigger recursive retrieval.
3. **Requirement Alignment**: If requirements remain ambiguous after retrieval, output a guiding question list to the user until boundaries are clear (no omissions, no redundancy).

### 5.2 Phase 2: Strategic Analysis (Codex Interface)

**Execution condition**: After context is ready, before coding begins.

1. **Distribute Input**: Provide Codex the user’s **original requirements** (without preconceived interpretation). Codex has a complete CLI system; do not over-supply context.
2. **Solution Iteration**:
   - Request multi-perspective solutions.
   - Perform iterative cross-checking and refinement until a step-by-step implementation plan with no logical gaps is produced.
3. **User Confirmation**: Present the final implementation plan (with appropriate pseudocode) to the user.

### 5.3 Phase 3: Prototyping (Codex Kernel)

**Execution condition**: After the implementation plan is confirmed.

- **Core Capabilities**: Use Codex for logical computation, debugging, and implementation prototyping.
- **General Instructions**: Request an implementation prototype suitable for later refactoring.
- **Output Constraint (mandatory)**: Every Codex prompt **must explicitly request** output as a **Unified Diff Patch**. Codex must not make real modifications.

### 5.4 Phase 4: Implementation (Claude)

**Execution guidelines**:

1. **Refactor the Prototype**: Use the Phase 3 prototype as reference, remove redundancy, and rewrite into highly readable, maintainable, release-grade code.
2. **Documentation Standards**: Do not add comments or documentation unless necessary; prefer self-explanatory code.
3. **Minimal Scope**: Constrain changes strictly to requirements; perform a mandatory side-effect review and apply targeted fixes.

### 5.5 Phase 5: Audit & Delivery

1. **Automated Audit**: After changes take effect, immediately invoke Codex for code review.
   - Check items: logical correctness, requirement coverage, potential bugs/edge cases.
2. **Delivery**: Deliver results to the user only after the audit passes.

## 6. Resource Matrix

This matrix defines the **mandatory** resource invocation strategy per workflow phase. Claude, as the orchestrator, must schedule external resources strictly according to the current phase.

| Workflow Phase | Functionality | Designated Model / Tool | Input Strategy (Prompting) | Strict Output Constraints | Critical Constraints & Behavior |
| --- | --- | --- | --- | --- | --- |
| **Phase 1** | Context Retrieval | Auggie (`mcp__acemcp`) | Natural Language , focus on **What / Where / How** | Raw code / definitions (complete signatures) | **Forbidden:** `grep`/`rg` / keyword search. **Mandatory:** recursive retrieval until context is complete. |
| **Phase 2** | Analysis & Planning | Codex | Raw requirements ; minimal context | Step-by-step plan (text + pseudo-code) | Validate logic and approach; eliminate gaps before coding. |
| **Phase 3** | Prototyping | Codex | Focus on logic, algorithms, and implementation details | **Unified Diff Patch** (prototype only) | Use for debugging and prototyping. **Security:** no file system writes. |
| **Phase 4** | Refactoring / Implementation | Claude (self) | N/A | Production code | Claude is the final implementer. Style: clean, efficient, no redundancy; minimal comments. |
| **Phase 5** | Audit & QA | Codex | Unified diff + target file  | Review comments (bugs / edge cases) | Mandatory immediately after code changes; verify logic integrity. |