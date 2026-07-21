# rlm-cli

[![npm version](https://img.shields.io/npm/v/rlm-cli.svg)](https://www.npmjs.com/package/rlm-cli)
[![license](https://img.shields.io/npm/l/rlm-cli.svg)](https://github.com/viplismism/rlm-cli/blob/main/LICENSE)
[![node](https://img.shields.io/node/v/rlm-cli.svg)](https://nodejs.org/)

CLI for **Recursive Language Models** — based on the [RLM paper](https://arxiv.org/abs/2512.24601).

Instead of dumping a huge context into a single LLM call, RLM lets the model write Python code to process it — slicing, chunking, running sub-queries on pieces, and building up an answer across multiple iterations.

<p align="center">
  <img src="demo.png" alt="rlm-cli demo" width="750">
</p>

## Quickstart

```bash
npm install -g rlm-cli
rlm                                          # interactive — first run sets up your provider + key
rlm run --file big.log "which errors repeat most, and when?"
```

Uses current frontier families only: GPT 5.6+, Claude Fable 5+, and Kimi K3+. Registered models use their native provider; newly released models can use configurable OpenAI- or Anthropic-compatible endpoints.

## Security

rlm runs Python that the **LLM writes**, and a prompt-injected context document (a file or fetched URL) can steer that code — so the subprocess is treated as untrusted.

- **Sandboxed by default.** The Python subprocess runs inside an OS-level sandbox (macOS `sandbox-exec`/Seatbelt, Linux `bwrap`/bubblewrap) that **blocks all network** and **hides `~/.rlm`**, so injected code can't exfiltrate your API keys. All LLM calls are proxied through the parent process, so the child needs neither network nor credential access — nothing legitimate breaks. The sandbox is probe-tested before use.
- **Graceful fallback.** Where no sandbox is available (Windows, or `bwrap` not installed), rlm prints a warning and runs unsandboxed — the code then has full access as your user, so only point it at content you trust or run inside a container/VM.
- **Opt out** with `RLM_NO_SANDBOX=1` when model code legitimately needs network or local-file access and you trust the input.

> The sandbox confines the primary exfiltration paths (network + credential reads). It does not yet fully confine arbitrary local file *writes* — treat untrusted context with care regardless.

## What's New in v0.6.0

- **Sandboxed execution** — model-generated Python runs in an OS-level sandbox by default (no network, no access to `~/.rlm`), so prompt-injected code can't exfiltrate your keys — see [Security](#security)
- **Mixed-model mode** — `sub_model` lets Kimi K3 or another frontier model analyze chunks while the strongest model plans and synthesizes
- **Paper-aligned system prompt** — per-iteration budget awareness, sub-query strategy guidance, parallel async patterns from arXiv:2512.24601
- **Session-based trajectories** — runs grouped into `~/.rlm/sessions/<session-id>/` instead of a flat directory
- **Refreshed terminal UI** — Electric Amber RGB palette, two-column welcome panel with version in border, silent operation (no noise between queries)
- **Honest runtime limits** — `max_depth` is pinned to `1` because the current runtime implements flat paper-style sub-calls, not nested recursive RLM agents

---

## Install

```bash
npm install -g rlm-cli
```

Requires **Node.js >= 20** and **Python 3**.

Run `rlm` to start. First launch will prompt for a provider + API key (saved to `~/.rlm/credentials`).

---

## Frontier Models

Older model families are intentionally excluded from model selection. The accepted floors are GPT 5.6, Claude Fable 5, and Kimi K3; later versions of those families are accepted automatically.

### GPT-5.6 Sol

```dotenv
RLM_MODEL=gpt-5.6-sol
OPENAI_API_KEY=...
```

### Claude Fable 5

```dotenv
RLM_MODEL=claude-fable-5
ANTHROPIC_API_KEY=...
```

### GPT-5.6 Sol root with Kimi K3 sub-queries

This keeps planning and final synthesis on Sol while Kimi processes selected chunks:

```dotenv
RLM_MODEL=gpt-5.6-sol
OPENAI_API_KEY=...
RLM_SUB_MODEL=kimi-k3
RLM_SUB_API_KEY=...
RLM_SUB_BASE_URL=https://api.moonshot.ai/v1
RLM_SUB_CONTEXT_WINDOW=262144
```

Set `sub_model: kimi-k3` in `rlm_config.yaml`. If your Kimi host uses a different model ID or URL, use the exact values supplied by that host.

Unregistered root models use `RLM_API_KEY`, `RLM_BASE_URL`, and optional `RLM_API` (`openai-completions` or `anthropic-messages`). Sub-model overrides use the corresponding `RLM_SUB_*` variables. Never put real keys in `rlm_config.yaml` or commit them.

Keys are loaded from (highest priority wins):
1. Shell environment variables
2. `.env` file in the current working directory (falls back to the package root)
3. `~/.rlm/credentials`

### From Source

```bash
git clone https://github.com/viplismism/rlm-cli.git
cd rlm-cli
npm install
npm run build
npm link
```

---

## Usage

### Interactive Terminal

```bash
rlm
```

Persistent session with a two-column welcome panel showing your model, provider, context, and quick-ref slash commands. Everything auto-saves to a session folder.

**Slash commands:**

| Command | What it does |
|---------|-------------|
| `/file <path>` | Load file, directory, or glob as context |
| `/url <url>` | Fetch URL as context |
| `@file <query>` | Load file + run query in one step |
| `/model [id]` | List or switch model by ID (shows Ollama models too) |
| `/provider` | Switch provider (includes Ollama if running) |
| `/trace` | Open the live RLM trace window |
| `/trajectories` | Browse saved sessions |
| `/clear` | Clear the transcript |
| `/help` | Full command reference |
| `/quit` | Exit |

**Tips:**
- Just type a question — no context needed for general queries
- Paste a URL directly to fetch it as context
- **Ctrl+C** stops a running query, **Ctrl+C twice** exits

### Single-Shot Mode

```bash
rlm run "Explain recursive language models"
rlm run --file large-file.txt "List all classes and their methods"
rlm run --url https://example.com/data.txt "Summarize this"
cat data.txt | rlm run --stdin "Count the errors"
rlm run --model gpt-5.6-sol --file code.py "Find bugs"
```

Answer goes to stdout, progress to stderr — pipe-friendly.

### Trajectory Viewer

```bash
rlm viewer
```

Browse saved runs in a TUI. Navigate iterations, inspect code and output at each step, drill into sub-queries. Sessions are saved to `~/.rlm/sessions/`.

---

## Benchmarks

Compare direct LLM vs RLM on the same query from standard long-context datasets.

| Benchmark | Dataset | What it tests |
|-----------|---------|---------------|
| `oolong` | [Oolong Synth](https://huggingface.co/datasets/oolongbench/oolong-synth) | Synthetic long-context: timeline ordering, user tracking, counting |
| `longbench` | [LongBench NarrativeQA](https://huggingface.co/datasets/THUDM/LongBench) | Reading comprehension over long narratives |

```bash
rlm benchmark oolong          # default: index 4743
rlm benchmark longbench       # default: index 182
rlm benchmark oolong --idx 10
```

Python dependencies are auto-installed into a `.venv` on first run.

> **Note:** `rlm benchmark` requires a source checkout of the repo (see [From Source](#from-source)) — it is not available in npm installs.

---

## How It Works

1. Your full context is loaded into a persistent Python REPL as a `context` variable
2. The LLM gets metadata about the context (size, preview) plus your query
3. It writes Python code that can slice `context`, call `llm_query(chunk, instruction)` for sub-questions, and call `FINAL(answer)` when done
4. Code runs, output is captured and fed back for the next iteration
5. Loop continues until `FINAL()` is called or max iterations are reached

For large documents, the model chunks the text and runs parallel sub-queries with `async_llm_query()` + `asyncio.gather()`, then aggregates the results.

---

## Configuration

Create `rlm_config.yaml` in your working directory:

```yaml
max_iterations: 20       # Max iterations before giving up (1-100)
max_depth: 1             # Fixed at 1 in the current runtime
max_sub_queries: 50      # Max total sub-queries (1-500)
truncate_len: 5000       # Truncate REPL output beyond this (500-50000)
metadata_preview_lines: 20
sub_model: kimi-k3       # Optional frontier model for chunk analysis
max_total_tokens: 0      # Hard run budget; 0 disables
max_cost_usd: 0          # Hard run budget; 0 disables
max_concurrency: 8       # Concurrent sub-queries
min_request_interval_ms: 0
max_retries: 2
chunk_size_chars: 0      # 0 derives from the sub-model context window
```

For custom endpoints, set `RLM_INPUT_COST_PER_M`/`RLM_OUTPUT_COST_PER_M` (or `RLM_SUB_*`) so cost budgets use the host's current pricing. Token budgets work from provider-reported usage.

---

## Project Structure

```
src/
  main.ts          CLI entry point and command router
  interactive.ts   Interactive terminal REPL
  rlm.ts           Core RLM loop (Algorithm 1 from paper)
  repl.ts          Python REPL subprocess manager
  sandbox.ts       OS-level sandbox for the Python subprocess (Seatbelt/bubblewrap)
  runtime.py       Python runtime (FINAL, FINAL_VAR, llm_query, async_llm_query)
  cli.ts           Single-shot CLI mode
  viewer.ts        Trajectory viewer TUI
  colors.ts        Terminal color palette (Electric Amber RGB)
  ollama.ts        Ollama local model integration
  config.ts        Config loader
  env.ts           Environment variable loader
benchmarks/
  oolong_synth.ts
  longbench_narrativeqa.ts
  requirements.txt
bin/
  rlm.mjs          Global CLI shim
```

---

## License

MIT
