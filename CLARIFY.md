# CLARIFY.md

Project-local clarification rules for agent-ready task briefs.

## Priority Scale

- `P0` / `0`: Critical. Production breakage, data loss, security exposure, or work that blocks all execution.
- `P1` / `1`: High. Important correctness regression, blocked milestone, or risky runtime behavior needing prompt attention.
- `P2` / `2`: Normal. Default priority for planned experiments, offline evaluation, feature development, or cleanup work.
- `P3` / `3`: Low. Nice-to-have polish, documentation, or non-urgent maintenance.

## Labels

- `nvfp4`: Use for NVFP4 quantization, dequantization, matmul, and cache-path tasks.
- `kv-cache`: Use for K/V cache storage, replay, or validation work.
- `tensor-export`: Use for runtime tensor export plumbing and artifacts.
- `attention-replay`: Use for offline attention score/probability replay and comparison work.
- `ggml-cuda`: Use for CUDA backend and CUDA experiment implementation changes.
- `ggml-cpu`: Use for CPU reference paths and offline validation helpers.
- `llama-graph`: Use for graph build, tensor binding, and dispatch-path tasks.
- `tensor-export-eval`: Use for `llama-tensor-export-eval` tool code and tests.
- `experiments`: Use for `experiments/` run folders, scripts, artifacts, and summaries.
- `server-validation`: Use for `llama-server` startup and request-validation tasks.
- `docs-development`: Use for development workflow and experiment documentation.
- `model-runtime`: Use for model binding and runtime execution-path changes.

## Label Rules

- Prefer functional subsystems or experiment surfaces over generic work-type labels.
- Use lowercase kebab-case labels.
- Reuse the labels above unless a new recurring subsystem clearly needs its own label.
- Avoid generic labels such as `bug`, `feature`, `cleanup`, or `misc` when a subsystem label fits.

## Clarification Defaults

- Ask one question at a time unless the user explicitly requests batch mode.
- If enough repo context exists to proceed safely, provide an agent-ready brief instead of over-clarifying.
- For tasks below `85/100` maturity, include unresolved assumptions in the brief or task draft.
