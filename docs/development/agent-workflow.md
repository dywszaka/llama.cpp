# Agent Workflow Guidelines

Reusable guidance for agent-assisted development across projects. Project-specific
instructions should live in the project's root `AGENTS.md` or equivalent local
guide and should reference this document instead of duplicating generic rules.

## Scope

Use this document for durable, cross-project working agreements:

- documentation hygiene;
- change and experiment-switch policy;
- logging and diagnostic discipline;
- validation and reporting expectations.

Do not put project-specific paths, hardware settings, model locations, concrete
validation commands, or local environment-variable names here. Keep those in the
project-local agent guide or another local contract document.

## Documentation Hygiene

- Keep long-lived guidance separate from dated session notes.
- Do not record temporary build directories, one-off validation results, old
  commit lists, or raw experiment outputs in durable guidance documents.
- When a path, runtime description, switch, or validation entry point changes,
  update the relevant index, map, or reference in the same change.
- Prefer one authoritative document for each contract. Link to it instead of
  copying the same rule into many places.
- Make the scope of each document clear:
  - durable workflow guidance;
  - project-local contracts;
  - experiment summaries;
  - raw logs and artifacts.

## Change Policy and Feature Switches

- Treat experimental, uncertain, performance-sensitive, or behavior-changing work
  as switch-gated unless the project explicitly says otherwise.
- New experimental switches should preserve existing behavior by default.
- Switch definitions should be centralized in one clear location that documents:
  - the switch name;
  - the default state;
  - the behavior it enables;
  - any priority or interaction with related switches.
- Prefer one narrow helper or accessor per switch.
- Avoid repeated environment parsing, duplicated string checks, or scattered
  direct switch reads across unrelated call sites.
- Keep experiment control separate from core algorithm code where practical.
- Prefer small functions with one reason to change and narrow helper APIs over
  broad configuration plumbing.
- When touching older switch code, migrate toward centralized helpers and
  documentation instead of adding another scattered check.

## Logging and Diagnostics

- Normal or release execution should not contain high-volume diagnostic logs.
- Per-item, per-request, per-token, per-kernel, or tight-loop diagnostics should
  be debug-only or protected by an explicit diagnostic switch.
- Important switch state or path-selection logs may be useful when they print
  once during startup or first use.
- Logs should confirm runtime configuration and code paths without becoming a
  meaningful performance input.
- For performance experiments, record enough diagnostic information to explain
  the result, but avoid logging that changes the behavior being measured.

## Validation and Reporting

- Report only validation that was actually run.
- If validation is skipped because hardware, toolkit, data, credentials, build
  artifacts, or other local prerequisites are unavailable, say so explicitly.
- Distinguish validation types when reporting results:
  - build or compile validation;
  - unit or focused tests;
  - smoke tests;
  - performance measurements;
  - profiler captures;
  - manual inspection.
- Do not present an unrun check as evidence. If a check cannot be run locally,
  provide the project-specific command or plan that should be used downstream.
- When a validation result depends on a switch, runtime mode, or environment
  setting, include that context with the result.
