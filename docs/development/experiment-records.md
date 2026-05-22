# Experiment Record Guidelines

Reusable guidance for recording experiments, baselines, comparisons, profiling
runs, and evidence across projects. Project-specific documents should define the
local experiment root, baseline contract, scripts, workloads, hardware settings,
and validation commands.

## Scope

Use this document for experiments, profiling runs, performance comparisons, and
A/B validation. Keep project-specific values out of this document, including
model paths, device IDs, local datasets, repository-specific source paths,
concrete environment variables, and benchmark command lines.

## Experiment Directory Structure

- Use one directory per experiment or validation run family.
- Keep related artifacts together so the run can be understood without searching
  through unrelated directories.
- Store the exact command or script used for each run.
- Store input references or request payloads needed to reproduce the run.
- Store raw logs, raw tool output, profiler reports, parsed metrics, summaries,
  and environment/runtime notes together with the run.
- Keep baseline and experiment results in sibling or otherwise clearly linked
  locations when they are intended to be compared.
- Do not put transient raw output into durable workflow documents. Link to the
  run folder or summarize the stable conclusion instead.
- Prefer timestamped or otherwise sortable run directory names when a project
  accumulates many experiments.

## Baseline and Comparison Rules

- Start each direct comparison from a documented baseline.
- Change only the parameter, switch, workload, or implementation detail under
  test.
- Keep unrelated inputs fixed by default, such as:
  - code revision;
  - build configuration;
  - model or dataset;
  - request or prompt data;
  - runtime sizes and batching;
  - precision or cache settings;
  - hardware and device selection;
  - thread counts and affinity;
  - service/runtime modes;
  - diagnostic environment variables.
- If a baseline parameter must change, record the reason in the experiment
  folder.
- Do not claim a strict A/B comparison when unrelated parameters changed. Instead,
  state the limitation and the likely confounders.
- Comparison summaries should state:
  - what changed;
  - what stayed fixed;
  - why the comparison is valid or limited;
  - the baseline result;
  - the experiment result;
  - the measured delta;
  - known confounders.

## Evidence Standard

Use multiple independent signals before drawing a conclusion about runtime
behavior or performance:

1. Runtime configuration: captured command, script, environment, configuration
   file, request payload, and logs should show the intended settings.
2. Code-path confirmation: logs, counters, traces, or test assertions should
   confirm that the intended path or switch state was reached when practical.
3. Measurement evidence: raw test output, benchmark metrics, profiler reports, or
   exported tool summaries should support the claim.

Additional guidance:

- Do not infer behavior from a single indirect signal when direct evidence is
  available.
- Do not rely on profiler kernel, thread, task, or stage names alone when tools
  can rename, fuse, hide, or internally select implementations.
- For profiler captures, save the profiler command, target selection, output
  files, and summary extraction method.
- Before expensive captures, do a dry run or lower-cost validation to confirm the
  command, workload, and artifact paths.
- Record tool versions, hardware, drivers, runtimes, or other environment details
  when they affect reproducibility.

## Summaries and Claims

- Separate observed facts from interpretation, hypotheses, and unverified
  assumptions.
- Include enough measurement context for readers to understand what was tested.
- Keep failed experiments when the failure is informative; record the failure
  mode, logs, and likely cause.
- Do not delete or overwrite raw evidence when writing a later summary. Add a new
  summary or follow-up note instead.
- If a result depends on a diagnostic-only setting, say so and avoid presenting it
  as production-equivalent evidence unless the baseline used the same setting.
