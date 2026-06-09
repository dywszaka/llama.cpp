# Scripts

- `prepare-small-wikitext.py`: selects a small number of complete Wikitext documents and writes a manifest with source offsets and hashes.
- `run-kld-small.sh`: prepares the dataset by default. Set `RUN_BASELINE=1` to create baseline log-prob files. Set `RUN_KLD=1` to run experiment cases against those files.
- `parse-kld-results.py`: parses `logs/kld_*.raw.log` and counts bounded diagnostic artifacts under `diagnostics/<case>/`.

Common examples:

```bash
tools/kld/scripts/run-kld-small.sh
RUN_BASELINE=1 tools/kld/scripts/run-kld-small.sh
RUN_KLD=1 tools/kld/scripts/run-kld-small.sh
PARSE_ONLY=1 tools/kld/scripts/run-kld-small.sh
```

`CASE_MATRIX` entries are whitespace-separated and use
`case_name:k_cache_type:v_cache_type:ENV_KEY=VALUE`. Multiple environment
assignments are comma-separated. Use `-` when a case has no extra switch.
