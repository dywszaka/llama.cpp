#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT_DIR}/c100-sim-scripts/run-ppl-one-chunk.sh"
DOCKERFILE="${ROOT_DIR}/c100-sim-scripts/c100-ppl-runtime.Dockerfile"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

CAPTURE="${TMP_DIR}/docker-args.txt"
FAKE_BIN_DIR="${TMP_DIR}/bin"
mkdir -p "${FAKE_BIN_DIR}"

cat > "${FAKE_BIN_DIR}/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "${CAPTURE}"
EOF
chmod +x "${FAKE_BIN_DIR}/docker"

CAPTURE="${CAPTURE}" \
PATH="${FAKE_BIN_DIR}:$PATH" \
"${SCRIPT}" baseline

grep -Fxq 'local/llama.cpp:c100-ppl-runtime' "${CAPTURE}"

test -f "${DOCKERFILE}"
grep -Eq '^FROM nvidia/cuda:13\.0\.0-runtime-ubuntu24\.04$' "${DOCKERFILE}"
grep -Eq '\blibcurl4\b' "${DOCKERFILE}"
grep -Eq '\blibgomp1\b' "${DOCKERFILE}"
grep -Eq '\bdevice-tree-compiler\b' "${DOCKERFILE}"
