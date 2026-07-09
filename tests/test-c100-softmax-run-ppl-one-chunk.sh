#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT_DIR}/c100-sim-scripts/run-ppl-one-chunk.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

FAKE_ROOT="${TMP_DIR}/root"
FAKE_BIN_DIR="${FAKE_ROOT}/build-cuda-c100/bin"
FAKE_TOOLS_DIR="${TMP_DIR}/tools"
OUTPUT_LOG="${TMP_DIR}/stdout.log"

mkdir -p "${FAKE_BIN_DIR}" "${FAKE_TOOLS_DIR}"

cat > "${FAKE_BIN_DIR}/llama-perplexity" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "LLAMA_C100_REGISTER_DEVICE=${LLAMA_C100_REGISTER_DEVICE:-}"
echo "ARGS=$*"
echo "Final estimate: PPL = 1.0"
EOF
chmod +x "${FAKE_BIN_DIR}/llama-perplexity"

cat > "${FAKE_TOOLS_DIR}/ldconfig" <<'EOF'
#!/usr/bin/env bash
cat <<'OUT'
	libcurl.so.4 (libc6,x86-64) => /usr/lib/libcurl.so.4
	libgomp.so.1 (libc6,x86-64) => /usr/lib/libgomp.so.1
OUT
EOF
chmod +x "${FAKE_TOOLS_DIR}/ldconfig"

cat > "${FAKE_TOOLS_DIR}/dtc" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "${FAKE_TOOLS_DIR}/dtc"

PATH="${FAKE_TOOLS_DIR}:$PATH" \
LLAMA_IN_DOCKER=1 \
CONTAINER_ROOT="${FAKE_ROOT}" \
"${SCRIPT}" c100-softmax > "${OUTPUT_LOG}" 2>&1

grep -q '^LLAMA_C100_REGISTER_DEVICE=1$' "${OUTPUT_LOG}"
grep -q 'ARGS=--device CUDA0,C100 --tensor-split 1,0' "${OUTPUT_LOG}"
