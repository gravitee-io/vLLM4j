#!/usr/bin/env bash
#
# Copyright © 2015 The Gravitee team (http://gravitee.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Sets up the uv virtual environment and installs vLLM with the appropriate
# backend-specific packages. This script is invoked from the backend Maven
# profile during the 'initialize' phase.
#
# Usage:
#   ./setup-venv.sh -d <project_dir> -v <python_version> -b <backend>
#
#   -d  project root directory (where .venv will be created)
#   -v  Python version (e.g. 3.12)
#   -b  backend: metal | cuda | cpu
#

set -euo pipefail

PROJECT_DIR="${PROJECT_BASEDIR:-.}"
PYTHON_VERSION="3.12"
BACKEND=""
VLLM_VERSION="0.19.0"  # minimum version floor; CUDA/CPU pull latest nightly >= this

print_usage() {
  echo "Usage: $0 -d <project_dir> -v <python_version> -b <backend>"
  echo "  backend: metal | cuda | cpu"
}

while getopts ":d:v:b:h" opt; do
  case ${opt} in
    d) PROJECT_DIR=$OPTARG ;;
    v) PYTHON_VERSION=$OPTARG ;;
    b) BACKEND=$OPTARG ;;
    h) print_usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; print_usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; print_usage; exit 1 ;;
  esac
done

if [[ -z "$BACKEND" ]]; then
  echo "Missing required argument: -b <backend>" >&2
  print_usage
  exit 1
fi

VENV_DIR="${PROJECT_DIR}/.venv"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Ensure uv is installed
# ═══════════════════════════════════════════════════════════════════════════════

if ! command -v uv &>/dev/null; then
  echo "uv not found — installing via official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

UV_BIN="$(command -v uv)"
echo "Using uv: $UV_BIN  ($(uv --version))"

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Create venv
# ═══════════════════════════════════════════════════════════════════════════════

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR (Python $PYTHON_VERSION) ..."
  "$UV_BIN" venv "$VENV_DIR" --python "$PYTHON_VERSION"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Install packages based on backend
# ═══════════════════════════════════════════════════════════════════════════════

install_common() {
  # jinja2 is the only dependency vLLM does not pull in transitively.
  # ninja, setuptools, and transformers are all bundled by vllm>=0.16.0.
  "$UV_BIN" pip install --python "$VENV_PYTHON" jinja2
}

case "$BACKEND" in

  metal)
    # Check if the correct vllm version is already installed.
    # We do NOT skip vllm-metal here: it is installed from git@main, so there
    # is no pinned version to compare against.  Always reinstalling it is the
    # only reliable way to avoid a stale, incompatible build being reused from
    # a previous CI run — the symptom being:
    #   AttributeError: 'SchedulerConfig' object has no attribute
    #   'max_num_scheduled_tokens'
    # which occurs when the cached vllm-metal expects a newer vllm API than
    # the installed vllm core.
    VLLM_OK=false
    if "$VENV_PYTHON" -c "
import vllm, importlib.metadata as m
assert m.version('vllm') == '${VLLM_VERSION}', f'wrong vllm {m.version(\"vllm\")}'
" &>/dev/null; then
      VLLM_OK=true
    fi

    if [[ "$VLLM_OK" == "false" ]]; then
      echo "Installing vLLM ${VLLM_VERSION} (Apple Silicon Metal/MLX) ..."

      # vLLM is not on PyPI for metal — install from the GitHub release tarball,
      # same as the official vllm-metal install.sh does.
      VLLM_TARBALL="/tmp/vllm-${VLLM_VERSION}.tar.gz"
      if [[ ! -f "$VLLM_TARBALL" ]]; then
        echo "  Downloading vLLM ${VLLM_VERSION} tarball..."
        curl -fsSL "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}.tar.gz" \
          -o "$VLLM_TARBALL"
      fi

      # Always extract fresh — a prior failed build leaves stale CMake cache
      # (baked ninja paths from uv's temp build-isolation dir) that causes
      # subsequent builds to fail. Fresh extraction is cheap (~1s) and ensures
      # idempotent builds.
      VLLM_SRC="/tmp/vllm-${VLLM_VERSION}"
      rm -rf "$VLLM_SRC"
      tar xf "$VLLM_TARBALL" -C /tmp

      # Patch chained comparisons in CPU attention headers that newer Clang
      # (Apple CLT ≥ 21.0 / macOS 26) rejects with -Werror=parentheses.
      # See: static_assert(0 < M <= 8) → static_assert(0 < M && M <= 8)
      sed -i '' 's/static_assert(0 < M <= 8)/static_assert(0 < M \&\& M <= 8)/' \
        "$VLLM_SRC/csrc/cpu/cpu_attn_vec.hpp"
      sed -i '' 's/static_assert(0 < M <= 16)/static_assert(0 < M \&\& M <= 16)/' \
        "$VLLM_SRC/csrc/cpu/cpu_attn_vec16.hpp"

      "$UV_BIN" pip install --python "$VENV_PYTHON" \
        -r "${VLLM_SRC}/requirements/cpu.txt" \
        --index-strategy unsafe-best-match

      "$UV_BIN" pip install --python "$VENV_PYTHON" "$VLLM_SRC"
    else
      echo "vllm ${VLLM_VERSION} already installed — skipping vllm core install."
    fi

    # vllm-metal is always reinstalled: it tracks git@main with no pinned
    # version, so any cached copy may be out of sync with the installed vllm
    # core. The install is fast (~10 s) and prevents hard-to-debug mismatches.
    echo "Installing vllm-metal (always fresh from git@main) ..."
    "$UV_BIN" pip install --python "$VENV_PYTHON" \
      "vllm-metal @ git+https://github.com/vllm-project/vllm-metal.git@main"

    install_common
    ;;

  cuda)
    if "$VENV_PYTHON" -c "import vllm" &>/dev/null; then
      echo "vllm already installed — skipping."
    else
      echo "Installing vLLM ${VLLM_VERSION} (CUDA wheel) ..."

      "$UV_BIN" pip install --python "$VENV_PYTHON" \
        "vllm==${VLLM_VERSION}" --torch-backend=auto
    fi

    install_common
    ;;

  cpu)
    if "$VENV_PYTHON" -c "import vllm" &>/dev/null; then
      echo "vllm already installed — skipping."
    else
      echo "Installing vLLM ${VLLM_VERSION} (CPU-only wheel) ..."

      "$UV_BIN" pip install --python "$VENV_PYTHON" \
        "vllm==${VLLM_VERSION}" --torch-backend cpu
    fi

    install_common
    ;;

  *)
    echo "ERROR: Unknown backend '$BACKEND'. Supported: metal, cuda, cpu" >&2
    exit 1
    ;;
esac

echo "venv setup complete for backend '$BACKEND' at $VENV_DIR"
