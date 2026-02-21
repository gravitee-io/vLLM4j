#!/usr/bin/env bash
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
VLLM_VERSION="0.16.0"  # minimum version floor; CUDA/CPU pull latest nightly >= this

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
    if "$VENV_PYTHON" -c "import vllm; import vllm_metal" &>/dev/null; then
      echo "vllm + vllm-metal already installed — skipping."
    else
      echo "Installing vLLM ${VLLM_VERSION} + vllm-metal (Apple Silicon Metal/MLX) ..."

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

      "$UV_BIN" pip install --python "$VENV_PYTHON" \
        -r "${VLLM_SRC}/requirements/cpu.txt" \
        --index-strategy unsafe-best-match

      "$UV_BIN" pip install --python "$VENV_PYTHON" "$VLLM_SRC"

      # vllm-metal: install from git main (the PyPI release has a stale import path)
      "$UV_BIN" pip install --python "$VENV_PYTHON" \
        "vllm-metal @ git+https://github.com/vllm-project/vllm-metal.git@main"
    fi

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
