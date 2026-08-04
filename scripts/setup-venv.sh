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
VLLM_VERSION="0.23.0"  # minimum version floor; CUDA/CPU pull latest nightly >= this
# See install_common() — newer xgrammar segfaults on import.
XGRAMMAR_VERSION="0.2.2"
TVM_FFI_VERSION="0.1.12"

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

# Always a uv-managed standalone CPython, never a system/Homebrew one.
#
# Homebrew's macOS pythons are *framework* builds, where the interpreter on PATH
# is a shim whose prefix holds no stdlib — so a venv built on one records a
# pyvenv.cfg `home` that PYTHONHOME cannot be derived from naively (CPython then
# aborts with "No module named 'encodings'"). PythonLibLoader handles that
# layout, but pinning to a standalone build keeps CI, local dev and the jextract
# toolchain on one identical interpreter instead of whatever brew happens to
# ship.
export UV_PYTHON_PREFERENCE=only-managed
"$UV_BIN" python install "$PYTHON_VERSION"

# A venv cached from before this change may still be built on a system python,
# so check what it actually points at rather than just that it exists.
if [[ -f "${VENV_DIR}/pyvenv.cfg" ]]; then
  VENV_HOME="$(sed -n 's/^home[[:space:]]*=[[:space:]]*//p' "${VENV_DIR}/pyvenv.cfg" | head -1)"
  if [[ "$VENV_HOME" != *"/uv/python/"* ]]; then
    echo "Existing venv uses a non-managed Python ($VENV_HOME) — recreating."
    rm -rf "$VENV_DIR"
  fi
fi

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
  # ninja, setuptools, and transformers are all bundled by vllm>=0.23.0.
  "$UV_BIN" pip install --python "$VENV_PYTHON" jinja2

  # Pin xgrammar and its tvm-ffi runtime.
  #
  # vLLM only asks for "xgrammar>=0.2.0,<1.0.0", so a fresh resolve picks up
  # whatever is newest. xgrammar 0.2.4 (with apache-tvm-ffi 0.1.13) segfaults in
  # its own static initialiser the moment `import vllm` loads it:
  #   !!!!!!! Segfault encountered !!!!!!!
  #     TVMFFIEnvRegisterCAPI / xgrammar::__TVMFFIStaticInitFunc0()
  # taking the whole process down (exit 139) before any vLLM code runs. It is
  # not JVM-specific — a plain `python -c "from vllm import LLM"` crashes too.
  #
  # 0.2.2/0.1.12 is the last combination verified to import cleanly. Revisit
  # when bumping VLLM_VERSION; xgrammar is only used for guided decoding, which
  # vLLM4j does not currently expose.
  "$UV_BIN" pip install --python "$VENV_PYTHON" \
    "xgrammar==${XGRAMMAR_VERSION}" "apache-tvm-ffi==${TVM_FFI_VERSION}"
}

# Aligns the CUDA build toolchain (nvcc, nvvm/cicc, crt, cccl) to the CUDA
# version torch was built against.
#
# vLLM's CUDA wheel pins the CUDA *runtime* (nvidia-cuda-runtime) to torch's
# CUDA version (e.g. 13.0), but the build-time toolchain wheels are not pinned
# and float to the newest available patch (e.g. nvcc/nvvm 13.2, crt/cccl 13.3).
# flashinfer JIT-compiles its sampling kernels at engine init using this
# toolchain, and a mixed toolchain fails to build:
#   - cccl headers reject a compiler whose version differs from the CTK headers
#     ("CUDA compiler and CUDA toolkit headers are incompatible")
#   - a newer cicc emits PTX the older ptxas can't assemble
#     ("Unsupported .version 9.2; current version is '9.0'")
#   - a newer crt host_runtime.h and older cudafe++ disagree on the device-stub
#     ABI ("'__cudaLaunch' was not declared in this scope")
#
# Pinning the toolchain to torch's CUDA major.minor (==X.Y.*) keeps the
# compiler, assembler and headers coherent so the kernels build.
#
# Only CUDA 13 is affected. NVIDIA publishes these toolchain wheels on PyPI for
# 13.x only — for a CUDA 12 torch (what --torch-backend=auto picks on a pre-580
# driver) they come from torch's own cu12 index as one coherent set, nothing
# floats, and there is nothing to align. Asking for "==12.9.*" there is not a
# no-op but a hard resolve failure ("only nvidia-cuda-nvcc<12.9.dev0 and
# >12.10.dev0 are available"), so bail out before that instead.
align_cuda_toolchain() {
  local cuda_mm
  cuda_mm="$("$VENV_PYTHON" -c 'import torch; print(torch.version.cuda or "")')"
  if [[ -z "$cuda_mm" ]]; then
    echo "Could not determine torch CUDA version — skipping toolchain alignment."
    return 0
  fi

  if [[ "${cuda_mm%%.*}" -lt 13 ]]; then
    echo "torch is a CUDA ${cuda_mm} build — toolchain already coherent, skipping alignment."
    return 0
  fi

  echo "Aligning CUDA build toolchain to ${cuda_mm}.* (matches torch) ..."
  "$UV_BIN" pip install --python "$VENV_PYTHON" \
    "nvidia-cuda-nvcc==${cuda_mm}.*" \
    "nvidia-nvvm==${cuda_mm}.*" \
    "nvidia-cuda-crt==${cuda_mm}.*" \
    "nvidia-cuda-cccl==${cuda_mm}.*"
}

# Builds and installs the vLLM core from the GitHub release tarball.
#
# Used by both `metal` and `cpu`: neither has a usable wheel on PyPI. The
# published `vllm` wheel is CUDA-only — installing it on a machine without a GPU
# leaves `current_platform.device_type` empty and every engine construction dies
# with "Device string must not be empty". Compiling from source with
# VLLM_TARGET_DEVICE=cpu produces a genuine CPU build (`0.23.0+cpu`).
#
# For metal this provides the core that the vllm-metal plugin sits on top of;
# for cpu it is the whole thing.
#
# Needs a C/C++ toolchain (build-essential / Xcode CLT) to compile the CPU
# kernels.
install_vllm_from_source() {
  # importlib reports the *local* version (e.g. "0.23.0+cpu"), so compare only
  # the part before "+" — otherwise this never matches and every run rebuilds.
  if "$VENV_PYTHON" -c "
import importlib.metadata as m, sys
sys.exit(0 if m.version('vllm').split('+')[0] == '${VLLM_VERSION}' else 1)
" &>/dev/null; then
    echo "vllm ${VLLM_VERSION} already installed — skipping vllm core install."
    return 0
  fi

  echo "Building vLLM ${VLLM_VERSION} from source (CPU kernels) ..."

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

  # Build requirements first, then runtime, then build with isolation OFF.
  #
  # The isolation matters: pyproject.toml asks for a bare "torch == 2.11.0",
  # which PEP 517 re-resolves from PyPI into a throwaway environment — and on
  # Linux that is the CUDA build, so find_package(Torch) drags in Caffe2Config
  # and configure dies with:
  #   Your installed Caffe2 version uses CUDA but I cannot find the CUDA
  #   libraries.
  # even though -DVLLM_TARGET_DEVICE=cpu was passed. On macOS the same resolve
  # yields a CPU-only torch (no CUDA build is published for Darwin), which is
  # why this only bites on Linux.
  #
  # requirements/build/cpu.txt is vLLM's own build set: it carries the PyTorch
  # CPU index and platform markers that select torch==2.11.0+cpu on Linux and
  # plain 2.11.0 on Darwin. Installing it into the venv and then building with
  # --no-build-isolation makes the build use that torch instead of re-resolving.
  "$UV_BIN" pip install --python "$VENV_PYTHON" \
    -r "${VLLM_SRC}/requirements/build/cpu.txt" \
    --index-strategy unsafe-best-match

  "$UV_BIN" pip install --python "$VENV_PYTHON" \
    -r "${VLLM_SRC}/requirements/cpu.txt" \
    --index-strategy unsafe-best-match

  # Explicit rather than inferred: setup.py defaults to a CUDA build on Linux,
  # which is exactly what must not happen here.
  VLLM_TARGET_DEVICE=cpu "$UV_BIN" pip install --python "$VENV_PYTHON" \
    --no-build-isolation "$VLLM_SRC"
}

case "$BACKEND" in

  metal)
    # vllm-metal is a plugin over the CPU core built above; it is always
    # reinstalled because it is version-pinned separately and a stale copy that
    # expects a newer vllm API fails at runtime with:
    #   AttributeError: 'SchedulerConfig' object has no attribute
    #   'max_num_scheduled_tokens'
    install_vllm_from_source

    # Install prebuilt vllm-metal wheel from GitHub release (includes Metal kernels compiled and ready to use)
    echo "Installing vllm-metal (prebuilt wheel) ..."
    "$UV_BIN" pip install --python "$VENV_PYTHON" \
      "https://github.com/vllm-project/vllm-metal/releases/download/v0.3.0.dev20260616093506/vllm_metal-0.3.0.dev20260616093506-cp312-cp312-macosx_11_0_arm64.whl"

    install_common
    ;;

  cuda)
    {
      # Pick the wheel that matches the *driver*, not the newest build.
      #
      # The wheel on PyPI is a CUDA 13 build — vllm._C links libcudart.so.13 —
      # and --torch-backend=auto pairs it with whatever torch the driver allows.
      # On a pre-580 driver those two disagree and the engine dies at import:
      #   ImportError: libcudart.so.13: cannot open shared object file
      # Forcing torch to cu130 instead only moves the failure one step later:
      #   RuntimeError: The NVIDIA driver on your system is too old (found
      #   version 12080)
      # because CUDA 13 needs r580+, which no 12.x driver satisfies.
      #
      # The release page carries a +cu129 wheel alongside the default one, and
      # that whole stack (vllm._C, torch, the nvidia-*-cu12 runtime) runs on any
      # 12.x driver. Choose between them on the driver version, and pin
      # --torch-backend to match so uv cannot resolve torch into the other major.
      # A build-only box has no nvidia-smi at all, and under `set -e` a missing
      # command inside a command substitution takes the whole script down with
      # 127 before a single package is installed — with stderr swallowed by the
      # redirect, so the log just stops after "Creating virtual environment".
      # Absent driver means "cannot tell": fall through to the default wheel,
      # which is what a machine that only compiles wants anyway.
      if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
          | head -1 | cut -d. -f1 || true)"
      else
        DRIVER_MAJOR=""
      fi

      if [[ -n "$DRIVER_MAJOR" && "$DRIVER_MAJOR" -lt 580 ]]; then
        echo "Driver ${DRIVER_MAJOR}.x predates CUDA 13 (needs r580+) — using the +cu129 wheel."
        VLLM_PACKAGE="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_$(uname -m).whl"
        TORCH_BACKEND="cu129"
        WANT_CUDA_MAJOR="12"
      else
        VLLM_PACKAGE="vllm==${VLLM_VERSION}"
        TORCH_BACKEND="auto"
        WANT_CUDA_MAJOR="13"
      fi
    }

    # "Already installed" is not the same as "usable". A venv built before this
    # check existed — or on a machine with a different driver — can hold a CUDA
    # 13 vllm beside a cu129 torch, a pairing that only fails at model load:
    #   ImportError: libcudart.so.13: cannot open shared object file
    # Compare what is actually installed against what the driver needs and
    # reinstall over it, so re-running this script repairs a venv instead of
    # declaring it fine.
    CURRENT_CUDA_MAJOR="$("$VENV_PYTHON" -c \
      'import torch; print((torch.version.cuda or "").split(".")[0])' 2>/dev/null || true)"

    # Probe vllm._C, not vllm. The mismatch this repairs lives in the compiled
    # extension, and `import vllm` sails straight past it:
    #   >>> import vllm            # fine
    #   >>> import vllm._C         # ImportError: libcudart.so.13
    # The engine only trips over it at model load, which is exactly the late,
    # opaque failure this check exists to prevent.
    if "$VENV_PYTHON" -c "import vllm._C" &>/dev/null &&
       [[ "$CURRENT_CUDA_MAJOR" == "$WANT_CUDA_MAJOR" ]]; then
      echo "vllm already installed and built for CUDA ${CURRENT_CUDA_MAJOR} — skipping."
    else
      if "$VENV_PYTHON" -c "import importlib.metadata as m; m.version('vllm')" &>/dev/null; then
        # Two different breakages, and the torch version alone cannot tell them
        # apart: a cu13 vllm wheel next to a cu12 torch has the *right* torch and
        # still fails to load its kernels.
        if "$VENV_PYTHON" -c "import vllm._C" &>/dev/null; then
          echo "Repairing venv: torch targets CUDA ${CURRENT_CUDA_MAJOR:-unknown}," \
               "this driver needs CUDA ${WANT_CUDA_MAJOR}."
        else
          echo "Repairing venv: vllm's compiled extension does not load — its CUDA runtime" \
               "does not match torch (CUDA ${CURRENT_CUDA_MAJOR:-unknown})."
        fi
      fi

      echo "Installing vLLM ${VLLM_VERSION} (CUDA wheel, torch backend ${TORCH_BACKEND}) ..."

      # --reinstall-package is a no-op on a fresh venv and the whole point on a
      # stale one: without it uv keeps the satisfied-but-wrong versions.
      "$UV_BIN" pip install --python "$VENV_PYTHON" \
        --reinstall-package vllm --reinstall-package torch \
        "$VLLM_PACKAGE" --torch-backend="$TORCH_BACKEND"
    fi

    install_common
    align_cuda_toolchain
    ;;

  cpu)
    install_vllm_from_source

    install_common
    ;;

  *)
    echo "ERROR: Unknown backend '$BACKEND'. Supported: metal, cuda, cpu" >&2
    exit 1
    ;;
esac

echo "venv setup complete for backend '$BACKEND' at $VENV_DIR"
