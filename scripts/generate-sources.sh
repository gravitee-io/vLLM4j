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
# Resolves the Python.h include path, libpython path, and PYTHONHOME, then
# runs jextract to generate Java FFM bindings for CPython.
#
# This script assumes the .venv already exists (created by setup-venv.sh).
# It is invoked from the OS/arch Maven profile during 'generate-sources'.
#
# Usage:
#   ./generate-sources.sh \
#     -d <project_dir>     \
#     -v <python_version>  \
#     -j <jextract_bin>    \
#     -o <output_dir>      \
#     -p <cpython_package> \
#     -c <cpython_class>   \
#     -l <libpython_name>
#

set -euo pipefail

PROJECT_DIR="${PROJECT_BASEDIR:-.}"
PYTHON_VERSION="3.12"
JEXTRACT_BIN=""
OUTPUT_DIR=""
CPYTHON_PACKAGE="io.gravitee.vllm"
CPYTHON_CLASS="CPython"
LIBPYTHON_NAME="python3.12"
JEXTRACT_OS=""
JEXTRACT_PLATFORM=""

print_usage() {
  echo "Usage: $0 -d <project_dir> -v <python_version> -j <jextract_bin> -o <output_dir> [-p <package>] [-c <class>] [-l <libpython>] [-O <os>] [-P <platform>]"
}

while getopts ":d:v:j:o:p:c:l:O:P:h" opt; do
  case ${opt} in
    d) PROJECT_DIR=$OPTARG ;;
    v) PYTHON_VERSION=$OPTARG ;;
    j) JEXTRACT_BIN=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    p) CPYTHON_PACKAGE=$OPTARG ;;
    c) CPYTHON_CLASS=$OPTARG ;;
    l) LIBPYTHON_NAME=$OPTARG ;;
    O) JEXTRACT_OS=$OPTARG ;;
    P) JEXTRACT_PLATFORM=$OPTARG ;;
    h) print_usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; print_usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; print_usage; exit 1 ;;
  esac
done

if [[ -z "$JEXTRACT_BIN" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required arguments (-j and -o)." >&2
  print_usage
  exit 1
fi

# Build the platform-specific target package for jextract.
# e.g. io.gravitee.vllm.macosx.aarch64 or io.gravitee.vllm.linux.x86_64
if [[ -n "$JEXTRACT_OS" && -n "$JEXTRACT_PLATFORM" ]]; then
  CPYTHON_PACKAGE="${CPYTHON_PACKAGE}.${JEXTRACT_OS}.${JEXTRACT_PLATFORM}"
  echo "Platform-specific package: ${CPYTHON_PACKAGE}"
fi

VENV_DIR="${PROJECT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -f "$VENV_PYTHON" ]]; then
  echo "ERROR: .venv not found at $VENV_DIR. Run setup-venv.sh first." >&2
  exit 1
fi

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Resolve Python.h location and libpython directory via sysconfig
# ═════════════════════════════════════════════════════════════════════════════

# Ask the venv's own interpreter where its headers are.
# This correctly follows the base interpreter (framework, Homebrew, apt, etc.)
# rather than looking inside the venv directory itself.
PYTHON_INCLUDE_DIR="$("$VENV_PYTHON" -c "import sysconfig; print(sysconfig.get_path('include'))")"

if [[ ! -f "${PYTHON_INCLUDE_DIR}/Python.h" ]]; then
  echo "ERROR: Python.h not found at ${PYTHON_INCLUDE_DIR}/Python.h" >&2
  echo "" >&2
  echo "  macOS:  ensure python${PYTHON_VERSION} was installed via python.org or Homebrew." >&2
  echo "  Linux:  sudo apt install python${PYTHON_VERSION}-dev   (Debian/Ubuntu)" >&2
  echo "          sudo dnf install python${PYTHON_VERSION}-devel (Fedora/RHEL)" >&2
  exit 1
fi

echo "Python.h resolved at: ${PYTHON_INCLUDE_DIR}/Python.h"

# Resolve the libpython directory — the directory containing libpython3.x.dylib/.so
LIBPYTHON_ABS="$("$VENV_PYTHON" -c "
import sysconfig, os, glob
libdir = sysconfig.get_config_var('LIBDIR') or ''
# On macOS framework builds, try the dylib symlink first (resolves to the real binary)
for pat in ['libpython*.dylib', 'libpython*.so', 'libpython*.so.*']:
    matches = glob.glob(os.path.join(libdir, pat))
    if matches:
        print(os.path.realpath(matches[0]))
        break
else:
    # Fallback: framework binary
    ldlib = sysconfig.get_config_var('LDLIBRARY') or ''
    if ldlib and not os.path.isabs(ldlib):
        ldlib = os.path.join(libdir, '..', ldlib)
    if ldlib and os.path.exists(ldlib):
        print(os.path.realpath(ldlib))
    else:
        print('')
")"

if [[ -z "$LIBPYTHON_ABS" || ! -f "$LIBPYTHON_ABS" ]]; then
  echo "ERROR: Could not resolve absolute path to libpython." >&2
  echo "  LIBDIR=$(${VENV_PYTHON} -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" >&2
  exit 1
fi

echo "libpython resolved at: ${LIBPYTHON_ABS}"
echo "${LIBPYTHON_ABS}" > "${PROJECT_DIR}/.libpython-path"
LIBPYTHON_DIR="$(dirname "$LIBPYTHON_ABS")"
echo "${LIBPYTHON_DIR}" > "${PROJECT_DIR}/.libpython-dir"
echo "libpython.path=${LIBPYTHON_ABS}" > "${PROJECT_DIR}/python.properties"

# Resolve the base Python prefix (sys.base_prefix) — this is what PYTHONHOME must be
# set to. A uv venv only contains site-packages; the stdlib lives in the base install.
PYTHON_HOME="$("$VENV_PYTHON" -c "import sys; print(sys.base_prefix)")"
echo "Python base prefix (PYTHONHOME): ${PYTHON_HOME}"
echo "${PYTHON_HOME}" > "${PROJECT_DIR}/.python-home"

# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Run jextract
# ═════════════════════════════════════════════════════════════════════════════

echo "Running jextract..."
echo "  header : ${PYTHON_INCLUDE_DIR}/Python.h"
echo "  output : ${OUTPUT_DIR}"
echo "  package: ${CPYTHON_PACKAGE}"
echo "  class  : ${CPYTHON_CLASS}"
echo "  library: ${LIBPYTHON_ABS} (absolute path)"

mkdir -p "$OUTPUT_DIR"

"$JEXTRACT_BIN" \
  -t "$CPYTHON_PACKAGE" \
  --header-class-name "$CPYTHON_CLASS" \
  --include-dir "$PYTHON_INCLUDE_DIR" \
  --output "$OUTPUT_DIR" \
  --library ":${LIBPYTHON_ABS}" \
  --use-system-load-library \
  --include-function Py_InitializeEx \
  --include-function Py_FinalizeEx \
  --include-function PyGILState_Ensure \
  --include-function PyGILState_Release \
  --include-function PyEval_SaveThread \
  --include-function PyEval_RestoreThread \
  --include-function PyErr_Occurred \
  --include-function PyErr_Fetch \
  --include-function PyErr_Clear \
  --include-function PyErr_Print \
  --include-function PyErr_NormalizeException \
  --include-function PyImport_ImportModule \
  --include-function PyObject_GetAttrString \
  --include-function PyObject_SetAttrString \
  --include-function PyObject_CallObject \
  --include-function PyObject_CallMethod \
  --include-function PyObject_CallMethodObjArgs \
  --include-function PyObject_IsTrue \
  --include-function PyObject_Str \
  --include-function PyObject_Repr \
  --include-function PyObject_Length \
  --include-function PyObject_GetItem \
  --include-function PyTuple_New \
  --include-function PyTuple_SetItem \
  --include-function PyTuple_GetItem \
  --include-function PyTuple_Size \
  --include-function PyList_New \
  --include-function PyList_Size \
  --include-function PyList_GetItem \
  --include-function PyList_Append \
  --include-function PyDict_New \
  --include-function PyDict_SetItemString \
  --include-function PyDict_GetItemString \
  --include-function PyUnicode_AsUTF8 \
  --include-function PyUnicode_FromString \
  --include-function PyUnicode_DecodeFSDefault \
  --include-function PyLong_AsLong \
  --include-function PyLong_FromLong \
  --include-function PyFloat_AsDouble \
  --include-function PyFloat_FromDouble \
  --include-function PyBool_FromLong \
  --include-function PySequence_Size \
  --include-function PySequence_GetItem \
  --include-function PyIter_Next \
  --include-function PyObject_GetIter \
  --include-function PyCallable_Check \
  --include-function Py_IncRef \
  --include-function Py_DecRef \
  --include-function Py_IsNone \
  --include-function Py_IsTrue \
  --include-function Py_IsFalse \
  --include-struct PyStatus \
  --include-struct PyWideStringList \
  "${PYTHON_INCLUDE_DIR}/Python.h"

echo "jextract completed. Generated sources at: ${OUTPUT_DIR}"
