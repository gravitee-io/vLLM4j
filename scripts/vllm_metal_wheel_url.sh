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
# Prints the download URL of the newest vllm-metal wheel for a vLLM version,
# with GitHub's sha256 for it appended as #sha256=<hex>.
#
# vllm-metal tracks vLLM's version numbers. Until the stable vX.Y.Z tag ships
# it only publishes vX.Y.Z.dev<timestamp> pre-releases, and it keeps just the
# latest one — older dev releases are deleted, so a pinned dev URL stops
# resolving within a day. This picks, in order:
#   1. the stable vX.Y.Z wheel, if released
#   2. otherwise the newest vX.Y.Z.dev* / vX.Y.Z.rc* pre-release wheel
#
# Usage:
#   ./vllm_metal_wheel_url.sh -v <vllm_version> [-p <python_version>]
#
#   -v  vLLM version (e.g. 0.31.0)
#   -p  Python version the wheel is built for (default: 3.12)
#
# The URL goes to stdout, diagnostics to stderr. The releases are public, so
# the API is queried anonymously — no token is ever sent.
#
# uv only verifies a direct-URL wheel when the URL carries a hash: unlike an
# index, a GitHub download gives it nothing to compare against. With the
# #sha256 fragment, uv refuses a wheel whose bytes differ from the digest
# GitHub recorded at upload. A wheel without a digest is refused here rather
# than installed unchecked.
#

set -euo pipefail

REPO="vllm-project/vllm-metal"
VERSION=""
PYTHON_VERSION="3.12"

print_usage() {
  echo "Usage: $0 -v <vllm_version> [-p <python_version>]" >&2
}

while getopts ":v:p:h" opt; do
  case ${opt} in
    v) VERSION=$OPTARG ;;
    p) PYTHON_VERSION=$OPTARG ;;
    h) print_usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; print_usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; print_usage; exit 1 ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "Missing required argument: -v <vllm_version>" >&2
  print_usage
  exit 1
fi

PY_TAG="cp${PYTHON_VERSION//./}"

# Plain grep over the JSON rather than jq: this runs from setup-venv.sh on CI
# images that only guarantee curl. browser_download_url is the one field that
# carries both the release tag and the wheel filename; each asset lists its
# digest before it, so the awk pairs every URL with the digest just read and
# emits "<url>#sha256=<hex>" (or the bare URL when GitHub has no digest).
URLS="$(curl -fsSL -H "Accept: application/vnd.github+json" "https://api.github.com/repos/${REPO}/releases?per_page=100" \
  | grep -oE '"digest"[[:space:]]*:[[:space:]]*(null|"sha256:[0-9a-f]{64}")|"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]+"' \
  | awk '
      /^"digest"/ { digest = ""; if (match($0, /sha256:[0-9a-f]+/)) digest = substr($0, RSTART + 7, RLENGTH - 7); next }
      {
        url = $0; sub(/^"browser_download_url"[[:space:]]*:[[:space:]]*"/, "", url); sub(/"$/, "", url)
        print (digest == "" ? url : url "#sha256=" digest); digest = ""
      }' \
  | grep -E '\.whl(#|$)' \
  | grep -F -- "-${PY_TAG}-${PY_TAG}-" || true)"

# Escape the dots so a version like 1.2.3 cannot match 1x2y3.
VERSION_RE="${VERSION//./\\.}"

require_digest() {
  if [[ "$1" != *"#sha256="* ]]; then
    echo "ERROR: GitHub has no sha256 digest for ${1} — refusing to install it unverified." >&2
    exit 1
  fi
}

STABLE="$(grep -E "/download/v${VERSION_RE}/" <<<"$URLS" | head -1 || true)"
if [[ -n "$STABLE" ]]; then
  require_digest "$STABLE"
  echo "Found stable vllm-metal v${VERSION}." >&2
  echo "$STABLE"
  exit 0
fi

# dev tags are UTC timestamps, so a version sort on the tag puts the newest
# last.
PRE="$(grep -E "/download/v${VERSION_RE}\.?(dev|rc)[0-9]+/" <<<"$URLS" \
  | awk -F/ '{ print $(NF-1) " " $0 }' \
  | sort -V -k1,1 | tail -1 | cut -d' ' -f2- || true)"
if [[ -n "$PRE" ]]; then
  require_digest "$PRE"
  echo "No stable vllm-metal v${VERSION} yet — using pre-release $(basename "$(dirname "$PRE")")." >&2
  echo "$PRE"
  exit 0
fi

echo "ERROR: no vllm-metal wheel for v${VERSION} (${PY_TAG}) in ${REPO} releases." >&2
exit 1
