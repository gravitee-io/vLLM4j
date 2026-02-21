#!/usr/bin/env bash
#
# Downloads the jextract early access binary from OpenJDK.
# Skips download if the binary already exists (idempotent).
#
# Usage: ./download-jextract.sh -o <os> -p <platform> [-d <jextract_dir>]
#   -o  OS:       macosx | linux
#   -p  platform: aarch64 | x86_64
#   -d  target directory (default: ${PROJECT_BASEDIR}/.jextract)
#

set -euo pipefail

OS=""
PLATFORM=""
JEXTRACT_DIR="${PROJECT_BASEDIR:-.}/.jextract"

print_usage() {
  echo "Usage: $0 -o <os> -p <platform> [-d <jextract_dir>]"
  echo "Example: $0 -o macosx -p aarch64"
  echo "         $0 -o linux  -p x86_64 -d /path/to/jextract"
}

while getopts ":o:p:d:h" opt; do
  case ${opt} in
    o) OS=$OPTARG ;;
    p) PLATFORM=$OPTARG ;;
    d) JEXTRACT_DIR=$OPTARG ;;
    h) print_usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; print_usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; print_usage; exit 1 ;;
  esac
done

if [[ -z "$OS" || -z "$PLATFORM" ]]; then
  echo "Missing required arguments." >&2
  print_usage
  exit 1
fi

# ── URL table ──────────────────────────────────────────────────────────────────
# jextract EA build 25+2-4 from https://jdk.java.net/jextract/
JEXTRACT_URL=""
case "$OS/$PLATFORM" in
  macosx/aarch64)
    JEXTRACT_URL="https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz" ;;
  macosx/x86_64)
    JEXTRACT_URL="https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_macos-x64_bin.tar.gz" ;;
  linux/x86_64)
    JEXTRACT_URL="https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_linux-x64_bin.tar.gz" ;;
  *)
    echo "Unsupported OS/platform: $OS/$PLATFORM" >&2; exit 1 ;;
esac

JEXTRACT_BIN="${JEXTRACT_DIR}/bin/jextract"
if [[ -f "$JEXTRACT_BIN" && -x "$JEXTRACT_BIN" ]]; then
  echo "jextract already present at $JEXTRACT_BIN — skipping download."
  exit 0
fi

mkdir -p "$JEXTRACT_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading jextract for $OS/$PLATFORM ..."
echo "  $JEXTRACT_URL"
curl -fsSL -o "$TMP_DIR/jextract.tar.gz" "$JEXTRACT_URL"

echo "Extracting to $JEXTRACT_DIR ..."
tar -xzf "$TMP_DIR/jextract.tar.gz" -C "$JEXTRACT_DIR" --strip-components=1

if [[ ! -f "$JEXTRACT_BIN" ]]; then
  echo "ERROR: jextract binary not found at $JEXTRACT_BIN after extraction." >&2
  exit 1
fi

chmod +x "$JEXTRACT_BIN"
rm -f "$JEXTRACT_DIR"/bin/jextract.bat "$JEXTRACT_DIR"/bin/jextract.ps1
find "$JEXTRACT_DIR" -type d -empty -delete

echo "jextract ready at $JEXTRACT_BIN  ($(${JEXTRACT_BIN} --version 2>&1 | head -1))"
