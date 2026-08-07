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

set -euo pipefail

# --- Configuration ---
UPSTREAM_REPO="vllm-project/vllm"
PROJECT_DIR="$HOME_DIR/vLLM4j"
SETUP_VENV_FILE="$PROJECT_DIR/scripts/setup-venv.sh"
README_FILE="$PROJECT_DIR/README.md"

# --- Get latest stable vLLM version ---
echo "Fetching latest vLLM release from GitHub..."
# Tags are "v0.23.0"; strip the leading v to match VLLM_VERSION.
NEW_VLLM_VERSION=$(gh release list -R "$UPSTREAM_REPO" --exclude-pre-releases --exclude-drafts --limit 1 --json tagName | jq -r '.[0].tagName' | sed 's/^v//')

# --- Get current version from project ---
cd "$PROJECT_DIR"
OLD_VLLM_VERSION=$(grep -oE '^VLLM_VERSION="[0-9]+\.[0-9]+\.[0-9]+"' "$SETUP_VENV_FILE" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')

echo "Current: $OLD_VLLM_VERSION — Latest: $NEW_VLLM_VERSION"
if [[ "$NEW_VLLM_VERSION" == "$OLD_VLLM_VERSION" ]]; then
  echo "Already on the latest vLLM release — nothing to do."
  exit 0
fi

# --- Create a branch for the update ---
branch_name="chore/vllm-$OLD_VLLM_VERSION-to-$NEW_VLLM_VERSION"
echo "Creating branch $branch_name..."
git checkout -b "$branch_name"

# --- Update version in files ---
# Blanket replace: the version appears in the VLLM_VERSION pin, in comments
# documenting version-specific behaviour, and in the README target note.
echo "Updating versions from $OLD_VLLM_VERSION to $NEW_VLLM_VERSION..."
sed -i'' -E "s/$OLD_VLLM_VERSION/$NEW_VLLM_VERSION/g" "$SETUP_VENV_FILE"
sed -i'' -E "s/$OLD_VLLM_VERSION/$NEW_VLLM_VERSION/g" "$README_FILE"

# --- Commit and push changes ---
echo "Committing and pushing changes..."
git add "$SETUP_VENV_FILE" "$README_FILE"

TITLE="feat(deps): update vLLM from $OLD_VLLM_VERSION to $NEW_VLLM_VERSION"
git commit -m "$TITLE"
git push origin "$branch_name"

# --- Create GitHub PR ---
echo "Creating pull request..."
gh pr create --title "$TITLE" \
  --body "This PR updates vLLM from $OLD_VLLM_VERSION to $NEW_VLLM_VERSION.

Release notes: https://github.com/$UPSTREAM_REPO/releases/tag/v$NEW_VLLM_VERSION"
