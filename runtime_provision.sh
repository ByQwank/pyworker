#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE:-/workspace}"
PROVISIONING_DONE_MARKER="${PROVISIONING_DONE_MARKER:-${WORKSPACE_ROOT}/.provisioning-complete}"
PROVISIONING_FAILED_MARKER="${PROVISIONING_FAILED_MARKER:-${WORKSPACE_ROOT}/.provisioning-failed}"
MANIFEST_PATH="${WORKFLOW_IMAGE_MANIFEST_PATH:-/opt/workflow-images/workflow-1/manifest.json}"
BUILD_METADATA_PATH="${WORKFLOW_IMAGE_BUILD_METADATA_PATH:-/opt/workflow-images/workflow-1/state/build-metadata.json}"
BOOTSTRAP_MANIFEST_B64="${PYWORKER_BOOTSTRAP_MANIFEST_B64:-}"

trap 'touch "$PROVISIONING_FAILED_MARKER"' ERR

rm -f "$PROVISIONING_DONE_MARKER" "$PROVISIONING_FAILED_MARKER"

python3 - "$MANIFEST_PATH" "$BUILD_METADATA_PATH" "$BOOTSTRAP_MANIFEST_B64" <<'PY'
import base64
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
build_metadata_path = Path(sys.argv[2])
bootstrap_b64 = sys.argv[3].strip()

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

if build_metadata_path.exists():
    json.loads(build_metadata_path.read_text(encoding="utf-8"))

expected_workflow = manifest.get("workflowProfileSlug")
expected_dependency = manifest.get("dependencyProfileSlug")

if bootstrap_b64:
    padding = "=" * (-len(bootstrap_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(f"{bootstrap_b64}{padding}").decode("utf-8"))
    workflow = payload.get("workflowProfileSlug")
    dependency = payload.get("dependencyProfileSlug")

    if workflow and expected_workflow and workflow != expected_workflow:
        raise SystemExit(
            f"Workflow image mismatch. expected '{expected_workflow}', got '{workflow}'."
        )

    if dependency and expected_dependency and dependency != expected_dependency:
        raise SystemExit(
            f"Dependency image mismatch. expected '{expected_dependency}', got '{dependency}'."
        )

missing = []
for asset in manifest.get("assetDownloads", []):
    target = Path(asset["targetPath"])
    if not target.exists():
        missing.append(str(target))

if missing:
    raise SystemExit("Missing baked workflow assets:\n" + "\n".join(missing))
PY

touch "$PROVISIONING_DONE_MARKER"
echo "Baked workflow image verified"
