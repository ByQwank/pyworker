#!/bin/bash

set -e -o pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

SERVER_DIR="$WORKSPACE_DIR/vast-pyworker"
ENV_PATH="$WORKSPACE_DIR/worker-env"
DEBUG_LOG="$WORKSPACE_DIR/debug.log"
PYWORKER_LOG="$WORKSPACE_DIR/pyworker.log"
STATUS_FILE="${PYWORKER_STATUS_FILE:-$WORKSPACE_DIR/pyworker-status.json}"
PROVISIONING_DONE_MARKER="${PROVISIONING_DONE_MARKER:-$WORKSPACE_DIR/.provisioning-complete}"
PROVISIONING_FAILED_MARKER="${PROVISIONING_FAILED_MARKER:-$WORKSPACE_DIR/.provisioning-failed}"
PROVISIONING_WAIT_TIMEOUT_SECONDS="${PROVISIONING_WAIT_TIMEOUT_SECONDS:-2700}"
PROVISIONING_WAIT_INTERVAL_SECONDS="${PROVISIONING_WAIT_INTERVAL_SECONDS:-5}"

REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
USE_SSL="${USE_SSL:-true}"
WORKER_PORT="${WORKER_PORT:-3000}"
READY_ROUTE="${PYWORKER_READY_ROUTE:-/readyz}"
STATUS_ROUTE="${PYWORKER_STATUS_ROUTE:-/statusz}"
BOOTSTRAP_STATUS_PID=""
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

exec &> >(tee -a "$DEBUG_LOG")

function echo_var(){
    echo "$1: ${!1}"
}

function start_bootstrap_status_server(){
    if ! command -v python3 >/dev/null 2>&1; then
        echo "WARNING: python3 not found, bootstrap status server disabled"
        return 0
    fi

    if [[ -n "${BOOTSTRAP_STATUS_PID:-}" ]] && kill -0 "$BOOTSTRAP_STATUS_PID" 2>/dev/null; then
        return 0
    fi

    echo "Starting bootstrap status server on port ${WORKER_PORT}"

    python3 -u - <<'PY' &
import json
import os
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKER_PORT = int(os.getenv("WORKER_PORT", "3000"))
READY_ROUTE = os.getenv("PYWORKER_READY_ROUTE", "/readyz").strip() or "/readyz"
STATUS_ROUTE = os.getenv("PYWORKER_STATUS_ROUTE", "/statusz").strip() or "/statusz"
STATUS_FILE = Path(os.getenv("PYWORKER_STATUS_FILE", "/workspace/pyworker-status.json"))
DEBUG_LOG_FILE = Path(os.getenv("PYWORKER_DEBUG_LOG_FILE", "/workspace/debug.log"))
PYWORKER_LOG_FILE = Path(os.getenv("PYWORKER_LOG_FILE", "/workspace/pyworker.log"))
MODEL_LOG_FILE = Path(os.getenv("MODEL_LOG", "/var/log/portal/comfyui.log"))
PROVISIONING_DONE_MARKER = Path(
    os.getenv("PROVISIONING_DONE_MARKER", "/workspace/.provisioning-complete")
)
PROVISIONING_FAILED_MARKER = Path(
    os.getenv("PROVISIONING_FAILED_MARKER", "/workspace/.provisioning-failed")
)
LOG_TAIL_MAX_BYTES = int(os.getenv("PYWORKER_LOG_TAIL_MAX_BYTES", "8192"))
LOW_DISK_FREE_BYTES = int(
    os.getenv("PYWORKER_LOW_DISK_FREE_BYTES", str(5 * 1024 * 1024 * 1024))
)
STATUS_BODY_PREVIEW_LIMIT = int(os.getenv("PYWORKER_STATUS_BODY_PREVIEW_LIMIT", "500"))
AUTH_TOKEN = (
    os.getenv("OPEN_BUTTON_TOKEN", "").strip()
    or os.getenv("WEB_PASSWORD", "").strip()
)
AUTH_REQUIRED = (
    os.getenv("ENABLE_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}
    or os.getenv("UNSECURED", "").strip().lower() in {"0", "false", "no", "off"}
)

TERMINAL_PATTERNS = (
    (
        "disk_full",
        "No space left on device",
        "Disk is full. Worker should be terminated and reprovisioned.",
    ),
    (
        "comfy_prestartup_failed",
        "PRESTARTUP FAILED",
        "A ComfyUI custom node failed during startup.",
    ),
    (
        "torch_dynamo_circular_import",
        "torch._dynamo' has no attribute 'guards'",
        "Torch/einops import failed during ComfyUI startup.",
    ),
    (
        "provisioning_failed",
        "Provisioning failed",
        "Provisioning reported a fatal failure before the worker became ready.",
    ),
    (
        "provisioning_script_failed",
        "[ERROR] Provisioning Script failed",
        "Provisioning script exited with a fatal error.",
    ),
    (
        "pyworker_exited",
        "PyWorker exited with status",
        "PyWorker process exited unexpectedly during startup.",
    ),
)


def safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def tail_text(path: Path, limit: int) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - limit), os.SEEK_SET)
            data = file.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return data[-limit:]


def disk_snapshot(path: str) -> dict[str, int | None]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {"totalBytes": None, "usedBytes": None, "freeBytes": None}
    return {
        "totalBytes": usage.total,
        "usedBytes": usage.used,
        "freeBytes": usage.free,
    }


def find_fatal_signals(status_payload: dict, log_map: dict[str, str | None]) -> list[dict[str, str]]:
    haystacks: list[tuple[str, str]] = []
    message = status_payload.get("message")
    if isinstance(message, str) and message:
        haystacks.append(("status.message", message))
    error_message = status_payload.get("errorMessage")
    if isinstance(error_message, str) and error_message:
        haystacks.append(("status.errorMessage", error_message))
    for name, content in log_map.items():
        if isinstance(content, str) and content:
            haystacks.append((name, content))

    matches: list[dict[str, str]] = []
    for code, needle, description in TERMINAL_PATTERNS:
        for source, content in haystacks:
            if needle in content:
                matches.append(
                    {
                        "code": code,
                        "message": description,
                        "source": source,
                        "match": needle,
                    }
                )
                break
    return matches


def build_phase(
    recorded_phase: str | None,
    provisioning_done: bool,
    provisioning_failed: bool,
    fatal_signals: list[dict[str, str]],
) -> str:
    if provisioning_failed or fatal_signals:
        return "failed"
    if recorded_phase == "failed":
        return "failed"
    if recorded_phase in {"provisioning", "starting", "booting"}:
        return recorded_phase
    if provisioning_done:
        return "starting"
    return "booting"


def build_status() -> dict:
    status_payload = safe_read_json(STATUS_FILE)
    logs = {
        "debugLogTail": tail_text(DEBUG_LOG_FILE, LOG_TAIL_MAX_BYTES),
        "pyworkerLogTail": tail_text(PYWORKER_LOG_FILE, LOG_TAIL_MAX_BYTES),
        "modelLogTail": tail_text(MODEL_LOG_FILE, LOG_TAIL_MAX_BYTES),
    }
    provisioning_done = PROVISIONING_DONE_MARKER.exists()
    provisioning_failed = PROVISIONING_FAILED_MARKER.exists()
    disk = {
        "workspace": disk_snapshot(str(PROVISIONING_DONE_MARKER.parent)),
        "tmp": disk_snapshot("/tmp"),
    }
    fatal_signals = find_fatal_signals(status_payload, logs)
    workspace_free_bytes = disk["workspace"].get("freeBytes")
    if isinstance(workspace_free_bytes, int) and workspace_free_bytes <= LOW_DISK_FREE_BYTES:
        fatal_signals.append(
            {
                "code": "workspace_low_disk",
                "message": "Workspace free disk is below the low-water mark.",
                "source": "disk.workspace",
                "match": str(workspace_free_bytes),
            }
        )

    phase = build_phase(
        status_payload.get("phase") if isinstance(status_payload.get("phase"), str) else None,
        provisioning_done=provisioning_done,
        provisioning_failed=provisioning_failed,
        fatal_signals=fatal_signals,
    )
    should_terminate = phase == "failed"
    model_health = {
        "ok": True if phase == "ready" else False,
        "source": "bootstrap-status-server",
        "url": None,
        "status": "bootstrapping" if phase != "failed" else "failed",
    }

    return {
        "ok": phase == "ready",
        "phase": phase,
        "shouldTerminate": should_terminate,
        "message": status_payload.get("message"),
        "errorMessage": status_payload.get("errorMessage"),
        "fatalSignals": fatal_signals,
        "provisioning": {
            "doneMarkerExists": provisioning_done,
            "failedMarkerExists": provisioning_failed,
        },
        "disk": disk,
        "modelHealth": model_health,
        "logs": logs,
        "bootstrapServer": True,
        "updatedAt": int(time.time() * 1000),
    }


class BootstrapStatusHandler(BaseHTTPRequestHandler):
    server_version = "PyworkerBootstrapStatus/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[bootstrap-status] {self.address_string()} - {fmt % args}", flush=True)

    def _authorized(self) -> bool:
        if not AUTH_REQUIRED or not AUTH_TOKEN:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {AUTH_TOKEN}"

    def _write_json(self, payload: dict, status_code: int) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path not in {READY_ROUTE, STATUS_ROUTE, "/health"}:
            self._write_json({"error": "not_found"}, 404)
            return

        if not self._authorized():
            self._write_json({"error": "unauthorized"}, 401)
            return

        status_payload = build_status()
        if self.path == STATUS_ROUTE:
            self._write_json(status_payload, 200)
            return

        if status_payload["shouldTerminate"] is True:
            code = 500
        elif status_payload["ok"] is True:
            code = 200
        else:
            code = 425

        preview = dict(status_payload)
        if isinstance(preview.get("logs"), dict):
            preview["logs"] = {
                key: (value[-STATUS_BODY_PREVIEW_LIMIT:] if isinstance(value, str) else value)
                for key, value in preview["logs"].items()
            }
        self._write_json(preview, code)


ThreadingHTTPServer(("0.0.0.0", WORKER_PORT), BootstrapStatusHandler).serve_forever()
PY

    BOOTSTRAP_STATUS_PID=$!
    echo "Bootstrap status server PID: ${BOOTSTRAP_STATUS_PID}"
}

function stop_bootstrap_status_server(){
    local pid="${BOOTSTRAP_STATUS_PID:-}"
    if [[ -z "$pid" ]]; then
        return 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping bootstrap status server PID: ${pid}"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    BOOTSTRAP_STATUS_PID=""
}

trap 'stop_bootstrap_status_server' EXIT

function update_status_file(){
    local phase="$1"
    local message="${2:-}"

    if ! python3 - "$STATUS_FILE" "$phase" "$message" >/dev/null 2>&1 <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
phase = sys.argv[2]
message = sys.argv[3]

data = {}
if path.exists():
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw
    except Exception:
        data = {}

data["phase"] = phase
data["message"] = message
data["updatedAt"] = int(time.time() * 1000)
data["pid"] = os.getpid()

if phase == "failed":
    data["errorMessage"] = message

path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
PY
    then
        return 0
    fi
}

function wait_for_provisioning_completion(){
    local wait_mode="${WAIT_FOR_PROVISIONING_MARKER:-auto}"
    local now
    local deadline

    update_status_file "provisioning" "Waiting for provisioning to complete"

    if [ "$wait_mode" = "false" ]; then
        echo "Skipping provisioning wait because WAIT_FOR_PROVISIONING_MARKER=false"
        return 0
    fi

    if [ "$wait_mode" = "auto" ] && [ -f "/.noprovisioning" ]; then
        echo "Skipping provisioning wait because /.noprovisioning is present"
        return 0
    fi

    if [ "$wait_mode" = "auto" ] && [ -z "${PROVISIONING_SCRIPT:-}" ]; then
        echo "Skipping provisioning wait because PROVISIONING_SCRIPT is not set"
        return 0
    fi

    deadline=$(( $(date +%s) + PROVISIONING_WAIT_TIMEOUT_SECONDS ))
    echo "Waiting for provisioning marker: $PROVISIONING_DONE_MARKER"

    while true; do
        if [ -f "$PROVISIONING_DONE_MARKER" ]; then
            echo "Provisioning marker found"
            return 0
        fi

        if [ -f "$PROVISIONING_FAILED_MARKER" ]; then
            report_error_and_exit "Provisioning failed before pyworker startup"
        fi

        now=$(date +%s)
        if [ "$now" -ge "$deadline" ]; then
            report_error_and_exit "Timed out waiting for provisioning marker: $PROVISIONING_DONE_MARKER"
        fi

        sleep "$PROVISIONING_WAIT_INTERVAL_SECONDS"
    done
}

function report_error_and_exit(){
    local error_msg="$1"
    echo "ERROR: $error_msg"
    update_status_file "failed" "$error_msg"

    MTOKEN="${MASTER_TOKEN:-}"
    VERSION="${PYWORKER_VERSION:-0}"

    IFS=',' read -r -a REPORT_ADDRS <<< "${REPORT_ADDR}"
    for addr in "${REPORT_ADDRS[@]}"; do
        curl -sS -X POST -H 'Content-Type: application/json' \
            -d "$(cat <<JSON
{
  "id": ${CONTAINER_ID:-0},
  "mtoken": "${MTOKEN}",
  "version": "${VERSION}",
  "error_msg": "${error_msg}",
  "url": "${URL:-}"
}
JSON
)" "${addr%/}/worker_status/" || true
    done

    exit 1
}

function install_vastai_sdk() {
    # If SDK_BRANCH is set, install vastai-sdk from the vast-sdk repo at that branch/tag/commit.
    if [ -n "${SDK_BRANCH:-}" ]; then
        if [ -n "${SDK_VERSION:-}" ]; then
            echo "WARNING: Both SDK_BRANCH and SDK_VERSION are set; using SDK_BRANCH=${SDK_BRANCH}"
        fi
        echo "Installing vastai-sdk from https://github.com/vast-ai/vast-sdk/ @ ${SDK_BRANCH}"
        if ! uv pip install "vastai-sdk @ git+https://github.com/vast-ai/vast-sdk.git@${SDK_BRANCH}"; then
            report_error_and_exit "Failed to install vastai-sdk from vast-ai/vast-sdk@${SDK_BRANCH}"
        fi
        return 0
    fi

    if [ -n "${SDK_VERSION:-}" ]; then
        echo "Installing vastai-sdk version ${SDK_VERSION}"
        if ! uv pip install "vastai-sdk==${SDK_VERSION}"; then
            report_error_and_exit "Failed to install vastai-sdk==${SDK_VERSION}"
        fi
        return 0
    fi

    echo "Installing default vastai-sdk"
    if ! uv pip install vastai-sdk; then
        report_error_and_exit "Failed to install vastai-sdk"
    fi
}

update_status_file "booting" "start_server.sh bootstrapping"

[ -z "$CONTAINER_ID" ] && report_error_and_exit "CONTAINER_ID must be set!"
[ "$BACKEND" = "comfyui" ] && [ -z "$COMFY_MODEL" ] && report_error_and_exit "For comfyui backends, COMFY_MODEL must be set!"

start_bootstrap_status_server

echo "start_server.sh"
date

echo_var BACKEND
echo_var REPORT_ADDR
echo_var WORKER_PORT
echo_var WORKSPACE_DIR
echo_var SERVER_DIR
echo_var ENV_PATH
echo_var DEBUG_LOG
echo_var PYWORKER_LOG
echo_var STATUS_FILE
echo_var MODEL_LOG
echo_var PROVISIONING_DONE_MARKER
echo_var PROVISIONING_FAILED_MARKER

ROTATE_MODEL_LOG="${ROTATE_MODEL_LOG:-false}"
if [ "$ROTATE_MODEL_LOG" = "true" ] && [ -e "$MODEL_LOG" ]; then
    echo "Rotating model log at $MODEL_LOG to $MODEL_LOG.old"
    if ! cat "$MODEL_LOG" >> "$MODEL_LOG.old"; then
        report_error_and_exit "Failed to rotate model log"
    fi
    if ! : > "$MODEL_LOG"; then
        report_error_and_exit "Failed to truncate model log"
    fi
fi

# Populate /etc/environment with quoted values
if ! grep -q "VAST" /etc/environment; then
    if ! env -0 | grep -zEv "^(HOME=|SHLVL=)|CONDA" | while IFS= read -r -d '' line; do
            name=${line%%=*}
            value=${line#*=}
            printf '%s="%s"\n' "$name" "$value"
        done > /etc/environment; then
        echo "WARNING: Failed to populate /etc/environment, continuing anyway"
    fi
fi

if [ ! -d "$ENV_PATH" ]
then
    echo "setting up venv"
    if ! which uv; then
        if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
            report_error_and_exit "Failed to install uv package manager"
        fi
        if [[ -f ~/.local/bin/env ]]; then
            if ! source ~/.local/bin/env; then
                report_error_and_exit "Failed to source uv environment"
            fi
        else
            echo "WARNING: ~/.local/bin/env not found after uv installation"
        fi
    fi

    if [[ ! -d $SERVER_DIR ]]; then
        if ! git clone "${PYWORKER_REPO:-https://github.com/vast-ai/pyworker}" "$SERVER_DIR"; then
            report_error_and_exit "Failed to clone pyworker repository"
        fi
    fi
    if [[ -n ${PYWORKER_REF:-} ]]; then
        if ! (cd "$SERVER_DIR" && git checkout "$PYWORKER_REF"); then
            report_error_and_exit "Failed to checkout pyworker reference: $PYWORKER_REF"
        fi
    fi

    if ! uv venv --python-preference only-managed "$ENV_PATH" -p 3.10; then
        report_error_and_exit "Failed to create virtual environment"
    fi
    
    if ! source "$ENV_PATH/bin/activate"; then
        report_error_and_exit "Failed to activate virtual environment"
    fi

    if ! uv pip install -r "${SERVER_DIR}/requirements.txt"; then
        report_error_and_exit "Failed to install Python requirements"
    fi

    install_vastai_sdk

    if ! touch ~/.no_auto_tmux; then
        report_error_and_exit "Failed to create ~/.no_auto_tmux"
    fi
else
    if [[ -f ~/.local/bin/env ]]; then
        if ! source ~/.local/bin/env; then
            report_error_and_exit "Failed to source uv environment"
        fi
    fi
    if ! source "$WORKSPACE_DIR/worker-env/bin/activate"; then
        report_error_and_exit "Failed to activate existing virtual environment"
    fi
    echo "environment activated"
    echo "venv: $VIRTUAL_ENV"
fi

if [ "$USE_SSL" = true ]; then

    if ! cat << EOF > /etc/openssl-san.cnf
    [req]
    default_bits       = 2048
    distinguished_name = req_distinguished_name
    req_extensions     = v3_req

    [req_distinguished_name]
    countryName         = US
    stateOrProvinceName = CA
    organizationName    = Vast.ai Inc.
    commonName          = vast.ai

    [v3_req]
    basicConstraints = CA:FALSE
    keyUsage         = nonRepudiation, digitalSignature, keyEncipherment
    subjectAltName   = @alt_names

    [alt_names]
    IP.1   = 0.0.0.0
EOF
    then
        report_error_and_exit "Failed to write OpenSSL config"
    fi

    if ! openssl req -newkey rsa:2048 -subj "/C=US/ST=CA/CN=pyworker.vast.ai/" \
        -nodes \
        -sha256 \
        -keyout /etc/instance.key \
        -out /etc/instance.csr \
        -config /etc/openssl-san.cnf; then
        report_error_and_exit "Failed to generate SSL certificate request"
    fi

    if ! curl --header 'Content-Type: application/octet-stream' \
        --data-binary @/etc/instance.csr \
        -X \
        POST "https://console.vast.ai/api/v0/sign_cert/?instance_id=$CONTAINER_ID" > /etc/instance.crt; then
        report_error_and_exit "Failed to sign SSL certificate"
    fi
fi

export REPORT_ADDR WORKER_PORT USE_SSL UNSECURED

if ! cd "$SERVER_DIR"; then
    report_error_and_exit "Failed to cd into SERVER_DIR: $SERVER_DIR"
fi

wait_for_provisioning_completion
update_status_file "starting" "Launching pyworker process"
stop_bootstrap_status_server

echo "launching PyWorker server"

set +e

PY_STATUS=1

if [ -f "$SERVER_DIR/worker.py" ]; then
    echo "trying worker.py"
    python3 -m "worker" 2>&1 | tee -a "$PYWORKER_LOG"
    PY_STATUS=${PIPESTATUS[0]}
fi

if [ "${PY_STATUS}" -ne 0 ] && [ -f "$SERVER_DIR/workers/$BACKEND/worker.py" ]; then
    echo "trying workers.${BACKEND}.worker"
    python3 -m "workers.${BACKEND}.worker" 2>&1 | tee -a "$PYWORKER_LOG"
    PY_STATUS=${PIPESTATUS[0]}
fi

if [ "${PY_STATUS}" -ne 0 ] && [ -f "$SERVER_DIR/workers/$BACKEND/server.py" ]; then
    echo "trying workers.${BACKEND}.server"
    python3 -m "workers.${BACKEND}.server" 2>&1 | tee -a "$PYWORKER_LOG"
    PY_STATUS=${PIPESTATUS[0]}
fi

set -e

if [ "${PY_STATUS}" -ne 0 ]; then
    if [ ! -f "$SERVER_DIR/worker.py" ] && [ ! -f "$SERVER_DIR/workers/$BACKEND/worker.py" ] && [ ! -f "$SERVER_DIR/workers/$BACKEND/server.py" ]; then
        report_error_and_exit "Failed to find PyWorker"
    fi
    report_error_and_exit "PyWorker exited with status ${PY_STATUS}"
fi

echo "launching PyWorker server done"
