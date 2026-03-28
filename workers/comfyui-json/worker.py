import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import re
import shutil
import time
from urllib.parse import urlparse

import aiohttp
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# ComfyUI transport configuration
# Vast's generic HandlerConfig proxies the same client route to the configured model server
# route. For comfyui-json that means pyworker must talk to the API wrapper's
# /generate/sync endpoint, while the wrapper itself talks to raw ComfyUI via its own
# COMFYUI_API_BASE env var. Keep those env names separate so they don't fight each other.
MODEL_SERVER_BASE_URL = (
    os.getenv("PYWORKER_MODEL_SERVER_BASE_URL", "http://127.0.0.1:18288").strip().rstrip("/")
)
MODEL_HEALTHCHECK_BASE_URL = (
    os.getenv("PYWORKER_HEALTHCHECK_BASE_URL", MODEL_SERVER_BASE_URL).strip().rstrip("/")
    or MODEL_SERVER_BASE_URL
)
_MODEL_SERVER_PARSED = urlparse(MODEL_SERVER_BASE_URL)
MODEL_SERVER_SCHEME = _MODEL_SERVER_PARSED.scheme or "http"
MODEL_SERVER_HOST = _MODEL_SERVER_PARSED.hostname or "127.0.0.1"
MODEL_SERVER_PORT = _MODEL_SERVER_PARSED.port or (443 if MODEL_SERVER_SCHEME == "https" else 80)
MODEL_SERVER_URL = f"{MODEL_SERVER_SCHEME}://{MODEL_SERVER_HOST}"
MODEL_LOG_FILE = "/var/log/portal/comfyui.log"
MODEL_HEALTHCHECK_ENDPOINT = (
    os.getenv("PYWORKER_MODEL_HEALTHCHECK_ENDPOINT", "/health").strip() or "/health"
)
READY_ROUTE = os.getenv("PYWORKER_READY_ROUTE", "/readyz").strip() or "/readyz"
STATUS_ROUTE = os.getenv("PYWORKER_STATUS_ROUTE", "/statusz").strip() or "/statusz"
STATUS_SUMMARY_ROUTE = (
    os.getenv("PYWORKER_STATUS_SUMMARY_ROUTE", "/status-summaryz").strip()
    or "/status-summaryz"
)
PROVISIONING_DONE_MARKER = Path(
    os.getenv("PROVISIONING_DONE_MARKER", "/workspace/.provisioning-complete")
)
PROVISIONING_FAILED_MARKER = Path(
    os.getenv("PROVISIONING_FAILED_MARKER", "/workspace/.provisioning-failed")
)
STATUS_FILE = Path(os.getenv("PYWORKER_STATUS_FILE", "/workspace/pyworker-status.json"))
DEBUG_LOG_FILE = Path(os.getenv("PYWORKER_DEBUG_LOG_FILE", "/workspace/debug.log"))
PYWORKER_LOG_FILE = Path(os.getenv("PYWORKER_LOG_FILE", "/workspace/pyworker.log"))
LOG_TAIL_MAX_BYTES = int(os.getenv("PYWORKER_LOG_TAIL_MAX_BYTES", "8192"))
LOW_DISK_FREE_BYTES = int(
    os.getenv("PYWORKER_LOW_DISK_FREE_BYTES", str(5 * 1024 * 1024 * 1024))
)
STATUS_BODY_PREVIEW_LIMIT = int(os.getenv("PYWORKER_STATUS_BODY_PREVIEW_LIMIT", "500"))
STATUS_SUMMARY_CACHE_MS = int(os.getenv("PYWORKER_STATUS_SUMMARY_CACHE_MS", "10000"))

# ComyUI-specific log messages
MODEL_LOAD_LOG_MSG = ["To see the GUI go to: "]

MODEL_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",
    "[ERROR] Provisioning Script failed",
]

MODEL_INFO_LOG_MSGS = ['"message":"Downloading']

DEFAULT_WIDTH = 432
DEFAULT_HEIGHT = 768
DEFAULT_FRAMES = 17
DEFAULT_STEPS = 8

# Enforce 1 worker = 1 generation:
# if one request is already running, reject additional requests immediately
# so callers can re-route to another worker.
MAX_QUEUE_TIME = 0.0
WORKLOAD_MULTIPLIER = 0.6
ENABLE_BOOTSTRAP_BENCHMARK = os.getenv("PYWORKER_ENABLE_BOOTSTRAP_BENCHMARK", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

HF_LORA_REPO = os.getenv("HF_LORA_REPO", "Dylaaann/Lora")
HF_LORA_TOKEN = os.getenv("HF_TOKEN")
COMFY_LORA_DIR = Path(os.getenv("COMFY_LORA_DIR", "/workspace/ComfyUI/models/loras"))
MANIFEST_SECRET = os.getenv("PYWORKER_MANIFEST_SECRET", "").strip()
MANIFEST_MAX_AGE_SECONDS = int(os.getenv("PYWORKER_MANIFEST_MAX_AGE_SECONDS", "900"))
MANIFEST_ENDPOINT = os.getenv("PYWORKER_MANIFEST_ENDPOINT", "direct-instance").strip()

_require_manifest_raw = os.getenv("PYWORKER_REQUIRE_MANIFEST", "auto").strip().lower()
if _require_manifest_raw in {"1", "true", "yes", "on"}:
    REQUIRE_SIGNED_MANIFEST = True
elif _require_manifest_raw in {"0", "false", "no", "off"}:
    REQUIRE_SIGNED_MANIFEST = False
else:
    # Auto: require signed manifests whenever a verification secret is configured.
    REQUIRE_SIGNED_MANIFEST = bool(MANIFEST_SECRET)

if REQUIRE_SIGNED_MANIFEST and not MANIFEST_SECRET:
    raise RuntimeError(
        "PYWORKER_REQUIRE_MANIFEST is enabled but PYWORKER_MANIFEST_SECRET is not configured."
    )

IGNORED_LORA_NAMES = {"", "none", "null"}
LORA_INPUT_KEY_REGEX = re.compile(r"^lora(?:_\d+)?_name$", re.IGNORECASE)

log = logging.getLogger("custom-comfyui-json-worker")

TERMINAL_LOG_PATTERNS = (
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
        "torch_dynamo_guards_missing",
        "torch._dynamo' has no attribute 'guards",
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
        "Provisioning script logged a fatal error.",
    ),
)

JOB_STAGE_PATTERNS = (
    (
        "failed",
        re.compile(r"GenerationWorker \d+ failed job (?P<job>\S+): (?P<detail>.+)"),
        "Generation failed.",
    ),
    (
        "cancelled",
        re.compile(r"Job (?P<job>\S+) was cancelled during generation"),
        "Generation was cancelled.",
    ),
    (
        "postprocess",
        re.compile(r"PostprocessWorker \d+ processing job: (?P<job>\S+)"),
        "Postprocessing generated output.",
    ),
    (
        "generation",
        re.compile(r"GenerationWorker \d+ processing job: (?P<job>\S+)"),
        "Running ComfyUI generation.",
    ),
    (
        "preprocess",
        re.compile(r"PreprocessWorker \d+ processing job: (?P<job>\S+)"),
        "Preparing generation inputs.",
    ),
    (
        "completed",
        re.compile(r"PostprocessWorker \d+ completed job: (?P<job>\S+)"),
        "Generation completed.",
    ),
)

PROGRESS_LINE_RE = re.compile(r"(?P<percent>\d{1,3})%\|.*?(?P<current>\d+)/(?P<total>\d+)")
GENERATION_PROGRESS_RE = re.compile(
    r"Progress update:\s*Progress:\s*(?P<percent>\d+(?:\.\d+)?)%\s*\((?P<current>\d+)/(?P<total>\d+)\)"
)
PROVISIONING_PROGRESS_PERCENT_RE = re.compile(r"\((?P<percent>\d{1,3})%\)")
PROMPT_EXECUTED_RE = re.compile(r"Prompt executed in (?P<seconds>[\d.]+) seconds")

_status_summary_cache: dict[str, Any] | None = None
_status_summary_cache_expires_at = 0


def safe_read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return raw if isinstance(raw, dict) else {}


def safe_tail_file(path: Path, max_bytes: int = LOG_TAIL_MAX_BYTES) -> str | None:
    if max_bytes <= 0 or not path.exists() or not path.is_file():
        return None

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def compact_log_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()[:240]


def parse_progress_line(line: str, source: str) -> dict[str, Any] | None:
    generation_match = GENERATION_PROGRESS_RE.search(line)
    if generation_match:
        try:
            percent = float(generation_match.group("percent"))
            current = int(generation_match.group("current"))
            total = int(generation_match.group("total"))
        except ValueError:
            return None

        return {
            "kind": "generation",
            "percent": max(0, min(100, round(percent, 1))),
            "current": current,
            "total": total,
            "source": source,
        }

    match = PROGRESS_LINE_RE.search(line)
    if match:
        try:
            percent = int(match.group("percent"))
            current = int(match.group("current"))
            total = int(match.group("total"))
        except ValueError:
            return None

        return {
            "kind": "generation",
            "percent": max(0, min(100, percent)),
            "current": current,
            "total": total,
            "source": source,
        }

    if "[DL:" in line:
        percentages = [
            int(value)
            for value in PROVISIONING_PROGRESS_PERCENT_RE.findall(line)
            if value.isdigit()
        ]
        if not percentages:
            return None

        return {
            "kind": "provisioning",
            "percent": round(sum(percentages) / len(percentages), 1),
            "source": source,
            "detail": compact_log_line(line),
        }

    return None


def extract_recent_progress(logs: dict[str, str | None]) -> dict[str, Any] | None:
    for source in ("modelTail", "pyworkerTail", "debugTail"):
        text = logs.get(source)
        if not isinstance(text, str):
            continue

        for raw_line in reversed(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue

            progress = parse_progress_line(line, source)
            if progress:
                return progress

    return None


def extract_activity(
    *,
    status_file: dict[str, Any],
    logs: dict[str, str | None],
    phase: str,
) -> dict[str, Any]:
    progress = extract_recent_progress(logs)

    for source in ("pyworkerTail", "modelTail", "debugTail"):
        text = logs.get(source)
        if not isinstance(text, str):
            continue

        for raw_line in reversed(text.splitlines()):
            line = raw_line.strip()
            if not line:
                continue

            progress_line = parse_progress_line(line, source)
            if progress_line and progress_line.get("kind") == "generation":
                current = progress_line.get("current")
                total = progress_line.get("total")
                percent = progress_line.get("percent")
                message = "Running ComfyUI generation."
                if isinstance(current, int) and isinstance(total, int):
                    message = f"Running ComfyUI generation ({current}/{total})."
                return {
                    "stage": "generation",
                    "source": source,
                    "message": message,
                    "progress": progress_line,
                    "progressPct": percent,
                }

            if (
                phase == "provisioning"
                and progress_line
                and progress_line.get("kind") == "provisioning"
            ):
                percent = progress_line.get("percent")
                message = "Downloading provisioning assets."
                if isinstance(percent, (int, float)):
                    message = f"Downloading provisioning assets ({percent}%)."
                return {
                    "stage": "provisioning",
                    "source": source,
                    "message": message,
                    "progress": progress_line,
                    "progressPct": progress_line.get("percent"),
                }

            for stage, pattern, default_message in JOB_STAGE_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue

                activity: dict[str, Any] = {
                    "stage": stage,
                    "source": source,
                    "message": default_message,
                }
                job_id = match.groupdict().get("job")
                if job_id:
                    activity["jobId"] = job_id
                detail = match.groupdict().get("detail")
                if detail:
                    activity["message"] = compact_log_line(detail)
                if progress and stage in {"preprocess", "generation", "postprocess"}:
                    activity["progress"] = progress
                    activity["progressPct"] = progress["percent"]
                return activity

            lower = line.lower()
            if "processing interrupted" in lower:
                return {
                    "stage": "cancelled",
                    "source": source,
                    "message": "Generation processing was interrupted.",
                }

            prompt_executed = PROMPT_EXECUTED_RE.search(line)
            if prompt_executed:
                return {
                    "stage": "completed",
                    "source": source,
                    "message": compact_log_line(line),
                }

            if "waiting for jobs" in lower and phase == "ready":
                return {
                    "stage": "idle",
                    "source": source,
                    "message": "Worker ready and waiting for jobs.",
                }

    status_message = status_file.get("message")
    if isinstance(status_message, str) and status_message.strip():
        activity: dict[str, Any] = {
            "stage": phase if phase != "ready" else "idle",
            "source": "statusFile.message",
            "message": compact_log_line(status_message),
        }
        if progress and phase in {"provisioning", "preprocess", "generation", "postprocess"}:
            activity["progress"] = progress
            activity["progressPct"] = progress["percent"]
        return activity

    fallback_message = "Worker ready and waiting for jobs." if phase == "ready" else f"Worker phase: {phase}"
    activity: dict[str, Any] = {
        "stage": "idle" if phase == "ready" else phase,
        "source": "phase",
        "message": fallback_message,
    }
    if progress and phase in {"provisioning", "preprocess", "generation", "postprocess"}:
        activity["progress"] = progress
        activity["progressPct"] = progress["percent"]
    return activity


def collect_log_signals(source: str, text: str | None) -> list[dict[str, str]]:
    if not text:
        return []

    signals: list[dict[str, str]] = []
    for code, needle, message in TERMINAL_LOG_PATTERNS:
        if needle.lower() in text.lower():
            signals.append(
                {
                    "code": code,
                    "source": source,
                    "message": message,
                }
            )

    lower = text.lower()
    if lower.count("warn exited: comfyui") >= 3:
        signals.append(
            {
                "code": "comfy_crash_loop",
                "source": source,
                "message": "ComfyUI appears to be restarting repeatedly.",
            }
        )
    if lower.count("502 bad gateway") >= 3:
        signals.append(
            {
                "code": "bad_gateway_loop",
                "source": source,
                "message": "Repeated 502 health failures indicate the model server is not becoming healthy.",
            }
        )

    deduped: dict[tuple[str, str], dict[str, str]] = {}
    for signal in signals:
        deduped[(signal["code"], signal["source"])] = signal
    return list(deduped.values())


async def get_model_health_snapshot() -> dict[str, Any]:
    health_url = f"{MODEL_HEALTHCHECK_BASE_URL}{MODEL_HEALTHCHECK_ENDPOINT}"
    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(health_url) as response:
                body = await response.text()
                return {
                    "ok": response.status == 200,
                    "status": response.status,
                    "url": health_url,
                    "bodyPreview": body[:STATUS_BODY_PREVIEW_LIMIT],
                }
    except Exception as error:
        return {
            "ok": False,
            "url": health_url,
            "error": str(error),
        }


def get_disk_snapshot(path: str) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
    except Exception as error:
        return {
            "path": path,
            "error": str(error),
        }

    return {
        "path": path,
        "totalBytes": usage.total,
        "usedBytes": usage.used,
        "freeBytes": usage.free,
        "isLow": usage.free <= LOW_DISK_FREE_BYTES,
    }


def build_phase(
    *,
    provisioning_failed: bool,
    provisioning_done: bool,
    model_health_ok: bool,
    fatal_signals: list[dict[str, str]],
    recorded_phase: str | None,
) -> str:
    if provisioning_failed:
        return "failed"
    if fatal_signals:
        return "failed"
    if not provisioning_done:
        return "provisioning"
    if model_health_ok:
        return "ready"
    if recorded_phase == "failed":
        return "failed"
    return "starting"


async def build_worker_status() -> dict[str, Any]:
    status_file = safe_read_json_file(STATUS_FILE)
    debug_tail = safe_tail_file(DEBUG_LOG_FILE)
    pyworker_tail = safe_tail_file(PYWORKER_LOG_FILE)
    model_tail = safe_tail_file(Path(MODEL_LOG_FILE))
    logs = {
        "debugTail": debug_tail,
        "pyworkerTail": pyworker_tail,
        "modelTail": model_tail,
    }

    health = await get_model_health_snapshot()
    log_signals = [
        *collect_log_signals("debug", debug_tail),
        *collect_log_signals("pyworker", pyworker_tail),
        *collect_log_signals("model", model_tail),
    ]

    provisioning_failed = PROVISIONING_FAILED_MARKER.exists()
    provisioning_done = PROVISIONING_DONE_MARKER.exists()
    disk = {
        "workspace": get_disk_snapshot(str(PROVISIONING_DONE_MARKER.parent)),
        "tmp": get_disk_snapshot("/tmp"),
    }
    if disk["workspace"].get("isLow"):
        log_signals.append(
            {
                "code": "workspace_low_disk",
                "source": "workspace",
                "message": "Workspace disk free space is below the safety threshold.",
            }
        )
    if disk["tmp"].get("isLow"):
        log_signals.append(
            {
                "code": "tmp_low_disk",
                "source": "tmp",
                "message": "Temporary disk free space is below the safety threshold.",
            }
        )

    phase = build_phase(
        provisioning_failed=provisioning_failed,
        provisioning_done=provisioning_done,
        model_health_ok=health.get("ok") is True,
        fatal_signals=log_signals,
        recorded_phase=status_file.get("phase")
        if isinstance(status_file.get("phase"), str)
        else None,
    )
    activity = extract_activity(
        status_file=status_file,
        logs=logs,
        phase=phase,
    )

    return {
        "ok": phase == "ready",
        "phase": phase,
        "shouldTerminate": phase == "failed",
        "instance": {
            "containerId": os.getenv("CONTAINER_ID"),
            "machineId": os.getenv("MACHINE_ID"),
            "hostName": os.getenv("HOSTNAME"),
        },
        "provisioning": {
            "doneMarkerExists": provisioning_done,
            "failedMarkerExists": provisioning_failed,
        },
        "modelHealth": health,
        "disk": disk,
        "activity": activity,
        "fatalSignals": log_signals,
        "statusFile": status_file,
        "logs": logs,
        "updatedAt": int(time.time() * 1000),
    }


def build_worker_status_summary_payload(status: dict[str, Any]) -> dict[str, Any]:
    status_file = (
        status.get("statusFile") if isinstance(status.get("statusFile"), dict) else {}
    )
    activity = status.get("activity") if isinstance(status.get("activity"), dict) else {}
    message = activity.get("message")
    if not isinstance(message, str) or not message.strip():
        message = status_file.get("message")
    error_message = status_file.get("errorMessage")

    return {
        "ok": status.get("ok"),
        "phase": status.get("phase"),
        "shouldTerminate": status.get("shouldTerminate"),
        "instance": status.get("instance"),
        "provisioning": status.get("provisioning"),
        "modelHealth": status.get("modelHealth"),
        "activity": status.get("activity"),
        "fatalSignals": status.get("fatalSignals"),
        "message": message,
        "errorMessage": error_message if isinstance(error_message, str) else None,
        "updatedAt": status.get("updatedAt"),
    }


async def build_worker_status_summary(force_refresh: bool = False) -> dict[str, Any]:
    global _status_summary_cache, _status_summary_cache_expires_at

    now = int(time.time() * 1000)
    if (
        not force_refresh
        and _status_summary_cache is not None
        and now < _status_summary_cache_expires_at
    ):
        return dict(_status_summary_cache)

    status = await build_worker_status()
    summary = build_worker_status_summary_payload(status)
    _status_summary_cache = summary
    _status_summary_cache_expires_at = now + max(0, STATUS_SUMMARY_CACHE_MS)
    return dict(summary)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sanitize_lora_name(value: str) -> str | None:
    trimmed = value.strip()
    if not trimmed:
        return None

    normalized = trimmed.split("/")[-1].split("\\")[-1]
    if normalized.lower() in IGNORED_LORA_NAMES:
        return None

    if not re.fullmatch(r"[^/\\]+\.safetensors", normalized):
        raise ValueError(f"Invalid LoRA filename '{value}'.")

    return normalized


def extract_workflow_loras(workflow_json: dict[str, Any]) -> list[str]:
    names: set[str] = set()

    for node in workflow_json.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue

        for key, raw_value in inputs.items():
            if not isinstance(key, str) or not LORA_INPUT_KEY_REGEX.fullmatch(key):
                continue
            if not isinstance(raw_value, str):
                continue

            normalized = sanitize_lora_name(raw_value)
            if normalized:
                names.add(normalized)

    return sorted(names)


def parse_and_verify_manifest(manifest: dict[str, Any]) -> list[str]:
    required = manifest.get("requiredLoras")
    if not isinstance(required, list):
        raise ValueError("lora_manifest.requiredLoras must be an array.")

    required_loras: list[str] = []
    for entry in required:
        if not isinstance(entry, str):
            raise ValueError("lora_manifest.requiredLoras entries must be strings.")
        normalized = sanitize_lora_name(entry)
        if normalized:
            required_loras.append(normalized)
    required_loras = sorted(set(required_loras))

    if not MANIFEST_SECRET:
        return required_loras

    generation_id = manifest.get("generationId")
    endpoint = manifest.get("endpoint")
    issued_at = manifest.get("issuedAt")
    signature = manifest.get("signature")

    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError("lora_manifest.generationId must be a non-empty string.")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("lora_manifest.endpoint must be a non-empty string.")
    if MANIFEST_ENDPOINT and endpoint != MANIFEST_ENDPOINT:
        raise ValueError(
            f"lora_manifest.endpoint mismatch. expected='{MANIFEST_ENDPOINT}' got='{endpoint}'"
        )
    if not isinstance(issued_at, (int, float)):
        raise ValueError("lora_manifest.issuedAt must be a number.")
    if not isinstance(signature, str) or not signature:
        raise ValueError(
            "lora_manifest.signature is required when PYWORKER_MANIFEST_SECRET is set."
        )

    age_ms = abs(int(time.time() * 1000) - int(issued_at))
    if age_ms > MANIFEST_MAX_AGE_SECONDS * 1000:
        raise ValueError("lora_manifest signature expired.")

    payload_to_sign = {
        "endpoint": endpoint,
        "generationId": generation_id,
        "issuedAt": int(issued_at),
        "requiredLoras": required_loras,
    }
    expected_signature = hmac.new(
        MANIFEST_SECRET.encode("utf-8"),
        canonical_json(payload_to_sign).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, signature):
        raise ValueError("lora_manifest signature mismatch.")

    return required_loras


def ensure_lora_downloaded(lora_name: str) -> Path:
    COMFY_LORA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = COMFY_LORA_DIR / lora_name
    lock_path = COMFY_LORA_DIR / f"{lora_name}.lock"

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        if target_path.exists():
            return target_path

        log.info("Downloading LoRA '%s' from '%s'", lora_name, HF_LORA_REPO)
        downloaded_path = hf_hub_download(
            repo_id=HF_LORA_REPO,
            filename=lora_name,
            repo_type="model",
            local_dir=str(COMFY_LORA_DIR),
            local_dir_use_symlinks=False,
            token=HF_LORA_TOKEN or None,
        )

        downloaded_file = Path(downloaded_path)
        if (
            downloaded_file.exists()
            and downloaded_file != target_path
            and not target_path.exists()
        ):
            downloaded_file.replace(target_path)

        if not target_path.exists():
            raise RuntimeError(f"LoRA download finished but file is missing: {target_path}")

    return target_path


def ensure_required_loras(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object.")

    input_payload = payload.get("input")
    if not isinstance(input_payload, dict):
        raise ValueError("payload.input must be an object.")

    workflow_json = input_payload.get("workflow_json")
    if not isinstance(workflow_json, dict):
        # Allow modifier-mode requests that do not send a workflow_json payload.
        return payload

    requested_loras_raw = input_payload.get("required_loras")
    requested_loras: list[str] = []
    if requested_loras_raw is not None:
        if not isinstance(requested_loras_raw, list):
            raise ValueError("input.required_loras must be an array when provided.")
        for entry in requested_loras_raw:
            if not isinstance(entry, str):
                raise ValueError("input.required_loras entries must be strings.")
            normalized = sanitize_lora_name(entry)
            if normalized:
                requested_loras.append(normalized)

    manifest_raw = input_payload.get("lora_manifest")
    manifest_loras: list[str] = []
    if manifest_raw is not None:
        if not isinstance(manifest_raw, dict):
            raise ValueError("input.lora_manifest must be an object when provided.")
        manifest_loras = parse_and_verify_manifest(manifest_raw)
    elif REQUIRE_SIGNED_MANIFEST:
        raise ValueError("Signed lora_manifest is required but missing.")

    workflow_loras = extract_workflow_loras(workflow_json)

    required_set = set(workflow_loras)
    required_set.update(requested_loras)

    if MANIFEST_SECRET and manifest_loras:
        approved = set(manifest_loras)
        unexpected = sorted(name for name in required_set if name not in approved)
        if unexpected:
            raise ValueError(
                "Workflow requested LoRAs not allowed by signed manifest: "
                + ", ".join(unexpected)
            )
        required_set = approved
    elif manifest_loras:
        required_set.update(manifest_loras)

    required_loras = sorted(required_set)

    for lora_name in required_loras:
        ensure_lora_downloaded(lora_name)

    if required_loras:
        log.info("LoRA set ready (%d): %s", len(required_loras), ", ".join(required_loras))

    input_payload["required_loras"] = required_loras
    payload["input"] = input_payload
    return payload


def parse_numeric_value(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        try:
            parsed = float(trimmed)
            return parsed if math.isfinite(parsed) else None
        except ValueError:
            return None
    return None


def resolve_numeric_input(input_value, workflow, visited, depth=0):
    if depth > 6:
        return None
    direct = parse_numeric_value(input_value)
    if direct is not None:
        return direct

    if isinstance(input_value, list) and len(input_value) > 0:
        node_id = str(input_value[0])
        if node_id in visited:
            return None
        node = workflow.get(node_id)
        if not node or "inputs" not in node:
            return None
        visited.add(node_id)
        inputs = node.get("inputs") or {}

        value = parse_numeric_value(inputs.get("value"))
        if value is not None:
            return value

        a = resolve_numeric_input(inputs.get("a"), workflow, visited, depth + 1)
        b = resolve_numeric_input(inputs.get("b"), workflow, visited, depth + 1)
        operation = inputs.get("operation")
        op = operation.lower() if isinstance(operation, str) else None
        if a is not None and b is not None:
            if op in {"add", "+"}:
                return a + b
            if op in {"subtract", "-"}:
                return a - b
            if op in {"multiply", "*"}:
                return a * b
            if op in {"divide", "/"}:
                return None if b == 0 else a / b

        expression = inputs.get("value")
        if isinstance(expression, str) and a is not None and b is not None:
            expr = expression.replace(" ", "")
            if "a*b" in expr:
                return a * b
            if "a+b" in expr:
                return a + b
            if "a-b" in expr:
                return a - b
            if "a/b" in expr:
                return None if b == 0 else a / b

    return None


def extract_workflow(payload):
    if not isinstance(payload, dict):
        return None

    if isinstance(payload.get("workflow_json"), dict):
        return payload.get("workflow_json")

    input_data = payload.get("input")
    if isinstance(input_data, dict) and isinstance(input_data.get("workflow_json"), dict):
        return input_data.get("workflow_json")

    nested = payload.get("payload")
    if isinstance(nested, dict):
        return extract_workflow(nested)

    data = payload.get("data")
    if isinstance(data, dict):
        return extract_workflow(data)

    return None


def estimate_workload(workflow):
    width = None
    height = None
    frames = None

    for node in workflow.values():
        if node.get("class_type") == "WanImageToVideo":
            inputs = node.get("inputs") or {}
            width = resolve_numeric_input(inputs.get("width"), workflow, set())
            height = resolve_numeric_input(inputs.get("height"), workflow, set())
            frames = resolve_numeric_input(inputs.get("length"), workflow, set())
            break

    width = width or DEFAULT_WIDTH
    height = height or DEFAULT_HEIGHT
    frames = frames or DEFAULT_FRAMES

    steps_total = None
    for node in workflow.values():
        if node.get("class_type") != "KSamplerAdvanced":
            continue
        inputs = node.get("inputs") or {}
        end_at = resolve_numeric_input(inputs.get("end_at_step"), workflow, set())
        steps = resolve_numeric_input(inputs.get("steps"), workflow, set())
        candidate = end_at or steps
        if candidate is None:
            continue
        steps_total = candidate if steps_total is None else max(steps_total, candidate)

    steps_total = steps_total or DEFAULT_STEPS

    grid_w = math.ceil(width / 512)
    grid_h = math.ceil(height / 512)
    base_workload = grid_w * grid_h * frames * steps_total

    class_types = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if isinstance(class_type, str):
            class_types.append(class_type.lower())

    has_rife = any("rife" in class_type for class_type in class_types)
    has_upscale = any("upscale" in class_type for class_type in class_types)
    video_combine_count = sum(
        1 for class_type in class_types if "video" in class_type and "combine" in class_type
    )

    if has_rife:
        base_workload *= 1.35
    if has_upscale:
        base_workload *= 1.25
    if video_combine_count > 1:
        base_workload *= 1 + (video_combine_count - 1) * 0.15

    multiplier = WORKLOAD_MULTIPLIER if WORKLOAD_MULTIPLIER > 0 else 1.0
    return max(1.0, base_workload * multiplier)


def workload_calculator(payload):
    workflow = extract_workflow(payload)
    if not workflow:
        return estimate_workload({})
    return estimate_workload(workflow)


if ENABLE_BOOTSTRAP_BENCHMARK:
    log.info(
        "Pyworker heavy benchmark is disabled. "
        "Using lightweight bootstrap ping benchmark only."
    )
else:
    log.info(
        "Pyworker heavy benchmark is disabled and bootstrap ping benchmark is also disabled."
    )


def bootstrap_ping_generator() -> dict[str, Any]:
    # Lightweight synthetic payload used only to satisfy pyworker startup benchmark.
    return {"ping": True, "ts": time.time()}


async def bootstrap_ping_remote(**params):
    return {"ok": True, "params": params}


async def readyz_remote(**params):
    status = await build_worker_status_summary()
    if status["ok"] is not True:
        raise RuntimeError(
            f"Worker not ready phase={status['phase']} "
            f"should_terminate={status['shouldTerminate']} "
            f"fatal_signals={status['fatalSignals']} "
            f"model_health={status['modelHealth']}"
        )

    return {
        "ok": True,
        "phase": status["phase"],
        "healthcheck_url": status["modelHealth"].get("url"),
        "model_server_base_url": MODEL_SERVER_BASE_URL,
        "params": params,
    }


async def statusz_remote(**params):
    status = await build_worker_status()
    status["params"] = params
    return status


async def status_summary_remote(**params):
    status = await build_worker_status_summary()
    status["params"] = params
    return status


bootstrap_handler_config = (
    HandlerConfig(
        route="/benchmark/ping",
        allow_parallel_requests=True,
        max_queue_time=MAX_QUEUE_TIME,
        benchmark_config=BenchmarkConfig(
            generator=bootstrap_ping_generator,
            runs=1,
            concurrency=1,
            do_warmup=False,
        ),
        remote_function=bootstrap_ping_remote,
    )
    if ENABLE_BOOTSTRAP_BENCHMARK
    else None
)

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        bootstrap_handler_config,
        HandlerConfig(
            route=READY_ROUTE,
            allow_parallel_requests=True,
            max_queue_time=0.0,
            benchmark_config=None,
            remote_function=readyz_remote,
        ),
        HandlerConfig(
            route=STATUS_ROUTE,
            allow_parallel_requests=True,
            max_queue_time=0.0,
            benchmark_config=None,
            remote_function=statusz_remote,
        ),
        HandlerConfig(
            route=STATUS_SUMMARY_ROUTE,
            allow_parallel_requests=True,
            max_queue_time=0.0,
            benchmark_config=None,
            remote_function=status_summary_remote,
        ),
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=MAX_QUEUE_TIME,
            request_parser=ensure_required_loras,
            workload_calculator=workload_calculator,
            benchmark_config=None,
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

# Remove optional placeholder when benchmark is enabled.
worker_config.handlers = [handler for handler in worker_config.handlers if handler is not None]

Worker(worker_config).run()
