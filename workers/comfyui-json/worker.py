import fcntl
import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
import time
from urllib.parse import urlparse

import aiohttp
from aiohttp import web
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
PROVISIONING_DONE_MARKER = Path(
    os.getenv("PROVISIONING_DONE_MARKER", "/workspace/.provisioning-complete")
)
PROVISIONING_FAILED_MARKER = Path(
    os.getenv("PROVISIONING_FAILED_MARKER", "/workspace/.provisioning-failed")
)

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
MODEL_DOWNLOAD_ALLOCATION_RULE = "first_request_that_causes_download_pays; cache_hits_pay_zero"
REQUEST_TELEMETRY_LOCK = threading.Lock()
REQUEST_TELEMETRY: dict[str, dict[str, Any]] = {}

log = logging.getLogger("custom-comfyui-json-worker")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def now_ms() -> int:
    return int(time.time() * 1000)


def build_download_event(
    *,
    disposition: str,
    duration_ms: int,
    lora_name: str,
    started_at: int,
    completed_at: int,
    transfer_bytes: int = 0,
    file_size_bytes: int | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "completedAt": completed_at,
        "disposition": disposition,
        "durationMs": max(0, int(duration_ms)),
        "modelName": lora_name,
        "modelType": "lora",
        "startedAt": started_at,
        "transferBytes": max(0, int(transfer_bytes)),
    }

    if file_size_bytes is not None:
        event["fileSizeBytes"] = max(0, int(file_size_bytes))
    if error_message:
        event["errorMessage"] = error_message

    return event


def store_request_telemetry(
    request_id: str,
    *,
    downloads: list[dict[str, Any]],
    generation_id: str | None = None,
) -> None:
    cache_hit_count = 0
    cache_miss_count = 0
    shared_cache_hit_count = 0
    total_download_bytes = 0
    total_download_duration_ms = 0

    for event in downloads:
        total_download_bytes += int(event.get("transferBytes", 0) or 0)
        total_download_duration_ms += int(event.get("durationMs", 0) or 0)

        disposition = event.get("disposition")
        if disposition == "downloaded":
            cache_miss_count += 1
        elif disposition == "cache_hit_existing":
            cache_hit_count += 1
        elif disposition == "cache_hit_after_lock":
            cache_hit_count += 1
            shared_cache_hit_count += 1

    payload: dict[str, Any] = {
        "allocationRule": MODEL_DOWNLOAD_ALLOCATION_RULE,
        "cacheHitCount": cache_hit_count,
        "cacheMissCount": cache_miss_count,
        "downloads": downloads,
        "requestId": request_id,
        "sharedCacheHitCount": shared_cache_hit_count,
        "totalDownloadBytes": total_download_bytes,
        "totalDownloadDurationMs": total_download_duration_ms,
    }
    if generation_id:
        payload["generationId"] = generation_id

    with REQUEST_TELEMETRY_LOCK:
        REQUEST_TELEMETRY[request_id] = payload


def pop_request_telemetry(request_id: str | None) -> dict[str, Any] | None:
    if not request_id:
        return None

    with REQUEST_TELEMETRY_LOCK:
        return REQUEST_TELEMETRY.pop(request_id, None)


def extract_request_id(input_payload: dict[str, Any]) -> str | None:
    request_id = input_payload.get("request_id")
    return request_id if isinstance(request_id, str) and request_id else None


def extract_generation_id(manifest_raw: Any) -> str | None:
    if not isinstance(manifest_raw, dict):
        return None

    generation_id = manifest_raw.get("generationId")
    return generation_id if isinstance(generation_id, str) and generation_id else None


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


def ensure_lora_downloaded(lora_name: str) -> tuple[Path, dict[str, Any]]:
    COMFY_LORA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = COMFY_LORA_DIR / lora_name
    lock_path = COMFY_LORA_DIR / f"{lora_name}.lock"
    started_at = now_ms()
    existed_before_lock = target_path.exists()

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        if target_path.exists():
            completed_at = now_ms()
            file_size_bytes = target_path.stat().st_size
            disposition = "cache_hit_existing" if existed_before_lock else "cache_hit_after_lock"
            return target_path, build_download_event(
                disposition=disposition,
                duration_ms=completed_at - started_at,
                lora_name=lora_name,
                started_at=started_at,
                completed_at=completed_at,
                file_size_bytes=file_size_bytes,
            )

        if not HF_LORA_TOKEN:
            raise RuntimeError("HF_TOKEN is required to download LoRAs.")

        log.info("Downloading LoRA '%s' from '%s'", lora_name, HF_LORA_REPO)
        downloaded_path = hf_hub_download(
            repo_id=HF_LORA_REPO,
            filename=lora_name,
            repo_type="model",
            local_dir=str(COMFY_LORA_DIR),
            local_dir_use_symlinks=False,
            token=HF_LORA_TOKEN,
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

    completed_at = now_ms()
    file_size_bytes = target_path.stat().st_size
    return target_path, build_download_event(
        disposition="downloaded",
        duration_ms=completed_at - started_at,
        lora_name=lora_name,
        started_at=started_at,
        completed_at=completed_at,
        transfer_bytes=file_size_bytes,
        file_size_bytes=file_size_bytes,
    )


def ensure_required_loras(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object.")

    input_payload = payload.get("input")
    if not isinstance(input_payload, dict):
        raise ValueError("payload.input must be an object.")
    request_id = extract_request_id(input_payload)

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
    generation_id = extract_generation_id(manifest_raw)
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
    downloads: list[dict[str, Any]] = []

    for lora_name in required_loras:
        started_at = now_ms()
        try:
            _, event = ensure_lora_downloaded(lora_name)
            downloads.append(event)
        except Exception as exc:
            downloads.append(
                build_download_event(
                    disposition="error",
                    duration_ms=now_ms() - started_at,
                    lora_name=lora_name,
                    started_at=started_at,
                    completed_at=now_ms(),
                    error_message=str(exc),
                )
            )
            if request_id:
                store_request_telemetry(request_id, downloads=downloads, generation_id=generation_id)
            raise

    if required_loras:
        log.info("LoRA set ready (%d): %s", len(required_loras), ", ".join(required_loras))

    if request_id:
        store_request_telemetry(request_id, downloads=downloads, generation_id=generation_id)

    input_payload["required_loras"] = required_loras
    payload["input"] = input_payload
    return payload


async def generate_response_with_telemetry(client_request: Any, model_response: Any) -> web.Response:
    request_id: str | None = None
    try:
        client_payload = await client_request.json()
        if isinstance(client_payload, dict):
            input_payload = client_payload.get("input")
            if isinstance(input_payload, dict):
                request_id = extract_request_id(input_payload)
    except Exception:
        request_id = None

    telemetry = pop_request_telemetry(request_id)
    response_body = await model_response.read()
    content_type = model_response.headers.get("Content-Type", "application/json")

    if telemetry is None:
        return web.Response(body=response_body, status=model_response.status, content_type=content_type)

    try:
        decoded = response_body.decode("utf-8")
        payload = json.loads(decoded)
    except Exception:
        return web.Response(body=response_body, status=model_response.status, content_type=content_type)

    if not isinstance(payload, dict):
        return web.Response(body=response_body, status=model_response.status, content_type=content_type)

    payload["workerTelemetry"] = telemetry
    return web.json_response(payload, status=model_response.status)


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
    if PROVISIONING_FAILED_MARKER.exists():
        raise RuntimeError(f"Provisioning failed: {PROVISIONING_FAILED_MARKER}")
    if not PROVISIONING_DONE_MARKER.exists():
        raise RuntimeError(f"Provisioning not complete: {PROVISIONING_DONE_MARKER}")

    health_url = f"{MODEL_HEALTHCHECK_BASE_URL}{MODEL_HEALTHCHECK_ENDPOINT}"
    timeout = aiohttp.ClientTimeout(total=10)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(health_url) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"Model healthcheck failed status={response.status} url={health_url} body={body[:500]}"
                )

    return {
        "ok": True,
        "healthcheck_url": health_url,
        "model_server_base_url": MODEL_SERVER_BASE_URL,
        "params": params,
    }


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
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=MAX_QUEUE_TIME,
            request_parser=ensure_required_loras,
            response_generator=generate_response_with_telemetry,
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
