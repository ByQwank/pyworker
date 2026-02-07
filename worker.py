import fcntl
import hashlib
import hmac
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from vastai import BenchmarkConfig, HandlerConfig, LogActionConfig, Worker, WorkerConfig

# ComfyUI model configuration
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18288
MODEL_LOG_FILE = "/var/log/portal/comfyui.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# LoRA runtime configuration
HF_LORA_REPO = os.getenv("HF_LORA_REPO", "Dylaaann/Lora")
HF_LORA_TOKEN = os.getenv("HF_TOKEN")
COMFY_LORA_DIR = Path(os.getenv("COMFY_LORA_DIR", "/workspace/ComfyUI/models/loras"))
MANIFEST_SECRET = os.getenv("PYWORKER_MANIFEST_SECRET", "").strip()
MANIFEST_MAX_AGE_SECONDS = int(os.getenv("PYWORKER_MANIFEST_MAX_AGE_SECONDS", "900"))
REQUIRE_SIGNED_MANIFEST = os.getenv("PYWORKER_REQUIRE_MANIFEST", "false").lower() == "true"

log = logging.getLogger("custom-comfyui-worker")

# ComfyUI-specific log messages
MODEL_LOAD_LOG_MSG = ["To see the GUI go to: "]

MODEL_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",
    "Value not in list: ",
    "[ERROR] Provisioning Script failed",
]

MODEL_INFO_LOG_MSGS = ['"message":"Downloading']

IGNORED_LORA_NAMES = {"", "none", "null"}
LORA_INPUT_KEY_REGEX = re.compile(r"^lora(?:_\d+)?_name$", re.IGNORECASE)


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
    if not isinstance(issued_at, (int, float)):
        raise ValueError("lora_manifest.issuedAt must be a number.")
    if not isinstance(signature, str) or not signature:
        raise ValueError("lora_manifest.signature is required when PYWORKER_MANIFEST_SECRET is set.")

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
        if downloaded_file.exists() and downloaded_file != target_path and not target_path.exists():
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
                "Workflow requested LoRAs not allowed by signed manifest: " + ", ".join(unexpected)
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


benchmark_prompts = [
    "Cartoon hoodie hero; orc, anime cat, bunny; black goo; buff; vector on white.",
    "Cozy farming-game scene with fine details.",
    "2D vector child with soccer ball; airbrush chrome; swagger; antique copper.",
    "Realistic futuristic downtown of low buildings at sunset.",
    "Perfect wave front view; sunny seascape; ultra-detailed water; artful feel.",
    "Clear cup with ice, fruit, mint; creamy swirls; fluid-sim CGI; warm glow.",
    "Male biker with backpack on motorcycle; oilpunk; award-worthy magazine cover.",
]

benchmark_dataset = [
    {
        "input": {
            "request_id": f"benchmark-{random.randint(1000, 99999)}",
            "modifier": "Text2Image",
            "modifications": {
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "steps": 20,
                "seed": random.randint(0, sys.maxsize),
            },
        }
    }
    for prompt in benchmark_prompts
]

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=float(os.getenv("PYWORKER_MAX_QUEUE_TIME_SECONDS", "1800")),
            request_parser=ensure_required_loras,
            workload_calculator=lambda _payload: 100.0,
            benchmark_config=BenchmarkConfig(dataset=benchmark_dataset),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS,
    ),
)

if __name__ == "__main__":
    Worker(worker_config).run()
