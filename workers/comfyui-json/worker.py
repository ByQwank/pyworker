import json
import math
import os
import random
import sys

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# ComyUI model configuration
MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18288
MODEL_LOG_FILE = "/var/log/portal/comfyui.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# ComyUI-specific log messages
MODEL_LOAD_LOG_MSG = ["To see the GUI go to: "]

MODEL_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",
    "Value not in list: ",
    "[ERROR] Provisioning Script failed",
]

MODEL_INFO_LOG_MSGS = ['"message":"Downloading']

DEFAULT_WIDTH = 432
DEFAULT_HEIGHT = 768
DEFAULT_FRAMES = 17
DEFAULT_STEPS = 8

MAX_QUEUE_TIME = 900.0
WORKLOAD_MULTIPLIER = 0.6

BENCHMARK_WORKFLOW_PATH = os.path.join(
    os.path.dirname(__file__), "benchmark-wan22-fast.json"
)


def load_benchmark_workflow():
    try:
        with open(BENCHMARK_WORKFLOW_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        print(f"Failed to load benchmark workflow: {exc}")
        return None


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
    multiplier = WORKLOAD_MULTIPLIER if WORKLOAD_MULTIPLIER > 0 else 1.0
    return max(1.0, base_workload * multiplier)


def workload_calculator(payload):
    workflow = extract_workflow(payload)
    if not workflow:
        return estimate_workload({})
    return estimate_workload(workflow)


benchmark_workflow = load_benchmark_workflow()
benchmark_dataset = (
    [
        {
            "input": {
                "request_id": f"benchmark-{random.randint(1000, 99999)}",
                "workflow_json": benchmark_workflow,
            }
        }
    ]
    if benchmark_workflow
    else []
)

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=MAX_QUEUE_TIME,
            workload_calculator=workload_calculator,
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
            )
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()
