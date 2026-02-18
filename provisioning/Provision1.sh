#!/bin/bash
# Custom Wan 2.2 I2V provisioning.
# Benchmark preparation/diagnostics are optional and controlled by PYWORKER_ENABLE_BENCHMARK.

set -euo pipefail

### Configuration ###
WORKSPACE_DIR="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_DIR}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
INPUTS_DIR="${COMFYUI_DIR}/input"
WORKFLOWS_DIR="${COMFYUI_DIR}/user/default/workflows"
HF_SEMAPHORE_DIR="${WORKSPACE_DIR}/hf_download_sem_$$"
HF_MAX_PARALLEL="${HF_MAX_PARALLEL:-1}"
HF_CURL_MAX_TIME_SEC="${HF_CURL_MAX_TIME_SEC:-5400}"
HF_STALL_SPEED_LIMIT_BYTES="${HF_STALL_SPEED_LIMIT_BYTES:-131072}"
HF_STALL_SPEED_TIME_SEC="${HF_STALL_SPEED_TIME_SEC:-180}"
MODEL_LOG="${MODEL_LOG:-/var/log/portal/comfyui.log}"
LORA_PATH="${MODELS_DIR}/loras"
PRELOAD_ALL_LORAS="${PRELOAD_ALL_LORAS:-false}"
ROUTE_REFERENCE_WORKLOAD="${ROUTE_REFERENCE_WORKLOAD:-1811}"
ROUTE_TARGET_COST_PER_REFERENCE_WORKLOAD="${ROUTE_TARGET_COST_PER_REFERENCE_WORKLOAD:-2}"
ROUTE_COST_MIN="${ROUTE_COST_MIN:-1}"
ROUTE_COST_MAX="${ROUTE_COST_MAX:-30}"
PYWORKER_ENABLE_BENCHMARK="${PYWORKER_ENABLE_BENCHMARK:-false}"

# Custom Nodes
NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/city96/ComfyUI-GGUF"
    "https://github.com/kijai/ComfyUI-WanVideoWrapper"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
    "https://github.com/SeanScripts/ComfyUI-Unload-Model"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/WASasquatch/was-node-suite-comfyui"
)

# Only these are required for your runtime workflow path.
CRITICAL_CUSTOM_NODES=(
    "ComfyUI-Easy-Use"
    "ComfyUI-WanVideoWrapper"
    "ComfyUI-KJNodes"
    "ComfyUI-VideoHelperSuite"
)

# HuggingFace Models: "URL|OUTPUT_PATH"
HF_MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors|$MODELS_DIR/vae/wan_2.1_vae.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors|$MODELS_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors|$MODELS_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    "https://huggingface.co/hfmaster/models-moved/resolve/cab6dcee2fbb05e190dbb8f536fbdaa489031a14/rife/rife49.pth|${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/models/rife/rife49.pth"
    # Runtime path used by many Comfy-VFI builds.
    "https://huggingface.co/hfmaster/models-moved/resolve/cab6dcee2fbb05e190dbb8f536fbdaa489031a14/rife/rife49.pth|${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth"
    # Always install benchmark-critical LoRAs up front.
    "https://huggingface.co/Dylaaann/Lora/resolve/main/high_4step.safetensors|$LORA_PATH/high_4step.safetensors"
    "https://huggingface.co/Dylaaann/Lora/resolve/main/low_4step.safetensors|$LORA_PATH/low_4step.safetensors"
)

### End Configuration ###

mkdir -p "$(dirname "$MODEL_LOG")"

log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$MODEL_LOG"
}

is_truthy() {
    local value="${1:-}"
    value="$(echo "$value" | tr '[:upper:]' '[:lower:]')"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

script_cleanup() {
    log "Cleaning up semaphore directory..."
    rm -rf "$HF_SEMAPHORE_DIR"
    find "$MODELS_DIR" -name "*.lock" -type f -mmin +60 -delete 2>/dev/null || true
    find "$INPUTS_DIR" -name "*.lock" -type f -mmin +60 -delete 2>/dev/null || true
}

script_error() {
    local exit_code=$?
    local line_number=$1
    log "[ERROR] Provisioning script failed at line $line_number with exit code $exit_code"
    exit "$exit_code"
}

trap script_cleanup EXIT
trap 'script_error $LINENO' ERR

acquire_slot() {
    local prefix="$1"
    local max_slots="$2"
    while true; do
        local count
        count=$(find "$(dirname "$prefix")" -name "$(basename "$prefix")_*" 2>/dev/null | wc -l)
        if [ "$count" -lt "$max_slots" ]; then
            local slot="${prefix}_$$_$RANDOM"
            touch "$slot"
            echo "$slot"
            return 0
        fi
        sleep 0.5
    done
}

release_slot() {
    rm -f "$1"
}

parse_hf_repo_and_path() {
    local url="$1"
    local repo file_path
    repo=$(echo "$url" | sed -n 's|https://huggingface.co/\([^/]*/[^/]*\)/resolve/.*|\1|p')
    file_path=$(echo "$url" | sed -n 's|https://huggingface.co/[^/]*/[^/]*/resolve/[^/]*/\(.*\)|\1|p')
    if [[ -z "$repo" || -z "$file_path" ]]; then
        return 1
    fi
    echo "$repo|$file_path"
}

download_hf_file() {
    local url="$1"
    local output_path="$2"
    local lockfile="${output_path}.lock"
    local max_retries=5
    local retry_delay=2

    local slot
    slot=$(acquire_slot "$HF_SEMAPHORE_DIR/hf" "$HF_MAX_PARALLEL")
    mkdir -p "$(dirname "$output_path")"

    local free_gb
    free_gb=$(df -BG "$WORKSPACE_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')
    if [[ -n "$free_gb" ]]; then
        log "[HF] Pre-download free disk: ${free_gb}G for $output_path"
    fi

    (
        if ! flock -x -w 300 200; then
            log "[ERROR] Could not acquire lock for $output_path after 300s"
            release_slot "$slot"
            exit 1
        fi

        if [ -f "$output_path" ]; then
            log "File already exists: $output_path (skipping)"
            release_slot "$slot"
            exit 0
        fi

        local parsed repo file_path
        if ! parsed=$(parse_hf_repo_and_path "$url"); then
            log "[ERROR] Invalid HuggingFace URL: $url"
            release_slot "$slot"
            exit 1
        fi
        repo="${parsed%%|*}"
        file_path="${parsed##*|}"

        local attempt=1
        local current_delay=$retry_delay
        local part_path="${output_path}.part"
        local curl_auth_header=()
        if [[ -n "${HF_TOKEN:-}" ]]; then
            curl_auth_header=(-H "Authorization: Bearer ${HF_TOKEN}")
        fi

        while [ $attempt -le $max_retries ]; do
            local partial_before=0
            if [[ -f "$part_path" ]]; then
                partial_before=$(stat -c%s "$part_path" 2>/dev/null || echo 0)
            fi

            log "Downloading $repo/$file_path (attempt $attempt/$max_retries, resume_from=${partial_before}B)..."

            if curl -fL \
                --retry 5 \
                --retry-delay 3 \
                --retry-all-errors \
                --connect-timeout 30 \
                --max-time "$HF_CURL_MAX_TIME_SEC" \
                --speed-limit "$HF_STALL_SPEED_LIMIT_BYTES" \
                --speed-time "$HF_STALL_SPEED_TIME_SEC" \
                "${curl_auth_header[@]}" \
                -C - \
                -o "$part_path" \
                "$url" 2>&1 | tee -a "$MODEL_LOG"; then
                if [[ -s "$part_path" ]]; then
                    mv "$part_path" "$output_path"
                    local free_after
                    free_after=$(df -BG "$WORKSPACE_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')
                    if [[ -n "$free_after" ]]; then
                        log "[HF] Post-download free disk: ${free_after}G for $output_path"
                    fi
                    release_slot "$slot"
                    log "[OK] Downloaded: $output_path"
                    exit 0
                fi
                log "Download reported success but file missing: $part_path"
            fi

            log "[WARN] curl download failed for $repo/$file_path; trying hf CLI fallback..."
            local temp_dir
            temp_dir=$(mktemp -d)

            if hf download "$repo" "$file_path" --local-dir "$temp_dir" --cache-dir "$temp_dir/.cache" 2>&1 | tee -a "$MODEL_LOG"; then
                if [ -f "$temp_dir/$file_path" ]; then
                    mv "$temp_dir/$file_path" "$output_path"
                    local free_after_fallback
                    free_after_fallback=$(df -BG "$WORKSPACE_DIR" | awk 'NR==2 {gsub("G","",$4); print $4}')
                    if [[ -n "$free_after_fallback" ]]; then
                        log "[HF] Post-download free disk: ${free_after_fallback}G for $output_path"
                    fi
                    rm -rf "$temp_dir"
                    rm -f "$part_path"
                    release_slot "$slot"
                    log "[OK] Downloaded via hf CLI fallback: $output_path"
                    exit 0
                fi
                log "hf CLI fallback reported success but file missing: $temp_dir/$file_path"
            fi
            rm -rf "$temp_dir"

            local partial_after=0
            if [[ -f "$part_path" ]]; then
                partial_after=$(stat -c%s "$part_path" 2>/dev/null || echo 0)
            fi

            log "Download failed (attempt $attempt/$max_retries, partial=${partial_after}B), retrying in ${current_delay}s..."
            sleep "$current_delay"
            current_delay=$((current_delay * 2))
            attempt=$((attempt + 1))
        done

        log "[ERROR] Failed to download $output_path after $max_retries attempts"
        release_slot "$slot"
        exit 1
    ) 200>"$lockfile"

    local result=$?
    rm -f "$lockfile"
    return $result
}

find_benchmark_json_path() {
    local candidates=(
        "${WORKSPACE_DIR}/vast-pyworker/workers/comfyui-json/misc/benchmark.json"
        "${WORKSPACE_DIR}/pyworker/workers/comfyui-json/misc/benchmark.json"
        "/workspace/vast-pyworker/workers/comfyui-json/misc/benchmark.json"
        "/workspace/pyworker/workers/comfyui-json/misc/benchmark.json"
    )

    local path
    for path in "${candidates[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    path=$(find "$WORKSPACE_DIR" -type f -path "*/workers/comfyui-json/misc/benchmark.json" 2>/dev/null | head -n 1 || true)
    if [[ -n "$path" ]]; then
        echo "$path"
        return 0
    fi

    return 1
}

create_benchmark_image() {
    local target="$1"
    if [[ -f "$target" ]]; then
        log "[BENCHMARK] benchmark image already exists at $target"
        return 0
    fi

    mkdir -p "$(dirname "$target")"
    # 1x1 transparent PNG
    echo 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=' | base64 -d > "$target"
    log "[BENCHMARK] created benchmark image at $target"
}

ensure_lora_present() {
    local lora_name="$1"
    local target_lora_path="${LORA_PATH}/${lora_name}"

    if [[ -f "$target_lora_path" ]]; then
        log "[BENCHMARK] required LoRA present: $target_lora_path"
        return 0
    fi

    log "[BENCHMARK] downloading required benchmark LoRA: $lora_name"
    local temp_dir
    temp_dir=$(mktemp -d)

    if hf download Dylaaann/Lora "$lora_name" --repo-type model --local-dir "$temp_dir" 2>&1 | tee -a "$MODEL_LOG"; then
        if [[ -f "$temp_dir/$lora_name" ]]; then
            mkdir -p "$LORA_PATH"
            mv "$temp_dir/$lora_name" "$target_lora_path"
            rm -rf "$temp_dir"
            log "[BENCHMARK] downloaded required LoRA: $target_lora_path"
            return 0
        fi
    fi

    # fallback direct URL
    local direct_url="https://huggingface.co/Dylaaann/Lora/resolve/main/${lora_name}"
    if curl -fL --retry 5 --retry-delay 3 --retry-all-errors "$direct_url" -o "$target_lora_path" 2>&1 | tee -a "$MODEL_LOG"; then
        rm -rf "$temp_dir"
        log "[BENCHMARK] downloaded required LoRA via direct URL: $target_lora_path"
        return 0
    fi

    rm -rf "$temp_dir"
    log "[ERROR] Failed to download required benchmark LoRA: $lora_name"
    return 1
}

ensure_benchmark_runtime_assets() {
    local benchmark_path
    if ! benchmark_path=$(find_benchmark_json_path); then
        log "[ERROR] benchmark.json not found under workspace; cannot prepare benchmark assets"
        return 1
    fi

    local benchmark_image_path="${INPUTS_DIR}/benchmark.png"
    create_benchmark_image "$benchmark_image_path"

    local prep_output
    if ! prep_output=$(python - "$benchmark_path" <<'PY'
import json
import re
import sys

benchmark_path = sys.argv[1]
with open(benchmark_path, "r", encoding="utf-8") as f:
    workflow = json.load(f)

patched = 0
loras = set()
ignored = {"", "none", "null", "undefined"}
lora_key = re.compile(r"^lora(?:_\d+)?_name$", re.IGNORECASE)

for node in workflow.values():
    if not isinstance(node, dict):
        continue
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        continue

    class_type = str(node.get("class_type") or "")
    if class_type == "LoadImage":
        image = inputs.get("image")
        image_value = str(image).strip().lower() if image is not None else ""
        if image is None or image_value in ignored:
            inputs["image"] = "benchmark.png"
            patched += 1

    for k, v in inputs.items():
        if not isinstance(v, str):
            continue
        if not (lora_key.match(k) or str(k).lower() == "lora_name"):
            continue
        name = v.strip()
        if name.lower() in ignored:
            continue
        loras.add(name)

# Enforce benchmark profile used for autoscaler calibration:
# - true 2-2-2 stage split on KSamplerAdvanced windows
# - short 1s clip (17 frames) to keep benchmark runtime low
defaults = {
    "565": 1,  # Video Length in Seconds
    "632": 2,  # 1st End Step
    "633": 4,  # 2nd End Step
    "634": 6,  # Last End Step
}
updated_defaults = []
for node_id, expected in defaults.items():
    node = workflow.get(node_id)
    if not isinstance(node, dict):
        continue
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        continue
    current = inputs.get("value")
    if current != expected:
        inputs["value"] = expected
        updated_defaults.append(f"{node_id}:{current}->{expected}")

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(workflow, f, indent=2)

if patched > 0:
    print(f"[BENCHMARK] patched LoadImage nodes with invalid 'undefined' image: {patched}")
else:
    print("[BENCHMARK] no invalid LoadImage image values found")

sorted_loras = sorted(loras)
if sorted_loras:
    print(f"[BENCHMARK] required benchmark loras ({len(sorted_loras)}): {', '.join(sorted_loras)}")
else:
    print("[BENCHMARK] required benchmark loras: none")

if updated_defaults:
    print(f"[BENCHMARK] normalized benchmark profile: {', '.join(updated_defaults)}")
else:
    print("[BENCHMARK] benchmark profile already normalized (2-2-2, short clip)")

print("__BENCHMARK_LORAS__=" + ",".join(sorted_loras))
PY
); then
        log "[ERROR] Failed patching benchmark runtime assets"
        return 1
    fi

    echo "$prep_output" | tee -a "$MODEL_LOG"

    local required_loras_csv
    required_loras_csv=$(echo "$prep_output" | sed -n 's/^__BENCHMARK_LORAS__=//p' | tail -n 1)

    if [[ -n "$required_loras_csv" ]]; then
        IFS=',' read -r -a required_loras <<< "$required_loras_csv"
        local lora
        for lora in "${required_loras[@]}"; do
            lora=$(echo "$lora" | xargs)
            [[ -z "$lora" ]] && continue
            ensure_lora_present "$lora"
        done
    fi

    log "[BENCHMARK] runtime asset prep complete"
}

log_benchmark_diagnostics() {
    local benchmark_path
    if ! benchmark_path=$(find_benchmark_json_path); then
        log "[BENCHMARK] benchmark.json not found in expected pyworker paths; skipping diagnostics."
        return 0
    fi

    local checksum size_bytes
    checksum=$(sha256sum "$benchmark_path" | awk '{print $1}')
    size_bytes=$(wc -c < "$benchmark_path" | xargs)

    log "[BENCHMARK] benchmark_path=$benchmark_path"
    log "[BENCHMARK] benchmark_sha256=$checksum size_bytes=$size_bytes"
    log "[BENCHMARK] route_cost_tuning reference_workload=${ROUTE_REFERENCE_WORKLOAD} target_cost=${ROUTE_TARGET_COST_PER_REFERENCE_WORKLOAD} min=${ROUTE_COST_MIN} max=${ROUTE_COST_MAX} one_worker_mode=$([[ \"$ROUTE_COST_MAX\" == \"1\" ]] && echo true || echo false)"

    python - "$benchmark_path" "$ROUTE_REFERENCE_WORKLOAD" "$ROUTE_TARGET_COST_PER_REFERENCE_WORKLOAD" "$ROUTE_COST_MIN" "$ROUTE_COST_MAX" <<'PY' 2>&1 | tee -a "$MODEL_LOG"
import json
import math
import re
import sys
from collections import Counter

benchmark_path = sys.argv[1]
reference = float(sys.argv[2]) if len(sys.argv) > 2 else 1811.0
target = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
min_cost = int(float(sys.argv[4])) if len(sys.argv) > 4 else 1
max_cost = int(float(sys.argv[5])) if len(sys.argv) > 5 else 1

with open(benchmark_path, "r", encoding="utf-8") as f:
    workflow = json.load(f)


def parse_num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def resolve_num(inp, wf, visited=None, depth=0):
    if visited is None:
        visited = set()
    if depth > 6:
        return None
    direct = parse_num(inp)
    if direct is not None:
        return direct
    if isinstance(inp, list) and inp:
        node_id = str(inp[0])
        if node_id in visited:
            return None
        visited.add(node_id)
        node = wf.get(node_id) or {}
        ni = node.get("inputs") or {}

        val = parse_num(ni.get("value"))
        if val is not None:
            return val

        a = resolve_num(ni.get("a"), wf, visited, depth + 1)
        b = resolve_num(ni.get("b"), wf, visited, depth + 1)
        op = str(ni.get("operation") or "").lower()
        if a is not None and b is not None:
            if op in ("add", "+"):
                return a + b
            if op in ("subtract", "-"):
                return a - b
            if op in ("multiply", "*"):
                return a * b
            if op in ("divide", "/"):
                return None if b == 0 else a / b

        expr = str(ni.get("value") or "").replace(" ", "")
        if a is not None and b is not None:
            if "a*b" in expr:
                return a * b
            if "a+b" in expr:
                return a + b
            if "a-b" in expr:
                return a - b
            if "a/b" in expr:
                return None if b == 0 else a / b
    return None

nodes = list(workflow.values())
print(f"[BENCHMARK] workflow_nodes={len(workflow)}")

wan_node = None
for n in nodes:
    if not isinstance(n, dict):
        continue
    if n.get("class_type") == "WanImageToVideo" or ((n.get("_meta") or {}).get("title") == "WanImageToVideo"):
        wan_node = n
        break

width = resolve_num((wan_node or {}).get("inputs", {}).get("width"), workflow) or 432
height = resolve_num((wan_node or {}).get("inputs", {}).get("height"), workflow) or 768
frames = resolve_num((wan_node or {}).get("inputs", {}).get("length"), workflow) or 17

samplers = []
for node_id, n in workflow.items():
    if not isinstance(n, dict):
        continue
    if n.get("class_type") != "KSamplerAdvanced":
        continue
    ni = n.get("inputs") or {}
    start = resolve_num(ni.get("start_at_step"), workflow)
    end = resolve_num(ni.get("end_at_step"), workflow)
    steps = resolve_num(ni.get("steps"), workflow)
    title = ((n.get("_meta") or {}).get("title") or f"KSamplerAdvanced:{node_id}")
    samplers.append((node_id, title, start, end, steps))

samplers.sort(key=lambda t: ((t[2] if t[2] is not None else 1e9), (t[3] if t[3] is not None else 1e9)))
steps_total = 8.0
for _, _, _, end, steps in samplers:
    candidate = end if end is not None else steps
    if candidate is not None:
        steps_total = max(steps_total, candidate)

w = int(width)
h = int(height)
f = int(frames)
st = float(steps_total)
grid_w = max(1, math.ceil(w / 512))
grid_h = max(1, math.ceil(h / 512))
base_workload = grid_w * grid_h * f * st

class_types = []
for n in nodes:
    if not isinstance(n, dict):
        continue
    ct = n.get("class_type")
    if isinstance(ct, str) and ct:
        class_types.append(ct)

lower_ct = [c.lower() for c in class_types]
has_rife = any("rife" in c for c in lower_ct)
has_upscale = any("upscale" in c for c in lower_ct)
video_combines = sum(1 for c in lower_ct if "video" in c and "combine" in c)

adjusted = base_workload
if has_rife:
    adjusted *= 1.35
if has_upscale:
    adjusted *= 1.25
if video_combines > 1:
    adjusted *= 1 + (video_combines - 1) * 0.15

multiplier = 0.6
final_workload = adjusted * multiplier

print(f"[BENCHMARK] wan_dimensions={w}x{h} frames={f} steps_total={int(st) if st.is_integer() else st} grid={grid_w}x{grid_h}")
print(f"[BENCHMARK] class_flags rife={has_rife} upscale={has_upscale} video_combines={video_combines}")

counts = Counter(class_types)
top = ", ".join([f"{k}:{v}" for k, v in counts.most_common(10)])
if top:
    print(f"[BENCHMARK] class_types_top={top}")

print(f"[BENCHMARK] sampler_stages_count={len(samplers)}")
for node_id, title, start, end, steps in samplers:
    print(f"[BENCHMARK] sampler_stage {title}({node_id}):start={start} end={end} steps={steps}")

lora_re = re.compile(r"^lora(?:_\d+)?_name$", re.IGNORECASE)
ignored = {"", "none", "null", "undefined"}
loras = set()
for n in nodes:
    if not isinstance(n, dict):
        continue
    ni = n.get("inputs")
    if not isinstance(ni, dict):
        continue
    for k, v in ni.items():
        if not isinstance(v, str):
            continue
        if not (lora_re.match(str(k)) or str(k).lower() == "lora_name"):
            continue
        if v.strip().lower() in ignored:
            continue
        loras.add(v.strip())

if loras:
    print(f"[BENCHMARK] active_loras_count={len(loras)} names={', '.join(sorted(loras))}")
else:
    print("[BENCHMARK] active_loras_count=0")

scale = (target / reference) if reference > 0 else 1
scaled = round(final_workload * scale)
clamped = max(min_cost, min(max_cost, scaled if scaled > 0 else min_cost))

print(f"[BENCHMARK] workload base={base_workload:g} adjusted={adjusted:g} multiplier={multiplier} final={final_workload:g}")
print(f"[BENCHMARK] route_cost_prediction scaled={scaled} clamped={clamped} (min={min_cost}, max={max_cost})")
PY
    local diag_status=${PIPESTATUS[0]}
    if [[ "$diag_status" -ne 0 ]]; then
        log "[BENCHMARK] diagnostics python failed"
    fi
}

provisioning_install_python_deps() {
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
    export PATH="$HOME/.local/bin:$PATH"

    local pip_cmd="pip"
    if command -v uv &> /dev/null; then pip_cmd="uv pip"; fi

    $pip_cmd install --upgrade pip >/dev/null 2>&1
    $pip_cmd install "huggingface_hub[hf_transfer]" gguf color-matcher colorama scipy matplotlib >/dev/null 2>&1
    $pip_cmd install "torchvision>=0.18" "torchaudio>=2.3" "transformers>=4.40" "diffusers>=0.28" "accelerate>=0.30" "einops>=0.8" >/dev/null 2>&1
    $pip_cmd install "numpy<2.3.0,>=2.1.0" "librosa>=0.11.0" "numba>=0.60.0" >/dev/null 2>&1

    log "Installing ComfyUI-Easy-Use core dependencies..."
    $pip_cmd install lark sentencepiece spandrel peft opencv-python-headless "clip_interrogator>=0.6.0" 2>&1 | tee -a "$MODEL_LOG" || true

    if command -v nvidia-smi &> /dev/null; then
        $pip_cmd install onnxruntime-gpu cupy-cuda12x >/dev/null 2>&1 || true
    else
        $pip_cmd install onnxruntime >/dev/null 2>&1 || true
    fi
}

provisioning_run_node_installers() {
    local repo dir path
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        if [[ -e "${path}/install.py" ]]; then
            log "Running install.py for ${dir}..."
            if ! python "${path}/install.py" 2>&1 | tee -a "$MODEL_LOG"; then
                log "[WARN] install.py for ${dir} had issues (non-fatal)"
            fi
        fi
    done
}

provisioning_install_node_requirements() {
    local req_files
    req_files=$(find "${COMFYUI_DIR}/custom_nodes" -name "requirements.txt" 2>/dev/null)
    if [[ -z "$req_files" ]]; then return; fi

    local pip_cmd="pip"
    if command -v uv &> /dev/null; then pip_cmd="uv pip"; fi

    local req_file node_name
    for req_file in $req_files; do
        node_name=$(basename "$(dirname "$req_file")")
        log "Installing requirements for ${node_name}..."
        if ! $pip_cmd install --no-cache-dir -r "$req_file" 2>&1 | tee -a "$MODEL_LOG"; then
            log "[WARN] Some requirements for ${node_name} failed (non-fatal)"
        fi
    done
}

download_loras() {
    log "Downloading LoRAs from Dylaaann/Lora..."
    mkdir -p "$LORA_PATH"

    # Exclude benchmark-critical 4step LoRAs from bulk preload because they are
    # installed explicitly via HF_MODELS above.
    if ! hf download Dylaaann/Lora --local-dir "$LORA_PATH" --repo-type model --exclude "*.md" ".git*" "high_4step.safetensors" "low_4step.safetensors" 2>&1 | tee -a "$MODEL_LOG"; then
        log "[ERROR] Failed to download LoRAs"
        return 1
    fi

    find "$LORA_PATH" -mindepth 2 -type f -name "*.safetensors" -exec mv {} "$LORA_PATH" \; 2>/dev/null
    find "$LORA_PATH" -type d -empty -delete 2>/dev/null

    local safetensors_count
    safetensors_count=$(find "$LORA_PATH" -maxdepth 1 -type f -name "*.safetensors" 2>/dev/null | wc -l | awk '{print $1}')
    if [[ -z "$safetensors_count" ]] || [[ "$safetensors_count" -lt 1 ]]; then
        log "[ERROR] No .safetensors LoRAs found in $LORA_PATH"
        return 1
    fi

    local required_loras=("high_4step.safetensors" "low_4step.safetensors")
    local lora
    for lora in "${required_loras[@]}"; do
        if [[ ! -f "$LORA_PATH/$lora" ]]; then
            log "[ERROR] Missing required LoRA: $lora (expected at $LORA_PATH/$lora)"
            return 1
        fi
    done

    log "[OK] LoRAs downloaded successfully ($safetensors_count files)"
}

clone_custom_nodes() {
    log "Cloning ${#NODES[@]} custom nodes..."
    local node_pids=()
    local repo dir path

    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        (
            local max_attempts=4
            local attempt=1
            local delay_seconds=2
            local ok=0

            while [[ "$attempt" -le "$max_attempts" ]]; do
                if [[ -d "$path/.git" ]]; then
                    if git -C "$path" pull --depth 1 >/dev/null 2>&1 || git -C "$path" pull >/dev/null 2>&1; then
                        ok=1
                        break
                    fi
                elif [[ -d "$path" ]]; then
                    log "[WARN] ${dir} exists but is not a git repo; replacing directory before clone"
                    rm -rf "$path"
                    if git clone --depth 1 --single-branch "$repo" "$path" --recursive >/dev/null 2>&1 || \
                       git clone "$repo" "$path" --recursive >/dev/null 2>&1; then
                        ok=1
                        break
                    fi
                else
                    if git clone --depth 1 --single-branch "$repo" "$path" --recursive >/dev/null 2>&1 || \
                       git clone "$repo" "$path" --recursive >/dev/null 2>&1; then
                        ok=1
                        break
                    fi
                fi

                log "[WARN] Clone/update failed for ${dir} (attempt ${attempt}/${max_attempts})"
                if [[ "$attempt" -lt "$max_attempts" ]]; then
                    sleep "$delay_seconds"
                    delay_seconds=$((delay_seconds * 2))
                fi
                attempt=$((attempt + 1))
            done

            if [[ "$ok" -ne 1 ]]; then
                local critical=0
                local required
                for required in "${CRITICAL_CUSTOM_NODES[@]}"; do
                    if [[ "$required" == "$dir" ]]; then
                        critical=1
                        break
                    fi
                done

                if [[ "$critical" -eq 1 ]]; then
                    log "[ERROR] Clone/update failed for critical node ${dir} after ${max_attempts} attempts"
                    exit 1
                fi

                log "[WARN] Clone/update failed for non-critical node ${dir} after ${max_attempts} attempts; continuing"
                exit 0
            fi
        ) &
        node_pids+=($!)
    done

    local clone_failed=0
    local pid
    for pid in "${node_pids[@]}"; do
        if ! wait "$pid"; then
            clone_failed=1
        fi
    done

    if [[ "$clone_failed" -eq 1 ]]; then
        log "[ERROR] One or more custom node clones failed"
        return 1
    fi

    log "[OK] Custom nodes cloned"
}

set_cleanup_job() {
    local script_dir="/opt/instance-tools/bin"
    local script_path="${script_dir}/clean-output.sh"

    mkdir -p "$script_dir"

    if [[ ! -f "$script_path" ]]; then
        cat > "$script_path" << 'CLEAN_OUTPUT'
#!/bin/bash
output_dir="${WORKSPACE:-/workspace}/ComfyUI/output/"
min_free_mb=512
available_space=$(df -m "${output_dir}" | awk 'NR==2 {print $4}')
if [[ "$available_space" -lt "$min_free_mb" ]]; then
    oldest=$(find "${output_dir}" -mindepth 1 -type f -printf "%T@\n" 2>/dev/null | sort -n | head -1 | awk '{printf "%.0f", $1}')
    if [[ -n "$oldest" ]]; then
        cutoff=$(awk "BEGIN {printf \"%.0f\", ${oldest}+86400}")
        find "${output_dir}" -mindepth 1 -type f ! -newermt "@${cutoff}" -delete
        find "${output_dir}" -mindepth 1 -xtype l -delete
        find "${output_dir}" -mindepth 1 -type d -empty -delete
    fi
fi
CLEAN_OUTPUT
        chmod +x "$script_path"
    fi

    local cron_exists=0
    if crontab -l 2>/dev/null | grep -qF 'clean-output.sh'; then
        cron_exists=1
    fi

    if [[ "$cron_exists" -eq 0 ]]; then
        (crontab -l 2>/dev/null || true; echo "*/10 * * * * ${script_path}") | crontab -
    fi
}

main() {
    log "Starting custom ComfyUI provisioning..."

    if [ -f /venv/main/bin/activate ]; then
        . /venv/main/bin/activate
    fi

    rm -rf "$HF_SEMAPHORE_DIR"
    mkdir -p "$HF_SEMAPHORE_DIR"
    mkdir -p "$WORKFLOWS_DIR" "$INPUTS_DIR"
    mkdir -p "$MODELS_DIR"/{checkpoints,text_encoders,diffusion_models,vae,loras}
    mkdir -p "${COMFYUI_DIR}/custom_nodes"

    log "Phase 1: Installing Python dependencies..."
    provisioning_install_python_deps

    export HF_HUB_ENABLE_HF_TRANSFER=1
    if [[ -n "${HF_TOKEN:-}" ]]; then
        log "Logging into HuggingFace..."
        hf auth login --token "$HF_TOKEN" 2>/dev/null || \
        huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    fi

    set_cleanup_job
    log "Phase 2: Starting model downloads (HF_MAX_PARALLEL=${HF_MAX_PARALLEL})..."
    local pids=()
    local model url output_path
    for model in "${HF_MODELS[@]}"; do
        url="${model%%|*}"
        output_path="${model##*|}"
        url=$(echo "$url" | xargs)
        output_path=$(echo "$output_path" | xargs)
        log "Queuing HF download: $url -> $output_path"
        download_hf_file "$url" "$output_path" &
        pids+=($!)
    done

    if [[ "$PRELOAD_ALL_LORAS" == "true" ]]; then
        download_loras &
        pids+=($!)
    else
        log "Skipping bulk LoRA preload (PRELOAD_ALL_LORAS=false); LoRAs will be downloaded on demand by PyWorker."
    fi

    clone_custom_nodes &
    pids+=($!)

    log "Waiting for ${#pids[@]} parallel operations..."
    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            log "[ERROR] Process $pid failed"
            failed=1
        fi
    done

    if [[ $failed -eq 1 ]]; then
        log "[ERROR] One or more downloads failed"
        exit 1
    fi

    if is_truthy "$PYWORKER_ENABLE_BENCHMARK"; then
        log "Phase 2.5: Ensuring benchmark runtime assets..."
        if ! ensure_benchmark_runtime_assets; then
            log "[ERROR] Benchmark runtime asset preparation failed"
            exit 1
        fi
    else
        log "Phase 2.5: Skipping benchmark runtime assets (PYWORKER_ENABLE_BENCHMARK=${PYWORKER_ENABLE_BENCHMARK})"
    fi

    log "Phase 3: Setting up custom nodes..."
    provisioning_run_node_installers
    provisioning_install_node_requirements

    # Ensure RIFE exists in both expected paths across node versions.
    local rife_model_src="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/models/rife/rife49.pth"
    local rife_ckpt_dst="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth"
    if [[ -f "$rife_model_src" && ! -f "$rife_ckpt_dst" ]]; then
        mkdir -p "$(dirname "$rife_ckpt_dst")"
        cp -f "$rife_model_src" "$rife_ckpt_dst"
        log "Seeded RIFE checkpoint into ckpts path: $rife_ckpt_dst"
    fi

    log "Phase 4: Verifying critical node installations..."
    local node
    for node in "${CRITICAL_CUSTOM_NODES[@]}"; do
        if [[ -d "${COMFYUI_DIR}/custom_nodes/${node}" ]]; then
            log "[OK] ${node} directory exists"
            if [[ -f "${COMFYUI_DIR}/custom_nodes/${node}/__init__.py" ]]; then
                log "[OK] ${node} has __init__.py"
            else
                log "[WARN] ${node} missing __init__.py"
            fi
        else
            log "[ERROR] Critical node ${node} not found"
        fi
    done

    log "[OK] All provisioning completed successfully"

    local NODE_COUNT LORA_COUNT MODEL_COUNT
    NODE_COUNT=$(ls -d "${COMFYUI_DIR}/custom_nodes"/*/ 2>/dev/null | wc -l)
    LORA_COUNT=$(find "$LORA_PATH" -name "*.safetensors" 2>/dev/null | wc -l)
    MODEL_COUNT=$(find "$MODELS_DIR" -name "*.safetensors" 2>/dev/null | wc -l)

    log "Custom Nodes: $NODE_COUNT"
    log "LoRAs: $LORA_COUNT"
    log "Models: $MODEL_COUNT"

    if is_truthy "$PYWORKER_ENABLE_BENCHMARK"; then
        log "Phase 5: Benchmark diagnostics..."
        log_benchmark_diagnostics
    else
        log "Phase 5: Benchmark diagnostics skipped (PYWORKER_ENABLE_BENCHMARK=${PYWORKER_ENABLE_BENCHMARK})"
    fi
}

main
