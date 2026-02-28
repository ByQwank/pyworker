#!/bin/bash
# Custom Wan 2.2 I2V Provisioning for vast.ai
# Parallel downloads via aria2c + hf_transfer for maximum bandwidth utilization

set -e -o pipefail

source /venv/main/bin/activate
WORKSPACE_ROOT="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_ROOT}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
LORA_PATH="${MODELS_DIR}/loras"
MODEL_LOG="${MODEL_LOG:-/var/log/portal/comfyui.log}"
PROVISIONING_DONE_MARKER="${PROVISIONING_DONE_MARKER:-${WORKSPACE_ROOT}/.provisioning-complete}"
PROVISIONING_FAILED_MARKER="${PROVISIONING_FAILED_MARKER:-${WORKSPACE_ROOT}/.provisioning-failed}"

APT_PACKAGES=(
    "aria2"
    "rsync"
)

PIP_PACKAGES=(
    "huggingface_hub[hf_transfer]"
    "lark"
    "sentencepiece"
    "opencv-python-headless"
    "spandrel"
    "peft"
    "clip_interrogator>=0.6.0"
    "color-matcher"
    "colorama"
    "scipy"
    "matplotlib"
    "gguf"
    "einops>=0.8"
)

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

WORKFLOWS=(
)

CHECKPOINT_MODELS=(
)

UNET_MODELS=(
)

# Preload the 4-step lighting LoRAs used in every generation
# Other LoRAs are downloaded on demand by pyworker via ensure_lora_downloaded()
LORA_MODELS=(
    "https://huggingface.co/Dylaaann/Lora/resolve/main/high_4step.safetensors"
    "https://huggingface.co/Dylaaann/Lora/resolve/main/low_4step.safetensors"
)

VAE_MODELS=(
)

ESRGAN_MODELS=(
)

CONTROLNET_MODELS=(
)

# HuggingFace models downloaded via hf_transfer for max speed
# Format: "URL|OUTPUT_PATH"
HF_MODELS=(
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|${MODELS_DIR}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors|${MODELS_DIR}/vae/wan_2.1_vae.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors|${MODELS_DIR}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors|${MODELS_DIR}/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
)

# RIFE model for frame interpolation (goes into custom node, not models dir)
RIFE_URL="https://huggingface.co/hfmaster/models-moved/resolve/cab6dcee2fbb05e190dbb8f536fbdaa489031a14/rife/rife49.pth"
RIFE_PATH="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/models/rife/rife49.pth"

### DO NOT EDIT BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###

log() {
    printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$1"
}

function provisioning_start() {
    provisioning_print_header
    rm -f "$PROVISIONING_DONE_MARKER" "$PROVISIONING_FAILED_MARKER"

    # Phase 1: System deps + pip (sequential, fast)
    provisioning_get_apt_packages || return 1
    provisioning_get_pip_packages || return 1

    # Enable HuggingFace fast transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1

    if [[ -n "${HF_TOKEN:-}" ]]; then
        log "Logging into HuggingFace..."
        huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    fi

    # Phase 2: Clone all custom nodes in parallel
    provisioning_get_nodes_parallel || return 1

    # Phase 3: Install node requirements (must be after nodes are cloned)
    provisioning_install_node_requirements || return 1

    # Phase 4: Download ALL models in parallel using aria2c
    provisioning_download_all_models_parallel || return 1

    # Phase 5: Verify critical files
    provisioning_verify || return 1

    touch "$PROVISIONING_DONE_MARKER"
    provisioning_print_end
}

function provisioning_get_apt_packages() {
    if ((${#APT_PACKAGES[@]} > 0)); then
        local sudo_prefix=()

        if [[ $(id -u) -ne 0 ]]; then
            if ! command -v sudo >/dev/null 2>&1; then
                log "[ERROR] sudo is required to install APT packages"
                return 1
            fi
            sudo_prefix=(sudo)
        fi

        log "Installing APT packages: ${APT_PACKAGES[*]}"
        "${sudo_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update
        "${sudo_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"

        if ! command -v aria2c >/dev/null 2>&1; then
            log "[ERROR] aria2c is still unavailable after APT install"
            return 1
        fi
    fi
}

function provisioning_get_pip_packages() {
    if ((${#PIP_PACKAGES[@]} > 0)); then
        log "Installing PIP packages..."
        pip install --no-cache-dir "${PIP_PACKAGES[@]}"
    fi
}

function clone_or_update_repo() {
    local repo="$1"
    local path="$2"
    local attempt

    if [[ -d "$path/.git" ]]; then
        if [[ ${AUTO_UPDATE,,} == "false" ]]; then
            return 0
        fi

        for attempt in 1 2 3; do
            if (
                cd "$path" &&
                git pull --ff-only >/dev/null 2>&1 &&
                git submodule update --init --recursive >/dev/null 2>&1
            ); then
                return 0
            fi
            sleep $((attempt * 2))
        done
        return 1
    fi

    rm -rf "$path"
    for attempt in 1 2 3; do
        if git clone --depth 1 --single-branch --recursive --jobs 8 "$repo" "$path" >/dev/null 2>&1; then
            return 0
        fi

        rm -rf "$path"
        if git clone --recursive --jobs 8 "$repo" "$path" >/dev/null 2>&1; then
            return 0
        fi

        rm -rf "$path"
        sleep $((attempt * 2))
    done

    return 1
}

# Clone all custom nodes in parallel (much faster than sequential)
function provisioning_get_nodes_parallel() {
    log "Cloning ${#NODES[@]} custom nodes in parallel..."
    local pids=()
    local failed=0

    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        (
            if ! clone_or_update_repo "$repo" "$path"; then
                log "[ERROR] Failed to prepare custom node ${dir}"
                exit 1
            fi
        ) &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid" 2>/dev/null; then
            log "[ERROR] Node clone PID $pid failed"
            failed=1
        fi
    done

    if [[ $failed -ne 0 ]]; then
        log "[ERROR] One or more custom node clones failed"
        return 1
    fi

    log "✓ Custom nodes cloned"
}

# Install requirements for all custom nodes
function provisioning_install_node_requirements() {
    log "Installing custom node requirements..."
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -e $requirements ]]; then
            pip install --no-cache-dir -r "$requirements" 2>&1 || \
                log "[WARN] Some requirements for ${dir} failed (non-fatal)"
        fi
        if [[ -e "${path}/install.py" ]]; then
            python "${path}/install.py" 2>&1 || \
                log "[WARN] install.py for ${dir} had issues (non-fatal)"
        fi
    done
    log "✓ Node requirements installed"
}

# Build an aria2c input file and download everything at once
# aria2c uses multiple connections per file + parallel files = saturates bandwidth
function provisioning_download_all_models_parallel() {
    log "Preparing parallel model downloads..."

    # Create all target directories
    mkdir -p "${MODELS_DIR}/text_encoders"
    mkdir -p "${MODELS_DIR}/diffusion_models"
    mkdir -p "${MODELS_DIR}/vae"
    mkdir -p "${MODELS_DIR}/loras"
    mkdir -p "$(dirname "$RIFE_PATH")"

    local aria2_input=$(mktemp)
    local auth_header=""

    if [[ -n "${HF_TOKEN:-}" ]]; then
        auth_header="Authorization: Bearer ${HF_TOKEN}"
    fi

    # Add HF models to aria2 input
    for model in "${HF_MODELS[@]}"; do
        local url="${model%%|*}"
        local output_path="${model##*|}"
        local dir=$(dirname "$output_path")
        local filename=$(basename "$output_path")

        # Skip if already downloaded
        if [[ -f "$output_path" ]]; then
            log "Already exists: $filename (skipping)"
            continue
        fi

        echo "$url" >> "$aria2_input"
        echo "  dir=$dir" >> "$aria2_input"
        echo "  out=$filename" >> "$aria2_input"
        if [[ -n "$auth_header" ]]; then
            echo "  header=$auth_header" >> "$aria2_input"
        fi
    done

    # Add RIFE model
    if [[ ! -f "$RIFE_PATH" ]]; then
        echo "$RIFE_URL" >> "$aria2_input"
        echo "  dir=$(dirname "$RIFE_PATH")" >> "$aria2_input"
        echo "  out=$(basename "$RIFE_PATH")" >> "$aria2_input"
        if [[ -n "$auth_header" ]]; then
            echo "  header=$auth_header" >> "$aria2_input"
        fi
    fi

    # Add LoRA models
    for url in "${LORA_MODELS[@]}"; do
        local filename=$(basename "$url")
        if [[ -f "${MODELS_DIR}/loras/$filename" ]]; then
            log "Already exists: $filename (skipping)"
            continue
        fi
        echo "$url" >> "$aria2_input"
        echo "  dir=${MODELS_DIR}/loras" >> "$aria2_input"
        echo "  out=$filename" >> "$aria2_input"
        if [[ -n "$auth_header" ]]; then
            echo "  header=$auth_header" >> "$aria2_input"
        fi
    done

    # Check if there's anything to download
    if [[ ! -s "$aria2_input" ]]; then
        log "All models already present, nothing to download"
        rm -f "$aria2_input"
        return 0
    fi

    local file_count=$(grep -c '^http' "$aria2_input" || echo 0)
    log "Downloading $file_count file(s) with aria2c — max bandwidth, all parallel..."

    if ! command -v aria2c >/dev/null 2>&1; then
        rm -f "$aria2_input"
        log "[ERROR] aria2c is unavailable before model download phase"
        return 1
    fi

    # aria2c tuned for vast.ai 700-5000 Mbps connections downloading ~35GB:
    #   -x 16: 16 connections per server per file (HF CDN supports this)
    #   -s 16: split each file into 16 segments for parallel chunk download
    #   -j 7:  ALL 7 files download concurrently (no reason to queue on fat pipes)
    #   -k 4M: 4MB minimum segment size (aggressive splitting for big files)
    #   --file-allocation=none: skip pre-allocation, start downloading immediately
    #   --optimize-concurrent-downloads=true: aria2 auto-tunes based on bandwidth
    #   --stream-piece-selector=geom: prioritize completing files faster
    #   --max-connection-per-server=16: matches -x
    #   --split=16: matches -s
    #   --uri-selector=adaptive: pick fastest mirror automatically
    #   --disk-cache=128M: buffer writes to reduce IO blocking on slow disks
    aria2c \
        --input-file="$aria2_input" \
        -x 16 \
        -s 16 \
        -j 7 \
        -k 4M \
        --max-connection-per-server=16 \
        --file-allocation=none \
        --optimize-concurrent-downloads=true \
        --stream-piece-selector=geom \
        --uri-selector=adaptive \
        --disk-cache=128M \
        --continue=true \
        --retry-wait=3 \
        --max-tries=10 \
        --timeout=60 \
        --connect-timeout=10 \
        --max-resume-failure-tries=5 \
        --console-log-level=notice \
        --summary-interval=15 \
        --auto-file-renaming=false \
        --allow-overwrite=false

    local exit_code=$?
    rm -f "$aria2_input"

    if [[ $exit_code -ne 0 ]]; then
        log "[ERROR] aria2c exited with code $exit_code — some downloads may have failed"
        return 1
    fi

    log "✓ All models downloaded successfully"
}

function provisioning_verify() {
    log "Verifying critical files..."
    local failed=0

    # Verify base models
    local critical_files=(
        "${MODELS_DIR}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        "${MODELS_DIR}/vae/wan_2.1_vae.safetensors"
        "${MODELS_DIR}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
        "${MODELS_DIR}/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
        "$RIFE_PATH"
        "${MODELS_DIR}/loras/high_4step.safetensors"
        "${MODELS_DIR}/loras/low_4step.safetensors"
    )

    for f in "${critical_files[@]}"; do
        if [[ -f "$f" ]]; then
            local size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
            log "✓ $(basename "$f") ($(numfmt --to=iec-i --suffix=B "$size" 2>/dev/null || echo "${size} bytes"))"
        else
            log "[ERROR] Missing: $f"
            failed=1
        fi
    done

    # Verify critical custom nodes
    local critical_nodes=("ComfyUI-Easy-Use" "ComfyUI-WanVideoWrapper" "ComfyUI-KJNodes" "ComfyUI-VideoHelperSuite" "ComfyUI-Frame-Interpolation")
    for node in "${critical_nodes[@]}"; do
        if [[ -d "${COMFYUI_DIR}/custom_nodes/${node}" ]]; then
            log "✓ Node: ${node}"
        else
            log "[ERROR] Missing node: ${node}"
            failed=1
        fi
    done

    if [[ $failed -eq 1 ]]; then
        log "[ERROR] Verification failed — some files are missing"
        return 1
    fi

    log "✓ All critical files verified"
}

function provisioning_print_header() {
    printf "\n##############################################\n#                                            #\n#      Wan 2.2 I2V Provisioning              #\n#                                            #\n#      aria2c parallel downloads             #\n#                                            #\n# Your container will be ready on completion #\n#                                            #\n##############################################\n\n"
}

function provisioning_print_end() {
    printf "\nProvisioning complete:  Application will start now\n\n"
}

function provisioning_has_valid_hf_token() {
    [[ -n "$HF_TOKEN" ]] || return 1
    url="https://huggingface.co/api/whoami-v2"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/json")

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

function provisioning_has_valid_civitai_token() {
    [[ -n "$CIVITAI_TOKEN" ]] || return 1
    url="https://civitai.com/api/v1/models?hidden=1&limit=1"

    response=$(curl -o /dev/null -s -w "%{http_code}" -X GET "$url" \
        -H "Authorization: Bearer $CIVITAI_TOKEN" \
        -H "Content-Type: application/json")

    if [ "$response" -eq 200 ]; then
        return 0
    else
        return 1
    fi
}

# Download from $1 URL to $2 file path (fallback for non-aria2 use)
function provisioning_download() {
    if [[ -n $HF_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?huggingface\.co(/|$|\?) ]]; then
        auth_token="$HF_TOKEN"
    elif
        [[ -n $CIVITAI_TOKEN && $1 =~ ^https://([a-zA-Z0-9_-]+\.)?civitai\.com(/|$|\?) ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi
    if [[ -n $auth_token ]];then
        wget --header="Authorization: Bearer $auth_token" -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes="${3:-4M}" -P "$2" "$1"
    fi
}

if [[ ! -f /.noprovisioning ]]; then
    if ! provisioning_start; then
        touch "$PROVISIONING_FAILED_MARKER"
        exit 1
    fi
fi
