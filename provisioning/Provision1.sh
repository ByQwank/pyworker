#!/usr/bin/env bash
# Fast Wan 2.2 i2v provisioning (non-serverless benchmark flow removed).
# Installs your core nodes + base models + high/low 4step LoRAs.

set -euo pipefail

if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

WORKSPACE_DIR="${WORKSPACE:-/workspace}"
COMFYUI_DIR="${WORKSPACE_DIR}/ComfyUI"
MODELS_DIR="${COMFYUI_DIR}/models"
LORA_DIR="${MODELS_DIR}/loras"
MODEL_LOG="${MODEL_LOG:-/var/log/portal/comfyui.log}"

HF_MAX_PARALLEL="${HF_MAX_PARALLEL:-8}"
GIT_MAX_PARALLEL="${GIT_MAX_PARALLEL:-5}"
AUTO_UPDATE_NODES="${AUTO_UPDATE_NODES:-true}"
PRELOAD_ALL_LORAS="${PRELOAD_ALL_LORAS:-false}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${WORKSPACE_DIR}/.hf-cache}"

APT_PACKAGES=(
  git
  rsync
  curl
  wget
)

PIP_PACKAGES=(
  "huggingface_hub[hf_transfer]"
  hf_transfer
  lark
  sentencepiece
  opencv-python-headless
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

# URL|ABSOLUTE_OUTPUT_PATH
HF_MODELS=(
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|${MODELS_DIR}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors|${MODELS_DIR}/vae/wan_2.1_vae.safetensors"
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors|${MODELS_DIR}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors|${MODELS_DIR}/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
  "https://huggingface.co/hfmaster/models-moved/resolve/cab6dcee2fbb05e190dbb8f536fbdaa489031a14/rife/rife49.pth|${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/models/rife/rife49.pth"
  "https://huggingface.co/hfmaster/models-moved/resolve/cab6dcee2fbb05e190dbb8f536fbdaa489031a14/rife/rife49.pth|${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth"
  "https://huggingface.co/Dylaaann/Lora/resolve/main/high_4step.safetensors|${LORA_DIR}/high_4step.safetensors"
  "https://huggingface.co/Dylaaann/Lora/resolve/main/low_4step.safetensors|${LORA_DIR}/low_4step.safetensors"
)

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  mkdir -p "$(dirname "$MODEL_LOG")"
  echo "$msg" | tee -a "$MODEL_LOG"
}

truthy() {
  local v="${1:-}"
  v="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

setup_dirs() {
  mkdir -p "${COMFYUI_DIR}/custom_nodes"
  mkdir -p "${MODELS_DIR}/"{text_encoders,diffusion_models,vae,loras}
  mkdir -p "$HF_CACHE_DIR"
}

install_apt_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  log "Installing apt packages..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y >/dev/null
  apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}" >/dev/null
}

install_python_packages() {
  log "Installing python packages..."
  local pip_cmd="pip"
  if command -v uv >/dev/null 2>&1; then
    pip_cmd="uv pip"
  fi

  $pip_cmd install --upgrade pip >/dev/null 2>&1 || true
  $pip_cmd install "${PIP_PACKAGES[@]}" >/dev/null

  export HF_HUB_ENABLE_HF_TRANSFER=1
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || true
  fi
}

retry_cmd() {
  local attempts="$1"
  shift
  local n=1
  local delay=2
  until "$@"; do
    if (( n >= attempts )); then
      return 1
    fi
    sleep "$delay"
    n=$((n + 1))
    delay=$((delay * 2))
  done
}

clone_or_update_node() {
  local repo="$1"
  local dir="${repo##*/}"
  local path="${COMFYUI_DIR}/custom_nodes/${dir}"

  if [[ -d "${path}/.git" ]]; then
    if truthy "$AUTO_UPDATE_NODES"; then
      retry_cmd 4 git -C "$path" pull --ff-only >/dev/null 2>&1
    fi
    return 0
  fi

  rm -rf "$path"
  retry_cmd 4 git clone --recursive --depth 1 "$repo" "$path" >/dev/null 2>&1
}

clone_nodes_parallel() {
  log "Cloning/updating custom nodes..."
  local running=0
  local failed=0
  local repo
  for repo in "${NODES[@]}"; do
    (
      if clone_or_update_node "$repo"; then
        log "[OK] node ready: ${repo##*/}"
      else
        log "[ERROR] node failed: ${repo##*/}"
        exit 1
      fi
    ) &
    running=$((running + 1))
    if (( running >= GIT_MAX_PARALLEL )); then
      if ! wait -n; then
        failed=1
      fi
      running=$((running - 1))
    fi
  done

  while (( running > 0 )); do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done

  (( failed == 0 ))
}

parse_hf_url() {
  local url="$1"
  local repo file_path
  repo="$(echo "$url" | sed -n 's|https://huggingface.co/\([^/]*/[^/]*\)/resolve/.*|\1|p')"
  file_path="$(echo "$url" | sed -n 's|https://huggingface.co/[^/]*/[^/]*/resolve/[^/]*/\(.*\)|\1|p')"
  if [[ -z "$repo" || -z "$file_path" ]]; then
    return 1
  fi
  echo "$repo|$file_path"
}

download_with_resume() {
  local url="$1"
  local out="$2"
  local part="${out}.part"
  local header=()
  if [[ -n "${HF_TOKEN:-}" && "$url" =~ ^https://huggingface\.co/ ]]; then
    header=(-H "Authorization: Bearer ${HF_TOKEN}")
  fi

  retry_cmd 5 curl -fL \
    --retry 5 \
    --retry-delay 3 \
    --retry-all-errors \
    --connect-timeout 30 \
    -C - \
    "${header[@]}" \
    -o "$part" \
    "$url" >/dev/null

  mv "$part" "$out"
}

download_model_spec() {
  local spec="$1"
  local url="${spec%%|*}"
  local out="${spec##*|}"
  local out_dir
  out_dir="$(dirname "$out")"
  mkdir -p "$out_dir"

  if [[ -f "$out" ]]; then
    log "Skip existing: $out"
    return 0
  fi

  if [[ "$url" =~ ^https://huggingface\.co/ ]] && command -v hf >/dev/null 2>&1; then
    local parsed repo file_path tmpdir
    if parsed="$(parse_hf_url "$url")"; then
      repo="${parsed%%|*}"
      file_path="${parsed##*|}"
      tmpdir="$(mktemp -d)"
      if retry_cmd 5 hf download "$repo" "$file_path" --repo-type model --local-dir "$tmpdir" --cache-dir "$HF_CACHE_DIR" >/dev/null 2>&1; then
        if [[ -f "$tmpdir/$file_path" ]]; then
          mkdir -p "$out_dir"
          mv "$tmpdir/$file_path" "$out"
          rm -rf "$tmpdir"
          log "[OK] Downloaded: $out"
          return 0
        fi
      fi
      rm -rf "$tmpdir"
    fi
  fi

  download_with_resume "$url" "$out"
  log "[OK] Downloaded: $out"
}

download_models_parallel() {
  log "Starting model downloads (HF_MAX_PARALLEL=${HF_MAX_PARALLEL})..."
  local running=0
  local failed=0
  local spec
  for spec in "${HF_MODELS[@]}"; do
    (
      if ! download_model_spec "$spec"; then
        log "[ERROR] Download failed: ${spec##*|}"
        exit 1
      fi
    ) &
    running=$((running + 1))
    if (( running >= HF_MAX_PARALLEL )); then
      if ! wait -n; then
        failed=1
      fi
      running=$((running - 1))
    fi
  done

  while (( running > 0 )); do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done

  (( failed == 0 ))
}

install_node_requirements() {
  log "Installing custom node requirements..."
  local pip_cmd="pip"
  if command -v uv >/dev/null 2>&1; then
    pip_cmd="uv pip"
  fi

  local req
  while IFS= read -r req; do
    log "requirements: $(basename "$(dirname "$req")")"
    $pip_cmd install --no-cache-dir -r "$req" >/dev/null 2>&1 || \
      log "[WARN] requirements had partial failures: $req"
  done < <(find "${COMFYUI_DIR}/custom_nodes" -type f -name "requirements.txt" 2>/dev/null | sort)
}

run_node_installers() {
  log "Running install.py (if present)..."
  local installer
  while IFS= read -r installer; do
    log "install.py: $(basename "$(dirname "$installer")")"
    python "$installer" >/dev/null 2>&1 || \
      log "[WARN] install.py failed (continuing): $installer"
  done < <(find "${COMFYUI_DIR}/custom_nodes" -type f -name "install.py" 2>/dev/null | sort)
}

download_all_loras_if_enabled() {
  if ! truthy "$PRELOAD_ALL_LORAS"; then
    log "Skipping bulk LoRA preload (PRELOAD_ALL_LORAS=false)"
    return 0
  fi

  log "Preloading all LoRAs from Dylaaann/Lora..."
  mkdir -p "$LORA_DIR"
  hf download Dylaaann/Lora --repo-type model --local-dir "$LORA_DIR" --exclude "*.md" ".git*" >/dev/null 2>&1 || \
    log "[WARN] bulk LoRA preload failed (on-demand still works)"
}

print_summary() {
  local node_count model_count lora_count
  node_count="$(find "${COMFYUI_DIR}/custom_nodes" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | xargs)"
  model_count="$(find "${MODELS_DIR}" -type f \( -name "*.safetensors" -o -name "*.pth" \) 2>/dev/null | wc -l | xargs)"
  lora_count="$(find "${LORA_DIR}" -type f -name "*.safetensors" 2>/dev/null | wc -l | xargs)"
  log "Custom Nodes: ${node_count}"
  log "Model Files: ${model_count}"
  log "LoRAs: ${lora_count}"
}

provisioning_start() {
  log "Starting provisioning..."
  setup_dirs
  install_apt_packages
  install_python_packages

  clone_nodes_parallel
  download_models_parallel
  download_all_loras_if_enabled

  # Keep requirements/installers after clone + base downloads for better startup reliability.
  install_node_requirements
  run_node_installers

  print_summary
  log "Provisioning complete."
}

if [[ ! -f /.noprovisioning ]]; then
  provisioning_start
fi
