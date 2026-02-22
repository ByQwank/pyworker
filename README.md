# PyWorker Fork (ComfyUI JSON only)

This fork is intentionally minimal and only keeps the `comfyui-json` backend used by this project.

## What this fork contains

- `workers/comfyui-json/worker.py` - primary PyWorker backend
- `start_server.sh` - startup bootstrap script used by Vast templates
- `provisioning/Provision1.sh` - current provisioning script to use from template `PROVISIONING_SCRIPT`

## What was removed

To reduce maintenance and confusion, unused template workers were removed from this fork:

- `workers/openai`
- `workers/tgi`
- `workers/wan`
- `workers/ace`

## Required environment variables (template/runtime)

- `BACKEND=comfyui-json`
- `PYWORKER_REPO=https://github.com/ByQwank/pyworker`
- `PROVISIONING_SCRIPT=<raw url to provisioning/Provision1.sh>`
- `HF_TOKEN=<huggingface token>`
- `PYWORKER_MANIFEST_SECRET=<shared HMAC secret for LoRA manifests>`

Optional:

- `PYWORKER_REF=<branch|tag|commit>`
- `SDK_BRANCH=<vast-sdk branch|tag|commit>`
- `SDK_VERSION=<vastai-sdk version>`
- `PYWORKER_REQUIRE_MANIFEST=auto|true|false` (default `auto`, requires signed manifests when secret is set)
- `PYWORKER_MANIFEST_ENDPOINT=direct-instance` (must match Trigger manifest endpoint)

## Provision script raw URL

Use this URL format in your Vast template:

`https://raw.githubusercontent.com/ByQwank/pyworker/main/provisioning/Provision1.sh`

If you pin to a commit, replace `main` with that commit SHA.

## Security

- Do not commit API keys, tokens, or secrets to this repo.
- Keep credentials in Vast/Trigger/Convex environment variables only.
- This repo is safe to keep public as long as secrets remain in env vars.
