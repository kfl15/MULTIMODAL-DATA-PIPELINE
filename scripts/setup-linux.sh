#!/usr/bin/env bash
# Ubuntu bootstrap. Re-running reuses software, images, model weights and data.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trap 'echo "Setup stopped at line $LINENO. Fix the reported issue and rerun; downloaded data is retained." >&2' ERR
mode=${1:-auto}
case "$mode" in auto|--cpu|--gpu) ;; *) echo 'Usage: bash scripts/setup-linux.sh [--cpu|--gpu]'; exit 2;; esac
. /etc/os-release
if [[ "$ID" != ubuntu || ! "$VERSION_ID" =~ ^(22.04|24.04)$ ]]; then
  echo 'Automatic host installation targets Ubuntu 22.04/24.04. See README for manual Docker setup.'; exit 1
fi
as_root() { if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi; }
free_kb=$(df -Pk . | awk 'END {print $4}')
if (( free_kb < 20 * 1024 * 1024 )); then echo 'At least 20 GiB free is required for initial setup.'; exit 1; fi
if ! command -v curl >/dev/null; then as_root apt-get update; as_root apt-get install -y curl ca-certificates; fi
if ! command -v docker >/dev/null; then
  as_root apt-get update
  as_root apt-get install -y ca-certificates curl
  as_root install -m 0755 -d /etc/apt/keyrings
  as_root curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  as_root chmod a+r /etc/apt/keyrings/docker.asc
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' "$(dpkg --print-architecture)" "$VERSION_CODENAME" | as_root tee /etc/apt/sources.list.d/docker.list >/dev/null
  as_root apt-get update
  as_root apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
if ! docker compose version >/dev/null 2>&1; then echo 'Docker Compose v2 is required. Install docker-compose-plugin and rerun.'; exit 1; fi
if ! docker buildx version >/dev/null 2>&1; then
  as_root apt-get update
  if apt-cache show docker-buildx-plugin >/dev/null 2>&1; then as_root apt-get install -y docker-buildx-plugin
  else as_root apt-get install -y docker-buildx; fi
fi
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if as_root docker info >/dev/null 2>&1; then DOCKER=(sudo docker)
  else echo 'Start the Docker daemon, then rerun.'; exit 1; fi
fi
runtime=cpu
if [[ "$mode" != --cpu ]] && command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  if ! command -v nvidia-ctk >/dev/null && ! "${DOCKER[@]}" info --format '{{json .Runtimes}}' | grep -q nvidia; then
    as_root apt-get install -y gnupg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | as_root gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | as_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    as_root apt-get update
    as_root apt-get install -y nvidia-container-toolkit
  fi
  if ! "${DOCKER[@]}" info --format '{{json .Runtimes}}' | grep -q nvidia; then
    as_root nvidia-ctk runtime configure --runtime=docker
    if [[ -n "$("${DOCKER[@]}" ps -q)" ]]; then
      echo 'NVIDIA runtime configured. Existing containers are running; restart Docker when convenient, then rerun setup.'; exit 1
    fi
    as_root systemctl restart docker
  fi
  runtime=gpu
elif [[ "$mode" == --gpu ]]; then
  echo 'No usable NVIDIA driver/GPU. Install the appropriate host driver and reboot, or use --cpu.'; exit 1
else
  echo 'CPU fallback: CPU mode selected or no usable NVIDIA GPU/driver detected.'
fi
if ! command -v ollama >/dev/null; then
  installer=$(mktemp)
  curl -fsSL https://ollama.com/install.sh -o "$installer"
  as_root sh "$installer"
  rm -f "$installer"
fi
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
  as_root systemctl start ollama || { echo 'Start host Ollama with ollama serve, then rerun.'; exit 1; }
fi
# Reuse the installed model without checking the registry or updating its tag.
if ollama show granite3.2-vision:2b >/dev/null 2>&1; then
  echo 'Reusing installed Ollama model: granite3.2-vision:2b'
else
  ollama pull granite3.2-vision:2b
fi
mkdir -p input/{audio,text,pdf,excel,image,video,docs} output/{audio,text,pdf,excel,image,video,docs} models runtime reference_data
[[ -f .env ]] || cp .env.example .env
# Change only bootstrap-owned keys; retain user endpoint and model settings.
for key in RUNTIME DEVICE LOCAL_UID LOCAL_GID COMPOSE_FILE; do sed -i "/^${key}=/d" .env; done
if [[ "$runtime" == gpu ]]; then device=auto; compose_files=compose.yaml:compose.gpu.yaml
elif [[ "$mode" == --cpu ]]; then device=cpu; compose_files=compose.yaml
else device=auto; compose_files=compose.yaml; fi
printf '\nRUNTIME=%s\nDEVICE=%s\nLOCAL_UID=%s\nLOCAL_GID=%s\nCOMPOSE_FILE=%s\n' "$runtime" "$device" "${SUDO_UID:-$(id -u)}" "${SUDO_GID:-$(id -g)}" "$compose_files" >> .env
"${DOCKER[@]}" compose build
"${DOCKER[@]}" compose run --rm --no-deps app python -m scripts.download_models
"${DOCKER[@]}" compose run --rm --no-deps app python -m scripts.verify_runtime
"${DOCKER[@]}" compose up -d
for attempt in {1..60}; do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null; then
    echo 'Ready: http://127.0.0.1:8000'
    if command -v xdg-open >/dev/null; then xdg-open http://127.0.0.1:8000 >/dev/null 2>&1 || true; fi
    exit 0
  fi
  sleep 2
done
"${DOCKER[@]}" compose logs --tail=50 app
exit 1
