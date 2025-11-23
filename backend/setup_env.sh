
#!/usr/bin/env bash
set -euo pipefail

# Local environment setup script for the backend using Conda
# Mirrors essential steps from the Dockerfile so tests and development
# run locally without missing system packages.
#
# Usage:
#   cd backend
#   ./setup_env.sh
#
# After running, activate the conda env with:
#   conda activate <env-name>

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Backend setup starting in $ROOT_DIR"

# System packages required (from Dockerfile)
PKGS=(
  python3-opencv
  build-essential
  libglib2.0-0
  libsm6
  libxext6
  libxrender1
  ffmpeg
)

if command -v apt-get >/dev/null 2>&1; then
  echo "Updating apt and installing system packages (requires sudo)"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends "${PKGS[@]}"
else
  echo "apt-get not found — please install these system packages manually: ${PKGS[*]}"
fi

# Conda environment name (can be overridden by setting CONDA_ENV_NAME)
ENV_NAME="${CONDA_ENV_NAME:-htx}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found in PATH. Please install Miniconda or Anaconda and re-run this script."
  exit 1
fi

echo "Using conda executable: $(command -v conda)"

# Create the conda env if it doesn't exist
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
else
  echo "Creating conda environment '$ENV_NAME' with Python 3.12"
  conda create -n "$ENV_NAME" python=3.12 -y
fi

echo "Installing/upgrading pip, setuptools, wheel inside '$ENV_NAME'"
conda run -n "$ENV_NAME" pip install --upgrade pip setuptools wheel

echo "Installing CPU-only PyTorch (to match Dockerfile) inside '$ENV_NAME'"
conda run -n "$ENV_NAME" pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu || {
  echo "Warning: torch installation failed inside conda env — you may need to install a different wheel for your platform"
}

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  echo "Installing Python requirements from requirements.txt into '$ENV_NAME'"
  conda run -n "$ENV_NAME" pip install --no-cache-dir -r "$ROOT_DIR/requirements.txt"
else
  echo "No requirements.txt found in $ROOT_DIR — please install project deps manually"
fi

echo
echo "Setup complete. Activate the conda environment with:"
echo "  conda activate $ENV_NAME"
echo
echo "Optional: to download model files used by the processing pipeline (MobileNet-SSD), run:" 
echo "  conda activate $ENV_NAME && python -c \"from routers.process.utils import ensure_mobilenet_model; ensure_mobilenet_model()\""

echo "Notes:"
echo "- If you are on a non-Debian system, install the equivalent system packages listed above."
echo "- If torch CPU wheel version 2.9.1 is incompatible with your system, install an appropriate torch package for your platform."

exit 0
