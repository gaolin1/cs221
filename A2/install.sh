#!/bin/bash

# --- Prevent crashing shell if sourced ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "⚠️  Please run this script with 'source install.sh' to activate the environment in your current shell."
  exit 1
fi

ENV_NAME="XCS221"
ENV_FILE="environment.yml"
REQ_FILE="requirements.txt"
REFRESH=false

# --- Parse command-line options ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--refresh)
      REFRESH=true
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      echo "Usage: source install.sh [-r|--refresh]"
      exit 1
      ;;
    *)
      shift
      ;;
  esac
done

# --- Step 1: Check prerequisites ---
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ Error: 'conda' is not installed or not available in PATH."
  echo "Please install Anaconda/Miniconda/Miniforge."
  exit 1
fi

if [[ -f "gradescope/environment.yml" ]]; then
  ENV_FILE="gradescope/environment.yml"
elif [[ -f "src/environment.yml" ]]; then
  ENV_FILE="src/environment.yml"
else
  echo "❌ Error: 'environment.yml' not found."
  exit 1
fi


if [[ -f "gradescope/requirements.txt" ]]; then
  REQ_FILE="gradescope/requirements.txt"
elif [[ -f "src/requirements.txt" ]]; then
  REQ_FILE="src/requirements.txt"
else
  echo "❌ Error: 'requirements.txt' not found."
  exit 1
fi

# --- Step 2: Initialize conda for bash script execution ---
eval "$(conda shell.bash hook)"

# --- Step 3: Handle refresh ---
if $REFRESH; then
  echo "🔄 Refreshing environment '$ENV_NAME'..."
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
fi

# --- Step 4: Create or update Anaconda environment from environment.yml ---
if conda env list | grep -q -E "^${ENV_NAME}\s"; then
  echo "📦 Anaconda environment '$ENV_NAME' already exists. Updating..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "📦 Creating Anaconda environment '$ENV_NAME' from $ENV_FILE..."
  conda env create -f "$ENV_FILE"
fi

# --- Step 5: Activate the XCS221 environment ---
echo "🚀 Activating Anaconda environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

# --- Step 6: Install dependencies using uv ---
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ Error: 'uv' is not installed."
  echo "Please install uv or ensure it is provided by the conda environment."
  exit 1
fi

echo "🔧 Installing packages from $REQ_FILE using uv..."
uv pip install -r "$REQ_FILE"

echo "✅ Setup complete. Anaconda environment '$ENV_NAME' is active."