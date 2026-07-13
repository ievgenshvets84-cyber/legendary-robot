#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LLM-Forge – automatische Installation
#
# Legt eine virtuelle Umgebung an, installiert alle Abhängigkeiten und
# richtet die Laufzeitverzeichnisse ein. Erkennt vorhandene NVIDIA-GPUs
# und installiert dann die CUDA-Variante von PyTorch.
#
# Verwendung:
#   ./install.sh            # Standardinstallation
#   ./install.sh --cpu      # explizit CPU-only PyTorch
#   ./install.sh --dev      # zusätzlich Entwicklungswerkzeuge (pytest, ruff, mypy)
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR=".venv"
CPU_ONLY=false
DEV_TOOLS=false

for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_ONLY=true ;;
        --dev) DEV_TOOLS=true ;;
        *) echo "Unbekannte Option: $arg"; exit 1 ;;
    esac
done

# Python 3.12+ finden (Fallback auf python3, wenn python3.12 fehlt)
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
fi
"$PYTHON_BIN" - <<'EOF'
import sys
assert sys.version_info >= (3, 12), f"Python 3.12+ erforderlich, gefunden: {sys.version}"
EOF

echo "==> Erzeuge virtuelle Umgebung in $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# PyTorch: CUDA-Variante, wenn eine NVIDIA-GPU sichtbar ist
if [ "$CPU_ONLY" = false ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> NVIDIA-GPU erkannt – installiere PyTorch mit CUDA 12.1"
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    echo "==> Installiere PyTorch (CPU)"
    pip install torch
fi

echo "==> Installiere Projektabhängigkeiten"
pip install -r requirements.txt

if [ "$DEV_TOOLS" = true ]; then
    echo "==> Installiere Entwicklungswerkzeuge"
    pip install pytest pytest-cov httpx ruff mypy
fi

echo "==> Lege Laufzeitverzeichnisse an"
mkdir -p datasets/raw datasets/processed checkpoints exports logs config

echo
echo "Installation abgeschlossen."
echo "Server starten mit:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -m backend.main"
echo "Webinterface anschließend unter http://localhost:8000 erreichbar."
