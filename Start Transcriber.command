#!/bin/bash
# Double-click this file in Finder to launch the Conference Video Transcriber

cd "$(dirname "$0")"
ulimit -n 4096 2>/dev/null

echo ""
echo "============================================"
echo "  Conference Video Transcriber"
echo "============================================"
echo ""

# ── Python ──────────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "  Install it:  brew install python@3.11"
    echo ""
    read -r -p "Press Enter to close…"
    exit 1
fi

# ── ffmpeg ───────────────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not found — installing via Homebrew…"
    if ! command -v brew &>/dev/null; then
        echo "ERROR: Homebrew is required to install ffmpeg."
        echo ""
        echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        echo ""
        read -r -p "Press Enter to close…"
        exit 1
    fi
    brew install ffmpeg
    echo ""
fi

# ── Virtual environment + dependencies ───────────────────────────────────────
# Recreate venv if missing or broken (e.g. after moving the project folder)
VENV_OK=false
if [ -d "venv" ]; then
    if venv/bin/python3 -c "import gradio, torch, faster_whisper, huggingface_hub, pyannote.audio" &>/dev/null 2>&1; then
        VENV_OK=true
    fi
fi

if [ "$VENV_OK" = false ]; then
    if [ -d "venv" ]; then
        echo "Virtual environment is broken (folder may have been moved) — rebuilding…"
        rm -rf venv
    else
        echo "First-time setup — creating virtual environment…"
    fi
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    echo "Installing dependencies (this takes a few minutes on first run)…"
    if ! pip install -r requirements.txt; then
        echo ""
        echo "ERROR: Dependency installation failed. See above for details."
        echo ""
        read -r -p "Press Enter to close…"
        exit 1
    fi
    echo "Done."
    echo ""
else
    source venv/bin/activate
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo "Starting — browser will open automatically at http://127.0.0.1:7860"
echo "Close this window (or press Ctrl+C) to stop the transcriber."
echo ""
python3 transcriber.py
