#!/bin/bash
# setup.sh — Development environment setup for tensorbit-run
set -euo pipefail

echo "=== tensorbit-run dev setup ==="

# Detect OS
if [[ "$(uname -s)" == "Linux" ]]; then
    echo "[INFO] Linux detected"
    echo "Installing dependencies..."
    sudo apt-get update -qq
    sudo apt-get install -y build-essential cmake g++-13 || {
        echo "Installing g++-13 from PPA..."
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update -qq
        sudo apt-get install -y g++-13
    }
elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[INFO] macOS detected"
    echo "Installing dependencies..."
    brew install cmake gcc
elif [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]]; then
    echo "[INFO] Windows detected (MSYS2)"
    echo "Installing dependencies..."
    pacman -S --noconfirm mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc
else
    echo "[WARN] Unknown OS. Make sure you have: cmake, g++ 13+ or clang 16+"
fi

echo ""
echo "[OK] Setup complete. Run:"
echo "  mkdir build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build . --parallel -j4"
