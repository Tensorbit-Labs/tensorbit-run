#!/bin/bash
# test_all.sh — tensorbit-run test runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== tensorbit-run test suite ==="
echo ""

# Check prerequisites
command -v cmake >/dev/null 2>&1 || { echo "[MISS] cmake"; exit 1; }
command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || { echo "[MISS] C++ compiler"; exit 1; }

echo "[OK] cmake found"
echo "[OK] C++ compiler found"

# Build
mkdir -p "$BUILD_DIR"

echo ""
echo "--- Configuring ---"
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORBIT_BUILD_TESTS=ON \
    -DTENSORBIT_BACKEND_CUDA=OFF \
    -DTENSORBIT_BACKEND_METAL=OFF \
    -DTENSORBIT_BACKEND_VULKAN=OFF \
    -DTENSORBIT_BACKEND_NPU=OFF \
    -DTENSORBIT_BACKEND_SIMD=OFF

echo ""
echo "--- Building ---"
cmake --build "$BUILD_DIR" --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "--- Running tests ---"
cd "$BUILD_DIR"
ctest --output-on-failure --verbose

echo ""
echo "=== All tests complete ==="
