#!/bin/bash
# ==============================================================================
# Build and Run Script for Hermitian Matrix Inversion
# Target: AMD AI100 (gfx908), < 4 ms for 341×341 matrix
# OS: Debian Linux
# ==============================================================================

set -e

# ==============================================================================
# Configuration
# ==============================================================================

BUILD_DIR="build"
GPU_ARCH="${GPU_ARCH:-gfx908}"  # Default: AI100 (gfx908)
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# Debian-specific: ensure ROCm environment is loaded
if [ -f /etc/debian_version ]; then
    if [ -f "${ROCM_PATH}/bin/rocm.profile" ]; then
        source "${ROCM_PATH}/bin/rocm.profile" 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH}"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==============================================================================
# Helper Functions
# ==============================================================================

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ==============================================================================
# Check Prerequisites
# ==============================================================================

check_prereqs() {
    print_info "Checking prerequisites..."
    
    # Check hipcc
    if ! command -v hipcc &> /dev/null; then
        if [ -f "${ROCM_PATH}/bin/hipcc" ]; then
            export PATH="${ROCM_PATH}/bin:$PATH"
        else
            print_error "hipcc not found. Please install ROCm."
            exit 1
        fi
    fi
    print_success "hipcc found: $(which hipcc)"
    
    # Check cmake
    if ! command -v cmake &> /dev/null; then
        print_error "cmake not found. Please install CMake >= 3.16"
        exit 1
    fi
    print_success "cmake found: $(cmake --version | head -1)"
    
    # Check rocprof (optional)
    if ! command -v rocprof &> /dev/null; then
        print_warn "rocprof not found. Profiling will not be available."
    else
        print_success "rocprof found: $(which rocprof)"
    fi
    
    # Check GPU
    if command -v rocm-smi &> /dev/null; then
        GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "GPU" | head -1 || echo "AMD AI100")
        print_success "GPU: ${GPU_NAME}"
    else
        print_warn "rocm-smi not found. Assuming AMD AI100."
    fi
    
    # Check Debian version
    if [ -f /etc/debian_version ]; then
        DEBIAN_VER=$(cat /etc/debian_version)
        print_success "OS: Debian ${DEBIAN_VER}"
    fi
    
    echo ""
}

# ==============================================================================
# Build
# ==============================================================================

do_build() {
    print_info "Building with GPU_ARCH=${GPU_ARCH}..."
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_ARCH="${GPU_ARCH}" \
        -DCMAKE_PREFIX_PATH="${ROCM_PATH}"
    
    cmake --build . --parallel $(nproc)
    
    cd ..
    
    if [ -f "${BUILD_DIR}/matrix_invert" ]; then
        print_success "Build successful: ${BUILD_DIR}/matrix_invert"
    else
        print_error "Build failed!"
        exit 1
    fi
    
    echo ""
}

# ==============================================================================
# Run
# ==============================================================================

do_run() {
    if [ ! -f "${BUILD_DIR}/matrix_invert" ]; then
        print_error "Executable not found. Run './build.sh build' first."
        exit 1
    fi
    
    print_info "Running basic benchmark..."
    echo ""
    
    cd "${BUILD_DIR}"
    ./matrix_invert
    cd ..
    
    echo ""
}

# ==============================================================================
# Run Advanced (Custom kernels + Batched)
# ==============================================================================

do_run_advanced() {
    if [ ! -f "${BUILD_DIR}/matrix_invert_advanced" ]; then
        print_error "Advanced executable not found. Run './build.sh build' first."
        exit 1
    fi
    
    print_info "Running ADVANCED benchmark (Custom Gauss-Jordan + Batched)..."
    echo ""
    
    cd "${BUILD_DIR}"
    ./matrix_invert_advanced
    cd ..
    
    echo ""
}

# ==============================================================================
# Profile
# ==============================================================================

do_profile() {
    if [ ! -f "${BUILD_DIR}/matrix_invert" ]; then
        print_error "Executable not found. Run './build.sh build' first."
        exit 1
    fi
    
    if ! command -v rocprof &> /dev/null; then
        print_error "rocprof not found. Cannot profile."
        exit 1
    fi
    
    print_info "Running profiling with rocprof..."
    
    cd "${BUILD_DIR}"
    rocprof --stats --basenames on ./matrix_invert
    
    if [ -f "results.stats.csv" ]; then
        print_success "Profiling complete. Results:"
        echo ""
        cat results.stats.csv
    fi
    cd ..
    
    echo ""
}

# ==============================================================================
# Profile Advanced
# ==============================================================================

do_profile_advanced() {
    if [ ! -f "${BUILD_DIR}/matrix_invert_advanced" ]; then
        print_error "Advanced executable not found. Run './build.sh build' first."
        exit 1
    fi
    
    if ! command -v rocprof &> /dev/null; then
        print_error "rocprof not found. Cannot profile."
        exit 1
    fi
    
    print_info "Running ADVANCED profiling with rocprof..."
    
    cd "${BUILD_DIR}"
    rocprof --stats --basenames on ./matrix_invert_advanced
    
    if [ -f "results.stats.csv" ]; then
        print_success "Advanced profiling complete. Results:"
        echo ""
        cat results.stats.csv
    fi
    cd ..
    
    echo ""
}

# ==============================================================================
# Profile Detailed
# ==============================================================================

do_profile_detailed() {
    if [ ! -f "${BUILD_DIR}/matrix_invert" ]; then
        print_error "Executable not found. Run './build.sh build' first."
        exit 1
    fi
    
    if ! command -v rocprof &> /dev/null; then
        print_error "rocprof not found. Cannot profile."
        exit 1
    fi
    
    print_info "Running detailed profiling..."
    
    cd "${BUILD_DIR}"
    
    if [ -f "../rocprof_counters.txt" ]; then
        rocprof --timestamp on -i ../rocprof_counters.txt ./matrix_invert
    else
        print_warn "rocprof_counters.txt not found. Using basic profiling."
        rocprof --timestamp on ./matrix_invert
    fi
    
    print_success "Detailed profiling complete."
    cd ..
    
    echo ""
}

# ==============================================================================
# Clean
# ==============================================================================

do_clean() {
    print_info "Cleaning build artifacts..."
    rm -rf "${BUILD_DIR}"
    rm -f *.csv *.json
    print_success "Clean complete."
}

# ==============================================================================
# Full Pipeline
# ==============================================================================

do_full() {
    check_prereqs
    do_build
    do_run
    
    if command -v rocprof &> /dev/null; then
        do_profile
    fi
}

# ==============================================================================
# Help
# ==============================================================================

show_help() {
    cat << EOF
Hermitian Matrix Inversion Benchmark - Build Script

Usage: $0 [command] [options]

Commands:
    build           Build both basic and advanced versions
    run             Run basic benchmark (LU, Hybrid, Cholesky)
    run-advanced    Run ADVANCED benchmark (+ Custom Gauss-Jordan + Batched 100)
    profile         Run basic with rocprof
    profile-advanced Run advanced with rocprof
    profile-detail  Run with rocprof (detailed metrics)
    clean           Remove build artifacts
    full            Build + Run basic + Profile
    full-advanced   Build + Run advanced + Profile
    help            Show this help

Options:
    GPU_ARCH=gfx908|gfx90a|gfx942   GPU architecture (default: gfx908 for AI100)
    ROCM_PATH=/opt/rocm             ROCm installation path

Examples:
    $0 build                        # Build both versions
    $0 run                          # Run basic benchmark
    $0 run-advanced                 # Run advanced (Custom + Batched)
    $0 full-advanced                # Complete advanced pipeline

Executables:
    matrix_invert          - Basic (LU, Hybrid, Cholesky)
    matrix_invert_advanced - Advanced (+ Custom Gauss-Jordan + Batched 100)

Target: 341×341 Hermitian matrix inversion in < 4 ms
EOF
}

# ==============================================================================
# Main
# ==============================================================================

echo ""
echo "=============================================="
echo "  Hermitian Matrix Inversion Benchmark"
echo "  Target: 341×341 matrix, < 4 ms on AI100"
echo "  OS: Debian Linux"
echo "=============================================="
echo ""

case "${1:-full}" in
    build)
        check_prereqs
        do_build
        ;;
    run)
        do_run
        ;;
    run-advanced|run_advanced)
        do_run_advanced
        ;;
    profile)
        do_profile
        ;;
    profile-advanced|profile_advanced)
        do_profile_advanced
        ;;
    profile-detail|profile_detailed)
        do_profile_detailed
        ;;
    clean)
        do_clean
        ;;
    full)
        do_full
        ;;
    full-advanced|full_advanced)
        check_prereqs
        do_build
        do_run_advanced
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

print_success "Done!"

