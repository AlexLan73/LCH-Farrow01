#!/bin/bash
# Build script for Matrix Inversion Profiler
# Handles compilation and profiling

set -e

# Configuration
BUILD_DIR="build"
INSTALL_PREFIX="${ROCM_PATH:-/opt/rocm}"
MATRIX_SIZE=341
GPU_ARCH="gfx908"  # For MI100/MI250/MI300. Use "gfx90a" for MI250X

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check for hipcc
    if ! command -v hipcc &> /dev/null; then
        print_error "hipcc not found. Please ensure ROCm is installed."
        exit 1
    fi
    
    # Check for rocprof
    if ! command -v rocprof &> /dev/null; then
        print_warning "rocprof not found. Profiling may not work."
    fi
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        print_error "cmake not found. Please install CMake."
        exit 1
    fi
    
    print_success "Prerequisites OK"
}

# Build function
build() {
    print_status "Building Matrix Inversion Profiler..."
    
    # Create and enter build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure with CMake
    print_status "Configuring CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
        -DHIP_ARCHITECTURES="$GPU_ARCH"
    
    # Build
    print_status "Compiling (using $(nproc) cores)..."
    cmake --build . --parallel $(nproc)
    
    cd ..
    print_success "Build completed successfully"
}

# Profile function
profile_basic() {
    print_status "Running basic profiling with rocprof..."
    
    cd "$BUILD_DIR"
    
    # Run with stats collection
    rocprof --stats --basenames on ./matrix_invert
    
    # Check for results
    if [ -f "results.csv" ]; then
        print_success "Profiling completed. Results in results.csv"
        echo ""
        echo "Summary of results.csv:"
        head -20 results.csv
    else
        print_warning "results.csv not found"
    fi
    
    cd ..
}

# Profile detailed function
profile_detailed() {
    print_status "Running detailed profiling with rocprof..."
    
    cd "$BUILD_DIR"
    
    # Run with detailed metrics
    if [ -f "../rocprof_counters.txt" ]; then
        rocprof --timestamp on -i ../rocprof_counters.txt ./matrix_invert
        print_success "Detailed profiling completed"
    else
        print_warning "rocprof_counters.txt not found, using basic profiling"
        rocprof --timestamp on ./matrix_invert
    fi
    
    cd ..
}

# Analyze function
analyze() {
    print_status "Analyzing profiling results..."
    
    if [ ! -f "build/results.csv" ]; then
        print_error "results.csv not found. Please run profiling first."
        exit 1
    fi
    
    # Check if Python script exists
    if [ ! -f "analyze_profile.py" ]; then
        print_error "analyze_profile.py not found"
        exit 1
    fi
    
    # Run analysis
    python3 analyze_profile.py build/results.csv --markdown --json --output reports
    
    if [ -f "reports/profiling_report.md" ]; then
        print_success "Analysis complete. Report: reports/profiling_report.md"
    fi
}

# Clean function
clean() {
    print_status "Cleaning build artifacts..."
    rm -rf "$BUILD_DIR" reports
    print_success "Cleanup complete"
}

# Help function
show_help() {
    cat << EOF
Matrix Inversion Profiler Build Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build               Build the profiler (default)
    profile             Run basic profiling
    profile-detailed    Run detailed profiling with metrics
    analyze             Analyze profiling results
    clean               Clean build artifacts
    full                Build, profile, and analyze
    help                Show this help message

Options:
    --gpu-arch ARCH     GPU architecture (default: gfx908)
                        Use "gfx908" for MI100/MI250
                        Use "gfx90a" for MI250X
    --matrix-size SIZE  Matrix size (default: 341)
    --rocm-path PATH    ROCm installation path

Examples:
    $0 build                           # Build the project
    $0 profile                         # Run basic profiling
    $0 profile-detailed                # Run detailed profiling
    $0 analyze                         # Analyze results
    $0 full                            # Complete workflow
    $0 --gpu-arch gfx90a build        # Build for MI250X

EOF
}

# Parse arguments
COMMAND="build"
while [ $# -gt 0 ]; do
    case $1 in
        build)
            COMMAND="build"
            shift
            ;;
        profile)
            COMMAND="profile"
            shift
            ;;
        profile-detailed)
            COMMAND="profile_detailed"
            shift
            ;;
        analyze)
            COMMAND="analyze"
            shift
            ;;
        clean)
            COMMAND="clean"
            shift
            ;;
        full)
            COMMAND="full"
            shift
            ;;
        help)
            show_help
            exit 0
            ;;
        --gpu-arch)
            GPU_ARCH="$2"
            shift 2
            ;;
        --matrix-size)
            MATRIX_SIZE="$2"
            shift 2
            ;;
        --rocm-path)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
echo "=========================================================================="
echo "Matrix Inversion GPU Profiler"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  GPU Architecture:  $GPU_ARCH"
echo "  Matrix Size:       $MATRIX_SIZE x $MATRIX_SIZE (complex)"
echo "  ROCm Path:         $INSTALL_PREFIX"
echo "  Build Directory:   $BUILD_DIR"
echo ""

check_prerequisites

case $COMMAND in
    build)
        build
        ;;
    profile)
        if [ ! -f "$BUILD_DIR/matrix_invert" ]; then
            build
        fi
        profile_basic
        ;;
    profile_detailed)
        if [ ! -f "$BUILD_DIR/matrix_invert" ]; then
            build
        fi
        profile_detailed
        ;;
    analyze)
        analyze
        ;;
    clean)
        clean
        ;;
    full)
        clean
        build
        profile_basic
        analyze
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

print_success "Done!"
