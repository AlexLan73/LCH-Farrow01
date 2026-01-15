#pragma once

namespace gpu {

// ════════════════════════════════════════════════════════════════════════════
// Enum для типов памяти GPU
// ════════════════════════════════════════════════════════════════════════════

enum class MemoryType {
    GPU_READ_ONLY,
    GPU_WRITE_ONLY,
    GPU_READ_WRITE
};

} // namespace gpu

