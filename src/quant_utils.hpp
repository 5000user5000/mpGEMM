#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// INT4 量化：fp16_val → uint8_t (低 4 bits)
inline uint8_t quantize_int4(float fp16_val, float scale, int zero_point = 0) {
    int q = static_cast<int>(std::round(fp16_val / scale)) + zero_point;
    q = std::clamp(q, 0, 15);          // INT4 無號 0‥15
    return static_cast<uint8_t>(q);
}

// INT4 反量化：uint8_t (低 4 bits) → float
inline float dequantize_int4(uint8_t q, float scale, int zero_point = 0) {
    int qi = static_cast<int>(q) - zero_point;
    return static_cast<float>(qi) * scale;
}
