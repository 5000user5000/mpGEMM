#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// INT4 量化：fp16_val → uint8_t (低 4 bits)
inline uint8_t quantize_int4(float fp16_val, float scale, int zero_point = 8) {
    // 先將值限制在有符號 int4 範圍內（考慮 scale）
    float min_val = -8.0f * scale;
    float max_val = 7.0f * scale;
    float clamped_val = std::clamp(fp16_val, min_val, max_val);
    
    // 將值轉換為整數，並加上 zero_point
    float scaled_val = clamped_val / scale;
    int q = static_cast<int>(std::round(scaled_val)) + zero_point;
    
    // 確保結果在 0-15 範圍內
    q = std::clamp(q, 0, 15);
    return static_cast<uint8_t>(q);
}

// INT4 反量化：uint8_t (低 4 bits) → float
inline float dequantize_int4(uint8_t q, float scale, int zero_point = 8) {
    // 將無符號值轉換為有符號值
    int qi = static_cast<int>(q) - zero_point;
    // 轉換回浮點數
    return static_cast<float>(qi) * scale;
}
