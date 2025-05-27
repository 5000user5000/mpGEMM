#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

inline uint8_t quantize_int4(float fp16_val, float scale, int zero_point = 8) {

    float min_val = -8.0f * scale;
    float max_val = 7.0f * scale;
    float clamped_val = std::clamp(fp16_val, min_val, max_val);
    

    float scaled_val = clamped_val / scale;
    int q = static_cast<int>(std::round(scaled_val)) + zero_point;
    

    q = std::clamp(q, 0, 15);
    return static_cast<uint8_t>(q);
}


inline float dequantize_int4(uint8_t q, float scale, int zero_point = 8) {

    int qi = static_cast<int>(q) - zero_point;
    return static_cast<float>(qi) * scale;
}
