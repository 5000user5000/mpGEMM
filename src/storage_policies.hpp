#pragma once
#include <cstdint>   // for uint8_t
#include <cstddef>   // for size_t

template<typename T>
struct PlainStorage {
    using StorageType = T;
    static constexpr size_t entries_per_unit = 1;
    static T get(const StorageType &unit, size_t /*offset*/) { return unit; }
    static void set(StorageType &unit, T value, size_t /*offset*/) { unit = value; }
};

struct Int4Storage {
    using StorageType = uint8_t;
    static constexpr size_t entries_per_unit = 2;
    static uint8_t get(const StorageType &b, size_t offset) {
        return offset == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
    }
    static void set(StorageType &b, uint8_t v, size_t offset) {
        if (offset == 0) b = (b & 0xF0) | (v & 0x0F);
        else             b = (b & 0x0F) | ((v & 0x0F) << 4);
    }
};
