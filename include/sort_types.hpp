#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace hsort {

// ----------------------------------------------------------------------------
// 1. 정렬 대상 원소 타입 정의 (컴파일 옵션 -DBENCH_RECORD 유무로 4B / 8B 전환)
// ----------------------------------------------------------------------------
#ifdef BENCH_RECORD
struct alignas(8) DistanceRecord {
    std::int32_t  key;
    std::uint32_t entity_id;

    DistanceRecord() = default;
    constexpr DistanceRecord(std::int32_t k, std::uint32_t id = 0) noexcept
        : key(k), entity_id(id) {}

    // 키 기준 비교 연산자
    constexpr bool operator<(const DistanceRecord& other) const noexcept  { return key < other.key; }
    constexpr bool operator>(const DistanceRecord& other) const noexcept  { return key > other.key; }
    constexpr bool operator<=(const DistanceRecord& other) const noexcept { return key <= other.key; }
    constexpr bool operator>=(const DistanceRecord& other) const noexcept { return key >= other.key; }
    constexpr bool operator==(const DistanceRecord& other) const noexcept { return key == other.key; }
};

using Item = DistanceRecord;
constexpr const char* FORMAT_NAME = "key_id_8b";
#else
using Item = std::int32_t;
constexpr const char* FORMAT_NAME = "key_only_4b";
#endif

static_assert(std::is_trivially_copyable_v<Item>, "Item must be trivially copyable.");

inline constexpr std::int32_t get_key(const Item& item) noexcept {
#ifdef BENCH_RECORD
    return item.key;
#else
    return item;
#endif
}

inline constexpr Item make_item(std::int32_t key, std::uint32_t id = 0) noexcept {
#ifdef BENCH_RECORD
    return DistanceRecord{key, id};
#else
    (void)id;
    return key;
#endif
}

// ----------------------------------------------------------------------------
// 2. 하드웨어 캐시 계층 상수 (64B Cache Line & 8KiB L1 Tile)
// ----------------------------------------------------------------------------
constexpr std::size_t CACHE_LINE_BYTES = 64;
constexpr std::size_t LEAF_SIZE        = CACHE_LINE_BYTES / sizeof(Item); // 4B -> 16개, 8B -> 8개
constexpr std::size_t L1_TILE_BYTES    = 8192;                            // 8 KiB
constexpr std::size_t TILE_SIZE        = L1_TILE_BYTES / sizeof(Item);    // 4B -> 2048개, 8B -> 1024개

// ----------------------------------------------------------------------------
// 3. 64바이트 캐시 라인 경계 정렬 메모리 버퍼 (RAII)
// ----------------------------------------------------------------------------
class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t count)
        : size_(count) {
        const std::size_t alloc_count = (count > 0) ? count : 1;
        data_ = static_cast<Item*>(
            ::operator new[](alloc_count * sizeof(Item), std::align_val_t{CACHE_LINE_BYTES})
        );
    }

    ~AlignedBuffer() {
        ::operator delete[](data_, std::align_val_t{CACHE_LINE_BYTES});
    }

    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    Item*       data() noexcept       { return data_; }
    const Item* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

private:
    Item*       data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace hsort