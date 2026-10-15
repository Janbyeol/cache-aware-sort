#include "benchmark_utils.hpp"
#include <algorithm>
#include <random>
#include <sched.h>
#include <stdexcept>

namespace hsort {

const char* scenario_name(Scenario s) noexcept {
    switch (s) {
        case Scenario::Fresh3DDistances: return "fresh_3d_distances";
        case Scenario::NextFrameMotion:  return "next_frame_motion";
        case Scenario::Append5PctNew:    return "append_5pct_new";
        case Scenario::DistanceBuckets:  return "distance_buckets";
    }
    return "unknown";
}

namespace {
struct Point3D {
    std::int32_t  x, y, z, key;
    std::uint32_t id;
};

inline constexpr std::int32_t squared_distance(std::int32_t x, std::int32_t y, std::int32_t z) noexcept {
    return static_cast<std::int32_t>(
        std::int64_t{x} * x + std::int64_t{y} * y + std::int64_t{z} * z
    );
}
} // anonymous namespace

void generate_game_workload(Item* output, std::size_t n, std::size_t batch,
                            Scenario scenario, std::uint64_t seed) {
    std::mt19937_64 rng(seed);

    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<Point3D> points(n);
        for (std::size_t i = 0; i < n; ++i) {
            auto rand_coord = [&]() { return static_cast<std::int32_t>(rng() % 20001) - 10000; };
            points[i].x   = rand_coord();
            points[i].y   = rand_coord();
            points[i].z   = rand_coord();
            points[i].id  = static_cast<std::uint32_t>(i);
            points[i].key = squared_distance(points[i].x, points[i].y, points[i].z);
        }

        auto by_key = [](const Point3D& a, const Point3D& b) { return a.key < b.key; };

        if (scenario == Scenario::NextFrameMotion) {
            std::sort(points.begin(), points.end(), by_key);
            for (auto& p : points) {
                auto jitter = [&]() { return static_cast<std::int32_t>(rng() % 11) - 5; };
                p.x  += jitter() - 3;
                p.y  += jitter() + 2;
                p.z  += jitter() - 1;
                p.key = squared_distance(p.x, p.y, p.z);
            }
        } else if (scenario == Scenario::Append5PctNew) {
            const std::size_t sorted_prefix = n - (n / 20);
            std::sort(points.begin(), points.begin() + sorted_prefix, by_key);
        } else if (scenario == Scenario::DistanceBuckets) {
            for (auto& p : points) p.key /= 1'000'000;
        }

        for (std::size_t i = 0; i < n; ++i) {
            output[b * n + i] = make_item(points[i].key, points[i].id);
        }
    }
}

ReferenceData build_reference(const Item* source, std::size_t n, std::size_t batch) {
    ReferenceData ref;
    ref.sorted_keys.resize(n * batch);
#ifdef BENCH_RECORD
    ref.original_key_by_id.resize(n * batch);
    ref.visited_stamp.assign(n * batch, 0);
#endif

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            const Item& item = source[b * n + i];
            ref.sorted_keys[b * n + i] = get_key(item);
#ifdef BENCH_RECORD
            ref.original_key_by_id[b * n + item.entity_id] = get_key(item);
#endif
        }
        std::sort(ref.sorted_keys.begin() + b * n, ref.sorted_keys.begin() + (b + 1) * n);
    }
    return ref;
}

void verify_sorted_batch(const Item* work, std::size_t n, std::size_t batch,
                         ReferenceData& ref, std::uint8_t stamp_id) {
    for (std::size_t b = 0; b < batch; ++b) {
        const Item* arr = work + b * n;

        if (!std::is_sorted(arr, arr + n)) {
            throw std::runtime_error("Verification failed: array is not sorted.");
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (get_key(arr[i]) != ref.sorted_keys[b * n + i]) {
                throw std::runtime_error("Verification failed: key mismatch against std::sort.");
            }
        }

#ifdef BENCH_RECORD
        for (std::size_t i = 0; i < n; ++i) {
            const auto id = arr[i].entity_id;
            if (id >= n ||
                ref.visited_stamp[b * n + id] == stamp_id ||
                get_key(arr[i]) != ref.original_key_by_id[b * n + id]) {
                throw std::runtime_error("Verification failed: entity_id corrupted or duplicated.");
            }
            ref.visited_stamp[b * n + id] = stamp_id;
        }
#endif
    }
}

int pin_current_thread_to_cpu(int cpu_id) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return -1;

    int target_cpu = cpu_id;
    if (target_cpu < 0 || !CPU_ISSET(target_cpu, &allowed)) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &allowed)) {
                target_cpu = c;
                break;
            }
        }
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(target_cpu, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
    return target_cpu;
}

} // namespace hsort