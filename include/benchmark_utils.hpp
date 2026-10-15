#pragma once
#include "sort_types.hpp"
#include <string>
#include <vector>

namespace hsort {

enum class Scenario : int {
    Fresh3DDistances = 0, // 완전 무작위 3D 거리제곱
    NextFrameMotion  = 1, // 이전 프레임 정렬 후 ±5cm 미세 이동
    Append5PctNew    = 2, // 앞쪽 95% 정렬 + 뒤쪽 5% 신규 스폰
    DistanceBuckets  = 3  // 0..300 구간으로 나눈 다중 중복 키
};

const char* scenario_name(Scenario s) noexcept;

// 게임 워크로드 합성 데이터 생성 (타이머 외부에서 호출)
void generate_game_workload(Item* output, std::size_t n, std::size_t batch,
                            Scenario scenario, std::uint64_t seed);

// 정렬 무결성 검증을 위한 정답 데이터 구조체
struct ReferenceData {
    std::vector<std::int32_t> sorted_keys;
#ifdef BENCH_RECORD
    std::vector<std::int32_t> original_key_by_id;
    std::vector<std::uint8_t> visited_stamp;
#endif
};

ReferenceData build_reference(const Item* source, std::size_t n, std::size_t batch);

// 정렬 결과 및 Entity ID 보존 여부를 엄격히 검증 (실패 시 예외 발생)
void verify_sorted_batch(const Item* work, std::size_t n, std::size_t batch,
                         ReferenceData& ref, std::uint8_t stamp_id);

// 리눅스 CPU 코어 고정 (캐시 문맥 교환 방지)
int pin_current_thread_to_cpu(int cpu_id);

} // namespace hsort