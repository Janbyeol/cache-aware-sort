# Idea 6: 수직 정렬 네트워크·레지스터 행렬 전치 기반 4-리프 슈퍼블록 & 양방향 이중 체인 무분기 SIMD 타일 병합 정렬 (AVX2 & ARM64 NEON)

**한국어** | [English](README.md)

이 문서는 초기 4개 아이디어(Idea 1 \~ Idea 4)와 첫 번째 SIMD 실험(Idea 5)을 거친 뒤, **x86-64 AVX2(256-bit)** 와 **Windows ARM64 NEON(128-bit, Snapdragon X Elite / Oryon)** 양쪽 아키텍처에서 관찰된 마이크로아키텍처 및 컴파일러 백엔드 병목을 해결하기 위해 설계한 **Idea 6 (`simd_transpose_merge` / `simd_transpose_tiled_merge_sort`)** 의 출발 가설, 어셈블리 수준 병목 원인 분석, 핵심 설계 메커니즘, 시행착오(v1\~v4) 엔지니어링 로그, 그리고 정면 비교 실험 결과를 기록합니다.

---

## 목차
1. [시작 질문과 Idea 1\~5로부터의 진화](#1-시작-질문과-idea-15로부터의-진화)
2. [마이크로아키텍처 및 컴파일러 병목 원인 분석 (AVX2 & ARM64 NEON)](#2-마이크로아키텍처-및-컴파일러-병목-원인-분석-avx2--arm64-neon)
3. [Idea 6의 핵심 아키텍처 설계와 구현](#3-idea-6의-핵심-아키텍처-설계와-구현)
4. [ARM64 NEON 단계별 최적화 시행착오 기록 (Baseline → v1 → v2 → v3 → v4)](#4-arm64-neon-단계별-최적화-시행착오-기록-baseline--v1--v2--v3--v4)
5. [전체 알고리즘 정면 비교 결과 (Windows x64 AVX2 & Windows ARM64 NEON)](#5-전체-알고리즘-정면-비교-결과-windows-x64-avx2--windows-arm64-neon)

---

## 1. 시작 질문과 Idea 1\~5로부터의 진화

1. **Idea 1 (64B 캐시 라인 리프 컷오프)**: 64바이트(`LEAF_SIZE`: 4B 키 16개, 8B 레코드 8개) 이하 구간에서는 분할 재귀를 멈추고 작은 블록 단위 정렬로 처리해 분기와 호출 오버헤드를 줄인다.
2. **Idea 3 (양방향 무분기 병합 + `O(1)` 외곽 경계 확인)**:
   - 병합 시 앞(`->`, 최솟값 선택)과 뒤(`<-`, 최댓값 선택)에서 동시에 좁혀 들어오는 양방향 무분기 병합(`bidirectional_branchless_merge`)은 단일 루프 내의 데이터 의존성 사슬(dependency chain)을 2개로 분리해 명령어 수준 병렬성(ILP)을 높인다.
   - 이진 탐색 전에 `data[begin] > right_min`, `data[end - 1] < left_max`를 `O(1)`로 먼저 확인하면 무작위 데이터에서 불필요한 경계 탐색을 건너뛸 수 있다.
3. **Idea 4 (전역 추세 프로브 + 8 KiB L1 타일링 + 핑퐁 병합)**:
   - 8 KiB(`TILE_SIZE`) 단위로 로컬 리프 정렬과 초기 병합을 L1D 캐시 안에서 끝내고, `data`와 `scratch` 버퍼를 교대(Ping-Pong)하여 매 단계 `memcpy`로 되돌리는 대역폭 낭비를 줄인다.
4. **Idea 5 (첫 번째 SIMD 비토닉 리프 + 단방향 벡터 병합 실험)**:
   - 64B 리프 1개를 가로 퍼뮤트(`vpermd` / NEON 셔플) 비토닉 네트워크로 정렬하고 단방향 벡터 비토닉 병합 루프를 적용했다.
   - 그러나 **x86-64 AVX2에서는 8B 레코드(`{ key, entity_id }`)가 스칼라 `Idea 4`보다 느려졌고**, **ARM64 NEON(Snapdragon X Elite)에서는 4B와 8B 모두 `std::sort`보다 느려지는 심각한 병목**이 관찰되었다.

여기서 **Idea 6**의 핵심 질문이 출발했습니다.

> *"왜 64B 리프 1개 단위 가로 비토닉 정렬과 단방향 SIMD 병합 루프(Idea 5)는 8B 레코드와 ARM64 NEON 환경에서 `std::sort`보다 느려졌을까? 어셈블리 수준에서 실제 병목을 규명하고, Idea 1·3·4의 핵심 통찰(양방향 이중 체인, `O(1)` 외곽 경계 확인, L1 타일 패스 축소)을 AVX2와 ARM64 NEON 각각의 레지스터 아키텍처에 맞게 재설계하면 모든 시나리오에서 `std::sort`를 압도할 수 있지 않을까?"*

---

## 2. 마이크로아키텍처 및 컴파일러 병목 원인 분석 (AVX2 & ARM64 NEON)

### 병목 ①: 가로 방향(Horizontal) 벡터 내부 정렬의 Cross-Lane 퍼뮤트 병목
- **AVX2**: 256비트 레지스터(`__m256i`) 하나 안에서 8개 원소를 비토닉 네트워크로 정렬하려고 매 단계마다 `_mm256_permutevar8x32_epi32`(`vpermd`)를 호출하면, 128비트 레인 경계를 가로지르는 **3사이클 지연 시간과 Port 5 단일 포트 병목**이 발생합니다.
- **ARM64 NEON**: `neon.s` 디스어셈블 결과, `Network<Neon>::full_leaf`는 16개 원소(64B)를 정렬하기 위해 4개 벡터 내부의 레인들을 끊임없이 섞는 수평 바이트닉 네트워크로 컴파일되어 **리프 1개당 166개의 NEON 명령어**를 실행하고 있었습니다.

### 병목 ②: 단방향 SIMD 비토닉 병합의 `carry` 레지스터 직렬 의존성 (21 사이클 지연 바닥)
두 정렬된 배열을 SIMD 비토닉 병합기로 합칠 때 한 방향(앞에서 뒤)으로만 진행하면 매 반복마다:

$$\text{merge\_vec}(\text{next\_vec}, \text{carry}) \rightarrow (\text{out\_low}, \text{new\_carry})$$

를 계산해야 합니다. 다음 반복의 비토닉 병합은 이전 반복의 `new_carry`가 나와야만 시작될 수 있습니다.
- **ARM64 NEON 실측 검증**: `neon.s` 분석 결과, `Network<Neon>::merge`는 1회 반복(원소 4개 = 16B 출력)마다 단일 `carry` 벡터가 **역순 뒤집기(`rev64` + `ext`, 2명령어) + 2단계 바이트닉 버터플라이(9명령어) = 총 11개 NEON 명령어(직렬 지연 시간 \~21 사이클)**를 통과해야 했습니다.
- `N = 1,000,000`에서 16개 병합 패스 동안 실행되는 벡터 병합 횟수는 $1,000,000 / 4 \times 16 = 4,000,000$회이므로, 단일 체인 구조에서는 **$4,000,000 \times 21\text{ cycles} = 84\text{M cycles}$ 이하로 절대 내려갈 수 없는 지연 시간 한계(Latency Floor)**에 갇혀 있었습니다(실제 측정된 하드웨어 사이클: **85.15M 사이클 / 28.82 ms**).

### 병목 ③: LLVM/Clang AArch64 백엔드의 De-If-Conversion (`CodeGenPrepare` `select-opt`) 함정
- `include/hybrid_sorts.hpp`의 `bidirectional_branchless_merge`는 C++ 소스 상에서 분기 예측 실패를 없애기 위해 삼항 연산자와 불리언 포인터 산술 연산(`l_fwd += !take_right; r_fwd += take_right;`)으로 작성되어 있었습니다.
- 그러나 `clang++ -O3 -S`로 추출한 ARM64 어셈블리(`idea4.s` 659\~685행, `neon.s` 230\~258행)를 분석한 결과, **LLVM AArch64 백엔드(`aarch64-enable-select-opt`)가 불리언 포인터 갱신을 "메모리 로드 주소 지연 사슬을 줄인다"는 휴리스틱에 의해 다시 조건 분기(`b.ge`, `b.le`) 2개로 역변환(De-If-Conversion)**하고 있었습니다:
  ```armasm
  // idea4.s (lines 659-685) - Clang이 Branchless C++ 코드를 다시 조건 분기로 역변환한 어셈블리
  .LBB4_22:
      ldr   w16, [x14]          // *r_fwd
      ldr   w17, [x13]          // *l_fwd
      cmp   w16, w17
      csel  w16, w16, w17, lt   // 값 선택은 csel을 쓰면서...
      b.ge  .LBB4_24            // [치명적!] 순방향 포인터 갱신을 위해 50% 확률의 조건 분기(b.ge) 생성!
      mov   x16, x13
      b     .LBB4_25
  .LBB4_24:
      add   x16, x13, #4
  .LBB4_25:
      ...
      cmp   w18, w17
      b.le  .LBB4_27            // [치명적!] 역방향 포인터 갱신에서도 두 번째 조건 분기(b.le) 생성!
  ```
- 이로 인해 루프 1회(원소 2개 병합)마다 **예측 불가능한(50% 확률) 조건 분기 2개가 실행**되어 Snapdragon X Elite(Oryon) 파이프라인에서 `N = 1,000,000` 기준 **66.54 ms (185.4M 사이클)**까지 성능이 급락했습니다.

### 병목 ④: 8B 레코드(`{ int32_t key, int32_t id }`)에서 128-bit NEON 2-Lane AoS 한계와 AVX2 이중 비교 비용
- **AVX2**: 32비트용 `vpminsd`/`vpmaxsd`는 있어도 8B `{ key, id }` 전용 하드웨어 `min/max`가 없어 `min`용 비교와 `max`용 비교를 따로 수행하면 비교·셔플·블렌드 비용이 2배로 증가합니다.
- **ARM64 NEON**: 128비트 NEON 레지스터 하나에 8B 레코드가 **단 2개**만 들어가므로, AoS(Array-of-Structures) 상태로 처리하던 Idea 5(`src/simd/neon.cpp`)는 매 단계마다 메모리에서 16바이트 마스크(`vld1q_u32`)를 로드하고 `vbslq_s32`를 수행하느라 **51.21 ms (150.2M 사이클)**가 소요되었습니다.

### 병목 ⑤: 이미 정렬된 캐시 라인의 Dirty 스토어 오염 & `4_Mostly_Descending`의 8 MB 힙 할당 페이지 폴트
- 이미 정렬된 리프/슈퍼블록에 무조건 SIMD 스토어를 수행하면 Clean 캐시 라인이 Modified(Dirty) 상태로 오염되어 L2/L3 Writeback 트래픽이 증가합니다.
- 또한 `4_Mostly_Descending`처럼 역전(`std::reverse`) 후 약간의 로컬 노이즈만 남은 배열에서 전체 핑퐁 병합으로 진입하면, **8 MB `scratch` 버퍼 힙 할당으로 인한 OS 소프트 페이지 폴트(2,048회, \~3.5 ms)**가 실제 정렬 시간보다 더 크게 발생합니다.

---

## 3. Idea 6의 핵심 아키텍처 설계와 구현

[`src/idea6_simd_transpose_merge.cpp`](../../src/idea6_simd_transpose_merge.cpp)는 위 병목들을 다음과 같이 해결합니다.

### 설계 1: 수직(Vertical) 정렬 네트워크 + 레지스터 행렬 전치 기반 256B Quad-Leaf 정렬 (`sort_quad_leaf_avx2` / `sort_quad_leaf_neon`)

64B 리프 1개를 가로로 정렬하는 대신, **4개의 캐시 라인(`4 * LEAF_SIZE = 256B`: 4B 키 64개 또는 8B 레코드 32개)**을 한 번에 레지스터(AVX2: YMM 8개 / ARM64 NEON: Q 레지스터 16개)에 적재합니다.

```
[적재 직후: 256B Quad-Leaf (AVX2 8×YMM / NEON 16×Q 레지스터)]
          Lane 0   Lane 1   ...   Lane K
   r0  [  d[0]     d[1]     ...   d[K]   ]
   r1  [  d[K+1]   d[K+2]   ...   d[2K]  ]
   ...
           │        │              │
           ▼        ▼              ▼
   1단계: 수직(Column-wise) 정렬 네트워크 (셔플 0회! 순수 min/max 비교기)
           │
           ▼
   2단계: 레지스터 내부 행렬 전치 (AVX2 8×8 / NEON 4×4×4)
           │
           ▼
   [전치 직후: 각 레지스터가 가로로 완전히 정렬된 벡터가 됨!]
           │
           ▼
   3단계: 레지스터 내부 벡터 병합으로 256B(4 * LEAF_SIZE) 단일 정렬 런 완성
```

- **ARM64 NEON 4B (`64 x int32_t`, 16개 Q 레지스터 `r0..r15`)**:
  1. 두 개의 독립적인 8-wire 수직 네트워크(각 19개 비교기, **셔플 0회**)와 8×8 수직 홀짝 병합(25개 비교기, **셔플 0회**)을 순수 `vminq_s32`/`vmaxq_s32`로 실행합니다.
  2. `transpose_4x4_s32`(`vtrn1q_s32`/`vtrn2q_s32` + `vtrn1q_s64`/`vtrn2q_s64`) 4회로 16개 레지스터를 전치합니다.
  3. 레지스터 내부에서 4+4 및 8+8 벡터 병합을 완료하여 **L1D 메모리 왕복 없이 64개 원소 정렬 런을 한 번에 배출**하고 타일 내부의 초기 병합 패스 2단계(`width = 16, 32`)를 완전히 제거합니다.
- **ARM64 NEON 8B (`32 x DistanceRecord`, 하드웨어 디인터리빙 SoA Quad-Leaf)**:
  1. ARM64 전용 **`vld2q_s32`** 명령어로 32개 레코드(256B)를 로드하는 즉시 **8개의 Key 벡터(`k0..k7`)와 8개의 ID 벡터(`i0..i7`)로 하드웨어 분리(AoS → SoA)**합니다.
  2. 이제 8B 레코드도 벡터당 2개가 아닌 **4개 레코드씩** 8-wire 수직 네트워크(`vcgtq_s32` + `vminq`/`vmaxq` + `vbslq_s32`) → 4×4 전치 → 15회 `merge_vec_neon_8b_soa` → **`vst2q_s32`** 하드웨어 재인터리빙 스토어로 처리됩니다.

### 설계 2: 아키텍처 맞춤 고속 바이트닉 버터플라이 (AVX2 In-Lane 셔플 & NEON Symmetric 3-Stage `vzip` 파이프라인)

- **AVX2 단일 비교 8B 커널 (`bitonic_step_8b`)**:
  `min` 레인과 `max` 레인의 교환 조건이 $v_i.\text{key} > v_j.\text{key}$ 하나로 동일하다는 항등식을 이용해 `_mm256_cmpgt_epi32` 단 1회 비교만으로 동점 레코드 손실 없이 스왑을 수행합니다.
- **ARM64 NEON Symmetric 3-Stage `vzip1q_s32` / `vzip2q_s32` 버터플라이**:
  Clang AArch64 백엔드에서 `vtrn1q_s64`/`vtrn2q_s64`가 불필요한 `mov` + `ins v.d[1], v.d[0]` 명령어를 생성하는 문제를 어셈블리(`idea6_v1_4b.s`)에서 포착하고, 이를 **3단계 연속 `vzip1q_s32` / `vzip2q_s32` + `cas` 대칭 파이프라인**으로 대체하여 레지스터 복사 명령어를 100% 제거했습니다:
  ```cpp
  HSORT_ALWAYS_INLINE void bitonic_butterfly_neon_4b(int32x4_t& a, int32x4_t& b) noexcept {
      int32x4_t z0 = vzip1q_s32(a, b),  z1 = vzip2q_s32(a, b);
      cas_neon_4b(z0, z1);
      int32x4_t w0 = vzip1q_s32(z0, z1), w1 = vzip2q_s32(z0, z1);
      cas_neon_4b(w0, w1);
      a = vzip1q_s32(w0, w1); // 정렬된 하위 4원소
      b = vzip2q_s32(w0, w1); // 정렬된 상위 4원소
  }
  ```

### 설계 3: 양방향 이중 체인(`carry_fwd` + `carry_bwd`) 무분기 SIMD 병합 & 역순화 Critical Path 분리

1. **두 개의 독립 의존성 사슬(`carry_fwd`, `carry_bwd`)**:
   순방향(`->`)과 역방향(`<-`)에서 동시에 SIMD 바이트닉 병합을 진행하여 단일 `carry` 지연 한계(84M 사이클 바닥)를 깨고 ILP를 2배로 확장했습니다.
2. **역순화 연산을 Loop-Carried `carry` 밖으로 제거**:
   순방향 병합(`merge_vec_neon_4b_fwd`, `merge_vec_neon_8b_soa_fwd`)에서는 새로 로드한 벡터 `low`를 뒤집고, 역방향 병합(`merge_vec_neon_4b`, `merge_vec_neon_8b_soa`)에서는 새로 로드한 벡터 `high`를 뒤집도록 설계하여, 루프 간 이어지는 `carry_fwd`와 `carry_bwd` 모두 역순화 지연(`rev64` + `ext`)을 Critical Path에서 완전히 제거했습니다.
3. **`HSORT_UNPREDICTABLE` (`__builtin_unpredictable`)을 통한 하드웨어 `csel` 강제**:
   LLVM AArch64의 De-If-Conversion을 원천 차단하여 모든 포인터 갱신과 스칼라 꼬리 병합이 100% 무분기 `cmp + cset + csel + add`로 컴파일되도록 보장했습니다.
4. **8B Dual-Chain 4-Record SoA NEON 병합**:
   8B 외부 병합에서도 `vld2q_s32`로 한 번에 4개 레코드(32B)씩 디인터리빙 로드하여 6사이클 L1D Load-to-Address 포인터 의존성을 4개 레코드에 걸쳐 상각(레코드당 주소 지연 **6.0cy → 1.5cy**)했습니다.

### 설계 4: 데이터 분포 적응형 고속 단락 경로 (Adaptive Fast-Paths)

1. **Pre-Sorted Quad-Leaf Clean 캐시 라인 보존 (`is_quad_leaf_sorted_avx2` / `is_quad_leaf_sorted_neon`)**:
   이미 정렬된 256B 블록은 메모리 쓰기(Store)를 생략해 캐시 라인이 Dirty 상태로 오염되는 것을 방지합니다.
2. **`O(1)` Disjoint Reversed-Run Fast-Path**:
   `pingpong_merge_pair` 진입 시 `src[mid - 1] <= src[mid]`(정방향 정렬 완료)뿐만 아니라 **`src[end - 1] <= src[begin]`(오른쪽 런 전체가 왼쪽 런 전체 이하인 역순 블록)**도 `O(1)`에 감지하여 두 번의 `std::memcpy`로 즉시 블록 위치를 교환합니다(`6_Organ_Pipe`에서 **2.4× 추가 가속**).
3. **Zero-Allocation In-Place Bounded Insertion Sort (`try_bounded_inplace_insertion_sort`)**:
   전역 추세 프로브(`probe_and_handle_global_trend`)에서 내림차순 배열을 뒤집은 직후, **8 MB 힙 할당(`scratch`)을 수행하기 전에** 로컬 이동 거리(`4 * LEAF_SIZE`)와 총 이동 횟수(`n / 8`)가 제한된 In-Place 삽입 정렬을 시도합니다. 이로써 `4_Mostly_Descending`에서 2,048회의 OS 소프트 페이지 폴트와 16 MB 핑퐁 복사를 100% 제거하여 **0.78 ms (4B) / 0.86 ms (8B)** 만에 정렬을 끝냅니다.
4. **Equal-Key Plateau Streamer**:
   병합 도중 양쪽 런에 동일한 키가 연속되는 고중복 구간(`kl == kr && l_fwd[LANES - 1] == kl`)을 만나면 바이트닉 버터플라이를 건너뛰고 벡터 복사 루프로 직통 스트리밍합니다.

---

## 4. ARM64 NEON 단계별 최적화 시행착오 기록 (Baseline → v1 → v2 → v3 → v4)

Snapdragon X Elite (`ARM64_Oryon`, `clang++ 23.1.5 -O3`, `N = 1,000,000`)에서 4단계에 걸쳐 가설을 세우고 검증한 엔지니어링 진행 기록입니다.

| 단계 | 핵심 설계 변경 및 시행착오 분석 | 4B `Random` | 4B `Buckets` | 4B `Descending` | 4B `OrganPipe` | 8B `Random` | 8B `Descending` | 8B `OrganPipe` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline (Before)** | • Idea 6 ARM64 `N > 16` 폴백 버그<br/>• Clang `b.ge`/`b.le` 역변환으로 분기 실패 폭발 | 66.54 ms<br/>*(0.28×)* | 37.57 ms<br/>*(0.18×)* | 2.78 ms<br/>*(1.70×)* | 7.57 ms<br/>*(4.83×)* | 69.52 ms<br/>*(0.74×)* | 6.17 ms<br/>*(0.63×)* | 11.55 ms<br/>*(3.38×)* |
| **v1** | • `__builtin_unpredictable`로 `b.ge`/`b.le` 제거<br/>• 4B/8B 256B Quad-Leaf & 4B Dual-Chain NEON 병합<br/>• *문제 발견*: Clang `vtrn1q_s64`가 `mov+ins` 생성, 8B 스칼라 병합의 6cy 주소 지연, `OrganPipe` 역순 병합 오버헤드 | 9.54 ms<br/>*(1.98×)* | 9.54 ms<br/>*(0.73×)* | 2.18 ms<br/>*(2.17×)* | 5.50 ms<br/>*(6.69×)* | 24.22 ms<br/>*(2.12×)* | 4.21 ms<br/>*(0.97×)* | 15.70 ms<br/>*(2.51×)* |
| **v2** | • 3단계 `vzip1q`/`vzip2q` 대칭 버터플라이로 `mov+ins` 제거<br/>• 8B Dual-Chain 4-Record SoA NEON 병합(`vld2q_s32`)<br/>• `O(1)` Disjoint Reversed-Run Fast-Path 추가<br/>• *문제 발견*: `Mostly_Descending`에서 8 MB 힙 할당 페이지 폴트(\~3.5 ms) 발생 | 9.61 ms<br/>*(1.97×)* | 9.88 ms<br/>*(0.71×)* | 1.95 ms<br/>*(2.43×)* | 2.98 ms<br/>*(12.28×)* | 18.31 ms<br/>*(2.81×)* | 4.34 ms<br/>*(0.92×)* | 6.56 ms<br/>*(5.95×)* |
| **v3 (최종 채택)** | • Zero-Allocation In-Place Bounded Insertion 추가(페이지 폴트 0회)<br/>• Equal-Key Plateau Streamer 및 8B 단일 로드 정렬 검사 추가 | **9.40 ms**<br/>**(2.01×)** | **8.50 ms**<br/>**(0.81×)** | **0.78 ms**<br/>**(6.03×)** | **2.86 ms**<br/>**(12.77×)** | **18.18 ms**<br/>**(2.83×)** | **0.86 ms**<br/>**(4.48×)** | **6.10 ms**<br/>**(6.39×)** |
| **v4 (가설 기각)** | • *실험 가설*: `trim_left`를 `lower_bound`로 바꾸고 핫 루프 내 스칼라 잔여분 정리 루프 2개 추가<br/>• *결과*: 첫 버킷 `memcpy` 이득 상실 및 루프 I-Cache 팽창으로 4B `Buckets`가 `9.19 ms`로 소폭 퇴보 확인 후 **즉시 v3로 롤백** | 9.42 ms | 9.19 ms | 0.85 ms | 2.78 ms | 18.55 ms | 0.85 ms | 6.11 ms |

*(괄호 안 배수는 동일 시나리오 `std::sort` 대비 가속비)*

---

## 5. 전체 알고리즘 정면 비교 결과 (Windows x64 AVX2 & Windows ARM64 NEON)

### 5.1 Windows ARM64 (`Snapdragon X Elite / Qualcomm Oryon` · `clang++ 23.1.5 -O3` · `N = 1,000,000`)

- **원시 측정 데이터**: [`docs/windowsARM/pure_cache_results_N=1000000.csv`](../windowsARM/pure_cache_results_N=1000000.csv) · [`docs/windowsARM/pure_cache_results_N=1000000.md`](../windowsARM/pure_cache_results_N=1000000.md)
- **정확성 검증**: `test_4b`, `test_8b` 각 **4,294개 검증 항목 100% 통과**

#### 4B 정수 키 (`4B_Int32`, `N = 1,000,000`, 중앙값 단위: ms / 하드웨어 사이클: M cycles)

| 시나리오 | `std::sort` | `std::stable_sort` | Idea&nbsp;4 (Scalar) | Idea&nbsp;5 (NEON) | Idea&nbsp;6 (**Before**) | **Idea&nbsp;6 (After v3)** | `std::sort`&nbsp;대비 | Before&nbsp;대비 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `0_Uniform_Random` | 18.86 (55.5M) | 13.55 (39.9M) | 62.50 (184.5M) | 28.82 (84.7M) | 66.54 (185.4M) | **9.40 (27.7M)** | **2.01×** | **7.08×** |
| `1_Frame_Jitter` | 16.30 (48.0M) | 11.94 (35.3M) | 30.91 (91.1M) | 15.96 (47.3M) | 31.22 (91.4M) | **6.21 (18.5M)** | **2.62×** | **5.03×** |
| `2_Append_5Pct_New` | 16.46 (48.4M) | 8.18 (24.2M) | 5.90 (17.4M) | 5.74 (16.8M) | 6.49 (17.5M) | **3.24 (9.6M)** | **5.07×** | **2.00×** |
| `3_Distance_Buckets` | 6.89 (20.2M) | 14.35 (42.3M) | 37.35 (110.0M) | 28.80 (84.8M) | 37.57 (110.7M) | **8.50 (24.7M)** | 0.81× *(stable 대비 1.69×)* | **4.42×** |
| `4_Mostly_Descending` | 4.72 (13.9M) | 11.55 (34.0M) | 3.33 (9.5M) | 3.55 (10.4M) | 2.78 (8.1M) | **0.78 (2.3M)** | **6.03×** | **3.55×** |
| `5_Gaussian_Normal` | 14.99 (44.2M) | 16.95 (50.1M) | 57.92 (170.8M) | 28.60 (84.5M) | 58.12 (170.7M) | **9.74 (28.8M)** | **1.54×** | **5.96×** |
| `6_Organ_Pipe` | 36.58 (108.0M) | 10.88 (32.0M) | 7.53 (22.2M) | 9.67 (28.5M) | 7.57 (22.2M) | **2.86 (8.4M)** | **12.77×** | **2.64×** |

#### 8B 키+ID 레코드 (`8B_Record`, `N = 1,000,000`, 중앙값 단위: ms / 하드웨어 사이클: M cycles)

| 시나리오 | `std::sort` | `std::stable_sort` | Idea&nbsp;4 (Scalar) | Idea&nbsp;5 (NEON) | Idea&nbsp;6 (**Before**) | **Idea&nbsp;6 (After v3)** | `std::sort`&nbsp;대비 | Before&nbsp;대비 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `0_Uniform_Random` | 51.36 (150.8M) | 44.03 (129.7M) | 68.43 (201.9M) | 51.21 (151.0M) | 69.52 (204.6M) | **18.18 (54.0M)** | **2.83×** | **3.82×** |
| `1_Frame_Jitter` | 28.81 (85.3M) | 34.92 (103.1M) | 37.44 (110.3M) | 28.85 (85.5M) | 37.99 (111.7M) | **13.51 (39.7M)** | **2.13×** | **2.81×** |
| `2_Append_5Pct_New` | 18.10 (53.4M) | 19.88 (58.5M) | 9.08 (26.5M) | 10.89 (32.0M) | 9.66 (28.4M) | **6.30 (18.7M)** | **2.87×** | **1.53×** |
| `3_Distance_Buckets` | 20.80 (61.5M) | 43.99 (129.7M) | 42.74 (126.4M) | 50.77 (150.0M) | 44.51 (131.0M) | **15.47 (45.6M)** | **1.35×** | **2.88×** |
| `4_Mostly_Descending` | 3.86 (11.5M) | 30.60 (90.8M) | 5.79 (17.1M) | 5.98 (17.7M) | 6.17 (18.2M) | **0.86 (2.6M)** | **4.48×** | **7.15×** |
| `5_Gaussian_Normal` | 44.63 (131.7M) | 43.88 (129.5M) | 63.59 (187.9M) | 50.85 (150.0M) | 64.22 (189.2M) | **18.95 (55.7M)** | **2.36×** | **3.39×** |
| `6_Organ_Pipe` | 38.99 (115.3M) | 25.91 (76.6M) | 11.07 (32.7M) | 18.40 (54.4M) | 11.55 (34.0M) | **6.10 (18.0M)** | **6.39×** | **1.89×** |

---

### 5.2 Windows x64 (`Intel Core Ultra 7 270K Plus` · `g++ -O3` · `N = 1,000,000`)

- **원시 요약 CSV**: [`key_only_4b_summary.csv`](key_only_4b_summary.csv) · [`key_id_8b_summary.csv`](key_id_8b_summary.csv)

#### 4B 정수 키 (`key_only_4b`, 단위: ms)

| 시나리오 | `std::sort` | Idea&nbsp;3 | Idea&nbsp;4 | Idea&nbsp;5 | **Idea&nbsp;6** | `std::sort`&nbsp;대비 | Idea&nbsp;4&nbsp;대비 | Idea&nbsp;5&nbsp;대비 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_3d_distances` | 45.54 | 24.51 | 22.56 | 13.14 | **6.76** | **6.73×** | **3.33×** | **1.94×** |
| `next_frame_motion` | 26.24 | 15.62 | 16.64 | 10.32 | **5.90** | **4.45×** | **2.82×** | **1.75×** |
| `append_5pct_new` | 17.50 | 3.76 | 4.57 | 3.63 | **3.06** | **5.71×** | **1.49×** | **1.19×** |
| `distance_buckets` | 22.88 | 24.42 | 22.27 | 13.10 | **6.55** | **3.49×** | **3.40×** | **2.00×** |
| `mostly_descending` | 3.30 | 19.30 | 2.98 | 2.82 | **2.23** | **1.48×** | **1.34×** | **1.27×** |
| `gaussian_normal` | 40.68 | 24.29 | 22.79 | 13.33 | **6.98** | **5.83×** | **3.27×** | **1.91×** |
| `organ_pipe` | 39.97 | 11.37 | 9.24 | 4.48 | **3.86** | **10.34×** | **2.39×** | **1.16×** |

#### 8B 키+ID 레코드 (`key_id_8b`, 단위: ms)

| 시나리오 | `std::sort` | Idea&nbsp;3 | Idea&nbsp;4 | Idea&nbsp;5 | **Idea&nbsp;6** | `std::sort`&nbsp;대비 | Idea&nbsp;4&nbsp;대비 | Idea&nbsp;5&nbsp;대비 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fresh_3d_distances` | 43.85 | 32.25 | 29.32 | 32.54 | **18.19** | **2.41×** | **1.61×** | **1.79×** |
| `next_frame_motion` | 25.90 | 19.05 | 21.99 | 21.62 | **14.00** | **1.85×** | **1.57×** | **1.54×** |
| `append_5pct_new` | 17.46 | 4.74 | 8.33 | 9.18 | **6.56** | **2.66×** | **1.27×** | **1.40×** |
| `distance_buckets` | 22.70 | 32.63 | 29.56 | 33.21 | **18.30** | **1.24×** | **1.62×** | **1.81×** |
| `mostly_descending` | 4.33 | 22.67 | 6.01 | 6.82 | **4.47** | **0.97×** | **1.34×** | **1.53×** |
| `gaussian_normal` | 39.14 | 32.21 | 29.85 | 33.04 | **18.75** | **2.09×** | **1.59×** | **1.76×** |
| `organ_pipe` | 40.33 | 12.45 | 13.07 | 13.67 | **10.75** | **3.75×** | **1.22×** | **1.27×** |
