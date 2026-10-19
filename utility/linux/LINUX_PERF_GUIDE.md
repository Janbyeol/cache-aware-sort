# Ubuntu Linux CPU 캐시 히트율 및 분기 예측 프로파일링 가이드 (`perf`)

우분투(Ubuntu) 서버에서 `perf stat` 하드웨어 성능 카운터(PMU)를 이용해 각 정렬 알고리즘의 **순수 L1 데이터 캐시 적중률, LLC(L3) 적중률, 분기 예측 실패율(Branch Miss Rate), IPC**를 측정하고 로컬 `.csv` 및 `.md` 파일로 저장하는 매뉴얼입니다.

---

## 1. `utility/linux/` 폴더 구성

| 파일 경로 | 역할 |
| :--- | :--- |
| **`utility/linux/pure_perf_bench.cpp`** | **(권장) 커널 `perf_event_open` 직접 제어 벤치마크** — 데이터 생성(`generate_game_workload`) 및 메모리 복사(`memcpy`) 구간을 100% 배제하고 **순수 정렬 함수 실행 구간의 하드웨어 카운터만 정밀 측정** |
| **`utility/linux/run_pure_perf.sh`** | **(권장)** `pure_perf_4b` / `pure_perf_8b` 자동 빌드 및 단일 $N$ 또는 전체 $N$ 일괄 측정(`--all`) 실행 |
| **`utility/linux/run_pure_perf_all.sh`** | **(권장) 10개 $N$ 스케일 일괄 측정 + 쿨링 대기** (`16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000`) |
| `utility/linux/profile_one.cpp` | 외부 `perf stat` 명령어와 함께 쓰는 단일 알고리즘 실행기 |
| `utility/linux/run_perf_csv.sh` | 외부 `perf stat` 래퍼 스크립트 |
| `utility/linux/LINUX_PERF_GUIDE.md` | 리눅스 프로파일링 명령어 사용 설명서 |

---

## 2. 최초 1회: `perf` 하드웨어 카운터 권한 해제

우분투는 기본적으로 일반 사용자의 하드웨어 카운터 접근을 차단(`perf_event_paranoid = 4`)합니다. 측정 전 아래 명령어로 제한을 해제합니다.

```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
sudo sysctl -w kernel.kptr_restrict=0
```

> **참고:** Windows에서 작성된 `.sh` 파일을 리눅스로 옮겼을 때 줄바꿈(`CRLF`) 오류가 나면 아래 한 줄로 변환 후 실행하세요:
> ```bash
> sed -i 's/\r$//' utility/linux/*.sh && chmod +x utility/linux/*.sh
> ```

---

## 3. [권장] 데이터 생성 노이즈 0% 순수 정렬 측정 (`utility/linux/run_pure_perf.sh`)

리눅스 커널 시스템 콜(`perf_event_open` + `ioctl(PERF_EVENT_IOC_ENABLE / DISABLE)`)을 C++ 코드 내부에서 직접 제어하여, **데이터 생성(`generate_game_workload`)과 버퍼 복사(`memcpy`) 동안에는 하드웨어 카운터를 완전히 꺼두고 오직 정렬 함수가 도는 순간만 100% 순수 측정**합니다.

### 3.1 전체 10개 $N$ 스케일 한번에 일괄 측정 + CPU 쿨링 대기 (`--all`)

$N \in \{16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000\}$ 전체를 순차 측정하며, **$N$이 바뀔 때마다 CPU 열/부스트 클럭 안정화를 위한 쿨링 대기 시간(기본 5초)**을 두어 공정하게 측정합니다. 결과는 `docs/linux/pure_cache_results_N=<N>.csv` (개별 파일) 및 `docs/linux/pure_cache_results_all.csv` (통합 파일)로 자동 저장됩니다.

- **문법:** `./utility/linux/run_pure_perf.sh --all [반복횟수(기본:3)] [쿨링시간_초(기본:5)] [저장폴더(기본:docs/linux)]` (또는 `./utility/linux/run_pure_perf_all.sh`)

```bash
# 기본 일괄 실행 (10개 N 전체, 각 3회 평균, N 전환 시 5초 쿨링 -> docs/linux/ 폴더에 저장)
./utility/linux/run_pure_perf.sh --all
# 또는
./utility/linux/run_pure_perf_all.sh

# 반복 횟수(5회)와 N 전환 쿨링 시간(10초)을 직접 지정하여 실행
./utility/linux/run_pure_perf.sh --all 5 10 docs/linux
```

### 3.2 단일 $N$ 지정 측정 (기존 명령어)

- **문법:** `./utility/linux/run_pure_perf.sh [원소개수_N] [반복횟수] [저장할_CSV파일명]`

```bash
# 기본 실행 (N=1,000,000, 3회 평균 -> pure_cache_results.csv 및 pure_cache_results.md 저장)
./utility/linux/run_pure_perf.sh

# 캐시 계층 크기별로 각각 다른 파일 이름으로 리눅스 서버 로컬에 저장하기
./utility/linux/run_pure_perf.sh 2048     10 pure_l1_2k.csv      # L1 캐시 크기 (2,048개, 10회 평균)
./utility/linux/run_pure_perf.sh 131072   5  pure_l2_131k.csv    # L2 캐시 크기 (131,072개, 5회 평균)
./utility/linux/run_pure_perf.sh 1000000  3  pure_l3_1m.csv      # L3/DRAM 경계 (1,000,000개, 3회 평균)
./utility/linux/run_pure_perf.sh 10000000 3  pure_dram_10m.csv   # 대용량 DRAM (10,000,000개, 3회 평균)
```

---

## 4. 외부 `perf stat` 래퍼 스크립트 (`utility/linux/run_perf_csv.sh`)

프로젝트 루트(`~/cache-aware-sort`) 또는 `utility/linux/` 폴더 어디서 실행하든 자동으로 경로를 찾아 `utility/linux/prof_4b`, `utility/linux/prof_8b`를 빌드하고 측정을 수행합니다.

- **문법:** `./utility/linux/run_perf_csv.sh [원소개수_N] [반복횟수] [저장할_CSV파일명]`

```bash
./utility/linux/run_perf_csv.sh
```

---

## 4. 수동 컴파일 및 개별 알고리즘 단발성 측정 (`prof_4b`, `prof_8b`)

### 4.1 수동 빌드 명령어 (프로젝트 루트 기준)

```bash
# 4B 정수 (int32_t) 프로파일러 빌드
g++ -O3 -march=native -std=c++20 -Iinclude \
    src/baseline_sorts.cpp src/idea1_cache_leaf_quick.cpp src/idea2_mean_pivot_quick.cpp \
    src/idea3_boundary_trimmed_merge.cpp src/idea4_tiled_pingpong_merge.cpp \
    src/benchmark_api.cpp utility/linux/profile_one.cpp -o utility/linux/prof_4b

# 8B 레코드 (Key + Entity ID, -DBENCH_RECORD=1) 프로파일러 빌드
g++ -O3 -march=native -std=c++20 -Iinclude -DBENCH_RECORD=1 \
    src/baseline_sorts.cpp src/idea1_cache_leaf_quick.cpp src/idea2_mean_pivot_quick.cpp \
    src/idea3_boundary_trimmed_merge.cpp src/idea4_tiled_pingpong_merge.cpp \
    src/benchmark_api.cpp utility/linux/profile_one.cpp -o utility/linux/prof_8b
```

### 4.2 개별 실행 인자 설명

- **문법:** `./utility/linux/prof_4b <알고리즘> <시나리오번호> <원소개수_N>`

| 인자 | 입력 값 | 설명 |
| :--- | :--- | :--- |
| **실행 파일** | `./utility/linux/prof_4b` / `./utility/linux/prof_8b` | `prof_4b`: 4B 정수(`int32_t`) / `prof_8b`: 8B 레코드(`Key+ID`) |
| **`<알고리즘>`** | `std`, `stable`, `idea1`, `idea2`, `idea3`, `idea4` | 측정할 정렬 알고리즘 선택 |
| **`<시나리오번호>`** | `0` ~ `6` | `0`: 완전 무작위 (`Fresh3DDistances`)<br>`1`: 프레임 미세 흔들림 (`NextFrameMotion`)<br>`2`: 앞 95% 정렬 + 뒤 5% 난수 (`Append5PctNew`)<br>`3`: 거리 버킷 중복 키 (`DistanceBuckets`)<br>`4`: 역순 + 1% 노이즈 (`MostlyDescending`)<br>`5`: 가우시안 정규분포 (`GaussianNormal`)<br>`6`: 오르간 파이프 대칭 (`OrganPipe`) |
| **`<원소개수_N>`** | 예: `2048`, `131072`, `1000000` | 정렬할 배열의 원소 개수 (생략 시 기본값 `1000000`) |

```bash
# 예시 1: 시나리오 2(95% 정렬 + 5% 난수)에서 std::sort와 Idea 3 비교
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses ./utility/linux/prof_4b std 2 1000000
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses ./utility/linux/prof_4b idea3 2 1000000

# 예시 2: 단발성 측정 결과를 로컬 텍스트 파일(single_result.txt)로 저장 (-o 옵션)
perf stat -o single_result.txt -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses ./utility/linux/prof_4b idea4 0 1000000
```

---

## 5. 로컬에 저장된 결과 파일 열람 및 계산 공식

### 5.1 터미널에서 결과 표 열람

```bash
# 1) 생성된 Markdown 표(.md) 파일 바로 보기
cat cache_hit_results.md

# 2) CSV 파일을 줄 맞춰서 열람 (좌우 화살표로 이동, q로 종료)
column -t -s, cache_hit_results.csv | less -S
```

### 5.2 CSV / MD 지표 산출 공식

| 컬럼명 | 계산 공식 (`perf` 카운터 기준) | 의미 |
| :--- | :--- | :--- |
| **`L1_Hit_Rate(%)`** | `(1 - L1-dcache-load-misses / L1-dcache-loads) * 100` | L1 데이터 캐시 적중률 (높을수록 우수) |
| **`L1_Miss_Rate(%)`** | `(L1-dcache-load-misses / L1-dcache-loads) * 100` | L1 데이터 캐시 미스율 (낮을수록 우수) |
| **`LLC_Hit_Rate(%)`** | `(1 - cache-misses / cache-references) * 100` | L3(Last-Level Cache) 적중률 (높을수록 우수) |
| **`LLC_Miss_Rate(%)`** | `(cache-misses / cache-references) * 100` | L3 미스 및 DRAM 접근 비율 (낮을수록 우수) |
| **`Branch_Hit_Rate(%)`** | `(1 - branch-misses / branches) * 100` | CPU 분기 예측 성공률 (높을수록 우수) |
| **`Branch_Miss_Rate(%)`** | `(branch-misses / branches) * 100` | CPU 분기 예측 실패율 (낮을수록 우수) |
| **`IPC`** | `instructions / cycles` | 클럭당 실행 명령어 수 (파이프라인 효율) |
