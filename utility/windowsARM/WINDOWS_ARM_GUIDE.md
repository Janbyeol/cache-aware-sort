# Windows ARM64 (Snapdragon X Elite / Oryon) 순수 정렬 벤치마크 가이드

Windows 11 ARM64 노트북(Qualcomm Snapdragon X Elite / Oryon 코어)에서 데이터 생성(`generate_game_workload`) 및 메모리 복사(`memcpy`) 구간을 100% 배제하고, **순수 정렬 함수 실행 구간의 하드웨어 Thread CPU Cycles(`QueryThreadCycleTime`) 및 고정밀 실행 시간(`QueryPerformanceCounter`)**을 측정해 `docs/windowsARM/` 폴더에 저장하는 매뉴얼입니다.

---

## 1. `utility/windowsARM/` 폴더 구성

| 파일 경로 | 역할 |
| :--- | :--- |
| **`utility/windowsARM/pure_perf_bench.cpp`** | **Windows ARM64 순수 정렬 프로파일러** — Oryon 코어 클러스터 간 스레드 이동 방지(`SetThreadAffinityMask`) + 데이터 생성/복사 구간을 100% 제외한 순수 정렬 구간의 Thread Cycles 및 QPC 런타임 측정 |
| **`utility/windowsARM/run_pure_perf.bat`** | **컴파일 및 실행 스크립트** — ARM64 아키텍처 자동 감지(`vcvarsall.bat arm64` / `clang++` / `g++`) 후 단일 $N$ 또는 전체 10개 $N$(`--all`) 측정 |
| **`utility/windowsARM/run_pure_perf_all.bat`** | **10개 $N$ 스케일 일괄 측정 + CPU 쿨링 대기** (`16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000`) 후 `docs/windowsARM/`에 자동 저장 |
| **`utility/windowsARM/run_pure_perf.ps1`** | PowerShell 백엔드 스크립트 (`run_pure_perf.bat`에서 자동 호출) |

---

## 2. Snapdragon X Elite (Oryon) 아키텍처 맞춤 설계 특징

1. **네이티브 ARM64(`arm64`) 자동 타겟팅:**
   - 호스트 OS가 Windows ARM64이면 터미널이 x64 에뮬레이션 모드로 열려 있더라도 `$env:PROCESSOR_ARCHITEW6432`를 감지하여 Visual Studio의 **`arm64` 네이티브 컴파일러(`vcvarsall.bat arm64`)**로 빌드합니다 (`Arch = ARM64_Oryon` 출력으로 확인 가능).
2. **Oryon 3-Cluster L2 캐시 이동 방지 (`SetThreadAffinityMask`):**
   - Snapdragon X Elite는 4코어 단위로 12MB L2 캐시를 공유하는 3개 클러스터 구조를 가집니다. 측정 도중 Windows 스케줄러가 스레드를 다른 클러스터로 옮기면 L2 캐시가 통째로 Cold 상태가 되므로, 단일 코어에 스레드를 고정(`Affinity = Core 0`, `HIGH_PRIORITY_CLASS`)하여 클러스터 점프 노이즈를 원천 차단합니다.
3. **소규모 $N$ ($N < 32,768$) 사전 배치(Pre-copied Batch) 정밀 측정:**
   - $N = 16 \sim 8,192$처럼 수십 나노초 만에 끝나는 구간은 타이머 바깥에서 여러 개의 배열 복사본을 미리 준비해 둔 뒤, 순수 정렬 루프만 연속 실행하고 배치 수로 나누어 단일 사이클 단위의 정밀도를 확보합니다.

---

## 3. 실행 명령어 (CMD 또는 PowerShell)

프로젝트 루트(`cache-aware-sort`) 폴더에서 아래 명령어를 실행합니다.

### 3.1 [권장] 10개 $N$ 스케일 한번에 일괄 측정 + CPU 쿨링 대기 (`--all`)

$N \in \{16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000\}$ 전체를 순차 측정하며, **$N$이 바뀔 때마다 CPU 열/부스트 클럭 안정화를 위한 쿨링 대기 시간(기본 5초)**을 두어 공정하게 측정합니다. 결과는 **`docs/windowsARM/pure_cache_results_N=<N>.csv`** (개별 파일 10개) 및 **`docs/windowsARM/pure_cache_results_all.csv`** (통합 파일)로 자동 저장됩니다.

```powershell
# 기본 일괄 실행 (10개 N 전체, 각 3회 반복 중앙값, N 전환 시 5초 쿨링 -> docs/windowsARM/ 폴더에 저장)
.\utility\windowsARM\run_pure_perf.bat --all
# 또는
.\utility\windowsARM\run_pure_perf_all.bat

# 반복 횟수(5회)와 N 전환 쿨링 시간(10초)을 직접 지정하여 실행
.\utility\windowsARM\run_pure_perf.bat --all 5 10 docs\windowsARM
```

### 3.2 단일 $N$ 지정 측정

```powershell
# 기본 실행 (N=1,000,000, 3회 반복 -> pure_cache_results.csv 및 pure_cache_results.md 저장)
.\utility\windowsARM\run_pure_perf.bat

# 특정 N 크기 지정 실행
.\utility\windowsARM\run_pure_perf.bat 32768 5 pure_l2_32k.csv
```

---

## 4. 출력 CSV 컬럼 설명 (`docs/windowsARM/pure_cache_results_N=*.csv`)

| 컬럼명 | 설명 |
| :--- | :--- |
| **`Data_Type`** | `4B_Int32` (4바이트 정수) 또는 `8B_Record` (8바이트 Key+ID 레코드) |
| **`Arch`** | 컴파일된 바이너리 아키텍처 (`ARM64_Oryon` 또는 `x86_64`) |
| **`Scenario`** | `0_Uniform_Random` ~ `6_Organ_Pipe_Symmetric` (7개 워크로드 시나리오) |
| **`N`** | 정렬 배열 원소 개수 |
| **`Algorithm`** | 측정 대상 정렬 알고리즘 이름 |
| **`Runtime_us` / `Runtime_ms`** | 순수 정렬 구간 실행 시간 ($\mu\text{s}$ / $\text{ms}$, `QueryPerformanceCounter` 기준) |
| **`Speedup_Time`** | 동일 시나리오·$N$의 `std::sort` 대비 실행 시간 가속비 (`Time_std / Time_algo`) |
| **`Cycles`** | 순수 정렬 구간 하드웨어 스레드 CPU 사이클 (`QueryThreadCycleTime` 기준) |
| **`Speedup_Cycles`** | 동일 시나리오·$N$의 `std::sort` 대비 CPU 사이클 가속비 (`Cycles_std / Cycles_algo`) |
| **`Cycles_Per_Elem`** | 원소 1개당 소요 CPU 사이클 (`Cycles / N`) |
| **`Ns_Per_Elem`** | 원소 1개당 소요 시간 ($\text{ns}$, `Runtime_ns / N`) |
| **`Throughput_Mops`** | 초당 정렬 처리량 (백만 개/초, `Million Elements / sec`) |
