#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "사용법: $0 [원소개수_N] [반복횟수] [저장할_CSV파일명]"
    echo "  기본값: N=1000000, 반복횟수=3, 파일명=pure_cache_results.csv"
    echo ""
    echo "예시:"
    echo "  $0                                   # 기본 실행 (N=1,000,000 -> pure_cache_results.csv, .md)"
    echo "  $0 131072 5 pure_l2_131k.csv         # L2 크기(131,072개) 5회 평균"
    echo "  $0 10000000 3 pure_dram_10m.csv      # 대용량(1,000만 개) 3회 평균"
    exit 0
fi

N=${1:-1000000}
REPEATS=${2:-3}
CSV_FILE=${3:-"pure_cache_results.csv"}
MD_FILE="${CSV_FILE%.csv}.md"

PURE_4B="$SCRIPT_DIR/pure_perf_4b"
PURE_8B="$SCRIPT_DIR/pure_perf_8b"

echo "[1/3] pure_perf_4b 및 pure_perf_8b 컴파일 중..."
g++ -O3 -march=native -std=c++20 -I"$ROOT_DIR/include" \
    "$ROOT_DIR/src/baseline_sorts.cpp" \
    "$ROOT_DIR/src/idea1_cache_leaf_quick.cpp" \
    "$ROOT_DIR/src/idea2_mean_pivot_quick.cpp" \
    "$ROOT_DIR/src/idea3_boundary_trimmed_merge.cpp" \
    "$ROOT_DIR/src/idea4_tiled_pingpong_merge.cpp" \
    "$ROOT_DIR/src/benchmark_api.cpp" \
    "$SCRIPT_DIR/pure_perf_bench.cpp" -o "$PURE_4B"

g++ -O3 -march=native -std=c++20 -I"$ROOT_DIR/include" -DBENCH_RECORD=1 \
    "$ROOT_DIR/src/baseline_sorts.cpp" \
    "$ROOT_DIR/src/idea1_cache_leaf_quick.cpp" \
    "$ROOT_DIR/src/idea2_mean_pivot_quick.cpp" \
    "$ROOT_DIR/src/idea3_boundary_trimmed_merge.cpp" \
    "$ROOT_DIR/src/idea4_tiled_pingpong_merge.cpp" \
    "$ROOT_DIR/src/benchmark_api.cpp" \
    "$SCRIPT_DIR/pure_perf_bench.cpp" -o "$PURE_8B"

echo "[2/3] 순수 정렬 구간 PMU 하드웨어 카운터 측정 중 (N=$N, Repeats=$REPEATS)..."
printf "%-9s | %-29s | %-22s | %10s | %10s | %11s | %6s | %11s\n" "Type" "Scenario" "Algorithm" "L1 Hit(%)" "LLC Hit(%)" "Br Miss(%)" "IPC" "Cycles"
echo "------------------------------------------------------------------------------------------------------------------------------------"

"$PURE_4B" "$N" "$REPEATS" "$CSV_FILE" "$MD_FILE"
"$PURE_8B" "$N" "$REPEATS" "$CSV_FILE" "$MD_FILE" --append

echo "------------------------------------------------------------------------------------------------------------------------------------"
echo "[3/3] 측정 완료!"
echo "  -> CSV 저장: $(pwd)/$CSV_FILE"
echo "  -> MD  저장: $(pwd)/$MD_FILE"
