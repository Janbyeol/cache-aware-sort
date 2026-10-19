#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "사용법:"
    echo "  1) 단일 N 측정 (기존 명령어):"
    echo "     $0 [원소개수_N] [반복횟수] [저장할_CSV파일명]"
    echo "     기본값: N=1000000, 반복횟수=3, 파일명=pure_cache_results.csv"
    echo ""
    echo "  2) 전체 N 일괄 측정 + 쿨링 대기 (--all 모드):"
    echo "     $0 --all [반복횟수] [쿨링시간_초] [저장할_폴더경로]"
    echo "     대상 N: 16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000"
    echo "     기본값: 반복횟수=3, 쿨링시간=5초, 저장폴더=docs/linux"
    echo ""
    echo "예시:"
    echo "  $0                                   # 기본 실행 (N=1,000,000 -> pure_cache_results.csv, .md)"
    echo "  $0 131072 5 pure_l2_131k.csv         # 단일 크기(131,072개) 5회 평균"
    echo "  $0 --all                             # 10개 N 전체 일괄 측정 (N 전환 시 5초 쿨링 -> docs/linux/ 저장)"
    echo "  $0 --all 5 10 docs/linux             # 10개 N 전체 5회 반복, N 전환 시 10초 쿨링 -> docs/linux/ 저장"
    exit 0
fi

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

# ------------------------------------------------------------------------------
# 일괄 측정 모드 (--all 또는 all)
# ------------------------------------------------------------------------------
if [[ "$1" == "--all" || "$1" == "all" ]]; then
    N_LIST=(16 64 256 1024 2048 8192 32768 262144 1000000 5000000)
    REPEATS=${2:-3}
    COOLING_SEC=${3:-5}
    OUT_DIR=${4:-"$ROOT_DIR/docs/linux"}

    mkdir -p "$OUT_DIR"
    ALL_CSV="$OUT_DIR/pure_cache_results_all.csv"
    rm -f "$ALL_CSV"

    TOTAL_N=${#N_LIST[@]}
    echo "[2/3] 전체 ${TOTAL_N}개 N 일괄 PMU 측정 시작 (Repeats=$REPEATS, 쿨링=${COOLING_SEC}초, 저장경로=$OUT_DIR)"
    echo "      대상 N: ${N_LIST[*]}"
    echo "===================================================================================================================================="

    for (( idx=0; idx<TOTAL_N; idx++ )); do
        N="${N_LIST[$idx]}"
        STEP=$((idx + 1))
        CSV_FILE="$OUT_DIR/pure_cache_results_N=${N}.csv"
        MD_FILE="$OUT_DIR/pure_cache_results_N=${N}.md"

        echo ""
        echo ">>> [${STEP}/${TOTAL_N}] N = ${N} 측정 중 (Repeats=${REPEATS}) -> $(basename "$CSV_FILE")"
        printf "%-9s | %-29s | %-22s | %10s | %10s | %11s | %6s | %11s\n" "Type" "Scenario" "Algorithm" "L1 Hit(%)" "LLC Hit(%)" "Br Miss(%)" "IPC" "Cycles"
        echo "------------------------------------------------------------------------------------------------------------------------------------"

        "$PURE_4B" "$N" "$REPEATS" "$CSV_FILE" "$MD_FILE"
        if (( COOLING_SEC > 0 )); then
            sleep 1
        fi
        "$PURE_8B" "$N" "$REPEATS" "$CSV_FILE" "$MD_FILE" --append

        # 통합 CSV 파일(pure_cache_results_all.csv)에 병합
        if (( idx == 0 )); then
            cat "$CSV_FILE" > "$ALL_CSV"
        else
            tail -n +2 "$CSV_FILE" >> "$ALL_CSV"
        fi

        # 다음 N으로 넘어가기 전 CPU 쿨링 대기
        if (( idx < TOTAL_N - 1 && COOLING_SEC > 0 )); then
            NEXT_N="${N_LIST[$((idx + 1))]}"
            echo "------------------------------------------------------------------------------------------------------------------------------------"
            echo "[Cooling] N=${N} 완료 -> 다음 N=${NEXT_N} 공정한 측정을 위해 ${COOLING_SEC}초간 CPU 쿨링 대기 중..."
            sleep "$COOLING_SEC"
        fi
    done

    echo "===================================================================================================================================="
    echo "[3/3] 전체 N 일괄 측정 완료!"
    echo "  -> 개별 CSV/MD 저장 폴더: $OUT_DIR/pure_cache_results_N=<N>.csv"
    echo "  -> 통합 CSV 저장 파일   : $ALL_CSV"
    exit 0
fi

# ------------------------------------------------------------------------------
# 단일 N 측정 모드 (기존 명령어 100% 호환)
# ------------------------------------------------------------------------------
N=${1:-1000000}
REPEATS=${2:-3}
CSV_FILE=${3:-"pure_cache_results.csv"}
MD_FILE="${CSV_FILE%.csv}.md"

echo "[2/3] 순수 정렬 구간 PMU 하드웨어 카운터 측정 중 (N=$N, Repeats=$REPEATS)..."
printf "%-9s | %-29s | %-22s | %10s | %10s | %11s | %6s | %11s\n" "Type" "Scenario" "Algorithm" "L1 Hit(%)" "LLC Hit(%)" "Br Miss(%)" "IPC" "Cycles"
echo "------------------------------------------------------------------------------------------------------------------------------------"

"$PURE_4B" "$N" "$REPEATS" "$CSV_FILE" "$MD_FILE"
"$PURE_8B" "$N" "$REPEATS" "$CSV_FILE" "$MD_FILE" --append

echo "------------------------------------------------------------------------------------------------------------------------------------"
echo "[3/3] 측정 완료!"
echo "  -> CSV 저장: $(pwd)/$CSV_FILE"
echo "  -> MD  저장: $(pwd)/$MD_FILE"
