#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "사용법: $0 [원소개수_N] [반복횟수] [저장할_CSV파일명]"
    echo "  기본값: N=1000000, 반복횟수=3, 파일명=cache_hit_results.csv"
    echo ""
    echo "예시:"
    echo "  $0                              # 기본 실행 (N=1,000,000 -> cache_hit_results.csv, .md)"
    echo "  $0 131072 5 l2_cache_131k.csv   # L2 크기(131,072개) 5회 평균 -> l2_cache_131k.csv, .md"
    echo "  $0 10000000 3 dram_10m.csv      # 대용량(1,000만 개) 3회 평균 -> dram_10m.csv, .md"
    exit 0
fi

N=${1:-1000000}
REPEATS=${2:-3}
CSV_FILE=${3:-"cache_hit_results.csv"}
MD_FILE="${CSV_FILE%.csv}.md"

PROF_4B="$SCRIPT_DIR/prof_4b"
PROF_8B="$SCRIPT_DIR/prof_8b"

if [[ ! -f "$PROF_4B" || ! -f "$PROF_8B" ]]; then
    echo "[Build] prof_4b 및 prof_8b 컴파일 중..."
    g++ -O3 -march=native -std=c++20 -I"$ROOT_DIR/include" \
        "$ROOT_DIR/src/baseline_sorts.cpp" \
        "$ROOT_DIR/src/idea1_cache_leaf_quick.cpp" \
        "$ROOT_DIR/src/idea2_mean_pivot_quick.cpp" \
        "$ROOT_DIR/src/idea3_boundary_trimmed_merge.cpp" \
        "$ROOT_DIR/src/idea4_tiled_pingpong_merge.cpp" \
        "$ROOT_DIR/src/benchmark_api.cpp" \
        "$SCRIPT_DIR/profile_one.cpp" -o "$PROF_4B"

    g++ -O3 -march=native -std=c++20 -I"$ROOT_DIR/include" -DBENCH_RECORD=1 \
        "$ROOT_DIR/src/baseline_sorts.cpp" \
        "$ROOT_DIR/src/idea1_cache_leaf_quick.cpp" \
        "$ROOT_DIR/src/idea2_mean_pivot_quick.cpp" \
        "$ROOT_DIR/src/idea3_boundary_trimmed_merge.cpp" \
        "$ROOT_DIR/src/idea4_tiled_pingpong_merge.cpp" \
        "$ROOT_DIR/src/benchmark_api.cpp" \
        "$SCRIPT_DIR/profile_one.cpp" -o "$PROF_8B"
fi

# 1) CSV 헤더 초기화
echo "Data_Type,Scenario,N,Algorithm,L1_Hit_Rate(%),L1_Miss_Rate(%),LLC_Hit_Rate(%),LLC_Miss_Rate(%),Branch_Hit_Rate(%),Branch_Miss_Rate(%),IPC,L1_Loads,L1_Misses,LLC_Refs,LLC_Misses,Branches,Branch_Misses,Cycles,Instructions" > "$CSV_FILE"

# 2) Markdown (.md) 파일 헤더 초기화
cat << MD_EOF > "$MD_FILE"
# CPU Cache Hit Rate & Branch Prediction Analysis ($N = $N, Repeats = $REPEATS)

| Data Type | Scenario | Algorithm | L1 Hit Rate | L1 Miss Rate | LLC (L3) Hit Rate | LLC Miss Rate | Branch Hit Rate | Branch Miss Rate | IPC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
MD_EOF

SCEN_NAMES=(
    "0_Uniform_Random"
    "1_Frame_Jitter_Nearly_Sorted"
    "2_Append_5Pct_New_Tail"
    "3_Distance_Buckets_Duplicates"
    "4_Mostly_Descending"
    "5_Gaussian_Normal"
    "6_Organ_Pipe_Symmetric"
)

ALGOS=("std" "stable" "idea3" "idea4")
EVENTS="cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses"

echo "[Run] N=$N, Repeats=$REPEATS -> 로컬 저장: $CSV_FILE 및 $MD_FILE"
printf "%-9s | %-29s | %-6s | %10s | %10s | %11s | %6s\n" "Type" "Scenario" "Algo" "L1 Hit(%)" "LLC Hit(%)" "Br Miss(%)" "IPC"
echo "-------------------------------------------------------------------------------------------------------"

for TYPE_INFO in "4B_Int32:$PROF_4B" "8B_Record:$PROF_8B"; do
    DTYPE="${TYPE_INFO%%:*}"
    BIN="${TYPE_INFO##*:}"

    for SCEN_ID in 0 1 2 3 4 5 6; do
        SNAME="${SCEN_NAMES[$SCEN_ID]}"

        for ALGO in "${ALGOS[@]}"; do
            PERF_OUT=$(LC_ALL=C perf stat -x, -r "$REPEATS" -e "$EVENTS" "$BIN" "$ALGO" "$SCEN_ID" "$N" 2>&1)

            ROW=$(echo "$PERF_OUT" | awk -F',' -v dtype="$DTYPE" -v sname="$SNAME" -v n="$N" -v algo="$ALGO" '
                $3 ~ /^cycles/                { cycles = $1 + 0 }
                $3 ~ /^instructions/          { instr = $1 + 0 }
                $3 ~ /^cache-references/      { llc_ref = $1 + 0 }
                $3 ~ /^cache-misses/          { llc_miss = $1 + 0 }
                $3 ~ /^L1-dcache-loads/       { l1_load = $1 + 0 }
                $3 ~ /^L1-dcache-load-misses/ { l1_miss = $1 + 0 }
                $3 ~ /^branches/              { br_tot = $1 + 0 }
                $3 ~ /^branch-misses/         { br_miss = $1 + 0 }
                END {
                    l1_miss_rate  = (l1_load > 0) ? (l1_miss / l1_load) * 100.0 : 0.0;
                    l1_hit_rate   = 100.0 - l1_miss_rate;
                    llc_miss_rate = (llc_ref > 0) ? (llc_miss / llc_ref) * 100.0 : 0.0;
                    llc_hit_rate  = 100.0 - llc_miss_rate;
                    br_miss_rate  = (br_tot > 0)  ? (br_miss / br_tot) * 100.0 : 0.0;
                    br_hit_rate   = 100.0 - br_miss_rate;
                    ipc           = (cycles > 0)  ? (instr / cycles) : 0.0;

                    printf "%s,%s,%d,%s,%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f%%,%.2f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
                        dtype, sname, n, algo,
                        l1_hit_rate, l1_miss_rate,
                        llc_hit_rate, llc_miss_rate,
                        br_hit_rate, br_miss_rate,
                        ipc,
                        l1_load, l1_miss, llc_ref, llc_miss, br_tot, br_miss, cycles, instr;
                }
            ')

            echo "$ROW" >> "$CSV_FILE"

            # Markdown 표 행 추가
            echo "$ROW" | awk -F',' '{
                printf "| %s | %s | `%s` | **%s** | %s | **%s** | %s | %s | **%s** | %s |\n",
                    $1, $2, $4, $5, $6, $7, $8, $9, $10, $11
            }' >> "$MD_FILE"

            # 터미널 출력
            echo "$ROW" | awk -F',' '{
                printf "%-9s | %-29s | %-6s | %10s | %10s | %11s | %6s\n", $1, $2, $4, $5, $7, $10, $11
            }'
        done
    done
done

echo "-------------------------------------------------------------------------------------------------------"
echo "[Done] CSV 저장 완료: $(pwd)/$CSV_FILE"
echo "[Done] MD  저장 완료: $(pwd)/$MD_FILE"
