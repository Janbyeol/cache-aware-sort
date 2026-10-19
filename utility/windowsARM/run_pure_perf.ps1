$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir   = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

$arg1 = if ($args.Count -ge 1) { [string]$args[0] } else { "" }

if ($arg1 -eq "-h" -or $arg1 -eq "--help" -or $arg1 -eq "/?") {
    Write-Host "Usage:"
    Write-Host "  1) Single N measurement:"
    Write-Host "     .\utility\windowsARM\run_pure_perf.bat [N] [Repeats] [CSV_Path]"
    Write-Host "     Defaults: N=1000000, Repeats=3, CSV_Path=pure_cache_results.csv"
    Write-Host ""
    Write-Host "  2) Full 10-Scale Sweep + CPU Cooling (--all mode):"
    Write-Host "     .\utility\windowsARM\run_pure_perf.bat --all [Repeats] [CoolingSec] [OutDir]"
    Write-Host "     Target N: 16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000"
    Write-Host "     Defaults: Repeats=3, CoolingSec=5, OutDir=docs/windowsARM"
    exit 0
}

$Pure4B = Join-Path $ScriptDir "pure_perf_4b.exe"
$Pure8B = Join-Path $ScriptDir "pure_perf_8b.exe"

$SrcFiles = @(
    (Join-Path $RootDir "src\baseline_sorts.cpp"),
    (Join-Path $RootDir "src\idea1_cache_leaf_quick.cpp"),
    (Join-Path $RootDir "src\idea2_mean_pivot_quick.cpp"),
    (Join-Path $RootDir "src\idea3_boundary_trimmed_merge.cpp"),
    (Join-Path $RootDir "src\idea4_tiled_pingpong_merge.cpp"),
    (Join-Path $RootDir "src\benchmark_api.cpp"),
    (Join-Path $ScriptDir "pure_perf_bench.cpp")
)
$IncDir = Join-Path $RootDir "include"

# Detect whether the host machine is ARM64 (Snapdragon X Elite) or x64
$IsArm64Host = ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") -or ($env:PROCESSOR_ARCHITEW6432 -eq "ARM64")
$TargetArch  = if ($IsArm64Host) { "arm64" } else { "x64" }

Write-Host "[1/3] Compiling pure_perf_4b.exe and pure_perf_8b.exe (Target: $TargetArch)..."

function Find-VcVarsAll {
    $vswhereCandidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    foreach ($vswhere in $vswhereCandidates) {
        if ($vswhere -and (Test-Path $vswhere)) {
            $installPath = & $vswhere -latest -products * -property installationPath 2>$null
            if ($installPath) {
                $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvarsall.bat"
                if (Test-Path $vcvars) { return $vcvars }
            }
        }
    }
    $fallbackPaths = @(
        "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    )
    foreach ($p in $fallbackPaths) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$vcvarsPath = Find-VcVarsAll
$cmdExe = if ($env:ComSpec -and (Test-Path $env:ComSpec)) { $env:ComSpec } else { "C:\Windows\System32\cmd.exe" }

if ($vcvarsPath) {
    $quotedSrc = ($SrcFiles | ForEach-Object { "`"$_`"" }) -join " "
    $objDir4B  = Join-Path $ScriptDir "obj_4b"
    $objDir8B  = Join-Path $ScriptDir "obj_8b"
    New-Item -ItemType Directory -Force -Path $objDir4B, $objDir8B | Out-Null

    $buildBat = Join-Path $objDir4B "build_msvc.bat"
    $batContent = @"
@echo off
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%PATH%"
call "$vcvarsPath" $TargetArch >nul
if errorlevel 1 exit /b 1
pushd "$objDir4B"
cl.exe /utf-8 /nologo /O2 /Oi /Ot /GL /std:c++20 /EHsc /DNDEBUG /I"$IncDir" $quotedSrc /Fe"$Pure4B" /link /LTCG >nul
if errorlevel 1 ( popd & exit /b 1 )
popd
pushd "$objDir8B"
cl.exe /utf-8 /nologo /O2 /Oi /Ot /GL /std:c++20 /EHsc /DNDEBUG /DBENCH_RECORD=1 /I"$IncDir" $quotedSrc /Fe"$Pure8B" /link /LTCG >nul
if errorlevel 1 ( popd & exit /b 1 )
popd
"@
    Set-Content -Path $buildBat -Value $batContent -Encoding ASCII
    & $cmdExe /c "`"$buildBat`""
    $buildExit = $LASTEXITCODE
    Remove-Item -Recurse -Force $objDir4B, $objDir8B -ErrorAction SilentlyContinue
    Remove-Item -Force "$ScriptDir\*.iobj", "$ScriptDir\*.ipdb" -ErrorAction SilentlyContinue
    if ($buildExit -ne 0) { throw "MSVC compilation failed (ExitCode=$buildExit)" }
} elseif (Get-Command clang++ -ErrorAction SilentlyContinue) {
    & clang++ -O3 -std=c++20 -I"$IncDir" @SrcFiles -o "$Pure4B"
    if ($LASTEXITCODE -ne 0) { throw "clang++ compilation failed for pure_perf_4b.exe" }
    & clang++ -O3 -std=c++20 -DBENCH_RECORD=1 -I"$IncDir" @SrcFiles -o "$Pure8B"
    if ($LASTEXITCODE -ne 0) { throw "clang++ compilation failed for pure_perf_8b.exe" }
} elseif (Get-Command g++ -ErrorAction SilentlyContinue) {
    & g++ -O3 -std=c++20 -I"$IncDir" @SrcFiles -o "$Pure4B"
    if ($LASTEXITCODE -ne 0) { throw "g++ compilation failed for pure_perf_4b.exe" }
    & g++ -O3 -std=c++20 -DBENCH_RECORD=1 -I"$IncDir" @SrcFiles -o "$Pure8B"
    if ($LASTEXITCODE -ne 0) { throw "g++ compilation failed for pure_perf_8b.exe" }
} else {
    throw "No C++20 compiler found. Please install Visual Studio (C++ Desktop workload) or LLVM clang++."
}

# ------------------------------------------------------------------------------
# Batch Sweep Mode (--all or all)
# ------------------------------------------------------------------------------
if ($arg1 -eq "--all" -or $arg1 -eq "all") {
    $NList      = @(16, 64, 256, 1024, 2048, 8192, 32768, 262144, 1000000, 5000000)
    $Repeats    = if ($args.Count -ge 2) { [int]$args[1] } else { 3 }
    $CoolingSec = if ($args.Count -ge 3) { [int]$args[2] } else { 5 }
    $OutDir     = if ($args.Count -ge 4) { [string]$args[3] } else { Join-Path $RootDir "docs\windowsARM" }

    if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
        $OutDir = Join-Path $RootDir $OutDir
    }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    $AllCsv = Join-Path $OutDir "pure_cache_results_all.csv"
    if (Test-Path $AllCsv) { Remove-Item -Force $AllCsv }

    $TotalN = $NList.Count
    Write-Host "[2/3] Starting $TotalN-scale pure sorting benchmark (Repeats=$Repeats, Cooling=${CoolingSec}s, OutDir=$OutDir)"
    Write-Host "      Target N: $($NList -join ', ')"
    Write-Host ("=" * 124)

    for ($idx = 0; $idx -lt $TotalN; $idx++) {
        $N       = $NList[$idx]
        $Step    = $idx + 1
        $CsvFile = Join-Path $OutDir "pure_cache_results_N=$N.csv"
        $MdFile  = Join-Path $OutDir "pure_cache_results_N=$N.md"

        Write-Host ""
        Write-Host ">>> [$Step/$TotalN] Measuring N = $N (Repeats=$Repeats) -> pure_cache_results_N=$N.csv"
        Write-Host ("{0,-9} | {1,-29} | {2,-22} | {3,13} | {4,9} | {5,11} | {6,9} | {7,12}" -f "Type", "Scenario", "Algorithm", "Runtime(us)", "Spd(Time)", "Cycles", "Spd(Cyc)", "Throughput")
        Write-Host ("-" * 124)

        & $Pure4B "$N" "$Repeats" "$CsvFile" "$MdFile"
        if ($LASTEXITCODE -ne 0) { throw "pure_perf_4b.exe failed at N=$N" }

        if ($CoolingSec -gt 0) {
            Start-Sleep -Seconds 1
        }

        & $Pure8B "$N" "$Repeats" "$CsvFile" "$MdFile" "--append"
        if ($LASTEXITCODE -ne 0) { throw "pure_perf_8b.exe failed at N=$N" }

        # Append into combined pure_cache_results_all.csv
        $lines = Get-Content -Path $CsvFile
        if ($idx -eq 0) {
            $lines | Set-Content -Path $AllCsv -Encoding UTF8
        } else {
            $lines | Select-Object -Skip 1 | Add-Content -Path $AllCsv -Encoding UTF8
        }

        # CPU cooling pause before transitioning to the next N
        if ($idx -lt ($TotalN - 1) -and $CoolingSec -gt 0) {
            $NextN = $NList[$idx + 1]
            Write-Host ("-" * 124)
            Write-Host "[Cooling] N=$N finished -> Waiting ${CoolingSec}s for CPU thermal/boost stabilization before N=$NextN..."
            Start-Sleep -Seconds $CoolingSec
        }
    }

    Write-Host ("=" * 124)
    Write-Host "[3/3] All $TotalN scales measured successfully!"
    Write-Host "  -> Per-N CSV/MD folder : $OutDir\pure_cache_results_N=<N>.csv"
    Write-Host "  -> Combined CSV file   : $AllCsv"
    exit 0
}

# ------------------------------------------------------------------------------
# Single-N Mode (Default / Legacy Command Compatible)
# ------------------------------------------------------------------------------
$N       = if ($args.Count -ge 1 -and $arg1 -ne "") { $arg1 } else { "1000000" }
$Repeats = if ($args.Count -ge 2) { $args[1] } else { "3" }
$CsvFile = if ($args.Count -ge 3) { $args[2] } else { "pure_cache_results.csv" }
$MdFile  = [System.IO.Path]::ChangeExtension($CsvFile, ".md")

Write-Host "[2/3] Measuring pure sorting region (N=$N, Repeats=$Repeats)..."
Write-Host ("{0,-9} | {1,-29} | {2,-22} | {3,13} | {4,9} | {5,11} | {6,9} | {7,12}" -f "Type", "Scenario", "Algorithm", "Runtime(us)", "Spd(Time)", "Cycles", "Spd(Cyc)", "Throughput")
Write-Host ("-" * 124)

& $Pure4B "$N" "$Repeats" "$CsvFile" "$MdFile"
if ($LASTEXITCODE -ne 0) { throw "pure_perf_4b.exe failed" }

& $Pure8B "$N" "$Repeats" "$CsvFile" "$MdFile" "--append"
if ($LASTEXITCODE -ne 0) { throw "pure_perf_8b.exe failed" }

Write-Host ("-" * 124)
Write-Host "[3/3] Measurement complete!"
Write-Host "  -> CSV saved: $CsvFile"
Write-Host "  -> MD  saved: $MdFile"
