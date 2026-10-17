@echo off
setlocal
cd /d "%~dp0"
set "SELF_PATH=%~f0"

echo [1/2] Generating Visual Studio Solution and Projects...
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "$s = [System.IO.File]::ReadAllText($env:SELF_PATH); $idx = $s.LastIndexOf('###_PS_START_###'); Invoke-Expression $s.Substring($idx + 16)"
if errorlevel 1 (
    echo [ERROR] Failed to generate solution files.
    exit /b 1
)

echo [2/2] Done! Open 'cache-aware-sort.sln' in Visual Studio.
exit /b 0

###_PS_START_###
$root = $PWD.Path
$vsDir = Join-Path $root "vs"
if (!(Test-Path $vsDir)) {
    New-Item -ItemType Directory -Path $vsDir | Out-Null
}

$pf64 = $env:ProgramFiles
$pf86 = ${env:ProgramFiles(x86)}
$vsRoot64 = if ($pf64) { Join-Path $pf64 "Microsoft Visual Studio" } else { "" }
$vsRoot86 = if ($pf86) { Join-Path $pf86 "Microsoft Visual Studio" } else { "" }

$toolset = "v143"
if ($vsRoot64 -and (Test-Path (Join-Path $vsRoot64 "18"))) {
    $toolset = "v145"
} elseif ($vsRoot64 -and (Test-Path (Join-Path $vsRoot64 "2022"))) {
    $toolset = "v143"
} elseif ($vsRoot86 -and (Test-Path (Join-Path $vsRoot86 "2019"))) {
    $toolset = "v142"
}
Write-Host ("      -> Detected PlatformToolset: " + $toolset)

function Write-Vcxproj([string]$name, [string]$guid, [string]$mainCpp, [bool]$define8B) {
    $extraDef = if ($define8B) { "BENCH_RECORD=1;" } else { "" }
    $xml = @"
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{$guid}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>$name</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="`$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'`$(Configuration)|`$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>$toolset</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'`$(Configuration)|`$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>$toolset</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="`$(VCTargetsPath)\Microsoft.Cpp.props" />
  <PropertyGroup>
    <OutDir>`$(SolutionDir)bin\`$(Configuration)\</OutDir>
    <IntDir>`$(SolutionDir)vs\obj\`$(ProjectName)\`$(Configuration)\</IntDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'`$(Configuration)|`$(Platform)'=='Debug|x64'">
    <ClCompile>
      <WarningLevel>Level4</WarningLevel>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>${extraDef}_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <AdditionalIncludeDirectories>`$(SolutionDir)include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
      <AdditionalOptions>/utf-8 %(AdditionalOptions)</AdditionalOptions>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'`$(Configuration)|`$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level4</WarningLevel>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <Optimization>MaxSpeed</Optimization>
      <FavorSizeOrSpeed>Speed</FavorSizeOrSpeed>
      <EnableEnhancedInstructionSet>AdvancedVectorExtensions2</EnableEnhancedInstructionSet>
      <PreprocessorDefinitions>${extraDef}NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <LanguageStandard>stdcpp20</LanguageStandard>
      <AdditionalIncludeDirectories>`$(SolutionDir)include;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
      <AdditionalOptions>/utf-8 %(AdditionalOptions)</AdditionalOptions>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClInclude Include="..\include\hybrid_sorts.hpp" />
  </ItemGroup>
  <ItemGroup>
    <ClCompile Include="..\src\baseline_sorts.cpp" />
    <ClCompile Include="..\src\idea1_cache_leaf_quick.cpp" />
    <ClCompile Include="..\src\idea2_mean_pivot_quick.cpp" />
    <ClCompile Include="..\src\idea3_boundary_trimmed_merge.cpp" />
    <ClCompile Include="..\src\idea4_tiled_pingpong_merge.cpp" />
    <ClCompile Include="..\src\benchmark_api.cpp" />
    <ClCompile Include="..\src\$mainCpp" />
  </ItemGroup>
  <Import Project="`$(VCTargetsPath)\Microsoft.Cpp.targets" />
</Project>
"@
    $enc = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText((Join-Path $vsDir ($name + ".vcxproj")), $xml, $enc)
}

Write-Vcxproj "bench_4b" "A1111111-1111-4111-8111-111111111111" "benchmark_main.cpp" $false
Write-Vcxproj "bench_8b" "A2222222-2222-4222-8222-222222222222" "benchmark_main.cpp" $true
Write-Vcxproj "test_4b"  "A3333333-3333-4333-8333-333333333333" "test_main.cpp"      $false
Write-Vcxproj "test_8b"  "A4444444-4444-4444-8444-444444444444" "test_main.cpp"      $true

$sln = @"

Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "bench_4b", "vs\bench_4b.vcxproj", "{A1111111-1111-4111-8111-111111111111}"
EndProject
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "bench_8b", "vs\bench_8b.vcxproj", "{A2222222-2222-4222-8222-222222222222}"
EndProject
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "test_4b", "vs\test_4b.vcxproj", "{A3333333-3333-4333-8333-333333333333}"
EndProject
Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "test_8b", "vs\test_8b.vcxproj", "{A4444444-4444-4444-8444-444444444444}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|x64 = Debug|x64
		Release|x64 = Release|x64
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{A1111111-1111-4111-8111-111111111111}.Debug|x64.ActiveCfg = Debug|x64
		{A1111111-1111-4111-8111-111111111111}.Debug|x64.Build.0 = Debug|x64
		{A1111111-1111-4111-8111-111111111111}.Release|x64.ActiveCfg = Release|x64
		{A1111111-1111-4111-8111-111111111111}.Release|x64.Build.0 = Release|x64
		{A2222222-2222-4222-8222-222222222222}.Debug|x64.ActiveCfg = Debug|x64
		{A2222222-2222-4222-8222-222222222222}.Debug|x64.Build.0 = Debug|x64
		{A2222222-2222-4222-8222-222222222222}.Release|x64.ActiveCfg = Release|x64
		{A2222222-2222-4222-8222-222222222222}.Release|x64.Build.0 = Release|x64
		{A3333333-3333-4333-8333-333333333333}.Debug|x64.ActiveCfg = Debug|x64
		{A3333333-3333-4333-8333-333333333333}.Debug|x64.Build.0 = Debug|x64
		{A3333333-3333-4333-8333-333333333333}.Release|x64.ActiveCfg = Release|x64
		{A3333333-3333-4333-8333-333333333333}.Release|x64.Build.0 = Release|x64
		{A4444444-4444-4444-8444-444444444444}.Debug|x64.ActiveCfg = Debug|x64
		{A4444444-4444-4444-8444-444444444444}.Debug|x64.Build.0 = Debug|x64
		{A4444444-4444-4444-8444-444444444444}.Release|x64.ActiveCfg = Release|x64
		{A4444444-4444-4444-8444-444444444444}.Release|x64.Build.0 = Release|x64
	EndGlobalSection
EndGlobal
"@
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText((Join-Path $root "cache-aware-sort.sln"), $sln, $enc)
