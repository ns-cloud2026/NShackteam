<#
.SYNOPSIS
    Collects local NShackteam benchmark telemetry from stderr.log + nvidia-smi
    and writes a ready-to-paste GitHub Issue report for https://github.com/4lwayz/portwatch-vi/issues
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [string]$RepoDir = $PSScriptRoot,
    [string]$OutFile = (Join-Path (Get-Location).Path "portwatch-report.md")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LogPath)) {
    throw "Log file not found: $LogPath"
}

$commit = (git -C $RepoDir rev-parse --short HEAD).Trim()
$branch = (git -C $RepoDir rev-parse --abbrev-ref HEAD).Trim()

$gpuCsv = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { @(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader) -join " / " } else { "Unavailable" }
$cpuObj = Get-CimInstance Win32_Processor | Select-Object -First 1
$osObj  = Get-CimInstance Win32_OperatingSystem
$totalRamGB = [math]::Round($osObj.TotalVisibleMemorySize / 1MB, 1)
$freeRamMB  = [math]::Round($osObj.FreePhysicalMemory / 1024, 0)
$freeComMB  = [math]::Round($osObj.FreeVirtualMemory / 1024, 0)

$proc = Get-Process -Name victory -ErrorAction SilentlyContinue | Select-Object -First 1
$procLine = if ($proc) {
    $uptime = [math]::Round(((Get-Date) - $proc.StartTime).TotalMinutes, 1)
    "ALIVE (PID $($proc.Id), Uptime ${uptime} min, WS $([math]::Round($proc.WorkingSet64/1MB,0)) MB, Priv $([math]::Round($proc.PrivateMemorySize64/1MB,0)) MB)"
} else {
    "NOT RUNNING (log snapshot)"
}

# Read log with FileShare.ReadWrite so we never lock or fail while victory.exe writes
$fs = [System.IO.File]::Open($LogPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
$sr = New-Object System.IO.StreamReader($fs)
$lines = New-Object System.Collections.Generic.List[string]
while (($l = $sr.ReadLine()) -ne $null) {
    $lines.Add(($l -replace '\x1b\[[0-9;]*m', ''))
}
$sr.Close(); $fs.Close()

$exceptions = @($lines | Where-Object { $_ -match 'unhandled exception|panic|device lost|DXGI_ERROR_DEVICE|\bERROR\b' })
$displayLine = ($lines | Where-Object { $_ -match 'display mode reported to the title' } | Select-Object -Last 1)
$budgetLine  = ($lines | Where-Object { $_ -match 'unified local VRAM budget active' } | Select-Object -Last 1)
$tilesLine   = ($lines | Where-Object { $_ -match 'level by level \(VICTORY_LEVELS_ON_DEMAND=1\)' } | Select-Object -Last 1)
$adapterLine = ($lines | Where-Object { $_ -match 'host adapter name=' } | Select-Object -Last 1)

$menu  = New-Object System.Collections.Generic.List[object]
$world = New-Object System.Collections.Generic.List[object]
foreach ($l in $lines) {
    if ($l -match '(\d+\.\d+)\s+fps,\s+(\d+\.\d+)\s+ms/frame.*?(\d+)\s+draws/frame') {
        $fps = [double]$Matches[1]
        $ms  = [double]$Matches[2]
        $drw = [int]$Matches[3]
        $hbMs = if ($l -match 'heap buffers\s+\d+\.\d+\s+passes\s+(\d+\.\d+)\s+ms') { [double]$Matches[1] } else { 0.0 }
        $obj = [pscustomobject]@{ Fps=$fps; Ms=$ms; Draws=$drw; HeapMs=$hbMs; Line=$l }
        if ($drw -ge 100) { $world.Add($obj) } else { $menu.Add($obj) }
    }
}

$peakDStorageMBs = 0.0
$totalSlabHits = 0
$totalSlabMB = 0.0
$totalFallbacks = 0
foreach ($l in $lines) {
    if ($l -match 'dstorage: read ahead.*fallbacks=(\d+).*mb_per_s=(\d+\.\d+)(?:.*slab_hits=(\d+).*slab_hit_mb=(\d+\.\d+))?') {
        $totalFallbacks += [int]$Matches[1]
        $mbs = [double]$Matches[2]
        if ($mbs -gt $peakDStorageMBs) { $peakDStorageMBs = $mbs }
        if ($Matches[3]) { $totalSlabHits += [int]$Matches[3] }
        if ($Matches[4]) { $totalSlabMB   += [double]$Matches[4] }
    }
}

$lastAlias    = ($lines | Where-Object { $_ -match 'gpu aliases synced on card per frame' } | Select-Object -Last 1)
$lastParallel = ($lines | Where-Object { $_ -match 'parallel submission per frame' } | Select-Object -Last 1)
$lastVeh      = ($lines | Where-Object { $_ -match 'VEH userspace copies per frame' } | Select-Object -Last 1)
$lastBarriers = ($lines | Where-Object { $_ -match 'copy barriers batched per frame' } | Select-Object -Last 1)
$lastSettle   = ($lines | Where-Object { $_ -match 'submissions sent ahead per frame:' } | Select-Object -Last 1)
$lastClock    = ($lines | Where-Object { $_ -match 'the card''s clock per frame:' } | Select-Object -Last 1)

function Get-Stats($list) {
    if ($list.Count -eq 0) { return $null }
    $sortedFps = @($list | Sort-Object Fps | ForEach-Object Fps)
    $mid = [int][math]::Floor($sortedFps.Count / 2)
    $medFps = if ($sortedFps.Count % 2) { $sortedFps[$mid] } else { ($sortedFps[$mid - 1] + $sortedFps[$mid]) / 2 }
    [pscustomobject]@{
        Count    = $list.Count
        MinFps   = ($list | Measure-Object Fps -Minimum).Minimum
        MedFps   = $medFps
        AvgFps   = [math]::Round(($list | Measure-Object Fps -Average).Average, 2)
        MaxFps   = ($list | Measure-Object Fps -Maximum).Maximum
        AvgMs    = [math]::Round(($list | Measure-Object Ms -Average).Average, 1)
        AvgDraws = [math]::Round(($list | Measure-Object Draws -Average).Average, 0)
        MaxDraws = ($list | Measure-Object Draws -Maximum).Maximum
        AvgHeap  = [math]::Round(($list | Measure-Object HeapMs -Average).Average, 1)
    }
}

$menuStats  = Get-Stats $menu
$worldStats = Get-Stats $world

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# NShackteam - Local Benchmark Report")
$md.Add("")
$md.Add("- **Commit / Branch:** ``$commit`` (``$branch``)")
$md.Add("- **Generated:** $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')")
$md.Add("- **Process State:** $procLine")
$md.Add("- **GPU (nvidia-smi):** $gpuCsv")
$md.Add("- **CPU:** $($cpuObj.Name.Trim()) ($($cpuObj.NumberOfCores)c/$($cpuObj.NumberOfLogicalProcessors)t)")
$md.Add("- **RAM:** ${totalRamGB} GB total (Free RAM: ${freeRamMB} MB, Free Commit: ${freeComMB} MB)")
$md.Add("- **Storage:** Record the drive type used for this run.")
$md.Add("- **Unhandled Exceptions / Device Removals:** **$($exceptions.Count)**")
$md.Add("")
$md.Add("## Measured Frame Rates (from ``stderr.log``)")
$md.Add("")
$md.Add("| Phase | Samples | Min FPS | Median FPS | Mean FPS | Max FPS | Mean Frame Time | Avg / Max Draws | Mean Heap Pass |")
$md.Add("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
if ($menuStats) {
    $md.Add("| Boot / Main Menu (< 100 draws) | $($menuStats.Count) | $($menuStats.MinFps) | $($menuStats.MedFps) | $($menuStats.AvgFps) | $($menuStats.MaxFps) | $($menuStats.AvgMs) ms | $($menuStats.AvgDraws) / $($menuStats.MaxDraws) | $($menuStats.AvgHeap) ms |")
} else {
    $md.Add("| Boot / Main Menu (< 100 draws) | 0 | - | - | - | - | - | - | - |")
}
if ($worldStats) {
    $md.Add("| **High-draw samples (>= 100 draws)** | **$($worldStats.Count)** | **$($worldStats.MinFps)** | **$($worldStats.MedFps)** | **$($worldStats.AvgFps)** | **$($worldStats.MaxFps)** | **$($worldStats.AvgMs) ms** | **$($worldStats.AvgDraws) / $($worldStats.MaxDraws)** | **$($worldStats.AvgHeap) ms** |")
} else {
    $md.Add("| High-draw samples (>= 100 draws) | 0 | - | - | - | - | - | - | - |")
}
$md.Add("")
$md.Add("## Subsystem Telemetry")
$md.Add("- **HDD DirectStorage C-SCAN + 128 MB Slab Cache:** Peak **$peakDStorageMBs MB/s**, **$totalSlabHits slab hits** ($([math]::Round($totalSlabMB, 1)) MB served from RAM), **$totalFallbacks fallbacks**")
if ($adapterLine)  { $md.Add("- **Adapter:** ``$adapterLine``") }
if ($displayLine)  { $md.Add("- **Display:** ``$displayLine``") }
if ($budgetLine)   { $md.Add("- **VRAM Cap:** ``$budgetLine``") }
if ($tilesLine)    { $md.Add("- **Tiled Levels:** ``$tilesLine``") }
if ($lastAlias)    { $md.Add("- **GPU Alias v2:** ``$lastAlias``") }
if ($lastParallel) { $md.Add("- **Parallel Submit:** ``$lastParallel``") }
if ($lastVeh)      { $md.Add("- **VEH AVX2 Copy:** ``$lastVeh``") }
if ($lastBarriers) { $md.Add("- **Batched Barriers:** ``$lastBarriers``") }
if ($lastSettle)   { $md.Add("- **Settles:** ``$lastSettle``") }
if ($lastClock)    { $md.Add("- **GPU Clock:** ``$lastClock``") }
if ($world.Count -gt 0) {
    $md.Add("")
    $md.Add("### Sample High-Draw Log Lines (Last 5)")
    $md.Add('```text')
    $world | Select-Object -Last 5 | ForEach-Object { $md.Add($_.Line) }
    $md.Add('```')
}

[System.IO.File]::WriteAllLines($OutFile, $md, [System.Text.UTF8Encoding]::new($false))
Write-Output "WROTE_REPORT=$OutFile"
$md | ForEach-Object { Write-Output $_ }
