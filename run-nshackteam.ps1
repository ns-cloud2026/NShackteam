<#
.SYNOPSIS
    NShackteam GTA VI PC Compatibility Layer Launcher (RTX 2060 12 GB)
.DESCRIPTION
    Combines Belobug lighting/pixel fixes, fxceless f85101b + PRs #827/#837/#838/#842/#845/#846/#853/#854/#855,
    VItoriaBr graphics-pipeline prioritization, Zettrox shader fixes, and the NS Hack Team RTX 2060 12 GB engine
    (GPU Alias v2 + GPU CopyBufferRegion, VEH AVX2 userspace copy_mapped, fast heap_buffers pass,
    unified 10.5 GB VRAM cap + dynamic mip bias + tile reclamation, batched copy barriers,
    and cross-lane C-SCAN HDD DirectStorage elevator).
    Set -GameDir to your local title directory outside the source checkout.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameDir,

    [ValidateSet("MaxFps", "UltraFps", "Baseline", "Balanced", "Safe", "RayTracing")]
    [string]$Profile = "MaxFps",

    [string]$Resolution = "",

    [string]$Ambient = "",

    [bool]$SkipBrokenTrees = $true,

    [ValidateSet("series-x", "series-s")]
    [string]$Console = "series-x",

    # A/B benchmark overrides for RTX 2060 12 GB & HDD switches (empty string = use profile default)
    [string]$AliasV2 = "",
    [string]$AliasIndex = "",
    [string]$GpuBufferCopy = "",
    [string]$VehCopy = "",
    [string]$FastHeapPass = "",
    [string]$ParallelHeap = "",
    [string]$FillWorkers = "",
    [string]$ParallelPreludes = "",
    [string]$EarlyDepthFirst = "",
    [string]$LocalBudgetBytes = "",
    [string]$MipBias = "",
    [string]$TileReclaim = "",
    [string]$BatchCopyBarriers = "",
    [string]$CardWindowsMB = "",
    [string]$Queue2 = "",
    [string]$CrossLaneElevator = "",
    [string]$DriveDepth = "",
    [string]$ReadAheadMaxKB = "",
    [string]$ReadAheadGapKB = "",
    [string]$SlabCacheMB = "",
    [string]$RawReadAhead = "",
    [string]$TextureRoomPercent = "",
    [string]$TargetsRoomMB = "",
    [string]$TextureStageMS = "",
    [string]$TextureCatchupMS = "",
    [string]$TextureSweep = "",
    [string]$HeapRecheck = "",
    [string]$UnwatchedFrames = "",
    [string]$GpuTiming = "",
    [hashtable]$ExtraEnv = @{},

    [switch]$Diag,
    [string]$WarmFile = "",
    [switch]$NoEarlyPipelines,
    [switch]$Background,
    [switch]$Wait,
    [switch]$NoTailWindow,
    [switch]$Tail,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
try {
    $Host.UI.RawUI.WindowTitle     = "NShackteam | GTA VI PC Launcher (RTX 2060 12 GB)"
    $Host.UI.RawUI.BackgroundColor = "Black"
    $Host.UI.RawUI.ForegroundColor = "Gray"
    [Console]::Write("$([char]27)]11;#000000$([char]7)")
    Clear-Host
} catch { }

$GameDir   = (Resolve-Path -LiteralPath $GameDir).Path
$TitleExe  = Join-Path $GameDir "game_scarlett_master.exe"
$RepoDir   = $PSScriptRoot
$StateDir  = Join-Path $GameDir ".nshackteam-state"
$CacheDir  = Join-Path $StateDir "cache"
$DataDir   = Join-Path $StateDir "data"
$LogDir    = Join-Path $StateDir "logs"
$TempDir   = Join-Path $StateDir "temp"
$Config    = Join-Path $RepoDir "victory.toml"
$ExePath   = if ($Diag) {
    Join-Path $RepoDir "target\diag\victory.exe"
} else {
    Join-Path $RepoDir "target\release\victory.exe"
}

foreach ($req in @($ExePath, $TitleExe)) {
    if (-not (Test-Path -LiteralPath $req -PathType Leaf)) {
        throw "Required file not found: $req"
    }
}

New-Item -ItemType Directory -Force -Path $CacheDir, $DataDir, $LogDir, $TempDir | Out-Null

$knownBinaries = @(
    (Join-Path $RepoDir "target\release\victory.exe")
    (Join-Path $RepoDir "target\diag\victory.exe")
)
$running = @(Get-Process -Name "victory" -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -in $knownBinaries })
if ($running -and -not $Restart) {
    Write-Host "[NShackteam] Already running (PID $($running[0].Id))." -ForegroundColor Cyan
    $latestErr = Join-Path $LogDir "stderr.log"
    if ($Tail -and (Test-Path -LiteralPath $latestErr -PathType Leaf)) {
        Get-Content -LiteralPath $latestErr -Wait -Tail 40
    }
    exit 0
}

if ($Restart -and $running) {
    $running | Stop-Process -ErrorAction Stop
    foreach ($process in $running) {
        if (-not $process.WaitForExit(5000)) {
            throw "Timed out waiting for NShackteam (PID $($process.Id)) to close."
        }
    }
}

# Clean stale VICTORY_* env vars so profile switches and A/B runs are deterministic
Get-ChildItem Env:VICTORY_* -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notin 'VICTORY_DXIL', 'VICTORY_DXC' } | ForEach-Object {
    Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
}

$env:TEMP             = $TempDir
$env:TMP              = $TempDir
if (Test-Path -LiteralPath $Config -PathType Leaf) {
    $env:VICTORY_CONFIG = $Config
}
$env:VICTORY_GAME_DIR  = $GameDir
$env:VICTORY_CACHE_DIR = $CacheDir
$env:VICTORY_DATA_DIR  = $DataDir

# Offline shader cache prewarm mode
if ($WarmFile) {
    Write-Host "[NShackteam] Pre-warming shader cache from $WarmFile..." -ForegroundColor Cyan
    & $ExePath translate --warm $WarmFile
    exit $LASTEXITCODE
}

# Logging filter (includes ambient, shadow/back-join, host FPS/VRAM, resource/tiles, pipeline library, dstorage)
$env:VICTORY_LOG = if ($Diag) {
    "warn,victory_gpu::host=info,victory_gpu::resource=info,victory_gpu::tiles=info,victory_gpu::ambient=info,victory_gpu::shadow=info,victory_gpu::dxgi=info,victory_gpu::queue=info,victory_gpu::pipeline=info,victory_gpu::pipeline::host=info,victory_gpu::pipeline::library=info,victory_hle::modules::dstorage=debug,victory_hle::modules::xuser=info,victory_loader::console_insn=info"
} else {
    "warn,victory_gpu::host=info,victory_gpu::resource=info,victory_gpu::tiles=info,victory_gpu::ambient=info,victory_gpu::shadow=info,victory_gpu::dxgi=info,victory_gpu::pipeline=info,victory_gpu::pipeline::host=info,victory_gpu::pipeline::library=info,victory_hle::modules::dstorage=info,victory_hle::modules::xuser=info,victory_loader::console_insn=info"
}

# Belobug 8 broken wind-volume distant tree imposter pipelines
$BrokenTreeHashes = "4e35b816f2f91d68,b3ba0ef9ee352720,5d627391703fcb4c,ec6e4c363cf9ef2b,c89489afd89ee7f0,45f081396c2c59ba,63083c777948927a,0bd4d80ad23b106b"

# === Common NShackteam Baseline (RTX 2060 12 GB + 10c/20t + 32 GB RAM + HDD) ===
$EffectiveResolution = if ($Resolution -ne "") {
    $Resolution
} elseif ($Profile -eq "UltraFps") {
    "1280x720"
} else {
    "1920x1080"
}
$env:VICTORY_DISPLAY_MODE             = $EffectiveResolution
$env:VICTORY_NAN_GUARD                = "1"
$env:VICTORY_MESH_LINK                = "1"
$env:VICTORY_WAVE_PAIRS               = "1"
$env:VICTORY_LEVELS_ON_DEMAND         = "1"
$env:VICTORY_VIEW_GUARD               = "1"
$env:VICTORY_REUPLOAD_WHEN_FULL       = "1"
$env:VICTORY_FULL_CARD_PASS           = "1"
$env:VICTORY_TEXTURE_ROOM_PERCENT     = "92"
$env:VICTORY_CARD_ROOM_FOR_TARGETS_MB = "2048"
$env:VICTORY_CARD_BUFFERS_MIN_MB      = "1"
$env:VICTORY_CARD_BUFFERS_MB          = "1024"
$env:VICTORY_CARD_BUFFER_RESERVE_MB   = "512"
$env:VICTORY_TEXTURE_NO_WAIT          = "1"
$env:VICTORY_TEXTURE_DETILE_WORKERS   = "4"
$env:VICTORY_TEXTURE_CATCHUP          = "1"
$env:VICTORY_PRELOAD_TEXTURES         = "0"
$env:VICTORY_EARLY_TEXTURES_MS        = "10"
$env:VICTORY_EARLY_TEXTURES_MB        = "64"
$env:VICTORY_TEXTURE_STAGE_MB         = "384"
$env:VICTORY_TEXTURE_STAGE_MS         = "16"
$env:VICTORY_TEXTURE_CATCHUP_MS       = "24"
$env:VICTORY_TABLED_SKIP_FRAMES       = "2"
$env:VICTORY_RAM_RELIEF_MB            = "1024"
$env:VICTORY_PIPELINE_WAIT_SHARED     = "1"
$env:VICTORY_PIPELINE_WAIT_MS         = "24"
$env:VICTORY_DSTORAGE_OVERLAPPED      = "1"
$env:VICTORY_DSTORAGE_ELEVATOR        = "1"
$env:VICTORY_DSTORAGE_CROSS_LANE      = "1"
$env:VICTORY_DSTORAGE_ELEVATOR_WINDOW = "1024"
$env:VICTORY_DSTORAGE_MAX_SKIPS       = "64"
$env:VICTORY_DSTORAGE_READ_AHEAD      = "1"
$env:VICTORY_DSTORAGE_RAW_READ_AHEAD  = "1"
$env:VICTORY_DSTORAGE_SLAB_CACHE_MB   = "128"
$env:VICTORY_DSTORAGE_READ_AHEAD_MAX_KB = "8192"
$env:VICTORY_DSTORAGE_READ_AHEAD_GAP_KB = "1024"
$env:VICTORY_DSTORAGE_READ_AHEAD_MAX_REQUESTS = "64"
$env:VICTORY_DSTORAGE_DRIVE_DEPTH     = "2"
$env:VICTORY_DSTORAGE_STAGING_MB      = "192"
$env:VICTORY_DSTORAGE_RETRIES         = "12"

if ($NoEarlyPipelines) {
    $env:VICTORY_EARLY_PIPELINES = "0"
}
if ($SkipBrokenTrees) {
    $env:VICTORY_SKIP_PIPELINES = $BrokenTreeHashes
}

switch ($Profile) {
    "MaxFps" {
        $env:VICTORY_RT_OFF             = "1"
        $env:VICTORY_AMBIENT            = if ($Ambient) { $Ambient } else { "0.6" }
        $env:VICTORY_ASYNC_SUBMIT       = "1"
        $env:VICTORY_QUEUE2             = "3"
        $env:VICTORY_INDIRECT_WINDOW    = "1"
        $env:VICTORY_INDIRECT_CARD_BUDGET = "1"
        $env:VICTORY_INDIRECT_CLAMP     = "1"
        $env:VICTORY_RELAY              = "2"
        $env:VICTORY_RELAY_COPY_QUEUE   = "1"
        $env:VICTORY_RELAY_WORKERS      = "6"
        $env:VICTORY_GPU_ALIAS          = "1"
        $env:VICTORY_NV_ALIAS_V2        = "1"
        $env:VICTORY_NV_ALIAS_INDEX     = "1"
        $env:VICTORY_NV_GPU_BUFFER_COPY = "1"
        $env:VICTORY_NV_VEH_COPY        = "1"
        $env:VICTORY_NV_FAST_HEAP_PASS  = "1"
        $env:VICTORY_NV_PARALLEL_HEAP   = "1"
        $env:VICTORY_FILL_WORKERS       = "4"
        $env:VICTORY_PARALLEL_PRELUDES  = "1"
        $env:VICTORY_EARLY_DEPTH_FIRST  = "1"
        $env:VICTORY_NV_BATCH_COPY_BARRIERS = "1"
        $env:VICTORY_LOCAL_BUDGET_BYTES = "10500000000"
        $env:VICTORY_MIP_BIAS           = "auto"
        $env:VICTORY_TILE_RECLAIM       = "1"
        $env:VICTORY_CARD_WINDOWS_MB    = "512"
        $env:VICTORY_CARD_FETCH_INFLIGHT = "1"
        $env:VICTORY_CARD_BATCH         = "1"
        $env:VICTORY_CARD_HASH_WIDE     = "1"
        $env:VICTORY_CARD_REHASH_RUNS   = "1"
        $env:VICTORY_WIDE_BACK          = "1"
        $env:VICTORY_WIDE_MERGE         = "1"
        $env:VICTORY_WIDE_REFRESH       = "1"
        $env:VICTORY_WIDE_WORKERS       = "8"
        $env:VICTORY_COPY_COMPARE       = "1"
        $env:VICTORY_TIGHT_REFRESH      = "1"
        $env:VICTORY_FILL_ON_CARD       = "1"
        $env:VICTORY_LATE_WAIT          = "3"
        $env:VICTORY_LATE_WAIT_LISTS    = "256"
        $env:VICTORY_MERGE_RUNS         = "2048"
        $env:VICTORY_BACK_JOIN          = "128"
    }
    "UltraFps" {
        # Aggressive 30 FPS push on RTX 2060 12 GB (720p default + 8-core parallel heap + tight gameplay texture cap)
        $env:VICTORY_RT_OFF             = "1"
        $env:VICTORY_AMBIENT            = if ($Ambient) { $Ambient } else { "0.6" }
        $env:VICTORY_ASYNC_SUBMIT       = "1"
        $env:VICTORY_QUEUE2             = "3"
        $env:VICTORY_INDIRECT_WINDOW    = "1"
        $env:VICTORY_INDIRECT_CARD_BUDGET = "1"
        $env:VICTORY_INDIRECT_CLAMP     = "1"
        $env:VICTORY_RELAY              = "2"
        $env:VICTORY_RELAY_COPY_QUEUE   = "1"
        $env:VICTORY_RELAY_WORKERS      = "6"
        $env:VICTORY_GPU_ALIAS          = "1"
        $env:VICTORY_NV_ALIAS_V2        = "1"
        $env:VICTORY_NV_ALIAS_INDEX     = "1"
        $env:VICTORY_NV_GPU_BUFFER_COPY = "1"
        $env:VICTORY_NV_VEH_COPY        = "1"
        $env:VICTORY_NV_FAST_HEAP_PASS  = "1"
        $env:VICTORY_NV_PARALLEL_HEAP   = "1"
        $env:VICTORY_FILL_WORKERS       = "4"
        $env:VICTORY_PARALLEL_PRELUDES  = "1"
        $env:VICTORY_EARLY_DEPTH_FIRST  = "1"
        $env:VICTORY_NV_BATCH_COPY_BARRIERS = "1"
        $env:VICTORY_LOCAL_BUDGET_BYTES = "10200000000"
        $env:VICTORY_MIP_BIAS           = "1"
        $env:VICTORY_TILE_RECLAIM       = "1"
        $env:VICTORY_PRELOAD_TEXTURES   = "0"
        $env:VICTORY_EARLY_TEXTURES_MS  = "3"
        $env:VICTORY_EARLY_TEXTURES_MB  = "32"
        $env:VICTORY_BAKE_TEXTURES_MS   = "150"
        $env:VICTORY_TEXTURE_STAGE_MB   = "192"
        $env:VICTORY_TEXTURE_STAGE_MS   = "4"
        $env:VICTORY_TEXTURE_CATCHUP_MS = "8"
        $env:VICTORY_TEXTURE_SWEEP      = "0"
        $env:VICTORY_HEAP_RECHECK       = "0"
        $env:VICTORY_UNWATCHED_FRAMES   = "2"
        $env:VICTORY_CARD_WINDOWS_MB    = "512"
        $env:VICTORY_CARD_FETCH_INFLIGHT = "1"
        $env:VICTORY_CARD_BATCH         = "1"
        $env:VICTORY_CARD_HASH_WIDE     = "1"
        $env:VICTORY_CARD_REHASH_RUNS   = "1"
        $env:VICTORY_WIDE_BACK          = "1"
        $env:VICTORY_WIDE_MERGE         = "1"
        $env:VICTORY_WIDE_REFRESH       = "1"
        $env:VICTORY_WIDE_WORKERS       = "8"
        $env:VICTORY_COPY_COMPARE       = "1"
        $env:VICTORY_TIGHT_REFRESH      = "1"
        $env:VICTORY_FILL_ON_CARD       = "1"
        $env:VICTORY_LATE_WAIT          = "3"
        $env:VICTORY_LATE_WAIT_LISTS    = "256"
        $env:VICTORY_MERGE_RUNS         = "2048"
        $env:VICTORY_BACK_JOIN          = "128"
    }
    "Baseline" {
        # Exact pre-upgrade (20dc2bf) control profile for A/B benchmarking
        $env:VICTORY_RT_OFF             = "1"
        $env:VICTORY_AMBIENT            = if ($Ambient) { $Ambient } else { "0.6" }
        $env:VICTORY_ASYNC_SUBMIT       = "1"
        $env:VICTORY_QUEUE2             = "1"
        $env:VICTORY_INDIRECT_WINDOW    = "1"
        $env:VICTORY_INDIRECT_CARD_BUDGET = "1"
        $env:VICTORY_INDIRECT_CLAMP     = "1"
        $env:VICTORY_RELAY              = "2"
        $env:VICTORY_RELAY_COPY_QUEUE   = "1"
        $env:VICTORY_RELAY_WORKERS      = "6"
        $env:VICTORY_GPU_ALIAS          = "1"
        $env:VICTORY_NV_ALIAS_V2        = "0"
        $env:VICTORY_NV_ALIAS_INDEX     = "0"
        $env:VICTORY_NV_GPU_BUFFER_COPY = "0"
        $env:VICTORY_NV_VEH_COPY        = "0"
        $env:VICTORY_NV_FAST_HEAP_PASS  = "0"
        $env:VICTORY_NV_PARALLEL_HEAP   = "0"
        $env:VICTORY_FILL_WORKERS       = "1"
        $env:VICTORY_PARALLEL_PRELUDES  = "0"
        $env:VICTORY_EARLY_DEPTH_FIRST  = "0"
        $env:VICTORY_NV_BATCH_COPY_BARRIERS = "0"
        $env:VICTORY_LOCAL_BUDGET_BYTES = "0"
        $env:VICTORY_MIP_BIAS           = "0"
        $env:VICTORY_TILE_RECLAIM       = "0"
        $env:VICTORY_TEXTURE_ROOM_PERCENT = "86"
        $env:VICTORY_CARD_ROOM_FOR_TARGETS_MB = "2560"
        $env:VICTORY_DSTORAGE_CROSS_LANE = "0"
        $env:VICTORY_DSTORAGE_RAW_READ_AHEAD = "0"
        $env:VICTORY_DSTORAGE_SLAB_CACHE_MB = "0"
        $env:VICTORY_DSTORAGE_ELEVATOR_WINDOW = "256"
        $env:VICTORY_DSTORAGE_MAX_SKIPS = "16"
        $env:VICTORY_DSTORAGE_READ_AHEAD_MAX_KB = "4096"
        $env:VICTORY_DSTORAGE_READ_AHEAD_GAP_KB = "512"
        $env:VICTORY_DSTORAGE_READ_AHEAD_MAX_REQUESTS = "32"
        $env:VICTORY_DSTORAGE_DRIVE_DEPTH = "3"
        $env:VICTORY_DSTORAGE_STAGING_MB = "128"
        $env:VICTORY_CARD_FETCH_INFLIGHT = "1"
        $env:VICTORY_CARD_BATCH         = "1"
        $env:VICTORY_CARD_HASH_WIDE     = "1"
        $env:VICTORY_CARD_REHASH_RUNS   = "1"
        $env:VICTORY_WIDE_BACK          = "1"
        $env:VICTORY_WIDE_MERGE         = "1"
        $env:VICTORY_WIDE_REFRESH       = "1"
        $env:VICTORY_WIDE_WORKERS       = "8"
        $env:VICTORY_COPY_COMPARE       = "1"
        $env:VICTORY_TIGHT_REFRESH      = "1"
        $env:VICTORY_FILL_ON_CARD       = "1"
        $env:VICTORY_LATE_WAIT          = "3"
        $env:VICTORY_LATE_WAIT_LISTS    = "256"
        $env:VICTORY_MERGE_RUNS         = "2048"
        $env:VICTORY_BACK_JOIN          = "128"
    }
    "Balanced" {
        $env:VICTORY_RT_OFF             = "1"
        $env:VICTORY_AMBIENT            = if ($Ambient) { $Ambient } else { "0.6" }
        $env:VICTORY_ASYNC_SUBMIT       = "1"
        $env:VICTORY_QUEUE2             = "1"
        $env:VICTORY_INDIRECT_WINDOW    = "1"
        $env:VICTORY_INDIRECT_CARD_BUDGET = "1"
        $env:VICTORY_INDIRECT_CLAMP     = "1"
        $env:VICTORY_RELAY              = "2"
        $env:VICTORY_RELAY_WORKERS      = "4"
        $env:VICTORY_NV_VEH_COPY        = "1"
        $env:VICTORY_NV_FAST_HEAP_PASS  = "1"
        $env:VICTORY_LOCAL_BUDGET_BYTES = "10500000000"
        $env:VICTORY_MIP_BIAS           = "auto"
        $env:VICTORY_TILE_RECLAIM       = "1"
        $env:VICTORY_NV_BATCH_COPY_BARRIERS = "1"
        $env:VICTORY_WIDE_BACK          = "1"
        $env:VICTORY_WIDE_MERGE         = "1"
        $env:VICTORY_WIDE_REFRESH       = "1"
        $env:VICTORY_WIDE_WORKERS       = "6"
        $env:VICTORY_COPY_COMPARE       = "1"
        $env:VICTORY_TIGHT_REFRESH      = "1"
        $env:VICTORY_LATE_WAIT          = "2"
        $env:VICTORY_MERGE_RUNS         = "1024"
    }
    "Safe" {
        $env:VICTORY_RT_OFF             = "1"
        $env:VICTORY_AMBIENT            = if ($Ambient) { $Ambient } else { "0.6" }
        $env:VICTORY_ASYNC_SUBMIT       = "0"
        $env:VICTORY_QUEUE2             = "0"
        $env:VICTORY_INDIRECT_WINDOW    = "0"
        $env:VICTORY_GPU_ALIAS          = "0"
        $env:VICTORY_NV_ALIAS_V2        = "0"
        $env:VICTORY_NV_ALIAS_INDEX     = "0"
        $env:VICTORY_NV_GPU_BUFFER_COPY = "0"
        $env:VICTORY_NV_VEH_COPY        = "0"
        $env:VICTORY_NV_FAST_HEAP_PASS  = "0"
        $env:VICTORY_NV_PARALLEL_HEAP   = "0"
        $env:VICTORY_FILL_WORKERS       = "1"
        $env:VICTORY_PARALLEL_PRELUDES  = "0"
        $env:VICTORY_EARLY_DEPTH_FIRST  = "0"
        $env:VICTORY_NV_BATCH_COPY_BARRIERS = "0"
        $env:VICTORY_LOCAL_BUDGET_BYTES = "10500000000"
        $env:VICTORY_MIP_BIAS           = "0"
        $env:VICTORY_TILE_RECLAIM       = "0"
        $env:VICTORY_RELAY              = "1"
        $env:VICTORY_WIDE_REFRESH       = "1"
        $env:VICTORY_WIDE_WORKERS       = "4"
        $env:VICTORY_COPY_COMPARE       = "1"
    }
    "RayTracing" {
        $env:VICTORY_RT_OFF             = "0"
        $env:VICTORY_BVH_BUILDER        = "1"
        $env:VICTORY_HOST_WAVE          = "32"
        $env:VICTORY_BACK_JOIN          = "256"
        $env:VICTORY_AMBIENT            = if ($Ambient) { $Ambient } else { "off" }
        $env:VICTORY_ASYNC_SUBMIT       = "1"
        $env:VICTORY_QUEUE2             = "3"
        $env:VICTORY_INDIRECT_WINDOW    = "1"
        $env:VICTORY_RELAY              = "2"
        $env:VICTORY_RELAY_COPY_QUEUE   = "1"
        $env:VICTORY_RELAY_WORKERS      = "6"
        $env:VICTORY_NV_VEH_COPY        = "1"
        $env:VICTORY_NV_FAST_HEAP_PASS  = "1"
        $env:VICTORY_LOCAL_BUDGET_BYTES = "10500000000"
        $env:VICTORY_MIP_BIAS           = "auto"
        $env:VICTORY_TILE_RECLAIM       = "1"
        $env:VICTORY_NV_BATCH_COPY_BARRIERS = "1"
        $env:VICTORY_WIDE_BACK          = "1"
        $env:VICTORY_WIDE_MERGE         = "1"
        $env:VICTORY_WIDE_REFRESH       = "1"
        $env:VICTORY_WIDE_WORKERS       = "8"
        $env:VICTORY_COPY_COMPARE       = "1"
        $env:VICTORY_TIGHT_REFRESH      = "1"
        $env:VICTORY_MERGE_RUNS         = "2048"
    }
}

# Apply per-feature A/B overrides cleanly after profile selection
if ($AliasV2 -ne "")           { $env:VICTORY_NV_ALIAS_V2 = $AliasV2 }
if ($AliasIndex -ne "")        { $env:VICTORY_NV_ALIAS_INDEX = $AliasIndex }
if ($GpuBufferCopy -ne "")     { $env:VICTORY_NV_GPU_BUFFER_COPY = $GpuBufferCopy }
if ($VehCopy -ne "")           { $env:VICTORY_NV_VEH_COPY = $VehCopy }
if ($FastHeapPass -ne "")      { $env:VICTORY_NV_FAST_HEAP_PASS = $FastHeapPass }
if ($ParallelHeap -ne "")      { $env:VICTORY_NV_PARALLEL_HEAP = $ParallelHeap }
if ($FillWorkers -ne "")       { $env:VICTORY_FILL_WORKERS = $FillWorkers }
if ($ParallelPreludes -ne "")  { $env:VICTORY_PARALLEL_PRELUDES = $ParallelPreludes }
if ($EarlyDepthFirst -ne "")   { $env:VICTORY_EARLY_DEPTH_FIRST = $EarlyDepthFirst }
if ($LocalBudgetBytes -ne "")  { $env:VICTORY_LOCAL_BUDGET_BYTES = $LocalBudgetBytes }
if ($MipBias -ne "")           { $env:VICTORY_MIP_BIAS = $MipBias }
if ($TileReclaim -ne "")       { $env:VICTORY_TILE_RECLAIM = $TileReclaim }
if ($BatchCopyBarriers -ne "") { $env:VICTORY_NV_BATCH_COPY_BARRIERS = $BatchCopyBarriers }
if ($CardWindowsMB -ne "")     { $env:VICTORY_CARD_WINDOWS_MB = $CardWindowsMB }
if ($Queue2 -ne "")            { $env:VICTORY_QUEUE2 = $Queue2 }
if ($CrossLaneElevator -ne "") { $env:VICTORY_DSTORAGE_CROSS_LANE = $CrossLaneElevator }
if ($DriveDepth -ne "")        { $env:VICTORY_DSTORAGE_DRIVE_DEPTH = $DriveDepth }
if ($ReadAheadMaxKB -ne "")    { $env:VICTORY_DSTORAGE_READ_AHEAD_MAX_KB = $ReadAheadMaxKB }
if ($ReadAheadGapKB -ne "")    { $env:VICTORY_DSTORAGE_READ_AHEAD_GAP_KB = $ReadAheadGapKB }
if ($SlabCacheMB -ne "")       { $env:VICTORY_DSTORAGE_SLAB_CACHE_MB = $SlabCacheMB }
if ($RawReadAhead -ne "")      { $env:VICTORY_DSTORAGE_RAW_READ_AHEAD = $RawReadAhead }
if ($TextureRoomPercent -ne "") { $env:VICTORY_TEXTURE_ROOM_PERCENT = $TextureRoomPercent }
if ($TargetsRoomMB -ne "")     { $env:VICTORY_CARD_ROOM_FOR_TARGETS_MB = $TargetsRoomMB }
if ($TextureStageMS -ne "")    { $env:VICTORY_TEXTURE_STAGE_MS = $TextureStageMS }
if ($TextureCatchupMS -ne "")  { $env:VICTORY_TEXTURE_CATCHUP_MS = $TextureCatchupMS }
if ($TextureSweep -ne "")      { $env:VICTORY_TEXTURE_SWEEP = $TextureSweep }
if ($HeapRecheck -ne "")       { $env:VICTORY_HEAP_RECHECK = $HeapRecheck }
if ($UnwatchedFrames -ne "")   { $env:VICTORY_UNWATCHED_FRAMES = $UnwatchedFrames }
if ($GpuTiming -ne "")         { $env:VICTORY_GPU_TIMING = $GpuTiming }
foreach ($k in $ExtraEnv.Keys) {
    Set-Item -Path "Env:$k" -Value ([string]$ExtraEnv[$k])
}

$StdoutLog = Join-Path $LogDir "stdout.log"
$StderrLog = Join-Path $LogDir "stderr.log"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  NShackteam - Unified GTA VI PC Layer (RTX 2060 12 GB)   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Profile     : $Profile | Display: $EffectiveResolution | Console: $Console"
Write-Host "  Lighting    : AMBIENT=$($env:VICTORY_AMBIENT) RT_OFF=$($env:VICTORY_RT_OFF) NAN_GUARD=$($env:VICTORY_NAN_GUARD) WAVE_PAIRS=$($env:VICTORY_WAVE_PAIRS)"
Write-Host "  Async/Q2    : ASYNC=$($env:VICTORY_ASYNC_SUBMIT) QUEUE2=$($env:VICTORY_QUEUE2) RELAY=$($env:VICTORY_RELAY) ALIAS=$($env:VICTORY_GPU_ALIAS)"
Write-Host "  CPU/GPU Opt : VEH_COPY=$($env:VICTORY_NV_VEH_COPY) FAST_HEAP=$($env:VICTORY_NV_FAST_HEAP_PASS) PAR_HEAP=$($env:VICTORY_NV_PARALLEL_HEAP) FILLS=$($env:VICTORY_FILL_WORKERS) PRELUDES=$($env:VICTORY_PARALLEL_PRELUDES) DEPTH_1ST=$($env:VICTORY_EARLY_DEPTH_FIRST)"
Write-Host "  VRAM 10.5GB : BUDGET=$($env:VICTORY_LOCAL_BUDGET_BYTES) ROOM=$($env:VICTORY_TEXTURE_ROOM_PERCENT)% TARGETS=$($env:VICTORY_CARD_ROOM_FOR_TARGETS_MB)MB MIP_BIAS=$($env:VICTORY_MIP_BIAS)"
Write-Host "  Storage     : ELEVATOR=$($env:VICTORY_DSTORAGE_ELEVATOR) CROSS_LANE=$($env:VICTORY_DSTORAGE_CROSS_LANE) SLAB=$($env:VICTORY_DSTORAGE_SLAB_CACHE_MB)MB RAW_RA=$($env:VICTORY_DSTORAGE_RAW_READ_AHEAD) DEPTH=$($env:VICTORY_DSTORAGE_DRIVE_DEPTH)"
Write-Host "  Binary      : $ExePath"
Write-Host "  Title       : $TitleExe"
Write-Host "  Logs        : $StderrLog"
Write-Host "==========================================================" -ForegroundColor Cyan

$ArgsList = @("run", $TitleExe, "--execute", "--console", $Console)

$ProcessArgs = 'run "{0}" --execute --console {1}' -f $TitleExe, $Console

if ($Background) {
    $proc = Start-Process -FilePath $ExePath -ArgumentList $ProcessArgs -WorkingDirectory $RepoDir `
        -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -WindowStyle Normal -PassThru
    Write-Host "[NShackteam] Started in background (PID $($proc.Id))." -ForegroundColor Green
    if ($Tail -and -not $NoTailWindow) {
        $tailCmd = "`$Host.UI.RawUI.BackgroundColor='Black'; `$Host.UI.RawUI.ForegroundColor='Gray'; [Console]::Write(`"`$([char]27)]11;#000000`$([char]7)`"); Clear-Host; Get-Content -LiteralPath '$StderrLog' -Wait -Tail 40"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $tailCmd
    }
    if ($Wait) {
        $proc.WaitForExit()
        exit $proc.ExitCode
    }
    exit 0
}

if ($Tail) {
    $proc = Start-Process -FilePath $ExePath -ArgumentList $ProcessArgs -WorkingDirectory $RepoDir `
        -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -WindowStyle Normal -PassThru
    Write-Host "[NShackteam] Running (PID $($proc.Id)) - tailing $StderrLog..." -ForegroundColor Green
    Start-Sleep -Milliseconds 800
    Get-Content -LiteralPath $StderrLog -Wait -Tail 30
} else {
    & $ExePath @ArgsList
    exit $LASTEXITCODE
}
