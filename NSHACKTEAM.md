# NShackteam architecture and credits

NShackteam Wave 5 combines work from the credited VIctory forks with team changes for an RTX 2060 12 GB, a 10-core CPU and SATA HDD.

## Unified Architecture & Sources

1. **Belobug (`Belobug/VIctory-Belobug-AMD`)**
   - `crates/victory-gpu/src/ambient.rs` (`VICTORY_AMBIENT=0.6`, `off` disables): replaces `CS_ScreenProbes_TemporalResolve`'s empty/garbage `R9G9B9E5` indirect diffuse UAV with a clean ambient fill — eliminating pitch-black shadows, magenta translucent foliage, and indoor color blocks while RT is off.
   - Pixel-shader helper-lane fixes in `lower.rs` & `lower_valu.rs`: quad reads include helper lanes (fixes triangle-edge lines on car paint, glass, skin) and helper-lane `v_readfirstlane_b32` reads its own value (stops helper-lane waterfall loops).
   - Default skip list for 8 broken wind-volume distant tree imposter pipelines (`-SkipBrokenTrees`).

2. **`fxceless/victory-fork` (`f85101b` *"Frame rate, second night"* + PRs #827, #837, #838, #842, #845, #846, #853, #854, #855)**
   - **Async frame pipeline**: `VICTORY_ASYNC_SUBMIT=1`, `VICTORY_QUEUE2=3` (copy-queue fetches + card uploads), guarded `VICTORY_INDIRECT_WINDOW=1` (`MOST_DRAW`), no-wait upload fills, 512 idle host lists, staging buffer reuse, `VICTORY_MERGE_RUNS=2048`.
   - **Shader coverage (99.0%, +230 programs on NVIDIA 32-lane GPUs — #854)**: paired wave64 execution on 32-lane hosts without barriers (`VICTORY_WAVE_PAIRS=1`), linked mesh-pixel output pruning + 16-bit barycentric packing fallback (`VICTORY_MESH_LINK=1`), `float64.rs`, `descriptor_data.rs`, `v_dot*` packed dot products, POPS ordering.
   - **VRAM & RAM efficiency**: `VICTORY_LEVELS_ON_DEMAND=1` (~3.1 GB VRAM saved), `VICTORY_VIEW_GUARD=1` (#853 zero `VirtualProtect` churn), `VICTORY_TABLED_SKIP_FRAMES=2` (#827 world-entry crash fix), `VICTORY_RAM_RELIEF_MB=2048`, `coalesce_pieces`, thread-local `fread` scratch buffer.
   - **Visual & stability fixes**: `VICTORY_NAN_GUARD=1` (#837 stops NaN black/overexposed boxes on NVIDIA), page-boundary title-memory load/store split (#846), fail-fast on device removal (#838), `VICTORY_BACK_JOIN=256` (#842 fast BVH copy-back for RT mode), DirectStorage late page-number volatile wait + retry (#855), `VICTORY_EARLY_PIPELINES=0` opt-out (#845).

3. **`uiliannn/VItoriaBr`**
   - `crates/victory-gpu/src/pipeline/make.rs`: traditional vertex+pixel graphics pipelines (HUD, screen composition, tonemapping) compile ahead of mesh and compute pipelines on cold cache.

4. **`Zettrox/ztrx` + NS Hack Team RTX 2060 12 GB Engine**
   - **GPU-to-GPU Aliasing v2 + GPU Buffer Copy Promotion (`gpu_alias.rs`, `host.rs`)**:
     - `VICTORY_NV_ALIAS_INDEX=1`: interval-indexed range overlap planner (exact submission order without O(N*M) scans).
     - `VICTORY_NV_ALIAS_V2=1`: unlocks GPU-to-GPU aliasing into carded destinations with `card_has_before()` by running a targeted post-copy GPU rehash (`Host::card_rehash_ranges`) on only the aliased byte ranges.
     - `VICTORY_NV_GPU_BUFFER_COPY=1`: promotes in-flight-conflicting `Step::Copy` (`CopyBufferRegion`) transfers onto GPU `CopyBufferRegion` + async relay writeback (`Flight::back` / `card_before`), cutting `by a buffer copy` pipeline stalls.
     - Fixes `Host::submit()` alias command-list acquisition ordering so failed allocations cleanly fall back to `relay_wait`/`settle` without dropping updates.
   - **Unified 10.5 GB VRAM Ceiling, High-Mip Cap, & Fence-Deferred Tile Reclamation (`resource.rs`, `tiles.rs`, `shadow.rs`)**:
     - `VICTORY_LOCAL_BUDGET_BYTES=10500000000` (or `auto` on <= 12 GiB NVIDIA `0x10DE` GPUs): unified local VRAM gate with atomic concurrent reservation across `resource::room()`, 64 MiB `tiles::take()` heaps, `shadow::card_for()`, and `shadow::window_card_for()`.
     - `VICTORY_MIP_BIAS=auto|1|2`: caps finest standard mip level(s) on title-filled levels-on-demand textures (`auto` triggers at >= 82% local VRAM pressure), protecting card-written levels, packed mip tails, and the last held level.
     - `VICTORY_TILE_RECLAIM=1`: unmaps shed/unheld standard mip tiles (`D3D12_TILE_RANGE_FLAG_NULL`) and returns them to the pool via `host::retire()` strictly after GPU fence completion.
   - **CPU Frame-Time Accelerators (`fast_copy.rs`, `host.rs`, `descriptor/host.rs`, `heap_pass.rs`, `draw.rs`)**:
     - `VICTORY_NV_VEH_COPY=1`: VEH-guarded AVX2/SSE2 userspace `try_copy` inside `copy_mapped` (eliminates ~27,000 `ReadProcessMemory` kernel syscalls/frame across ~290 MB/frame while catching unmapped source faults cleanly).
     - `VICTORY_NV_FAST_HEAP_PASS=1`: fixes `HostHeap::note_buffer` re-queueing already-seen buffer slots into `self.changed`, skips redundant `textures`/`placed` lock traffic on existing buffer slots, eliminates duplicate `shadow::find()` calls inside `buffer_copy()` and `mirror_buffer()`, and preserves `heap_pass::Picked` `tight_spans` cache when a slot's `(copy, from, to)` is unchanged.
     - `VICTORY_NV_PARALLEL_HEAP=1`: parallelizes `heap_buffers` D3D12 buffer-view creation + shadow span resolution across up to 8 CPU cores on disjoint descriptor heap handles while acquiring `made`/`buffers`/`kept` once per batch instead of once per slot.
     - `-Profile UltraFps`: aggressive gameplay FPS profile targeting 30 FPS (720p default, 8-core parallel heap, `VICTORY_TEXTURE_STAGE_MS=4`, `VICTORY_MIP_BIAS=1`, `VICTORY_UNWATCHED_FRAMES=2`, `VICTORY_TEXTURE_SWEEP=0`, `VICTORY_HEAP_RECHECK=0`).
   - **HDD DirectStorage Cross-Lane C-SCAN Streaming (`dstorage.rs`, `dstorage_elevator.rs`)**:
     - Fixes `read_ahead_elevator` aborting after `start_job_at` updated `lane.clock`.
     - `VICTORY_DSTORAGE_CROSS_LANE=1` + `VICTORY_DSTORAGE_RAW_READ_AHEAD=1` + `VICTORY_DSTORAGE_SLAB_CACHE_MB=128` + `VICTORY_DSTORAGE_ELEVATOR_WINDOW=1024` + `VICTORY_DSTORAGE_MAX_SKIPS=64` + `VICTORY_DSTORAGE_DRIVE_DEPTH=2` + 8 MiB read-ahead span: sweeps and coalesces same-file `.rpf` reads (including raw plain-address reads) across all same-priority `fiDeviceDirectStorage` queues on a single SATA HDD, backed by a bounded 128 MiB 1 MiB-aligned LRU slab cache in RAM to absorb scattered small reads on fragmented `.rpf` archives without breaking status ordering.
   - **Batched Copy Barriers & NVIDIA Telemetry (`list.rs`, `host.rs`)**:
     - `VICTORY_NV_BATCH_COPY_BARRIERS=1`: combines distinct texture `(src, COPY_SOURCE)` + `(dst, COPY_DEST)` transitions into a single `ResourceBarrier(&[b1, b2])` call in `copy_resource` and `copy_texture_region`.
     - Adapter vendor/VRAM detection (`VendorId == 0x10DE` / `Host::is_nvidia()`), `VICTORY_GPU_TIMING=0` safe parsing (never silently disables `VICTORY_LATE_WAIT`), AVX2 `fast_copy.rs`, HDD C-SCAN `dstorage_elevator.rs`.

5. **`DexNom/VIctory` (DexNom Runtime, #1 PortWatch VI — 78%) — Integrated into NShackteam Wave 5**
   - **Multi-Core Upload Fill Preparation (`parallel_submit.rs`, `host.rs`)**: `VICTORY_FILL_WORKERS=4` coalesces contiguous independent `Step::Fill` upload copies (128+ fills) into `Step::Fills` batches executed across CPU workers on `shadow::wide_run`, combining DexNom's multi-core fill fan-out with NShackteam's AVX2 VEH `copy_mapped`.
   - **Parallel Transition Prelude Recording (`parallel_submit.rs`, `host.rs`)**: `VICTORY_PARALLEL_PRELUDES=1` resolves texture state transitions and depth-twin syncs in strict guest order, then records independent D3D12 prelude command lists across CPU workers when submissions contain 4+ unique lists.
   - **Depth-Only Early Pipeline Warmup Priority (`pipeline/host.rs`, `pipeline/make.rs`)**: `VICTORY_EARLY_DEPTH_FIRST=1` warms previously used shadow-map & depth-prepass PSOs ahead of speculative material pipelines while preserving `VItoriaBr` traditional graphics priority for live draws.
   - **Crash-Safe Atomic Save Recovery (`victory-hle/src/modules/xgamesave.rs`)**: flushes kept blobs via `sync_all()` when falling back past NTFS's 1,024 hard-link limit, ignores non-canonical version folders (`v01`, `v0`), and preserves older readable save checkpoints if a newer folder is incomplete or corrupted.

## Build and benchmark

See the [project README](README.md) for launch instructions and the [RTX 2060 benchmark](docs/benchmark-2026-11-25.md) for the team's measured results.
