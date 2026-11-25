# NShackteam Wave 5 RTX 2060 12 GB benchmark

Maintainer-reported results from **November 25, 2026, 08:39:46 PST**. The measured local build was `ddc6acd` on `nshackteam`, running the release `UltraFps` profile with ray tracing off. The runtime reported a 1280×720 display mode to the title.

| Test system | Value |
| --- | --- |
| GPU | NVIDIA RTX 2060 12 GB, driver 617.42 |
| CPU | Intel Xeon E5-2660 v3, 10 cores and 20 threads |
| RAM and storage | 31.8 GB RAM, SATA HDD |
| Process at report time | Running for 18.3 min, 6,498 MB working set, 8,968 MB private memory |
| GPU at report time | 3,324 MiB used of 12,288 MiB, 20% utilization |
| Unhandled exceptions and device removals in the captured log | 0 |

## Frame telemetry

| Phase | Samples | Min FPS | Median FPS | Mean FPS | Max FPS | Mean frame time | Avg / max draws | Mean heap pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Boot / main menu (<100 draws) | 212 | 7.2 | 30.0 | 29.87 | 30.1 | 33.8 ms | 12 / 13 | 0 ms |
| Open world | 0 | — | — | — | — | — | — | — |

The 212 samples are periodic runtime summaries. This run records boot and menu telemetry only.

## Subsystem telemetry

- DirectStorage C-SCAN and 128 MB slab cache peaked at 43.7 MB/s, with 257 slab hits, 33.1 MB served from RAM and 0 fallbacks.
- The runtime configured a 9,727 MiB local VRAM budget and enabled on-demand texture levels.
- The last recorded AVX2/SSE2 copy sample showed 2 fast copies per frame and 0 unmapped fallbacks.
- The last recorded submission sample showed 4.0 submissions sent ahead per frame, 0.0 ms waiting for the GPU in settles and 2.0 fence signals placed behind flights.
- The last GPU clock sample measured 28.6 ms with no work from the layer queued and estimated 14% GPU activity.

```text
2026-11-25T16:21:33.894580Z  host adapter name="NVIDIA GeForce RTX 2060" vendor=0x10de vram_mb=12061 nvidia=true
2026-11-25T16:21:34.129226Z  display mode reported to the title width=1280 height=720
2026-11-25T16:21:59.584111Z  unified local VRAM budget active (VICTORY_LOCAL_BUDGET_BYTES) budget_bytes=10200000000 budget_mib=9727 nvidia=true
```
