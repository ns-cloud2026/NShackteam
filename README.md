# NShackteam Wave 5

NShackteam is a Windows compatibility-layer project tuned for an NVIDIA RTX 2060 12 GB and SATA HDD. It builds on the GPL-licensed VIctory ecosystem.

The team's [November 25 benchmark](docs/benchmark-2026-11-25.md) reached the main menu at **29.87 FPS mean** over 212 boot/menu samples at a reported 1280×720. Open-world FPS is unmeasured.

## Team changes

- NVIDIA VRAM budget, mip-level limits and fence-deferred tile reclamation.
- GPU buffer-copy aliasing and parallel descriptor-heap work.
- Cross-lane HDD read-ahead with a 128 MiB slab cache.
- Parallel upload fills, transition recording and depth-first warmup integrated in Wave 5.

See [NShackteam architecture and credits](NSHACKTEAM.md) for source attribution and implementation details. The [upstream fork notes](docs/upstream-fork-notes.md) describe prior upstream runs.

## Build and launch

On Windows, use the Rust toolchain pinned by `rust-toolchain.toml` and the MSVC build tools. The build requires your own lawfully obtained BCPack table file at `crates/victory-formats/src/bcpack-tables.bin`. It is git-ignored. Keep your local title files outside the checkout. See the [asset policy](docs/asset-policy.md).

```powershell
cargo build --release --locked -p victory
.\run-nshackteam.ps1 -GameDir 'D:\your-local-title-files' -Profile UltraFps -Background -Wait -NoTailWindow
```

The launcher stores logs in `.nshackteam-state\logs` under the local title directory. For a report from a captured run, use `collect-portwatch-report.ps1 -LogPath <path-to-stderr.log>`.

## License

The source is GPL-3.0-or-later. See [LICENSE](LICENSE). The repository contains source and documentation only.
