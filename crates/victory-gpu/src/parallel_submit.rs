//! Multi-core submission preparation helpers ported from DexNom Runtime and
//! adapted for NShackteam (`VICTORY_FILL_WORKERS`, `VICTORY_PARALLEL_PRELUDES`).
//!
//! Both features preserve exact guest submission order and are gated behind
//! environment variables so they can be A/B benchmarked independently.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

/// A host command list prelude pair: allocator and closed command list.
pub(crate) type PreludeCommands = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// One deferred upload-memory copy (`dst`, `src`, `len`) recorded on a list.
#[derive(Clone, Copy)]
pub(crate) struct Fill {
    pub(crate) dst: usize,
    pub(crate) src: usize,
    pub(crate) len: usize,
}

impl Fill {
    pub(crate) fn run(self) {
        crate::host::copy_mapped(self.dst as *mut u8, self.src as *const u8, self.len, false);
    }
}

/// Minimum fills per worker chunk before fanning out onto `shadow::wide_run`.
pub(crate) const FILLS_PER_JOB: usize = 128;

/// Maximum fills grouped into a single `Step::Fills` batch.
pub(crate) const FILL_GROUP_MAX: usize = 4096;

/// Parallel upload fill telemetry: `[batches_dispatched, fills_dispatched]`.
pub(crate) static PARALLEL_FILL_STATS: [AtomicU64; 2] =
    [const { AtomicU64::new(0) }; 2];

/// Parallel prelude recording telemetry: `[batches_dispatched, lists_recorded]`.
pub(crate) static PARALLEL_PRELUDE_STATS: [AtomicU64; 2] =
    [const { AtomicU64::new(0) }; 2];

/// Worker count for independent runs of upload fills at submission
/// (`VICTORY_FILL_WORKERS=1..=6`; default `1`, off for A/B testing).
pub(crate) fn fill_workers() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("VICTORY_FILL_WORKERS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 6)
    })
}

/// Increasing scratch allocations have distinct destinations. A source that
/// intersects the destination span (or an overflowing range) stays serial.
pub(crate) fn independent_fills(fills: &[Fill]) -> bool {
    let Some(first) = fills.first() else {
        return false;
    };
    let mut end = first.dst;
    for fill in fills {
        let Some(next) = fill.dst.checked_add(fill.len) else {
            return false;
        };
        if fill.dst < end {
            return false;
        }
        end = next;
    }
    fills.iter().all(|f| {
        f.src
            .checked_add(f.len)
            .is_some_and(|src_end| f.src >= end || src_end <= first.dst)
    })
}

/// Runs a contiguous series of upload fills. Workers join before the next step
/// can access title memory, release scratch, or submit lists to the GPU.
pub(crate) fn run_fills(fills: &[Fill], workers: usize) -> usize {
    let jobs = workers.min(fills.len() / FILLS_PER_JOB);
    if jobs > 1 && independent_fills(fills) {
        let queued = fills
            .chunks(fills.len().div_ceil(jobs))
            .map(|chunk| {
                let owned = chunk.to_vec();
                Box::new(move || owned.into_iter().for_each(Fill::run))
                    as Box<dyn FnOnce() + Send>
            })
            .collect();
        crate::shadow::wide_run(queued);
        PARALLEL_FILL_STATS[0].fetch_add(1, Ordering::Relaxed);
        PARALLEL_FILL_STATS[1].fetch_add(fills.len() as u64, Ordering::Relaxed);
        jobs
    } else {
        fills.iter().copied().for_each(Fill::run);
        1
    }
}

/// One texture state transition resolved on the queue thread in guest order.
pub(crate) struct PreludeTransition(pub(crate) ID3D12Resource, pub(crate) u32, pub(crate) u32);

fn record_transitions(list: &ID3D12GraphicsCommandList, transitions: &[PreludeTransition]) {
    if transitions.is_empty() {
        return;
    }
    let barriers: Vec<_> = transitions
        .iter()
        .map(|PreludeTransition(texture, from, to)| {
            crate::host::transition(
                texture,
                D3D12_RESOURCE_STATES(*from as i32),
                D3D12_RESOURCE_STATES(*to as i32),
            )
        })
        .collect();
    // SAFETY: the list is open and each transition owns a live resource ref.
    unsafe { list.ResourceBarrier(&barriers) };
}

/// A list's transition/twin prelude whose state tracking has already been
/// resolved in strict guest submission order and is ready to record.
pub(crate) struct PlannedPrelude {
    pub(crate) commands: PreludeCommands,
    pub(crate) to_copy: Vec<PreludeTransition>,
    pub(crate) syncs: Vec<(
        Arc<crate::resource::HostTexture>,
        Arc<crate::resource::Twin>,
        bool,
    )>,
    pub(crate) barriers: Vec<PreludeTransition>,
}

impl PlannedPrelude {
    pub(crate) fn record(self) {
        let (allocator, list) = self.commands;
        // SAFETY: the submission owns this allocator and list. No other
        // worker records it, and the previous flight has released it.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(&allocator, None)?;
                record_transitions(&list, &self.to_copy);
                for (depth, twin, colour) in &self.syncs {
                    crate::host::record_twin_copy(&list, &depth.texture, twin, *colour);
                }
                record_transitions(&list, &self.barriers);
                list.Close()
            })
        };
        if let Err(e) = recorded {
            crate::host::fail(format_args!("a host state prelude did not record ({e})"));
        }
    }
}

/// Whether independent transition preludes may record in parallel on CPU
/// workers (`VICTORY_PARALLEL_PRELUDES=1`; off by default for A/B testing).
pub(crate) fn parallel_preludes_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_PARALLEL_PRELUDES").is_some_and(|v| v == "1")
    })
}

/// Single-list and repeated-list submissions must record inline because each
/// list owns a single prelude command allocator.
pub(crate) fn should_record_preludes_parallel(
    enabled: bool,
    list_count: usize,
    all_unique: bool,
) -> bool {
    enabled && list_count >= 4 && all_unique
}

/// Records planned preludes, fanning out onto `shadow::wide_run` when the
/// batch is large enough to amortize worker dispatch overhead.
pub(crate) fn record_preludes(plans: Vec<PlannedPrelude>) {
    let barrier_count: usize = plans
        .iter()
        .map(|p| p.to_copy.len() + p.barriers.len())
        .sum();
    let includes_copies = plans.iter().any(|p| !p.syncs.is_empty());
    if plans.len() < 4 || (barrier_count < 256 && !includes_copies) {
        for plan in plans {
            plan.record();
        }
        return;
    }
    let count = plans.len() as u64;
    #[cfg(feature = "diagnostics")]
    {
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| {
            tracing::info!(
                lists = count,
                "recording submission preludes on CPU workers (VICTORY_PARALLEL_PRELUDES=1)"
            )
        });
    }
    let jobs = plans
        .into_iter()
        .map(|plan| Box::new(move || plan.record()) as Box<dyn FnOnce() + Send>)
        .collect();
    crate::shadow::wide_run(jobs);
    PARALLEL_PRELUDE_STATS[0].fetch_add(1, Ordering::Relaxed);
    PARALLEL_PRELUDE_STATS[1].fetch_add(count, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fills(source: &[u8], dst: &mut [u8], bytes: usize) -> Vec<Fill> {
        dst.chunks_exact_mut(bytes)
            .enumerate()
            .map(|(i, out)| Fill {
                dst: out.as_mut_ptr() as usize,
                src: source[i * bytes..].as_ptr() as usize,
                len: bytes,
            })
            .collect()
    }

    #[test]
    fn parallel_upload_fills_match_serial_bytes_and_follow_later_source_edits() {
        let mut input: Vec<u8> = (0..512 * 448).map(|i| (i % 251) as u8).collect();
        let mut serial = vec![0; input.len()];
        let mut parallel = vec![0; input.len()];
        let serial_jobs = make_fills(&input, &mut serial, 448);
        let jobs = make_fills(&input, &mut parallel, 448);
        for turn in [0x5a, 0x37] {
            for b in &mut input {
                *b ^= turn;
            }
            assert_eq!(run_fills(&serial_jobs, 1), 1);
            assert_eq!(run_fills(&jobs, 4), 4);
            assert_eq!(parallel, input);
            assert_eq!(parallel, serial);
        }
    }

    #[test]
    fn overlapping_fills_keep_the_last_write_and_source_aliases_remain_ordered() {
        let source: Vec<u8> = (0..256).map(|n| n as u8).collect();
        let mut output = [0; 256];
        let mut jobs = make_fills(&source, &mut output, 1);
        jobs[255].dst = jobs[0].dst;
        assert_eq!(run_fills(&jobs, 2), 1);
        assert_eq!(output[0], 255);
        let mut jobs = make_fills(&source, &mut output, 1);
        jobs[0].src = source[167..].as_ptr() as usize;
        jobs[1].src = jobs[0].dst;
        assert_eq!(run_fills(&jobs, 2), 1);
        assert_eq!(output[1], 167);
    }

    #[test]
    fn parallel_preludes_gate_requires_four_unique_lists() {
        assert!(!should_record_preludes_parallel(false, 8, true));
        assert!(!should_record_preludes_parallel(true, 3, true));
        assert!(!should_record_preludes_parallel(true, 8, false));
        assert!(should_record_preludes_parallel(true, 4, true));
        assert!(should_record_preludes_parallel(true, 16, true));
    }
}
