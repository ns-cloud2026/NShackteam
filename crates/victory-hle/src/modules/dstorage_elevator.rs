//! Mechanical HDD SCAN/C-SCAN elevator scheduler, windowed same-file read-ahead
//! coalescer, and per-physical-drive I/O concurrency gate for DirectStorage
//! (`VICTORY_DSTORAGE_ELEVATOR=1`).
//!
//! Profiling live gameplay on dual 7200 RPM HDDs (`D:` + `E:` SplitMap) showed
//! 16 concurrent DirectStorage workers issuing random FIFO seeks across 3,400+
//! queued `.ttpf`/`.rpf` requests, pinning `D:` at 99% busy while dropping
//! throughput to 1.5–6.8 MB/s and coalescing only ~2 requests per read.
//!
//! This module transforms random multi-thread HDD thrashing into smooth
//! sequential sweeps without violating DirectStorage's status-order contract:
//! - Resolves each opened file's real physical drive letter through symlinks
//!   (`GetFinalPathNameByHandleW`) so `D:` and `E:` are scheduled independently.
//! - Caps concurrent in-flight `ReadFile` calls per physical drive (default 3
//!   via `VICTORY_DSTORAGE_DRIVE_DEPTH`) while all 16 CPU workers decompress
//!   BCPack and zlib streams in parallel.
//! - Uses C-SCAN elevator selection within a bounded lookahead window (with
//!   strict anti-starvation on the queue head) and prefers idle drives so `E:`
//!   never sits waiting behind a busy `D:`.
//! - Coalesces scattered same-file requests anywhere in the lookahead window
//!   into single 2–4 MiB sequential reads.

use std::collections::HashMap;
use std::os::windows::io::RawHandle;
use std::sync::{Condvar, Mutex, OnceLock};

use windows_sys::Win32::Storage::FileSystem::{FILE_NAME_NORMALIZED, GetFinalPathNameByHandleW};

/// Default lookahead window inside a queue's pending reads for elevator
/// selection and windowed same-file coalescing (`VICTORY_DSTORAGE_ELEVATOR_WINDOW`).
pub(crate) const ELEVATOR_WINDOW: usize = 256;

/// Default maximum times the queue head may be bypassed before it is forced
/// next (`VICTORY_DSTORAGE_MAX_SKIPS`).
pub(crate) const MAX_FRONT_SKIPS: u32 = 16;

/// Configured elevator lookahead window (`VICTORY_DSTORAGE_ELEVATOR_WINDOW`,
/// default 256, clamped to `64..=4096`).
pub(crate) fn elevator_window() -> usize {
    static WINDOW: OnceLock<usize> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        std::env::var("VICTORY_DSTORAGE_ELEVATOR_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(ELEVATOR_WINDOW, |n| n.clamp(64, 4096))
    })
}

/// Configured anti-starvation head skip ceiling (`VICTORY_DSTORAGE_MAX_SKIPS`,
/// default 16, clamped to `4..=256`).
pub(crate) fn max_front_skips() -> u32 {
    static SKIPS: OnceLock<u32> = OnceLock::new();
    *SKIPS.get_or_init(|| {
        std::env::var("VICTORY_DSTORAGE_MAX_SKIPS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map_or(MAX_FRONT_SKIPS, |n| n.clamp(4, 256))
    })
}

/// Whether the HDD elevator and coalescer sweep across ALL queues in the same
/// priority class rather than bouncing round-robin between queues on the same
/// spinning platter (`VICTORY_DSTORAGE_CROSS_LANE=1`; off by default for A/B).
pub(crate) fn cross_lane_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_DSTORAGE_CROSS_LANE").is_some_and(|v| v == "1")
    })
}

/// Whether the HDD elevator scheduler is enabled (`VICTORY_DSTORAGE_ELEVATOR=1`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_DSTORAGE_ELEVATOR").is_some_and(|v| v == "1")
    })
}

/// Maximum concurrent disk `ReadFile` operations per physical drive letter
/// (`VICTORY_DSTORAGE_DRIVE_DEPTH`, default 3 when elevator is enabled; 0 = unlimited).
pub(crate) fn drive_depth() -> usize {
    static DEPTH: OnceLock<usize> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("VICTORY_DSTORAGE_DRIVE_DEPTH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(3, |n| n.min(16))
    })
}

/// Resolves the physical drive letter (`b'A'..=b'Z'`) behind an open file
/// handle, following NTFS symlinks/junctions (e.g. SplitMap links pointing to
/// `D:` vs `E:`). Returns `0` if no drive letter can be determined.
pub(crate) fn resolve_drive_id(handle: RawHandle) -> u8 {
    let mut buf = [0u16; 512];
    // SAFETY: `handle` is a valid open Windows file handle; `buf` holds 512 UTF-16 units.
    let len = unsafe {
        GetFinalPathNameByHandleW(handle, buf.as_mut_ptr(), buf.len() as u32, FILE_NAME_NORMALIZED)
    } as usize;
    if len == 0 || len >= buf.len() {
        return 0;
    }
    drive_letter_from_final_path(&buf[..len])
}

/// Extracts an uppercase ASCII drive letter from a `GetFinalPathNameByHandleW`
/// path (`\\?\D:\...`, `\\?\UNC\...`, or `D:\...`).
fn drive_letter_from_final_path(wide: &[u16]) -> u8 {
    // Strip leading `\\?\` if present.
    let rest = if wide.len() >= 4
        && wide[0] == b'\\' as u16
        && wide[1] == b'\\' as u16
        && wide[2] == b'?' as u16
        && wide[3] == b'\\' as u16
    {
        &wide[4..]
    } else {
        wide
    };
    if rest.len() >= 2 && rest[1] == b':' as u16 && rest[0] <= 0x7F {
        let ch = (rest[0] as u8).to_ascii_uppercase();
        if ch.is_ascii_alphabetic() {
            return ch;
        }
    }
    0
}

/// Per-physical-drive I/O concurrency gate. Permits are acquired strictly around
/// disk `ReadFile` calls and dropped immediately before CPU decompression so all
/// workers stay busy decoding while each spindle executes clean sequential sweeps.
pub(crate) struct DriveGate {
    limit: usize,
    active: Mutex<HashMap<u8, usize>>,
    freed: Condvar,
}

pub(crate) struct DrivePermit<'a> {
    gate: Option<&'a DriveGate>,
    drive: u8,
}

impl Drop for DrivePermit<'_> {
    fn drop(&mut self) {
        if let Some(gate) = self.gate {
            let mut active = gate.active.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(n) = active.get_mut(&self.drive) {
                *n = n.saturating_sub(1);
            }
            drop(active);
            gate.freed.notify_all();
        }
    }
}

impl DriveGate {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            active: Mutex::new(HashMap::new()),
            freed: Condvar::new(),
        }
    }

    /// Global drive gate configured from `VICTORY_DSTORAGE_DRIVE_DEPTH`.
    pub(crate) fn global() -> &'static Self {
        static GATE: OnceLock<DriveGate> = OnceLock::new();
        GATE.get_or_init(|| Self::new(drive_depth()))
    }

    /// Non-blocking check whether `drive` currently has an open I/O slot.
    pub(crate) fn has_room(&self, drive: u8) -> bool {
        if self.limit == 0 || drive == 0 {
            return true;
        }
        let active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        active.get(&drive).copied().unwrap_or(0) < self.limit
    }

    /// Blocks until `drive` has fewer than `limit` active disk reads, then
    /// returns an RAII permit that releases the slot on drop.
    pub(crate) fn acquire(&self, drive: u8) -> DrivePermit<'_> {
        if self.limit == 0 || drive == 0 {
            return DrivePermit { gate: None, drive };
        }
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let count = active.entry(drive).or_insert(0);
            if *count < self.limit {
                *count += 1;
                return DrivePermit {
                    gate: Some(self),
                    drive,
                };
            }
            active = self.freed.wait(active).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Candidate pending file read inside a queue's elevator lookahead window.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate {
    /// Owning lane id (`0` in single-lane mode).
    pub(crate) lane_id: u64,
    /// Index in `lane.reads`.
    pub(crate) index: usize,
    /// Opaque file identifier (`request.source as usize`).
    pub(crate) file_key: usize,
    /// Physical drive letter (`b'D'`, `b'E'`, etc., or `0` if memory/unknown).
    pub(crate) drive: u8,
    /// File byte offset.
    pub(crate) offset: u64,
    /// End byte offset (`offset + source_size`).
    pub(crate) end: u64,
    /// Staging bytes needed for this request.
    pub(crate) staging: u64,
}

/// Picks the best next candidate `(lane_id, index)` using a C-SCAN elevator
/// with idle-drive preference and strict head anti-starvation.
pub(crate) fn pick_next_candidate(
    candidates: &[Candidate],
    head_pos: &HashMap<usize, u64>,
    drive_free: impl Fn(u8) -> bool,
    front_skips: u32,
    max_skips: u32,
) -> Option<(u64, usize)> {
    if candidates.is_empty() {
        return None;
    }
    if front_skips >= max_skips || candidates.len() == 1 {
        return Some((candidates[0].lane_id, candidates[0].index));
    }

    // Score tuple (lower is better):
    //   0: 0 if physical drive has free I/O concurrency room, 1 if saturated
    //   1: 0 if forward sweep (offset >= head_pos[file]), 1 if wrap-around
    //   2: seek distance (forward: offset - head_pos; wrap: offset from start)
    //   3: original queue index (tie-breaker preserves FIFO)
    candidates
        .iter()
        .min_by_key(|c| {
            let busy = u8::from(!drive_free(c.drive));
            let head = head_pos.get(&c.file_key).copied().unwrap_or(0);
            let (wrap, dist) = match c.offset.checked_sub(head) {
                Some(fwd) => (0u8, fwd),
                None => (1u8, c.offset),
            };
            (busy, wrap, dist, c.index, c.lane_id)
        })
        .map(|c| (c.lane_id, c.index))
}

/// Picks the best next candidate index (into `lane.reads`) using a C-SCAN
/// elevator with idle-drive preference and strict head anti-starvation.
#[cfg(test)]
pub(crate) fn pick_next_index(
    candidates: &[Candidate],
    head_pos: &HashMap<usize, u64>,
    drive_free: impl Fn(u8) -> bool,
    front_skips: u32,
    max_skips: u32,
) -> Option<usize> {
    pick_next_candidate(candidates, head_pos, drive_free, front_skips, max_skips)
        .map(|(_, idx)| idx)
}

/// Picks up to `max_requests - 1` additional same-file candidates anywhere in
/// the lookahead window (single-lane or cross-lane) that can be coalesced with
/// `first` into ONE sequential disk read `[first.offset .. last.end]`,
/// returning `(lane_id, read_index)` pairs ordered by ascending file offset.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pick_coalesce_candidates(
    first_file: usize,
    first_start: u64,
    first_end: u64,
    candidates: &[Candidate],
    max_bytes: u64,
    max_requests: usize,
    max_gap: u64,
    staged: u64,
    staging_limit: u64,
) -> Vec<(u64, usize)> {
    if max_requests <= 1 || candidates.is_empty() {
        return Vec::new();
    }
    let span_limit = first_start.saturating_add(max_bytes);
    let mut matching: Vec<Candidate> = candidates
        .iter()
        .copied()
        .filter(|c| {
            c.file_key == first_file
                && c.offset >= first_end
                && c.end <= span_limit
                && c.end > c.offset
        })
        .collect();
    if matching.is_empty() {
        return Vec::new();
    }
    matching.sort_unstable_by_key(|c| (c.offset, c.lane_id, c.index));

    let max_total_gap = (max_bytes / 2).max(max_gap);
    let mut picked = Vec::new();
    let mut end = first_end;
    let mut gap_spent = 0u64;
    let mut staged_now = staged;

    for c in matching {
        if picked.len() + 1 >= max_requests {
            break;
        }
        // Skip overlapping/duplicate offset reads so each batch entry lies strictly forward.
        let Some(gap) = c.offset.checked_sub(end) else {
            continue;
        };
        if gap > max_gap || gap_spent.saturating_add(gap) > max_total_gap {
            // Because `matching` is sorted by offset, no later candidate can
            // have a smaller gap from `end`.
            break;
        }
        if staged_now.saturating_add(c.staging) > staging_limit {
            continue;
        }
        picked.push((c.lane_id, c.index));
        end = c.end;
        gap_spent += gap;
        staged_now += c.staging;
    }
    picked
}

/// Single-lane wrapper for [`pick_coalesce_candidates`] used by unit tests.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pick_coalesce_indices(
    first_file: usize,
    first_start: u64,
    first_end: u64,
    candidates: &[Candidate],
    max_bytes: u64,
    max_requests: usize,
    max_gap: u64,
    staged: u64,
    staging_limit: u64,
) -> Vec<usize> {
    pick_coalesce_candidates(
        first_file,
        first_start,
        first_end,
        candidates,
        max_bytes,
        max_requests,
        max_gap,
        staged,
        staging_limit,
    )
    .into_iter()
    .map(|(_, idx)| idx)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(index: usize, file_key: usize, drive: u8, offset: u64, len: u64) -> Candidate {
        Candidate {
            lane_id: 0,
            index,
            file_key,
            drive,
            offset,
            end: offset + len,
            staging: len,
        }
    }

    fn lane_cand(lane_id: u64, index: usize, file_key: usize, drive: u8, offset: u64, len: u64) -> Candidate {
        Candidate {
            lane_id,
            index,
            file_key,
            drive,
            offset,
            end: offset + len,
            staging: len,
        }
    }

    #[test]
    fn final_path_extracts_physical_drive_letter() {
        let to_u16 = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        assert_eq!(drive_letter_from_final_path(&to_u16(r"\\?\E:\VIctory\maps.ttpf")), b'E');
        assert_eq!(drive_letter_from_final_path(&to_u16(r"\\?\d:\game\terrain.ttpf")), b'D');
        assert_eq!(drive_letter_from_final_path(&to_u16(r"C:\test.bin")), b'C');
        assert_eq!(drive_letter_from_final_path(&to_u16(r"\\?\UNC\srv\share\a.rpf")), 0);
    }

    #[test]
    fn elevator_prefers_idle_drive_and_closest_forward_offset() {
        let mut heads = HashMap::new();
        heads.insert(1, 1000);
        heads.insert(2, 5000);
        let cands = vec![
            cand(0, 1, b'D', 9000, 100),
            cand(1, 1, b'D', 1100, 100), // closest forward on D
            cand(2, 2, b'E', 5200, 100), // forward on E
        ];
        // When both drives are free, closest forward seek (100 bytes on D) wins.
        assert_eq!(pick_next_index(&cands, &heads, |_| true, 0, 16), Some(1));
        // When D is saturated and E is idle, E wins immediately!
        assert_eq!(pick_next_index(&cands, &heads, |d| d == b'E', 0, 16), Some(2));
        // Anti-starvation forces queue head (index 0) once `front_skips >= max_skips`.
        assert_eq!(pick_next_index(&cands, &heads, |d| d == b'E', 16, 16), Some(0));
    }

    #[test]
    fn coalescer_gathers_out_of_order_same_file_requests_across_interleaved_files() {
        // Queue has interleaved files (file 1 and file 2) and out-of-order offsets for file 1.
        let cands = vec![
            cand(0, 2, b'E', 1000, 4096),       // different file
            cand(1, 1, b'D', 200_000, 4096),    // file 1, later offset
            cand(2, 2, b'E', 5000, 4096),       // different file
            cand(3, 1, b'D', 120_000, 4096),    // file 1, closer forward offset!
            cand(4, 1, b'D', 50_000, 4096),     // file 1, behind first (ignored)
            cand(5, 1, b'D', 2_000_000, 4096),  // file 1, gap too large (ignored)
        ];
        let picked = pick_coalesce_indices(
            1,
            100_000,
            104_096,
            &cands,
            4 << 20,
            16,
            128 << 10,
            4096,
            16 << 20,
        );
        // Must pick index 3 (120k) then index 1 (200k) in sorted offset order!
        assert_eq!(picked, vec![3, 1]);
    }

    #[test]
    fn drive_gate_limits_concurrency_per_drive_independently() {
        let gate = DriveGate::new(2);
        assert!(gate.has_room(b'D'));
        assert!(gate.has_room(b'E'));
        let p1 = gate.acquire(b'D');
        let p2 = gate.acquire(b'D');
        assert!(!gate.has_room(b'D'));
        assert!(gate.has_room(b'E'));
        let _pe = gate.acquire(b'E');
        assert!(gate.has_room(b'E'));
        drop(p1);
        assert!(gate.has_room(b'D'));
        drop(p2);
    }

    #[test]
    fn cross_lane_elevator_and_coalescer_sweep_same_file_across_queues() {
        let mut heads = HashMap::new();
        heads.insert(7, 500_000);
        let cands = vec![
            lane_cand(1, 0, 7, b'D', 900_000, 4096),
            lane_cand(2, 0, 7, b'D', 510_000, 4096), // closest forward across lanes
            lane_cand(3, 1, 7, b'D', 550_000, 4096),
            lane_cand(1, 1, 7, b'D', 520_000, 4096),
        ];
        assert_eq!(pick_next_candidate(&cands, &heads, |_| true, 0, 64), Some((2, 0)));
        let batch = pick_coalesce_candidates(
            7,
            510_000,
            514_096,
            &cands,
            8 << 20,
            64,
            1 << 20,
            4096,
            128 << 20,
        );
        // Must gather lane 1 idx 1 (520k), lane 3 idx 1 (550k), lane 1 idx 0 (900k) in offset order.
        assert_eq!(batch, vec![(1, 1), (3, 1), (1, 0)]);
    }
}
