//! Xbox unified-memory GPU-to-GPU alias synchronization (`VICTORY_GPU_ALIAS=0` disables).
//!
//! On Xbox Series X, GPU and CPU share a single unified address space, so two
//! overlapping views (e.g., the 174 MiB main heap buffer and a small `Shadow::exact`
//! copy created for a non-element-aligned typed SRV) refer to the exact same bytes.
//!
//! On PC D3D12, those views live in separate `Shadow` host buffers. Previously, when
//! an in-flight submission wrote `[a, b)` of `Shadow A` via shaders and the next
//! submission refreshed an overlapping range of `Shadow B`, the host had to stall
//! the entire CPU+GPU pipeline (`WHY_READ` / "another copy", ~70–105 ms/frame) until
//! `Shadow A` finished on the GPU and the relay copied it back to title memory.
//!
//! This module plans GPU-side `CopyBufferRegion` transfers on the main command queue
//! right before the next submission runs, eliminating synchronous pipeline stalls:
//! - `VICTORY_GPU_ALIAS=1` (default): aliases into pure non-carded / unhashed copies.
//! - `VICTORY_NV_ALIAS_INDEX=1`: interval-indexed overlap lookup preserving exact
//!   submission order (avoids O(R * F) scans on large submission batches).
//! - `VICTORY_NV_ALIAS_V2=1`: also aliases into carded destination copies that have
//!   GPU piece-hash buffers (`card_has_before`), scheduling an immediate GPU rehash
//!   of only the aliased sub-ranges so card hashes stay exact without stalling.
//! - `VICTORY_NV_GPU_BUFFER_COPY=1`: promotes `Step::Copy` (`CopyBufferRegion`)
//!   transfers reading from an in-flight GPU writer into a GPU-side copy plus async
//!   relay writeback so title RAM stays coherent without stalling the submit thread.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_UAV_BARRIER, ID3D12CommandAllocator,
    ID3D12CommandList, ID3D12CommandQueue, ID3D12GraphicsCommandList,
};
use windows::core::Interface;

use crate::shadow::Shadow;

pub(crate) type AliasList = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Whether GPU-to-GPU alias synchronization is enabled (on by default;
/// `VICTORY_GPU_ALIAS=0` disables it).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_ALIAS").is_none_or(|v| v != "0"))
}

/// Whether carded-destination GPU aliasing with post-copy GPU rehash is enabled
/// (`VICTORY_NV_ALIAS_V2=1`; off by default for A/B testing).
pub(crate) fn v2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NV_ALIAS_V2").is_some_and(|v| v == "1"))
}

/// Whether interval-indexed overlap planning is enabled (`VICTORY_NV_ALIAS_INDEX=1`,
/// or automatically enabled with `VICTORY_NV_ALIAS_V2=1` unless explicitly set to `0`).
pub(crate) fn index_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_NV_ALIAS_INDEX").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => v2_enabled(),
    })
}

/// Whether in-flight-conflicting `Step::Copy` buffer transfers are promoted to GPU
/// `CopyBufferRegion` + async relay writeback (`VICTORY_NV_GPU_BUFFER_COPY=1`; off
/// by default for A/B testing).
pub(crate) fn gpu_buffer_copy_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NV_GPU_BUFFER_COPY").is_some_and(|v| v == "1"))
}

/// Statistics since the last frame line: `[0]` submissions unblocked,
/// `[1]` GPU alias copies recorded, `[2]` total bytes copied on GPU,
/// `[3]` carded destination rehash ranges, `[4]` GPU buffer copies promoted.
static STATS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Takes `[submissions_unblocked, copies_recorded, bytes_copied, rehashes, gpu_buf_copies]`
/// since the last call.
pub(crate) fn take_stats() -> [u64; 5] {
    [
        STATS[0].swap(0, Ordering::Relaxed),
        STATS[1].swap(0, Ordering::Relaxed),
        STATS[2].swap(0, Ordering::Relaxed),
        STATS[3].swap(0, Ordering::Relaxed),
        STATS[4].swap(0, Ordering::Relaxed),
    ]
}

/// One GPU-side `CopyBufferRegion` from `src` at title address `src_addr`
/// into `dst` at title address `dst_addr` for `len` bytes.
#[derive(Clone)]
pub(crate) struct AliasCopy {
    pub(crate) src: Arc<Shadow>,
    pub(crate) dst: Arc<Shadow>,
    pub(crate) src_addr: usize,
    pub(crate) dst_addr: usize,
    pub(crate) len: usize,
}

/// A candidate `Step::Copy(dst, src, len)` that reads from an in-flight GPU writer
/// and can be promoted to a GPU copy + async relay writeback.
pub(crate) struct BufferCopyCandidate {
    pub(crate) list_idx: usize,
    pub(crate) step_idx: usize,
    pub(crate) dst_shadow: Arc<Shadow>,
    pub(crate) dst_addr: usize,
    pub(crate) src_addr: usize,
    pub(crate) len: usize,
}

/// Planned GPU alias transfers for one submission, plus per-destination title
/// ranges that the CPU refresh must skip so it does not race with the GPU copy.
pub(crate) struct AliasPlan {
    pub(crate) copies: Vec<AliasCopy>,
    /// Keyed by `Arc::as_ptr(&dst) as usize`: sorted, joined `[from, to)`
    /// title ranges handled on the GPU for that destination `Shadow`.
    pub(crate) skip_cpu: HashMap<usize, Vec<(usize, usize)>>,
    /// `(list_idx, step_idx)` of `Step::Copy` steps promoted to GPU copies
    /// (skipped on CPU during `HostList::run_deferred`).
    pub(crate) skip_step_copies: HashSet<(usize, usize)>,
    /// Non-carded (`shared_buffer`) destination ranges `(dst_ptr, from, to)`
    /// that will be written on the GPU by this flight's alias list.
    pub(crate) busy_dst: Vec<(usize, usize, usize)>,
    /// Carded destination ranges `(dst, from_title, to_title)` whose GPU piece
    /// hashes must be updated right after the alias copy (`VICTORY_NV_ALIAS_V2=1`).
    pub(crate) rehash_dst: Vec<(Arc<Shadow>, usize, usize)>,
    /// Destination ranges `(dst, from_title, to_title)` from promoted GPU
    /// `Step::Copy` operations that must be written back to title memory
    /// asynchronously by the relay when this flight completes.
    pub(crate) writeback_dst: Vec<(Arc<Shadow>, usize, usize)>,
}

impl AliasPlan {
    pub(crate) fn is_empty(&self) -> bool {
        self.copies.is_empty()
    }

    pub(crate) fn skipped_for(&self, shadow: &Arc<Shadow>) -> Option<&[(usize, usize)]> {
        let key = Arc::as_ptr(shadow) as usize;
        self.skip_cpu.get(&key).map(Vec::as_slice)
    }
}

/// Recycled D3D12 command lists for alias-sync preludes.
static POOL: Mutex<Vec<AliasList>> = Mutex::new(Vec::new());

/// Acquires a recycled or newly created alias command list BEFORE `Host::submit`
/// commits `batch.alias_skip`, so list-creation failures cleanly fall back to
/// normal CPU settle instead of skipping CPU refreshes without a GPU copy.
pub(crate) fn acquire_list(make_list: impl FnOnce() -> Option<AliasList>) -> Option<AliasList> {
    {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(list) = pool.pop() {
            return Some(list);
        }
    }
    make_list()
}

/// Returns an alias prelude list to the pool once its flight has finished.
pub(crate) fn recycle(list: AliasList) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if pool.len() < 32 {
        pool.push(list);
    }
}

/// Pure range-intersection planning core (testable without D3D12).
///
/// Given a list of refresh requests `(dst_id, from, to)` and in-flight writes
/// `(src_id, a, b)` in submission order (`src_id == 0` represents readbacks),
/// plus `immed` bottom-of-pipe writes and `ineligible_dsts`, returns
/// `Some(transfers)` where each transfer is `(src_id, dst_id, lo, hi)` in exact
/// submission order, or `None` if any conflict cannot be safely resolved.
pub(crate) fn plan_transfers(
    refreshes: &[(usize, usize, usize)],
    in_flight: &[(usize, usize, usize)],
    immed: &[(usize, usize)],
    ineligible_dsts: &HashSet<usize>,
    src_covers_and_clean: impl Fn(usize, usize, usize) -> bool,
) -> Option<Vec<(usize, usize, usize, usize)>> {
    if index_enabled() {
        plan_transfers_indexed(refreshes, in_flight, immed, ineligible_dsts, src_covers_and_clean)
    } else {
        plan_transfers_linear(refreshes, in_flight, immed, ineligible_dsts, src_covers_and_clean)
    }
}

/// Baseline O(R * F) range-intersection planner.
fn plan_transfers_linear(
    refreshes: &[(usize, usize, usize)],
    in_flight: &[(usize, usize, usize)],
    immed: &[(usize, usize)],
    ineligible_dsts: &HashSet<usize>,
    src_covers_and_clean: impl Fn(usize, usize, usize) -> bool,
) -> Option<Vec<(usize, usize, usize, usize)>> {
    let mut transfers: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut srcs_used: HashSet<usize> = HashSet::new();
    let mut dsts_used: HashSet<usize> = HashSet::new();

    for &(dst, from, to) in refreshes {
        if from >= to {
            continue;
        }
        // Bottom-of-pipe CPU writes always require a real settle.
        if immed.iter().any(|&(a, b)| a < to && b > from) {
            return None;
        }
        for &(src, a, b) in in_flight {
            if a >= to || b <= from || src == dst {
                continue;
            }
            // Readbacks (`src == 0`) have no GPU source buffer.
            if src == 0 {
                return None;
            }
            if ineligible_dsts.contains(&dst) {
                return None;
            }
            let lo = from.max(a);
            let hi = to.min(b);
            if !src_covers_and_clean(src, lo, hi) {
                return None;
            }
            srcs_used.insert(src);
            dsts_used.insert(dst);
            transfers.push((src, dst, lo, hi));
        }
    }

    finalize_transfers(transfers, &srcs_used, &dsts_used)
}

/// Interval-indexed range-intersection planner (`VICTORY_NV_ALIAS_INDEX=1`).
/// Uses a sorted start + prefix-max-end array over `in_flight` and `immed`,
/// re-sorting matched overlaps by original submission index `seq` so output
/// is bit-for-bit identical to [`plan_transfers_linear`].
fn plan_transfers_indexed(
    refreshes: &[(usize, usize, usize)],
    in_flight: &[(usize, usize, usize)],
    immed: &[(usize, usize)],
    ineligible_dsts: &HashSet<usize>,
    src_covers_and_clean: impl Fn(usize, usize, usize) -> bool,
) -> Option<Vec<(usize, usize, usize, usize)>> {
    // Sorted prefix-max index over `immed`.
    let mut immed_idx: Vec<(usize, usize, usize)> = immed
        .iter()
        .copied()
        .filter(|&(a, b)| a < b)
        .map(|(a, b)| (a, b, 0))
        .collect();
    immed_idx.sort_unstable_by_key(|&(a, b, _)| (a, b));
    let mut max_end = 0usize;
    for entry in &mut immed_idx {
        max_end = max_end.max(entry.1);
        entry.2 = max_end;
    }

    // Sorted prefix-max index over `in_flight`, preserving submission `seq`.
    let mut flight_idx: Vec<(usize, usize, usize, usize, usize)> = in_flight
        .iter()
        .enumerate()
        .filter(|&(_, &(_, a, b))| a < b)
        .map(|(seq, &(src, a, b))| (a, b, src, seq, 0))
        .collect();
    flight_idx.sort_unstable_by_key(|&(a, b, _, seq, _)| (a, b, seq));
    let mut fmax = 0usize;
    for entry in &mut flight_idx {
        fmax = fmax.max(entry.1);
        entry.4 = fmax;
    }

    let mut transfers: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut srcs_used: HashSet<usize> = HashSet::new();
    let mut dsts_used: HashSet<usize> = HashSet::new();
    let mut matched: Vec<(usize, usize, usize, usize)> = Vec::new();

    for &(dst, from, to) in refreshes {
        if from >= to {
            continue;
        }
        // Check `immed` overlap via prefix-max binary search.
        let im_hi = immed_idx.partition_point(|&(a, _, _)| a < to);
        if im_hi > 0 && immed_idx[im_hi - 1].2 > from {
            let mut k = im_hi;
            while k > 0 && immed_idx[k - 1].2 > from {
                k -= 1;
                if immed_idx[k].1 > from {
                    return None;
                }
            }
        }
        // Gather all `in_flight` intervals overlapping `[from, to)`.
        matched.clear();
        let hi = flight_idx.partition_point(|&(a, _, _, _, _)| a < to);
        let mut k = hi;
        while k > 0 && flight_idx[k - 1].4 > from {
            k -= 1;
            let (a, b, src, seq, _) = flight_idx[k];
            if b > from && src != dst {
                matched.push((seq, src, a, b));
            }
        }
        if matched.is_empty() {
            continue;
        }
        // Restore exact submission order before validating and pushing.
        matched.sort_unstable_by_key(|&(seq, _, _, _)| seq);
        for &(_, src, a, b) in &matched {
            if src == 0 || ineligible_dsts.contains(&dst) {
                return None;
            }
            let lo = from.max(a);
            let hi = to.min(b);
            if !src_covers_and_clean(src, lo, hi) {
                return None;
            }
            srcs_used.insert(src);
            dsts_used.insert(dst);
            transfers.push((src, dst, lo, hi));
        }
    }

    finalize_transfers(transfers, &srcs_used, &dsts_used)
}

fn finalize_transfers(
    transfers: Vec<(usize, usize, usize, usize)>,
    srcs_used: &HashSet<usize>,
    dsts_used: &HashSet<usize>,
) -> Option<Vec<(usize, usize, usize, usize)>> {
    // Never allow the same buffer to act as both COPY_SOURCE and COPY_DEST in
    // one alias command list (keeps implicit COMMON promotion 100% legal).
    if srcs_used.iter().any(|s| dsts_used.contains(s)) {
        return None;
    }

    // Coalesce consecutive identical/adjacent transfers for the same (src, dst)
    // while strictly preserving cross-source submission order.
    let mut coalesced: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(transfers.len());
    for (src, dst, lo, hi) in transfers {
        match coalesced.last_mut() {
            Some(last) if last.0 == src && last.1 == dst && lo <= last.3 && hi >= last.2 => {
                last.2 = last.2.min(lo);
                last.3 = last.3.max(hi);
            }
            _ => coalesced.push((src, dst, lo, hi)),
        }
    }
    Some(coalesced)
}

/// Pure planner for promoting `Step::Copy(dst_id, dst_addr, src_addr, len)`
/// buffer transfers against `in_flight` writers. Each candidate is resolved only
/// when a single non-zero `src_id` accounts for every in-flight overlap on
/// `[src_addr, src_addr + len)`, covers the full source range cleanly, and
/// introduces no `src`/`dst` hazard cycle with `existing_srcs`/`existing_dsts`.
pub(crate) fn plan_buffer_copies(
    candidates: &[(usize, usize, usize, usize)], // (dst_id, dst_addr, src_addr, len)
    in_flight: &[(usize, usize, usize)],
    immed: &[(usize, usize)],
    ineligible_dsts: &HashSet<usize>,
    existing_srcs: &HashSet<usize>,
    existing_dsts: &HashSet<usize>,
    src_covers_and_clean: impl Fn(usize, usize, usize) -> bool,
) -> Option<Vec<(usize, usize, usize, usize, usize)>> {
    let mut out: Vec<(usize, usize, usize, usize, usize)> = Vec::with_capacity(candidates.len());
    let mut srcs_used = existing_srcs.clone();
    let mut dsts_used = existing_dsts.clone();

    for &(dst_id, dst_addr, src_addr, len) in candidates {
        if len == 0 {
            continue;
        }
        let src_end = src_addr.checked_add(len)?;
        let dst_end = dst_addr.checked_add(len)?;
        if ineligible_dsts.contains(&dst_id) {
            return None;
        }
        if immed
            .iter()
            .any(|&(a, b)| (a < src_end && b > src_addr) || (a < dst_end && b > dst_addr))
        {
            return None;
        }
        // Reject if any in-flight flight writes to `dst_addr..dst_end` directly.
        if in_flight.iter().any(|&(_, a, b)| a < dst_end && b > dst_addr) {
            return None;
        }
        // Every in-flight write overlapping `src_addr..src_end` must belong to
        // one non-zero `src_id` that cleanly covers the full source span.
        let mut writer: Option<usize> = None;
        for &(src_id, a, b) in in_flight {
            if a >= src_end || b <= src_addr {
                continue;
            }
            if src_id == 0 || src_id == dst_id {
                return None;
            }
            match writer {
                None => writer = Some(src_id),
                Some(prev) if prev != src_id => return None,
                Some(_) => {}
            }
        }
        let src_id = writer?;
        if !src_covers_and_clean(src_id, src_addr, src_end) {
            return None;
        }
        srcs_used.insert(src_id);
        dsts_used.insert(dst_id);
        out.push((src_id, dst_id, src_addr, dst_addr, len));
    }

    if srcs_used.iter().any(|s| dsts_used.contains(s)) {
        return None;
    }
    Some(out)
}

/// Rejection reason counters for A/B diagnostics:
/// `[0]` dst ineligible (placed/pending/declared), `[1]` src ineligible/dirty/unmapped,
/// `[2]` immed write overlap, `[3]` plan_transfers failed (multi-writer/cycle/dst),
/// `[4]` buffer_copy promotion rejected.
static REJECT_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

pub(crate) fn note_reject(bucket: usize) {
    if let Some(c) = REJECT_STATS.get(bucket) {
        c.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn take_rejects() -> [u64; 5] {
    [
        REJECT_STATS[0].swap(0, Ordering::Relaxed),
        REJECT_STATS[1].swap(0, Ordering::Relaxed),
        REJECT_STATS[2].swap(0, Ordering::Relaxed),
        REJECT_STATS[3].swap(0, Ordering::Relaxed),
        REJECT_STATS[4].swap(0, Ordering::Relaxed),
    ]
}

/// Builds a concrete [`AliasPlan`] for `refreshes` and promoted `buffer_copies`
/// against `in_flight` writers (in submission order) and `immed` writes.
///
/// Returns `None` if any conflict cannot be resolved purely on the GPU
/// (caller falls back to normal settle / `relay_wait`).
pub(crate) fn build_plan(
    refreshes: &[(Arc<Shadow>, usize, usize)],
    buffer_copies: &[BufferCopyCandidate],
    in_flight: &[(Option<Arc<Shadow>>, usize, usize)],
    immed: &[(usize, usize)],
    declared_writes: &HashSet<usize>,
) -> Option<AliasPlan> {
    if !enabled() || (refreshes.is_empty() && buffer_copies.is_empty()) || in_flight.is_empty() {
        return None;
    }

    let allow_carded_rehash = v2_enabled();
    let mut dst_map: HashMap<usize, Arc<Shadow>> = HashMap::new();
    let mut ineligible_dsts: HashSet<usize> = declared_writes.clone();
    let mut refresh_ids: Vec<(usize, usize, usize)> = Vec::with_capacity(refreshes.len());
    for (shadow, from, to) in refreshes {
        if from >= to {
            continue;
        }
        let id = Arc::as_ptr(shadow) as usize;
        dst_map.entry(id).or_insert_with(|| shadow.clone());
        // Destination must be a pure consumer with no active shader writes.
        // Without `VICTORY_NV_ALIAS_V2=1`, carded buffers with a hash buffer are
        // ineligible; with `v2` enabled, we update their piece hashes on the GPU
        // right after the alias `CopyBufferRegion`.
        if shadow.is_placed()
            || shadow.writes_pending()
            || (!allow_carded_rehash && shadow.card_has_before())
        {
            ineligible_dsts.insert(id);
        }
        refresh_ids.push((id, *from, *to));
    }

    let mut copy_ids: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(buffer_copies.len());
    for cand in buffer_copies {
        let id = Arc::as_ptr(&cand.dst_shadow) as usize;
        dst_map.entry(id).or_insert_with(|| cand.dst_shadow.clone());
        if cand.dst_shadow.is_placed() || cand.dst_shadow.writes_pending() {
            ineligible_dsts.insert(id);
        }
        copy_ids.push((id, cand.dst_addr, cand.src_addr, cand.len));
    }

    let mut src_map: HashMap<usize, Arc<Shadow>> = HashMap::new();
    let mut flight_ids: Vec<(usize, usize, usize)> = Vec::with_capacity(in_flight.len());
    for (shadow, a, b) in in_flight {
        match shadow {
            Some(s) => {
                let id = Arc::as_ptr(s) as usize;
                src_map.entry(id).or_insert_with(|| s.clone());
                flight_ids.push((id, *a, *b));
            }
            None => flight_ids.push((0, *a, *b)),
        }
    }

    let src_ok = |src_id: usize, lo: usize, hi: usize| -> bool {
        let Some(src) = src_map.get(&src_id) else {
            note_reject(1);
            return false;
        };
        if src.is_placed() {
            note_reject(1);
            return false;
        }
        let len = (hi - lo) as u64;
        let ok = src.offset_in(lo as u64).is_some()
            && src.left(lo as u64) >= len
            && src.cpu_clean_for_alias(lo, hi);
        if !ok {
            note_reject(1);
        }
        ok
    };

    let Some(transfers) = plan_transfers(&refresh_ids, &flight_ids, immed, &ineligible_dsts, src_ok) else {
        note_reject(3);
        return None;
    };

    let mut srcs_used: HashSet<usize> = HashSet::new();
    let mut dsts_used: HashSet<usize> = HashSet::new();
    for &(s, d, _, _) in &transfers {
        srcs_used.insert(s);
        dsts_used.insert(d);
    }

    let promoted = if !copy_ids.is_empty() {
        let Some(p) = plan_buffer_copies(
            &copy_ids,
            &flight_ids,
            immed,
            &ineligible_dsts,
            &srcs_used,
            &dsts_used,
            src_ok,
        ) else {
            note_reject(4);
            return None;
        };
        p
    } else {
        Vec::new()
    };

    if transfers.is_empty() && promoted.is_empty() {
        return None;
    }

    let mut copies = Vec::with_capacity(transfers.len() + promoted.len());
    let mut skip_cpu: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut skip_step_copies: HashSet<(usize, usize)> = HashSet::new();
    let mut busy_dst: Vec<(usize, usize, usize)> = Vec::new();
    let mut rehash_dst: Vec<(Arc<Shadow>, usize, usize)> = Vec::new();
    let mut writeback_dst: Vec<(Arc<Shadow>, usize, usize)> = Vec::new();

    for (src_id, dst_id, lo, hi) in transfers {
        let src = src_map.get(&src_id)?.clone();
        let dst = dst_map.get(&dst_id)?.clone();
        let len = (hi - lo) as u64;
        if dst.offset_in(lo as u64).is_none() || dst.left(lo as u64) < len {
            return None;
        }
        if !dst.carded() {
            busy_dst.push((dst_id, lo, hi));
        } else if dst.card_has_before() {
            rehash_dst.push((dst.clone(), lo, hi));
        }
        skip_cpu.entry(dst_id).or_default().push((lo, hi));
        copies.push(AliasCopy {
            src,
            dst,
            src_addr: lo,
            dst_addr: lo,
            len: hi - lo,
        });
    }

    for (cand, (src_id, dst_id, src_addr, dst_addr, len)) in buffer_copies.iter().zip(promoted) {
        let src = src_map.get(&src_id)?.clone();
        let dst = dst_map.get(&dst_id)?.clone();
        if dst.offset_in(dst_addr as u64).is_none() || dst.left(dst_addr as u64) < len as u64 {
            return None;
        }
        let dst_end = dst_addr + len;
        if !dst.carded() {
            busy_dst.push((dst_id, dst_addr, dst_end));
        }
        // Promoted `Step::Copy` writes are copied back asynchronously to title
        // RAM by the relay (so we intentionally do NOT rehash `dst` here — its
        // pre-copy hashes/snapshot will detect the new GPU-copied bytes).
        writeback_dst.push((dst.clone(), dst_addr, dst_end));
        skip_step_copies.insert((cand.list_idx, cand.step_idx));
        skip_cpu.entry(dst_id).or_default().push((dst_addr, dst_end));
        copies.push(AliasCopy {
            src,
            dst,
            src_addr,
            dst_addr,
            len,
        });
    }

    // Sort and coalesce per-destination CPU skip spans.
    for spans in skip_cpu.values_mut() {
        spans.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
        for &(a, b) in spans.iter() {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        *spans = merged;
    }

    Some(AliasPlan {
        copies,
        skip_cpu,
        skip_step_copies,
        busy_dst,
        rehash_dst,
        writeback_dst,
    })
}

/// Records and submits the GPU-side `CopyBufferRegion` transfers on `queue` as
/// a standalone `ExecuteCommandLists` call using an already-acquired `list_pair`,
/// so every touched buffer enters and exits in `D3D12_RESOURCE_STATE_COMMON` via
/// D3D12's implicit promotion/decay.
///
/// Returns the command list handle to recycle once this flight finishes.
pub(crate) fn record_and_submit(
    queue: &ID3D12CommandQueue,
    list_pair: AliasList,
    plan: &AliasPlan,
) -> Option<AliasList> {
    if plan.copies.is_empty() {
        recycle(list_pair);
        return None;
    }

    let (allocator, list) = &list_pair;
    let mut bytes_total = 0u64;
    // SAFETY: `allocator`/`list` are idle (recycled after fence completion or
    // freshly created); all `src`/`dst` buffers are live `Arc<Shadow>`s held
    // by `plan` and kept alive by the submission's `HostList` / `Flight`.
    let recorded = unsafe {
        (|| -> windows::core::Result<()> {
            allocator.Reset()?;
            list.Reset(allocator, None)?;
            // Ensure prior in-flight shader UAV writes on `self.queue` are
            // visible before the copy engine reads `src.buffer`.
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            }]);
            for copy in &plan.copies {
                let len = copy.len as u64;
                let src_off = copy.src.offset(copy.src_addr as u64);
                let dst_off = copy.dst.offset(copy.dst_addr as u64);
                list.CopyBufferRegion(&copy.dst.buffer, dst_off, &copy.src.buffer, src_off, len);
                bytes_total += len;
            }
            list.Close()?;
            let cmd: ID3D12CommandList = list.cast()?;
            queue.ExecuteCommandLists(&[Some(cmd)]);
            Ok(())
        })()
    };

    match recorded {
        Ok(()) => {
            STATS[0].fetch_add(1, Ordering::Relaxed);
            STATS[1].fetch_add(plan.copies.len() as u64, Ordering::Relaxed);
            STATS[2].fetch_add(bytes_total, Ordering::Relaxed);
            if !plan.rehash_dst.is_empty() {
                STATS[3].fetch_add(plan.rehash_dst.len() as u64, Ordering::Relaxed);
            }
            if !plan.skip_step_copies.is_empty() {
                STATS[4].fetch_add(plan.skip_step_copies.len() as u64, Ordering::Relaxed);
            }
            Some(list_pair)
        }
        Err(e) => {
            tracing::warn!(error = %e, "gpu_alias: CopyBufferRegion submission failed");
            recycle(list_pair);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{plan_buffer_copies, plan_transfers_indexed, plan_transfers_linear};
    use std::collections::HashSet;

    #[test]
    fn resolves_overlapping_other_copy_and_preserves_submission_order() {
        let (a, b, c) = (0x1000, 0x2000, 0x3000);
        // Flight 1 wrote A:[0..400], Flight 2 wrote C:[200..600].
        // Next submission refreshes B:[100..500].
        let in_flight = [(a, 0, 400), (c, 200, 600)];
        let refreshes = [(b, 100, 500)];
        let linear = plan_transfers_linear(&refreshes, &in_flight, &[], &HashSet::new(), |_, _, _| true)
            .expect("resolvable");
        let indexed = plan_transfers_indexed(&refreshes, &in_flight, &[], &HashSet::new(), |_, _, _| true)
            .expect("resolvable indexed");
        assert_eq!(linear, vec![(a, b, 100, 400), (c, b, 200, 500)]);
        assert_eq!(indexed, linear);
    }

    #[test]
    fn rejects_readbacks_immed_dirty_cpu_and_src_dst_cycles() {
        let (a, b) = (0x1000, 0x2000);
        for indexed in [false, true] {
            let run = |r: &[(usize, usize, usize)],
                       f: &[(usize, usize, usize)],
                       im: &[(usize, usize)],
                       inel: &HashSet<usize>,
                       ok: bool| {
                if indexed {
                    plan_transfers_indexed(r, f, im, inel, |_, _, _| ok)
                } else {
                    plan_transfers_linear(r, f, im, inel, |_, _, _| ok)
                }
            };
            // Readback (`src == 0`) must settle.
            assert!(run(&[(b, 0, 100)], &[(0, 0, 100)], &[], &HashSet::new(), true).is_none());
            // Bottom-of-pipe immediate write must settle.
            assert!(run(&[(b, 0, 100)], &[(a, 0, 100)], &[(50, 60)], &HashSet::new(), true).is_none());
            // Dirty CPU pages on src must settle.
            assert!(run(&[(b, 0, 100)], &[(a, 0, 100)], &[], &HashSet::new(), false).is_none());
            // Ineligible dst (e.g., shader writes dst in same submission) must settle.
            let mut ineligible = HashSet::new();
            ineligible.insert(b);
            assert!(run(&[(b, 0, 100)], &[(a, 0, 100)], &[], &ineligible, true).is_none());
            // Cycle where A is both src and dst in the same list must settle.
            assert!(
                run(
                    &[(b, 0, 100), (a, 200, 300)],
                    &[(a, 0, 100), (b, 200, 300)],
                    &[],
                    &HashSet::new(),
                    true
                )
                .is_none()
            );
        }
    }

    #[test]
    fn indexed_matches_linear_on_nested_multi_flight_overlaps() {
        let (s1, s2, s3, d1, d2) = (0x10, 0x20, 0x30, 0x40, 0x50);
        // Deliberately unsorted starts & huge enclosing intervals to stress prefix-max walk.
        let in_flight = [
            (s3, 500, 900),
            (s1, 0, 1000),
            (s2, 200, 300),
            (s1, 900, 1200),
            (s2, 250, 600),
        ];
        let refreshes = [
            (d1, 220, 550),
            (d2, 0, 150),
            (d1, 550, 1100),
        ];
        let immed = [(2000, 2100), (1500, 1600)];
        let linear = plan_transfers_linear(&refreshes, &in_flight, &immed, &HashSet::new(), |_, _, _| true);
        let indexed = plan_transfers_indexed(&refreshes, &in_flight, &immed, &HashSet::new(), |_, _, _| true);
        assert_eq!(indexed, linear);
        assert!(indexed.is_some());
    }

    #[test]
    fn buffer_copy_promotion_validates_single_writer_and_rejects_hazards() {
        let (src, src2, dst) = (0x1000, 0x2000, 0x3000);
        let empty = HashSet::new();
        // Single clean in-flight writer covering src -> promoted.
        let got = plan_buffer_copies(
            &[(dst, 0x8000, 0x100, 64)],
            &[(src, 0x80, 0x200)],
            &[],
            &empty,
            &empty,
            &empty,
            |s, lo, hi| s == src && lo == 0x100 && hi == 0x140,
        );
        assert_eq!(got, Some(vec![(src, dst, 0x100, 0x8000, 64)]));
        // Two distinct writers overlapping src -> rejected (fallback to settle).
        assert!(
            plan_buffer_copies(
                &[(dst, 0x8000, 0x100, 64)],
                &[(src, 0x80, 0x120), (src2, 0x120, 0x200)],
                &[],
                &empty,
                &empty,
                &empty,
                |_, _, _| true,
            )
            .is_none()
        );
        // In-flight write touching dst -> rejected.
        assert!(
            plan_buffer_copies(
                &[(dst, 0x8000, 0x100, 64)],
                &[(src, 0x80, 0x200), (src2, 0x8020, 0x8040)],
                &[],
                &empty,
                &empty,
                &empty,
                |_, _, _| true,
            )
            .is_none()
        );
        // Cycle with existing refresh alias srcs -> rejected.
        let mut ex_srcs = HashSet::new();
        ex_srcs.insert(dst);
        assert!(
            plan_buffer_copies(
                &[(dst, 0x8000, 0x100, 64)],
                &[(src, 0x80, 0x200)],
                &[],
                &empty,
                &ex_srcs,
                &empty,
                |_, _, _| true,
            )
            .is_none()
        );
    }
}
