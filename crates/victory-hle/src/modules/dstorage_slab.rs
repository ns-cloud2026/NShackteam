//! Bounded in-RAM LRU aligned slab read cache for DirectStorage archive files
//! (`VICTORY_DSTORAGE_SLAB_CACHE_MB`, 0/unset = off).
//!
//! Profiling live gameplay on a fragmented mechanical SATA HDD (`levels.rpf`
//! with 46,634 NTFS extents, `audio.rpf` with 20,145 extents) showed the title
//! issuing hundreds of 16–256 KiB reads/sec across separate `Submit()` calls.
//! Reading aligned `1 MiB` slabs on misses and keeping the hottest slabs in a
//! strictly bounded LRU cache turns subsequent nearby reads into RAM hits with
//! zero HDD seeks, while capping RAM usage to the configured MiB ceiling.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Alignment and size of each cached archive slab (1 MiB).
pub(crate) const SLAB_SIZE: usize = 1 << 20;

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);
static SLAB_HITS: AtomicU64 = AtomicU64::new(0);
static SLAB_MISSES: AtomicU64 = AtomicU64::new(0);
static SLAB_HIT_BYTES: AtomicU64 = AtomicU64::new(0);

/// Returns a monotonic unique file identifier so closed/reallocated `File`
/// objects can never alias stale cached slabs.
pub(crate) fn next_file_id() -> u64 {
    NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Configured slab cache capacity in slabs (1 slab = 1 MiB), parsed from
/// `VICTORY_DSTORAGE_SLAB_CACHE_MB` (`0`/unset = disabled; clamped to `16..=512`).
pub(crate) fn capacity_slabs() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        let mb = std::env::var("VICTORY_DSTORAGE_SLAB_CACHE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0);
        if mb == 0 {
            0
        } else {
            let cap = mb.clamp(16, 512);
            tracing::info!(
                cache_mb = cap,
                slab_kb = SLAB_SIZE >> 10,
                "dstorage: bounded in-RAM archive slab cache active (VICTORY_DSTORAGE_SLAB_CACHE_MB)"
            );
            cap
        }
    })
}

pub(crate) fn enabled() -> bool {
    capacity_slabs() > 0
}

/// Takes `(hits, misses, hit_bytes)` since the last call for 5-second telemetry.
pub(crate) fn take_stats() -> [u64; 3] {
    [
        SLAB_HITS.swap(0, Ordering::Relaxed),
        SLAB_MISSES.swap(0, Ordering::Relaxed),
        SLAB_HIT_BYTES.swap(0, Ordering::Relaxed),
    ]
}

struct SlabCacheInner {
    max_slabs: usize,
    tick: u64,
    /// `(file_id, slab_index) -> (slab_bytes, last_used_tick)`.
    map: HashMap<(u64, u64), (Arc<[u8]>, u64)>,
    /// FIFO/LRU touch queue `(key, tick)` for O(1) amortized eviction.
    order: VecDeque<((u64, u64), u64)>,
}

impl SlabCacheInner {
    fn new(max_slabs: usize) -> Self {
        Self {
            max_slabs,
            tick: 0,
            map: HashMap::with_capacity(max_slabs),
            order: VecDeque::with_capacity(max_slabs * 2),
        }
    }

    fn get(&mut self, key: (u64, u64)) -> Option<Arc<[u8]>> {
        let entry = self.map.get_mut(&key)?;
        self.tick = self.tick.wrapping_add(1);
        entry.1 = self.tick;
        let data = Arc::clone(&entry.0);
        self.order.push_back((key, self.tick));
        if self.order.len() > self.max_slabs.saturating_mul(4).max(64) {
            self.compact_order();
        }
        Some(data)
    }

    fn insert(&mut self, key: (u64, u64), data: Arc<[u8]>) {
        if self.max_slabs == 0 {
            return;
        }
        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;
        if let Some(existing) = self.map.get_mut(&key) {
            existing.0 = data;
            existing.1 = tick;
            self.order.push_back((key, tick));
            return;
        }
        while self.map.len() >= self.max_slabs {
            let Some((old_key, old_tick)) = self.order.pop_front() else {
                // Fallback if queue was somehow drained: evict any single entry.
                if let Some(&any_key) = self.map.keys().next() {
                    self.map.remove(&any_key);
                } else {
                    break;
                }
                continue;
            };
            if self.map.get(&old_key).is_some_and(|(_, t)| *t == old_tick) {
                self.map.remove(&old_key);
            }
        }
        self.map.insert(key, (data, tick));
        self.order.push_back((key, tick));
    }

    fn invalidate_file(&mut self, file_id: u64) {
        self.map.retain(|&(fid, _), _| fid != file_id);
        self.order.retain(|&((fid, _), _)| fid != file_id);
    }

    fn compact_order(&mut self) {
        self.order.retain(|&(k, t)| self.map.get(&k).is_some_and(|(_, cur)| *cur == t));
    }
}

/// Thread-safe bounded LRU archive slab cache.
pub(crate) struct SlabCache {
    inner: Mutex<SlabCacheInner>,
}

impl SlabCache {
    pub(crate) fn new(max_slabs: usize) -> Self {
        Self {
            inner: Mutex::new(SlabCacheInner::new(max_slabs)),
        }
    }

    pub(crate) fn global() -> &'static Self {
        static CACHE: OnceLock<SlabCache> = OnceLock::new();
        CACHE.get_or_init(|| Self::new(capacity_slabs()))
    }

    pub(crate) fn invalidate_file(&self, file_id: u64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.invalidate_file(file_id);
        } else {
            self.inner.lock().unwrap_or_else(|e| e.into_inner()).invalidate_file(file_id);
        }
    }

    /// Reads `out.len()` bytes starting at `offset` of read-only file `file_id`,
    /// serving hits from RAM and calling `fetch_slab(slab_offset, &mut buf)` on
    /// misses WITHOUT holding the cache lock across disk I/O.
    ///
    /// `fetch_slab` must return `Ok(bytes_read)` (`0..=SLAB_SIZE`) or `Err(E)`.
    /// Returns `Ok(true)` if all `out.len()` bytes were filled, `Ok(false)` on
    /// short EOF before filling `out`, or `Err(E)` if `fetch_slab` failed.
    pub(crate) fn read_exact<E>(
        &self,
        file_id: u64,
        offset: u64,
        out: &mut [u8],
        mut fetch_slab: impl FnMut(u64, &mut [u8]) -> Result<usize, E>,
    ) -> Result<bool, E> {
        let mut done = 0usize;
        while done < out.len() {
            let cur = offset.saturating_add(done as u64);
            let slab_idx = cur / (SLAB_SIZE as u64);
            let in_slab = (cur % (SLAB_SIZE as u64)) as usize;
            let want = (out.len() - done).min(SLAB_SIZE - in_slab);
            let key = (file_id, slab_idx);

            let cached = {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                inner.get(key)
            };
            let slab = match cached {
                Some(s) => {
                    SLAB_HITS.fetch_add(1, Ordering::Relaxed);
                    SLAB_HIT_BYTES.fetch_add(want as u64, Ordering::Relaxed);
                    s
                }
                None => {
                    SLAB_MISSES.fetch_add(1, Ordering::Relaxed);
                    let mut buf = vec![0u8; SLAB_SIZE];
                    let got = fetch_slab(slab_idx * (SLAB_SIZE as u64), &mut buf)?;
                    buf.truncate(got);
                    let arc: Arc<[u8]> = Arc::from(buf);
                    if !arc.is_empty() {
                        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                        inner.insert(key, Arc::clone(&arc));
                    }
                    arc
                }
            };
            if in_slab + want > slab.len() {
                return Ok(false);
            }
            out[done..done + want].copy_from_slice(&slab[in_slab..in_slab + want]);
            done += want;
        }
        Ok(true)
    }
}

pub(crate) fn invalidate_file(file_id: u64) {
    if enabled() {
        SlabCache::global().invalidate_file(file_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_file(size: usize) -> Vec<u8> {
        (0..size).map(|i| ((i.wrapping_mul(31) ^ (i >> 8)) & 0xff) as u8).collect()
    }

    #[test]
    fn slab_cache_serves_hits_and_cross_slab_boundaries() {
        let data = mock_file(SLAB_SIZE * 2 + 12345);
        let cache = SlabCache::new(4);
        let disk_reads = std::cell::Cell::new(0usize);

        let fetch = |off: u64, buf: &mut [u8]| -> Result<usize, ()> {
            disk_reads.set(disk_reads.get() + 1);
            let start = off as usize;
            if start >= data.len() {
                return Ok(0);
            }
            let n = (data.len() - start).min(buf.len());
            buf[..n].copy_from_slice(&data[start..start + n]);
            Ok(n)
        };

        // First read inside slab 0 misses once.
        let mut out1 = vec![0u8; 4096];
        assert_eq!(cache.read_exact(1, 8192, &mut out1, fetch), Ok(true));
        assert_eq!(out1, &data[8192..8192 + 4096]);
        assert_eq!(disk_reads.get(), 1);

        // Nearby second read in slab 0 hits RAM (zero disk reads).
        let mut out2 = vec![0u8; 16384];
        assert_eq!(cache.read_exact(1, 65536, &mut out2, fetch), Ok(true));
        assert_eq!(out2, &data[65536..65536 + 16384]);
        assert_eq!(disk_reads.get(), 1);

        // Read crossing slab 0 -> slab 1 boundary hits slab 0 and fetches only slab 1.
        let cross_off = SLAB_SIZE - 512;
        let mut out3 = vec![0u8; 2048];
        assert_eq!(cache.read_exact(1, cross_off as u64, &mut out3, fetch), Ok(true));
        assert_eq!(out3, &data[cross_off..cross_off + 2048]);
        assert_eq!(disk_reads.get(), 2);

        // Short EOF returns Ok(false) cleanly.
        let mut eof_out = vec![0u8; 4096];
        assert_eq!(
            cache.read_exact(1, (data.len() - 100) as u64, &mut eof_out, fetch),
            Ok(false)
        );
    }

    #[test]
    fn slab_cache_enforces_lru_bound_and_invalidation() {
        let data = mock_file(SLAB_SIZE * 4);
        let cache = SlabCache::new(2);
        let disk_reads = std::cell::Cell::new(0usize);
        let fetch = |off: u64, buf: &mut [u8]| -> Result<usize, ()> {
            disk_reads.set(disk_reads.get() + 1);
            let start = off as usize;
            let n = (data.len() - start).min(buf.len());
            buf[..n].copy_from_slice(&data[start..start + n]);
            Ok(n)
        };

        let mut buf = [0u8; 64];
        // Load slab 0 and slab 1.
        assert_eq!(cache.read_exact(10, 0, &mut buf, fetch), Ok(true));
        assert_eq!(cache.read_exact(10, SLAB_SIZE as u64, &mut buf, fetch), Ok(true));
        assert_eq!(disk_reads.get(), 2);
        // Touch slab 0 so slab 1 becomes LRU.
        assert_eq!(cache.read_exact(10, 128, &mut buf, fetch), Ok(true));
        assert_eq!(disk_reads.get(), 2);
        // Load slab 2 -> must evict slab 1, keeping slab 0 hot!
        assert_eq!(cache.read_exact(10, (SLAB_SIZE * 2) as u64, &mut buf, fetch), Ok(true));
        assert_eq!(disk_reads.get(), 3);
        assert_eq!(cache.read_exact(10, 256, &mut buf, fetch), Ok(true));
        assert_eq!(disk_reads.get(), 3, "slab 0 must still be resident");
        // Invalidate file 10 -> next read of slab 0 must miss.
        cache.invalidate_file(10);
        assert_eq!(cache.read_exact(10, 256, &mut buf, fetch), Ok(true));
        assert_eq!(disk_reads.get(), 4);
    }
}
