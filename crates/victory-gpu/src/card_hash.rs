//! A wider form of the hash pass over copies in the card's memory
//! (`VICTORY_CARD_HASH_WIDE=1`; off by default): the same work as
//! `crate::host::CARD_COMPARE`, with the same root signature, constants and
//! results, but a group of 256 threads per 64 KiB piece instead of one
//! thread.
//!
//! The narrow form walks a piece 16 bytes at a time, 4,096 dependent steps
//! on one thread, with neighbouring threads 64 KiB apart, so a pass over a
//! few pieces takes the time of a whole piece on one lane of the card
//! (`crate::host`'s `rehash_runs`: 514 such passes a frame in run 102).
//! Here lane `l` of a piece's group takes the piece's 16-byte blocks
//! `l, l + 256, l + 512, ...` (16 steps for a whole piece; the group's loads
//! of one step are 4 KiB side by side), and the 256 lane hashes are then
//! joined in a fixed tree.
//!
//! The hash differs from the narrow form's. Nothing outside the shader
//! computes or reads it (the `hashes` buffer holds only what the shader
//! stored), and one form is chosen for the run, so the two never meet.
//!
//! What either form guarantees: a change confined to one 16-byte block of
//! a piece always changes its hash. Here because the lane's round is a
//! bijection of its state for a given block and of the block for a given
//! state, and [`join`] is a bijection of each side for a given other side.
//! A change spread over several blocks is missed only by a collision of
//! the 128-bit value, as before.

/// Groups per row of the dispatch: a dimension may not exceed 65,535
/// groups, and a run can hold more pieces than that.
pub(crate) const ROW: u32 = 32768;

/// The groups to dispatch for `pieces` pieces: the wide form takes a group
/// per piece, in rows of [`ROW`]; the narrow form a thread per piece, in
/// groups of 64.
pub(crate) fn groups(pieces: u32, wide: bool) -> (u32, u32) {
    if wide {
        (pieces.min(ROW), pieces.div_ceil(ROW))
    } else {
        (pieces.div_ceil(64), 1)
    }
}

/// The wide form's source (FXC, `cs_5_1`). Every thread reaches every
/// barrier: a group past the last piece hashes nothing and stores nothing,
/// but does not return early.
pub(crate) const CARD_COMPARE_WIDE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint4 lanes[256];

uint4 step16(uint4 h, uint4 x) {
    h = (h ^ x) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
    return h.yzwx ^ (h >> 15);
}

uint4 join(uint4 a, uint4 b) {
    return (a * uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu))
         ^ (b * uint4(0x01000193u, 0x2545F491u, 0x5851F42Du, 0x6C078965u)).wxyz;
}

[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint i = group.y * 32768u + group.x;
    bool live = i < pieces;
    uint piece = first + i;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    if (!live) hi = lo;
    uint4 h = uint4(0x9E3779B9u ^ lane, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo + lane * 16u; at < hi; at += 4096u) {
        h = step16(h, now.Load4(at));
    }
    lanes[lane] = h;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lane < s) lanes[lane] = join(lanes[lane], lanes[lane + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (live && lane == 0u) {
        h = lanes[0];
        if (compare != 0u) {
            differs.Store(out_at + i * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
            if (compare == 2u) hashes.Store4(piece * 16u, h);
        } else {
            hashes.Store4(piece * 16u, h);
        }
    }
}
"#;

/// Batched wide hash pass (FXC, `cs_5_1`): hashes an arbitrary list of `count`
/// 64 KiB piece indices stored in `piece_list` (`t1`) using the exact same
/// 256-thread-per-piece hash arithmetic as [`CARD_COMPARE_WIDE`]. Collapses
/// dozens of tiny per-range dispatches in `Host::card_before` into a single
/// wide GPU dispatch per dirty buffer.
pub(crate) const CARD_HASH_BATCH: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer Batch : register(b0) {
    uint skew;         // where the copy starts in its piece 0
    uint size;         // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint count;        // number of piece indices in `piece_list`
    uint list_offset;  // byte offset of the first piece index in `piece_list`
};
ByteAddressBuffer now : register(t0);
ByteAddressBuffer piece_list : register(t1);
RWByteAddressBuffer hashes : register(u0);
groupshared uint4 lanes[256];

uint4 step16(uint4 h, uint4 x) {
    h = (h ^ x) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
    return h.yzwx ^ (h >> 15);
}

uint4 join(uint4 a, uint4 b) {
    return (a * uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu))
         ^ (b * uint4(0x01000193u, 0x2545F491u, 0x5851F42Du, 0x6C078965u)).wxyz;
}

[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint i = group.y * 32768u + group.x;
    bool live = i < count;
    uint piece = live ? piece_list.Load(list_offset + i * 4u) : 0u;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    if (!live) hi = lo;
    uint4 h = uint4(0x9E3779B9u ^ lane, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo + lane * 16u; at < hi; at += 4096u) {
        h = step16(h, now.Load4(at));
    }
    lanes[lane] = h;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (lane < s) lanes[lane] = join(lanes[lane], lanes[lane + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (live && lane == 0u) {
        hashes.Store4(piece * 16u, lanes[0]);
    }
}
"#;

/// Whether the wide form is asked for (`VICTORY_CARD_HASH_WIDE=1`). Read
/// once, when the host makes the compare.
pub(crate) fn wide() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_WIDE").is_some_and(|v| v == "1"))
}

/// Flattens joined piece runs into a sorted, deduplicated list of 64 KiB piece
/// indices for [`CARD_HASH_BATCH`].
pub(crate) fn flatten_piece_runs(runs: &[(u32, u32)]) -> Vec<u32> {
    let total: usize = runs.iter().map(|&(_, n)| n as usize).sum();
    let mut out = Vec::with_capacity(total);
    for &(first, pieces) in runs {
        out.extend(first..first.saturating_add(pieces));
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    //! The shader's arithmetic, step for step in Rust, to pin what it
    //! computes (the shader itself runs only on a card).
    use super::{ROW, groups};

    /// Threads per piece (the shader's `numthreads`).
    const LANES: u32 = 256;

    const K: [u32; 4] = [0x0100_0193, 0x9E37_79B1, 0x85EB_CA77, 0xC2B2_AE3D];
    const JA: [u32; 4] = [0x9E37_79B1, 0x85EB_CA77, 0xC2B2_AE3D, 0x27D4_EB2F];
    const JB: [u32; 4] = [0x0100_0193, 0x2545_F491, 0x5851_F42D, 0x6C07_8965];

    fn mul(a: [u32; 4], k: [u32; 4]) -> [u32; 4] {
        std::array::from_fn(|i| a[i].wrapping_mul(k[i]))
    }

    fn step16(h: [u32; 4], x: [u32; 4]) -> [u32; 4] {
        let h = mul(std::array::from_fn(|i| h[i] ^ x[i]), K);
        // h.yzwx ^ (h >> 15)
        std::array::from_fn(|i| h[(i + 1) % 4] ^ (h[i] >> 15))
    }

    fn join(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
        let (a, b) = (mul(a, JA), mul(b, JB));
        // b.wxyz
        std::array::from_fn(|i| a[i] ^ b[(i + 3) % 4])
    }

    fn load4(buf: &[u8], at: usize) -> [u32; 4] {
        std::array::from_fn(|i| {
            u32::from_le_bytes(buf[at + 4 * i..at + 4 * i + 4].try_into().unwrap())
        })
    }

    /// The bounds the shader hashes for `piece`.
    fn bounds(piece: u32, skew: u32, size: u32) -> (u32, u32) {
        let lo = piece * 65536;
        let lo = if lo > skew { lo - skew } else { 0 };
        let hi = ((piece + 1) * 65536 - skew).min(size);
        (lo & !15, (hi + 15) & !15)
    }

    /// What the group of `piece` leaves in `lanes[0]`.
    fn piece_hash(buf: &[u8], piece: u32, skew: u32) -> [u32; 4] {
        let (lo, hi) = bounds(piece, skew, buf.len() as u32);
        let mut lanes: Vec<[u32; 4]> = (0..LANES)
            .map(|lane| {
                let mut h = [0x9E37_79B9 ^ lane, 0x85EB_CA6B, 0xC2B2_AE35, 0x27D4_EB2F];
                let mut at = lo + lane * 16;
                while at < hi {
                    h = step16(h, load4(buf, at as usize));
                    at += 4096;
                }
                h
            })
            .collect();
        let mut s = LANES / 2;
        while s > 0 {
            for lane in 0..s as usize {
                lanes[lane] = join(lanes[lane], lanes[lane + s as usize]);
            }
            s /= 2;
        }
        lanes[0]
    }

    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    /// Every 16-byte block of a piece is taken by exactly one lane, once.
    #[test]
    fn the_lanes_take_every_block_of_a_piece_once() {
        for (piece, skew, size) in [
            (0u32, 0u32, 1u32 << 20),
            (3, 0, 1 << 20),
            (0, 0x1f00, 1 << 20),
            (2, 0x1f00, 1 << 20),
            (15, 0, 1 << 20),
            (15, 0x8000, 1 << 20),
        ] {
            let (lo, hi) = bounds(piece, skew, size);
            let mut seen = vec![0u8; ((hi - lo) / 16) as usize];
            for lane in 0..LANES {
                let mut at = lo + lane * 16;
                while at < hi {
                    seen[((at - lo) / 16) as usize] += 1;
                    at += 4096;
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "piece {piece} skew {skew:#x}");
        }
    }

    /// A change of any one byte of a piece changes its hash; a change
    /// outside its bounds does not.
    #[test]
    fn one_changed_byte_always_shows() {
        let size = 4 * 65536;
        let base = noise(size, 7);
        for skew in [0u32, 0x1f00] {
            let piece = 1;
            let (lo, hi) = bounds(piece, skew, size as u32);
            let before = piece_hash(&base, piece, skew);
            let mut probe = noise(64, 99).into_iter();
            for at in (lo..hi)
                .step_by(97)
                .chain([lo, lo + 15, lo + 16, hi - 16, hi - 1])
            {
                let mut buf = base.clone();
                buf[at as usize] ^= probe.next().unwrap_or(1) | 1;
                assert_ne!(
                    piece_hash(&buf, piece, skew),
                    before,
                    "byte {at:#x} skew {skew:#x}"
                );
            }
            for at in [lo.wrapping_sub(1), hi] {
                if (at as usize) < size {
                    let mut buf = base.clone();
                    buf[at as usize] ^= 0x5a;
                    assert_eq!(
                        piece_hash(&buf, piece, skew),
                        before,
                        "byte {at:#x} outside"
                    );
                }
            }
        }
    }

    /// Pieces that hold the same bytes at different places, or the same
    /// blocks in another order, hash differently (the lane seeds and the
    /// tree's order are in the hash).
    #[test]
    fn swapped_blocks_show() {
        let size = 2 * 65536;
        let base = noise(size, 3);
        let before = piece_hash(&base, 0, 0);
        let mut swapped = base.clone();
        let (a, b) = (16 * 5, 16 * 300);
        for i in 0..16 {
            swapped.swap(a + i, b + i);
        }
        assert_ne!(piece_hash(&swapped, 0, 0), before);
        assert_eq!(piece_hash(&base, 0, 0), before);
    }

    /// The groups cover pieces `0..n` with each index once and stay within
    /// the 65,535-group limit per dimension.
    #[test]
    fn the_dispatch_covers_every_piece() {
        for n in [
            0u32,
            1,
            2,
            63,
            64,
            65,
            ROW - 1,
            ROW,
            ROW + 1,
            3 * ROW + 5,
            262_144,
        ] {
            let (x, y) = groups(n, true);
            assert!(x <= 65_535 && y <= 65_535);
            let live = (0..y)
                .flat_map(|gy| (0..x).map(move |gx| gy * ROW + gx))
                .filter(|&i| i < n)
                .count();
            assert_eq!(live as u32, n, "{n} pieces");
            let (x, _) = groups(n, false);
            assert!(x * 64 >= n && x * 64 < n + 64);
        }
    }

    #[test]
    fn flatten_piece_runs_emits_unique_sorted_pieces() {
        assert_eq!(
            super::flatten_piece_runs(&[(10, 2), (3, 1), (11, 3), (3, 1)]),
            vec![3, 10, 11, 12, 13]
        );
        assert!(super::flatten_piece_runs(&[]).is_empty());
    }
}
