//! Fast zero-bounce VEH-guarded AVX2/SSE2 compare-and-copy for `Shadow::copy_compared`.
//!
//! Profiling live gameplay frames showed `Shadow::copy_compared` scanning 130–205 MiB
//! of title memory per frame while 86–90% of 4 KiB pages are byte-for-byte unchanged,
//! plus taking thousands of repeated VEH exceptions when walking multi-megabyte
//! unmapped holes in sparse heaps/windows.
//!
//! This module compares each 4 KiB page directly in-process using a 128-byte unrolled
//! AVX2 loop (with automatic fallback to 64-byte unrolled SSE2) protected by a
//! Vectored Exception Handler:
//! - Unchanged pages take zero writes, zero scratch memory, and zero locks.
//! - Changed pages copy directly from title memory into the host shadow mapping.
//! - On an unmapped/faulting page (`EqStatus::Fault`), a single `VirtualQuery` detects
//!   contiguous uncommitted / `PAGE_NOACCESS` holes and fast-scans `held` against zero
//!   with zero additional VEH exceptions (preserving exact zero-fill semantics).

/// Outcome of a VEH-guarded direct memory equality comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EqStatus {
    /// All bytes in `[0..len)` matched.
    Equal,
    /// At least one byte differed.
    Different,
    /// Reading `a` faulted (unmapped / `PAGE_NOACCESS` / in-page error).
    Fault,
}

#[cfg(target_arch = "x86_64")]
mod simd_eq {
    use std::sync::{Once, OnceLock};

    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_IN_PAGE_ERROR, NTSTATUS,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };

    use super::EqStatus;

    // Windows x64 ABI:
    //   rcx = a (*const u8, may fault)
    //   rdx = b (*const u8, valid host memory)
    //   r8  = len (usize)
    //   al  = 0 (Equal) | 1 (Different) | 2 (Fault)
    // Touches only volatile registers (rax, rcx, rdx, r8, xmm0..xmm5 / ymm0..ymm5).
    std::arch::global_asm!(
        ".global victory_try_eq",
        ".global victory_try_eq_op",
        ".global victory_try_eq_end",
        ".global victory_try_eq_fault",
        "victory_try_eq:",
        "test r8, r8",
        "jz 4f",
        "victory_try_eq_op:",
        "1:",
        "cmp r8, 64",
        "jb 2f",
        "movdqu xmm0, [rcx]",
        "movdqu xmm1, [rdx]",
        "pcmpeqb xmm0, xmm1",
        "movdqu xmm2, [rcx + 16]",
        "movdqu xmm3, [rdx + 16]",
        "pcmpeqb xmm2, xmm3",
        "pand xmm0, xmm2",
        "movdqu xmm4, [rcx + 32]",
        "movdqu xmm5, [rdx + 32]",
        "pcmpeqb xmm4, xmm5",
        "pand xmm0, xmm4",
        "movdqu xmm2, [rcx + 48]",
        "movdqu xmm3, [rdx + 48]",
        "pcmpeqb xmm2, xmm3",
        "pand xmm0, xmm2",
        "pmovmskb eax, xmm0",
        "cmp eax, 0xFFFF",
        "jne 5f",
        "add rcx, 64",
        "add rdx, 64",
        "sub r8, 64",
        "jmp 1b",
        "2:",
        "cmp r8, 16",
        "jb 3f",
        "movdqu xmm0, [rcx]",
        "movdqu xmm1, [rdx]",
        "pcmpeqb xmm0, xmm1",
        "pmovmskb eax, xmm0",
        "cmp eax, 0xFFFF",
        "jne 5f",
        "add rcx, 16",
        "add rdx, 16",
        "sub r8, 16",
        "jmp 2b",
        "3:",
        "test r8, r8",
        "jz 4f",
        "mov al, [rcx]",
        "cmp al, [rdx]",
        "jne 5f",
        "inc rcx",
        "inc rdx",
        "dec r8",
        "jmp 3b",
        "victory_try_eq_end:",
        "4:",
        "xor eax, eax",
        "ret",
        "5:",
        "mov eax, 1",
        "ret",
        "victory_try_eq_fault:",
        "mov eax, 2",
        "ret",

        ".global victory_try_eq_avx2",
        ".global victory_try_eq_avx2_op",
        ".global victory_try_eq_avx2_end",
        ".global victory_try_eq_avx2_fault",
        "victory_try_eq_avx2:",
        "test r8, r8",
        "jz 14f",
        "victory_try_eq_avx2_op:",
        "11:",
        "cmp r8, 128",
        "jb 12f",
        "vmovdqu ymm0, [rcx]",
        "vpcmpeqb ymm0, ymm0, [rdx]",
        "vmovdqu ymm1, [rcx + 32]",
        "vpcmpeqb ymm1, ymm1, [rdx + 32]",
        "vpand ymm0, ymm0, ymm1",
        "vmovdqu ymm2, [rcx + 64]",
        "vpcmpeqb ymm2, ymm2, [rdx + 64]",
        "vpand ymm0, ymm0, ymm2",
        "vmovdqu ymm3, [rcx + 96]",
        "vpcmpeqb ymm3, ymm3, [rdx + 96]",
        "vpand ymm0, ymm0, ymm3",
        "vpmovmskb eax, ymm0",
        "cmp eax, -1",
        "jne 15f",
        "add rcx, 128",
        "add rdx, 128",
        "sub r8, 128",
        "jmp 11b",
        "12:",
        "cmp r8, 32",
        "jb 13f",
        "vmovdqu ymm0, [rcx]",
        "vpcmpeqb ymm0, ymm0, [rdx]",
        "vpmovmskb eax, ymm0",
        "cmp eax, -1",
        "jne 15f",
        "add rcx, 32",
        "add rdx, 32",
        "sub r8, 32",
        "jmp 12b",
        "13:",
        "test r8, r8",
        "jz 14f",
        "mov al, [rcx]",
        "cmp al, [rdx]",
        "jne 15f",
        "inc rcx",
        "inc rdx",
        "dec r8",
        "jmp 13b",
        "victory_try_eq_avx2_end:",
        "14:",
        "vzeroupper",
        "xor eax, eax",
        "ret",
        "15:",
        "vzeroupper",
        "mov eax, 1",
        "ret",
        "victory_try_eq_avx2_fault:",
        "vzeroupper",
        "mov eax, 2",
        "ret",

        // Windows x64 ABI:
        //   rcx = dst (*mut u8, valid writable host memory, non-overlapping with src)
        //   rdx = src (*const u8, may fault)
        //   r8  = len (usize)
        //   al  = 1 (Ok) | 0 (Fault)
        // Touches only volatile registers (rax, rcx, rdx, r8, xmm0..xmm3 / ymm0..ymm3).
        ".global victory_fast_try_copy",
        ".global victory_fast_try_copy_op",
        ".global victory_fast_try_copy_end",
        ".global victory_fast_try_copy_fault",
        "victory_fast_try_copy:",
        "test r8, r8",
        "jz 24f",
        "victory_fast_try_copy_op:",
        "21:",
        "cmp r8, 64",
        "jb 22f",
        "movdqu xmm0, [rdx]",
        "movdqu xmm1, [rdx + 16]",
        "movdqu xmm2, [rdx + 32]",
        "movdqu xmm3, [rdx + 48]",
        "movdqu [rcx], xmm0",
        "movdqu [rcx + 16], xmm1",
        "movdqu [rcx + 32], xmm2",
        "movdqu [rcx + 48], xmm3",
        "add rcx, 64",
        "add rdx, 64",
        "sub r8, 64",
        "jmp 21b",
        "22:",
        "cmp r8, 16",
        "jb 23f",
        "movdqu xmm0, [rdx]",
        "movdqu [rcx], xmm0",
        "add rcx, 16",
        "add rdx, 16",
        "sub r8, 16",
        "jmp 22b",
        "23:",
        "test r8, r8",
        "jz 24f",
        "mov al, [rdx]",
        "mov [rcx], al",
        "inc rcx",
        "inc rdx",
        "dec r8",
        "jmp 23b",
        "victory_fast_try_copy_end:",
        "24:",
        "mov eax, 1",
        "ret",
        "victory_fast_try_copy_fault:",
        "xor eax, eax",
        "ret",

        ".global victory_fast_try_copy_avx2",
        ".global victory_fast_try_copy_avx2_op",
        ".global victory_fast_try_copy_avx2_end",
        ".global victory_fast_try_copy_avx2_fault",
        "victory_fast_try_copy_avx2:",
        "test r8, r8",
        "jz 34f",
        "victory_fast_try_copy_avx2_op:",
        "31:",
        "cmp r8, 128",
        "jb 32f",
        "vmovdqu ymm0, [rdx]",
        "vmovdqu ymm1, [rdx + 32]",
        "vmovdqu ymm2, [rdx + 64]",
        "vmovdqu ymm3, [rdx + 96]",
        "vmovdqu [rcx], ymm0",
        "vmovdqu [rcx + 32], ymm1",
        "vmovdqu [rcx + 64], ymm2",
        "vmovdqu [rcx + 96], ymm3",
        "add rcx, 128",
        "add rdx, 128",
        "sub r8, 128",
        "jmp 31b",
        "32:",
        "cmp r8, 32",
        "jb 33f",
        "vmovdqu ymm0, [rdx]",
        "vmovdqu [rcx], ymm0",
        "add rcx, 32",
        "add rdx, 32",
        "sub r8, 32",
        "jmp 32b",
        "33:",
        "test r8, r8",
        "jz 34f",
        "mov al, [rdx]",
        "mov [rcx], al",
        "inc rcx",
        "inc rdx",
        "dec r8",
        "jmp 33b",
        "victory_fast_try_copy_avx2_end:",
        "34:",
        "vzeroupper",
        "mov eax, 1",
        "ret",
        "victory_fast_try_copy_avx2_fault:",
        "vzeroupper",
        "xor eax, eax",
        "ret",
    );

    unsafe extern "C" {
        fn victory_try_eq(a: *const u8, b: *const u8, len: usize) -> u8;
        static victory_try_eq_op: u8;
        static victory_try_eq_end: u8;
        static victory_try_eq_fault: u8;

        fn victory_try_eq_avx2(a: *const u8, b: *const u8, len: usize) -> u8;
        static victory_try_eq_avx2_op: u8;
        static victory_try_eq_avx2_end: u8;
        static victory_try_eq_avx2_fault: u8;

        fn victory_fast_try_copy(dst: *mut u8, src: *const u8, len: usize) -> u8;
        static victory_fast_try_copy_op: u8;
        static victory_fast_try_copy_end: u8;
        static victory_fast_try_copy_fault: u8;

        fn victory_fast_try_copy_avx2(dst: *mut u8, src: *const u8, len: usize) -> u8;
        static victory_fast_try_copy_avx2_op: u8;
        static victory_fast_try_copy_avx2_end: u8;
        static victory_fast_try_copy_avx2_fault: u8;
    }

    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: Windows passes valid exception pointers during VEH dispatch.
        let record = unsafe { &*(*info).ExceptionRecord };
        let code = record.ExceptionCode;
        if code != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && code != EXCEPTION_IN_PAGE_ERROR as NTSTATUS
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: `ContextRecord` is valid during VEH dispatch.
        let ctx = unsafe { &mut *(*info).ContextRecord };
        let rip = ctx.Rip as usize;

        let copy_avx_start = (&raw const victory_fast_try_copy_avx2_op) as usize;
        let copy_avx_end = (&raw const victory_fast_try_copy_avx2_end) as usize;
        if rip >= copy_avx_start && rip < copy_avx_end {
            ctx.Rip = (&raw const victory_fast_try_copy_avx2_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        let copy_sse_start = (&raw const victory_fast_try_copy_op) as usize;
        let copy_sse_end = (&raw const victory_fast_try_copy_end) as usize;
        if rip >= copy_sse_start && rip < copy_sse_end {
            ctx.Rip = (&raw const victory_fast_try_copy_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        let avx_start = (&raw const victory_try_eq_avx2_op) as usize;
        let avx_end = (&raw const victory_try_eq_avx2_end) as usize;
        if rip >= avx_start && rip < avx_end {
            ctx.Rip = (&raw const victory_try_eq_avx2_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        let sse_start = (&raw const victory_try_eq_op) as usize;
        let sse_end = (&raw const victory_try_eq_end) as usize;
        if rip >= sse_start && rip < sse_end {
            ctx.Rip = (&raw const victory_try_eq_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }

        EXCEPTION_CONTINUE_SEARCH
    }

    fn install() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            // SAFETY: `on_fault` only intercepts read faults inside the
            // `victory_try_eq*` and `victory_fast_try_copy*` instruction ranges.
            unsafe {
                AddVectoredExceptionHandler(1, Some(on_fault));
            }
        });
    }

    fn has_avx2() -> bool {
        static AVX2: OnceLock<bool> = OnceLock::new();
        *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
    }

    /// Compares `a[..len]` against `b[..len]` directly using AVX2 (128B/iter)
    /// or SSE2 (64B/iter).
    ///
    /// # Safety
    /// `b` must be valid readable memory for `len` bytes. Any fault reading `a`
    /// is intercepted and reported as [`EqStatus::Fault`].
    #[inline]
    pub(super) unsafe fn try_eq(a: *const u8, b: *const u8, len: usize) -> EqStatus {
        if len == 0 {
            return EqStatus::Equal;
        }
        install();
        // SAFETY: both assembly routines clobber only volatile registers and
        // redirect faults inside their guarded body to their fault label.
        let res = if len >= 32 && has_avx2() {
            unsafe { victory_try_eq_avx2(a, b, len) }
        } else {
            unsafe { victory_try_eq(a, b, len) }
        };
        match res {
            0 => EqStatus::Equal,
            1 => EqStatus::Different,
            _ => EqStatus::Fault,
        }
    }

    /// Copies `src[..len]` into non-overlapping valid writable `dst[..len]`
    /// directly using AVX2 (128B/iter) or SSE2 (64B/iter), returning `false`
    /// if reading `src` triggers an access violation or in-page error.
    ///
    /// # Safety
    /// `dst` must be valid writable memory for `len` bytes and must not overlap
    /// `src[..len]`.
    #[inline]
    pub(super) unsafe fn try_copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        install();
        // SAFETY: both assembly routines clobber only volatile registers and
        // redirect faults inside their guarded body to their fault label.
        let res = if len >= 32 && has_avx2() {
            unsafe { victory_fast_try_copy_avx2(dst, src, len) }
        } else {
            unsafe { victory_fast_try_copy(dst, src, len) }
        };
        res != 0
    }
}

/// Copies `src[..len]` into non-overlapping valid writable host memory `dst[..len]`
/// without kernel syscalls, catching any read fault cleanly via VEH.
///
/// # Safety
/// `dst` must be valid writable memory for `len` bytes and must not overlap `src[..len]`.
#[inline]
pub(crate) unsafe fn try_copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: caller guarantees `dst[..len]` is valid writable non-overlapping memory.
        unsafe { simd_eq::try_copy(dst, src, len) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dst, src, len);
        false
    }
}

/// Compares `a[..len]` against valid host memory `b[..len]` without scratch copies.
///
/// # Safety
/// `b` must be valid readable memory for `len` bytes.
#[inline]
pub(crate) unsafe fn try_eq(a: *const u8, b: *const u8, len: usize) -> EqStatus {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: caller guarantees `b[..len]` is valid readable memory.
        unsafe { simd_eq::try_eq(a, b, len) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (a, b, len);
        EqStatus::Fault
    }
}

/// Fast check whether all bytes in `bytes` are zero (used when skipping
/// contiguous unmapped title holes without taking per-page VEH faults).
#[inline]
pub(crate) fn is_all_zero(bytes: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if bytes.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime AVX2 detection; valid readable slice.
        return unsafe { is_all_zero_avx2(bytes) };
    }
    let (chunks, tail) = bytes.as_chunks::<8>();
    chunks.iter().all(|c| u64::from_ne_bytes(*c) == 0) && tail.iter().all(|&b| b == 0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn is_all_zero_avx2(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;
    let ptr = bytes.as_ptr();
    let len = bytes.len();
    let mut at = 0usize;
    while at + 128 <= len {
        // SAFETY: bounds checked above.
        unsafe {
            let v0 = _mm256_loadu_si256(ptr.add(at).cast());
            let v1 = _mm256_loadu_si256(ptr.add(at + 32).cast());
            let v2 = _mm256_loadu_si256(ptr.add(at + 64).cast());
            let v3 = _mm256_loadu_si256(ptr.add(at + 96).cast());
            let acc = _mm256_or_si256(_mm256_or_si256(v0, v1), _mm256_or_si256(v2, v3));
            if _mm256_testz_si256(acc, acc) == 0 {
                return false;
            }
        }
        at += 128;
    }
    while at + 32 <= len {
        // SAFETY: bounds checked above.
        unsafe {
            let v = _mm256_loadu_si256(ptr.add(at).cast());
            if _mm256_testz_si256(v, v) == 0 {
                return false;
            }
        }
        at += 32;
    }
    bytes[at..].iter().all(|&b| b == 0)
}

/// If `addr` lies in a contiguous uncommitted / unreadable virtual-memory
/// region, returns how many bytes from `addr` (capped at `remaining`) belong
/// to that hole so the caller can zero-check them without taking another VEH
/// fault on every 4 KiB page.
fn unmapped_hole_len(addr: usize, remaining: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    // SAFETY: fills a stack struct.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out-pointer of `size_of_val(&info)`.
    let n = unsafe { VirtualQuery(addr as *const _, &mut info, size_of_val(&info)) };
    if n != size_of_val(&info) {
        return 0;
    }
    let unreadable = info.State != MEM_COMMIT
        || info.Protect == 0
        || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0;
    if !unreadable {
        return 0;
    }
    let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    region_end.saturating_sub(addr).min(remaining)
}

/// Compares `held` against title memory starting at `title_addr` in 4 KiB pages,
/// copying only differing pages directly into `held` and recording coalesced
/// changed runs `(base_offset + start, base_offset + end)` into `runs`.
///
/// Returns the total number of changed bytes written to `held`.
pub(crate) fn compare_and_copy_slice(
    title_addr: usize,
    held: &mut [u8],
    base_offset: usize,
    runs: &mut Vec<(usize, usize)>,
) -> usize {
    const PIECE: usize = 4096;
    let mut changed = 0usize;
    let mut run: Option<usize> = None;
    let mut at = 0usize;
    let len = held.len();
    let mut fallback = [0u8; PIECE];

    while at < len {
        let end = (at + PIECE).min(len);
        let n = end - at;
        let src = (title_addr + at) as *const u8;
        let dst = &mut held[at..end];
        // SAFETY: `dst` is valid writable host memory for `n` bytes.
        let status = unsafe { try_eq(src, dst.as_ptr(), n) };
        match status {
            EqStatus::Equal => {
                if let Some(first) = run.take() {
                    runs.push((base_offset + first, base_offset + at));
                }
                at = end;
            }
            EqStatus::Different => {
                crate::host::copy_mapped(dst.as_mut_ptr(), src, n, false);
                run.get_or_insert(at);
                changed += n;
                at = end;
            }
            EqStatus::Fault => {
                // Detect multi-page unmapped holes with ONE VirtualQuery
                // instead of suffering a kernel VEH exception every 4 KiB.
                let hole = unmapped_hole_len(title_addr + at, len - at);
                if hole >= n {
                    let hole_end = at + hole;
                    while at < hole_end {
                        let piece_end = (at + PIECE).min(hole_end);
                        let slice = &mut held[at..piece_end];
                        if is_all_zero(slice) {
                            if let Some(first) = run.take() {
                                runs.push((base_offset + first, base_offset + at));
                            }
                        } else {
                            slice.fill(0);
                            run.get_or_insert(at);
                            changed += piece_end - at;
                        }
                        at = piece_end;
                    }
                } else {
                    let page = &mut fallback[..n];
                    crate::host::copy_mapped(page.as_mut_ptr(), src, n, false);
                    if page != dst {
                        dst.copy_from_slice(page);
                        run.get_or_insert(at);
                        changed += n;
                    } else if let Some(first) = run.take() {
                        runs.push((base_offset + first, base_offset + at));
                    }
                    at = end;
                }
            }
        }
    }
    if let Some(first) = run.take() {
        runs.push((base_offset + first, base_offset + len));
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::{EqStatus, compare_and_copy_slice, is_all_zero, try_eq};

    #[test]
    fn try_eq_matches_bytewise_across_sizes_and_alignments() {
        let base: Vec<u8> = (0..8256usize).map(|i| ((i * 37 + 13) & 0xFF) as u8).collect();
        for len in [0, 1, 2, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 4096, 8192] {
            let a = &base[3..3 + len];
            let mut b = a.to_vec();
            // SAFETY: both slices are valid readable memory for `len` bytes.
            assert_eq!(unsafe { try_eq(a.as_ptr(), b.as_ptr(), len) }, EqStatus::Equal, "eq len={len}");
            if len > 0 {
                for flip in [0, len / 2, len - 1] {
                    b[flip] ^= 0x5A;
                    // SAFETY: both slices are valid readable memory for `len` bytes.
                    assert_eq!(
                        unsafe { try_eq(a.as_ptr(), b.as_ptr(), len) },
                        EqStatus::Different,
                        "diff len={len} flip={flip}"
                    );
                    b[flip] ^= 0x5A;
                }
            }
        }
    }

    #[test]
    fn is_all_zero_handles_all_lengths_and_positions() {
        for len in [0, 1, 7, 8, 16, 31, 32, 64, 127, 128, 129, 257, 4096] {
            let mut buf = vec![0u8; len];
            assert!(is_all_zero(&buf), "len={len}");
            if len > 0 {
                for pos in [0, len / 2, len - 1] {
                    buf[pos] = 1;
                    assert!(!is_all_zero(&buf), "len={len} pos={pos}");
                    buf[pos] = 0;
                }
            }
        }
    }

    #[test]
    fn try_eq_catches_access_violation_on_unreadable_memory() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
        };
        let host = vec![0xAAu8; 4096];
        // Null pointer must report Fault cleanly rather than crashing.
        // SAFETY: `host` is valid; `try_eq` intercepts faults on `a`.
        assert_eq!(unsafe { try_eq(std::ptr::null(), host.as_ptr(), 64) }, EqStatus::Fault);

        // SAFETY: allocates a PAGE_NOACCESS page and frees it before returning.
        let no_access = unsafe {
            VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS)
        };
        assert!(!no_access.is_null());
        // SAFETY: `host` is valid; `try_eq` intercepts faults on `no_access`.
        let status = unsafe { try_eq(no_access.cast::<u8>(), host.as_ptr(), 4096) };
        // SAFETY: frees the allocation made above.
        unsafe {
            VirtualFree(no_access, 0, MEM_RELEASE);
        }
        assert_eq!(status, EqStatus::Fault);
    }

    #[test]
    fn compare_and_copy_slice_copies_only_changed_pages_and_coalesces_runs() {
        let mut title = vec![0x11u8; 4096 * 5];
        let mut held = title.clone();
        title[4096 + 17] = 0x22;
        title[4096 * 2 + 4000] = 0x33;
        title[4096 * 4 + 3] = 0x44;

        let mut runs = Vec::new();
        let changed = compare_and_copy_slice(title.as_ptr() as usize, &mut held, 8192, &mut runs);
        assert_eq!(changed, 4096 * 3);
        assert_eq!(held, title);
        assert_eq!(
            runs,
            vec![
                (8192 + 4096, 8192 + 4096 * 3),
                (8192 + 4096 * 4, 8192 + 4096 * 5),
            ]
        );
    }

    #[test]
    fn compare_and_copy_slice_skips_multi_page_unmapped_holes_cleanly() {
        use windows_sys::Win32::System::Memory::{
            MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
        };
        // Reserve 16 pages (64 KiB) without committing: an entire unmapped hole.
        // SAFETY: reserves virtual address space and frees it before returning.
        let hole = unsafe { VirtualAlloc(std::ptr::null(), 4096 * 16, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!hole.is_null());

        let mut held = vec![0u8; 4096 * 16];
        // Dirty pages 2, 3, and 11 only.
        held[4096 * 2 + 10] = 0x77;
        held[4096 * 3 + 99] = 0x88;
        held[4096 * 11] = 0x99;

        let mut runs = Vec::new();
        let changed = compare_and_copy_slice(hole as usize, &mut held, 0, &mut runs);
        assert_eq!(changed, 4096 * 3);
        assert!(is_all_zero(&held));
        assert_eq!(runs, vec![(4096 * 2, 4096 * 4), (4096 * 11, 4096 * 12)]);

        // Second pass finds zero changes across all 16 unmapped pages.
        runs.clear();
        let changed2 = compare_and_copy_slice(hole as usize, &mut held, 0, &mut runs);
        // SAFETY: frees the reservation made above.
        unsafe {
            VirtualFree(hole, 0, MEM_RELEASE);
        }
        assert_eq!(changed2, 0);
        assert!(runs.is_empty());
    }

    #[test]
    fn try_copy_matches_memcpy_and_catches_unreadable_memory() {
        use super::try_copy;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        for len in [0usize, 1, 7, 15, 16, 31, 32, 63, 64, 127, 128, 255, 513, 4096, 65536] {
            let src: Vec<u8> = (0..len + 7).map(|i| ((i * 37 + 13) & 0xff) as u8).collect();
            let mut dst = vec![0xCCu8; len + 7];
            // Test unaligned source (+3) and destination (+5) offsets.
            // SAFETY: disjoint valid slices of length `len`.
            let ok = unsafe { try_copy(dst[5..5 + len].as_mut_ptr(), src[3..3 + len].as_ptr(), len) };
            assert!(ok, "len={len}");
            assert_eq!(&dst[5..5 + len], &src[3..3 + len], "len={len}");
            assert_eq!(&dst[..5], &[0xCCu8; 5]);
        }

        let mut dst = vec![0u8; 8192];
        // Null pointer faults cleanly.
        // SAFETY: `dst` is valid writable host memory; `try_copy` catches faults on `src`.
        assert!(!unsafe { try_copy(dst.as_mut_ptr(), std::ptr::null(), 64) });

        // Committed page immediately followed by an uncommitted/PAGE_NOACCESS page.
        // SAFETY: allocates 2 pages (first RW, second reserved/NOACCESS) and frees before return.
        let region = unsafe { VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!region.is_null());
        let first = unsafe { VirtualAlloc(region, 4096, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(first, region);
        unsafe { first.cast::<u8>().write_bytes(0x5A, 4096) };
        // Full 4096-byte copy succeeds.
        assert!(unsafe { try_copy(dst.as_mut_ptr(), region.cast::<u8>(), 4096) });
        assert!(dst[..4096].iter().all(|&b| b == 0x5A));
        // Crossing into the uncommitted second page cleanly returns `false`.
        assert!(!unsafe { try_copy(dst.as_mut_ptr(), region.cast::<u8>(), 4097) });
        unsafe {
            VirtualFree(region, 0, MEM_RELEASE);
        }
    }
}
