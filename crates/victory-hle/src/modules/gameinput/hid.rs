//! The host's HID gamepads that XInput does not cover, read on one thread and
//! offered to `gameinput` as a pad in the first slot XInput leaves empty. One
//! HID pad at a time: the first device found, native drivers (`playstation`)
//! before anything generic.
//!
//! Reading is overlapped with a short wait, so a device that never sends a
//! report cannot pin the thread, and rumble goes out between reads over a
//! second handle of the same device (a write on the handle with a read
//! pending would wait behind the read).

use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    CM_GET_DEVICE_INTERFACE_LIST_PRESENT, CM_Get_Device_Interface_List_SizeW,
    CM_Get_Device_Interface_ListW, CR_SUCCESS,
};
use windows_sys::Win32::Devices::HumanInterfaceDevice::{
    HIDP_CAPS, HIDP_VALUE_CAPS, HidD_FreePreparsedData, HidD_GetFeature, HidD_GetHidGuid,
    HidD_GetPreparsedData, HidP_GetCaps, HidP_GetUsageValue, HidP_GetUsages, HidP_GetValueCaps,
    HidP_Input,
};
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_IO_PENDING, GENERIC_READ, GENERIC_WRITE, GetLastError, HANDLE,
    INVALID_HANDLE_VALUE, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
    WriteFile,
};
use windows_sys::Win32::System::IO::{CancelIoEx, GetOverlappedResult, OVERLAPPED};
use windows_sys::Win32::System::Threading::{CreateEventW, ResetEvent, WaitForSingleObject};

use super::{PadState, generic, nintendo, playstation};

/// How often to look for a pad while none is open.
const RESCAN: Duration = Duration::from_secs(1);
/// How long a read waits before the thread looks at rumble and the device.
const WAIT_MS: u32 = 20;

/// A reading and the pad's vendor and product ids.
pub(super) type Reading = (PadState, (u16, u16));

static LATEST: Mutex<Option<Reading>> = Mutex::new(None);
/// What the title asked of the rumble motors: low frequency, high frequency.
static RUMBLE: Mutex<(u16, u16)> = Mutex::new((0, 0));

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The pad's newest reading while one is open; starts the reader thread the
/// first time it is asked.
pub(super) fn connected_pad() -> Option<Reading> {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        if let Err(e) = std::thread::Builder::new()
            .name("hid-pad".into())
            .spawn(run)
        {
            tracing::warn!(error = %e, "no HID pad reader thread");
        }
    });
    *lock(&LATEST)
}

/// Rumble for the open pad (the title's 16-bit speeds); sent by the reader
/// thread, which sees it within a few milliseconds.
pub(super) fn set_rumble(low: u16, high: u16) {
    *lock(&RUMBLE) = (low, high);
}

/// What drives a device.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Driver {
    Sony(playstation::Kind),
    SwitchPro,
    /// Any other gamepad or joystick; mapped by guess (`generic`).
    Generic,
}

impl Driver {
    /// Native drivers come first when several pads are connected.
    fn rank(self) -> u8 {
        u8::from(self == Driver::Generic)
    }
}

struct Found {
    path: Vec<u16>,
    driver: Driver,
    vendor: u16,
    product: u16,
    bluetooth: bool,
}

/// `vid_054c&pid_0ce6` over USB, `vid&0002054c_pid&0ce6` over Bluetooth.
fn ids_of(path: &str) -> Option<(u16, u16)> {
    let path = path.to_ascii_lowercase();
    let hex = |at: usize| u16::from_str_radix(path.get(at..at + 4)?, 16).ok();
    let vid = path
        .find("vid_")
        .map(|i| i + 4)
        .or_else(|| path.find("vid&").map(|i| i + 4 + 4))?;
    let pid = path
        .find("pid_")
        .map(|i| i + 4)
        .or_else(|| path.find("pid&").map(|i| i + 4))?;
    Some((hex(vid)?, hex(pid)?))
}

/// Bluetooth HID devices carry the HID service's UUID in their path.
fn is_bluetooth(path: &str) -> bool {
    path.to_ascii_lowercase()
        .contains("{00001124-0000-1000-8000-00805f9b34fb}")
}

/// Whether a device is one we leave to someone else: XInput pads (their
/// path carries `ig_`), Microsoft's own, and Steam's virtual pads.
fn excluded(path: &str, vendor: u16) -> bool {
    path.to_ascii_lowercase().contains("ig_") || vendor == 0x045E || vendor == 0x28DE
}

/// `VICTORY_HID_GENERIC=1` reads every pad through the generic mapping, to
/// check that path with a pad whose layout is known.
fn forced_generic() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HID_GENERIC").is_some())
}

/// Whether the device presents itself as a gamepad or joystick.
fn is_gamepad(path: &[u16]) -> bool {
    let handle = open(path, 0, 0);
    if handle == INVALID_HANDLE_VALUE {
        return false;
    }
    let usage = usage_of(handle);
    // SAFETY: our handle.
    unsafe { CloseHandle(handle) };
    // Generic Desktop: Joystick (4) or Game Pad (5).
    matches!(usage, Some((1, 4 | 5)))
}

/// The top-level usage page and usage of a device.
fn usage_of(handle: HANDLE) -> Option<(u16, u16)> {
    // SAFETY: preparsed data from this handle, freed after the query.
    unsafe {
        let mut pp = 0isize;
        if !HidD_GetPreparsedData(handle, &mut pp) {
            return None;
        }
        let mut caps: HIDP_CAPS = std::mem::zeroed();
        let ok = HidP_GetCaps(pp, &mut caps) >= 0;
        HidD_FreePreparsedData(pp);
        ok.then_some((caps.UsagePage, caps.Usage))
    }
}

/// The connected devices we can read, native drivers first. `skip` holds the
/// paths already found to be no gamepad, so they are not opened again.
fn devices(skip: &mut std::collections::HashSet<String>) -> Vec<Found> {
    // SAFETY: plain queries into buffers we size from the answer.
    unsafe {
        let mut guid = std::mem::zeroed();
        HidD_GetHidGuid(&mut guid);
        let mut len = 0u32;
        let flags = CM_GET_DEVICE_INTERFACE_LIST_PRESENT;
        if CM_Get_Device_Interface_List_SizeW(&mut len, &guid, std::ptr::null(), flags)
            != CR_SUCCESS
            || len < 2
        {
            return Vec::new();
        }
        let mut buf = vec![0u16; len as usize];
        if CM_Get_Device_Interface_ListW(&guid, std::ptr::null(), buf.as_mut_ptr(), len, flags)
            != CR_SUCCESS
        {
            return Vec::new();
        }
        buf.split(|&c| c == 0)
            .filter(|p| !p.is_empty())
            .filter_map(|p| {
                let text = String::from_utf16_lossy(p);
                let (vendor, product) = ids_of(&text)?;
                let path: Vec<u16> = p.iter().copied().chain(Some(0)).collect();
                let native = if forced_generic() {
                    None
                } else if let Some(kind) = playstation::kind_of(vendor, product) {
                    Some(Driver::Sony(kind))
                } else {
                    nintendo::is_pro(vendor, product).then_some(Driver::SwitchPro)
                };
                let driver = match native {
                    Some(driver) => driver,
                    None if excluded(&text, vendor) || skip.contains(&text) => return None,
                    None if is_gamepad(&path) => Driver::Generic,
                    None => {
                        skip.insert(text);
                        return None;
                    }
                };
                Some(Found {
                    path,
                    driver,
                    vendor,
                    product,
                    bluetooth: is_bluetooth(&text),
                })
            })
            .collect()
    }
}

fn run() {
    let mut skip = std::collections::HashSet::new();
    loop {
        let mut found = devices(&mut skip);
        found.sort_by_key(|f| f.driver.rank());
        for found in found {
            if read_pad(&found) {
                break;
            }
        }
        *lock(&LATEST) = None;
        std::thread::sleep(RESCAN);
    }
}

fn open(path: &[u16], access: u32, flags: u32) -> HANDLE {
    // SAFETY: a NUL-terminated path.
    unsafe {
        CreateFileW(
            path.as_ptr(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_EXISTING,
            flags,
            std::ptr::null_mut(),
        )
    }
}

/// The report sizes of a device: input, output.
fn report_sizes(handle: HANDLE) -> Option<(usize, usize)> {
    // SAFETY: preparsed data from this handle, freed after the query.
    unsafe {
        let mut pp = 0isize;
        if !HidD_GetPreparsedData(handle, &mut pp) {
            return None;
        }
        let mut caps: HIDP_CAPS = std::mem::zeroed();
        let ok = HidP_GetCaps(pp, &mut caps) >= 0;
        HidD_FreePreparsedData(pp);
        ok.then(|| {
            (
                usize::from(caps.InputReportByteLength),
                usize::from(caps.OutputReportByteLength),
            )
        })
    }
}

/// A feature report, `size` bytes with `id` first; `None` when the device
/// does not answer.
fn feature(handle: HANDLE, id: u8, size: usize) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; size];
    buf[0] = id;
    // SAFETY: a buffer of `size` bytes.
    let ok = unsafe { HidD_GetFeature(handle, buf.as_mut_ptr().cast(), size as u32) };
    ok.then_some(buf)
}

/// Writes an output report over the synchronous handle.
fn write(control: HANDLE, report: &[u8]) -> u32 {
    let mut written = 0u32;
    // SAFETY: the report's bytes and a synchronous handle of our own.
    let ok = unsafe {
        WriteFile(
            control,
            report.as_ptr(),
            report.len() as u32,
            &mut written,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        // SAFETY: a plain query.
        let error = unsafe { GetLastError() };
        tracing::debug!(error, "HID write failed");
        return error;
    }
    0
}

/// What turns a pad's reports into readings.
enum Decoder {
    Sony(playstation::Kind),
    SwitchPro,
    Generic(GenericPad),
}

impl Decoder {
    /// The reading after `report`, given the one before.
    fn after(&self, report: &[u8], current: Option<PadState>) -> Option<PadState> {
        match self {
            Decoder::Sony(kind) => playstation::after(*kind, report, current),
            Decoder::SwitchPro => nintendo::parse(report).or(current),
            Decoder::Generic(pad) => pad.read(report).or(current),
        }
    }
}

/// An axis of a generic pad: where it lives in a report and its range.
struct ValueCap {
    link: u16,
    usage: u16,
    min: i64,
    max: i64,
    bits: u16,
}

/// A generic pad's report layout, from its HID descriptor.
struct GenericPad {
    preparsed: isize,
    values: Vec<ValueCap>,
}

impl GenericPad {
    fn new(handle: HANDLE) -> Option<Self> {
        // SAFETY: preparsed data owned by the result (freed in `drop`), and
        // buffers sized by the counts the API gave.
        unsafe {
            let mut preparsed = 0isize;
            if !HidD_GetPreparsedData(handle, &mut preparsed) {
                return None;
            }
            let mut pad = GenericPad {
                preparsed,
                values: Vec::new(),
            };
            let mut caps: HIDP_CAPS = std::mem::zeroed();
            if HidP_GetCaps(preparsed, &mut caps) < 0 {
                return None;
            }
            let mut n = caps.NumberInputValueCaps;
            let mut raw = vec![HIDP_VALUE_CAPS::default(); usize::from(n)];
            if n > 0 && HidP_GetValueCaps(HidP_Input, raw.as_mut_ptr(), &mut n, preparsed) < 0 {
                return None;
            }
            for c in &raw[..usize::from(n)] {
                if c.UsagePage != 1 {
                    continue;
                }
                let (first, last) = if c.IsRange {
                    (c.Anonymous.Range.UsageMin, c.Anonymous.Range.UsageMax)
                } else {
                    (c.Anonymous.NotRange.Usage, c.Anonymous.NotRange.Usage)
                };
                for usage in first..=last {
                    pad.values.push(ValueCap {
                        link: c.LinkCollection,
                        usage,
                        min: i64::from(c.LogicalMin),
                        max: i64::from(c.LogicalMax),
                        bits: c.BitSize,
                    });
                }
            }
            Some(pad)
        }
    }

    /// A reading from an input report (its id byte first); `None` for a
    /// report this layout does not describe.
    fn read(&self, report: &[u8]) -> Option<PadState> {
        const SUCCESS: i32 = 0x0011_0000;
        let mut data = report.to_vec();
        let len = data.len() as u32;
        let mut usages = [0u16; 64];
        let mut count = usages.len() as u32;
        // SAFETY: the report copy and buffers outlive the calls.
        let status = unsafe {
            HidP_GetUsages(
                HidP_Input,
                9,
                0,
                usages.as_mut_ptr(),
                &mut count,
                self.preparsed,
                data.as_mut_ptr().cast(),
                len,
            )
        };
        if status != SUCCESS {
            return None;
        }
        let mut axes = Vec::new();
        let mut hat = None;
        for cap in &self.values {
            let mut raw = 0u32;
            // SAFETY: as above.
            let status = unsafe {
                HidP_GetUsageValue(
                    HidP_Input,
                    1,
                    cap.link,
                    cap.usage,
                    &mut raw,
                    self.preparsed,
                    data.as_mut_ptr().cast(),
                    len,
                )
            };
            if status != SUCCESS {
                continue;
            }
            let value = if cap.min < 0 && (1..32).contains(&cap.bits) {
                let shift = 32 - u32::from(cap.bits);
                i64::from(((raw << shift) as i32) >> shift)
            } else {
                i64::from(raw)
            };
            if cap.usage == generic::HAT {
                hat = Some((value, cap.min, cap.max));
            } else {
                axes.push(generic::Axis {
                    usage: cap.usage,
                    value,
                    min: cap.min,
                    max: cap.max,
                });
            }
        }
        Some(generic::map(&usages[..count as usize], hat, &axes))
    }
}

impl Drop for GenericPad {
    fn drop(&mut self) {
        // SAFETY: the data from `new`.
        unsafe { HidD_FreePreparsedData(self.preparsed) };
    }
}

/// Sends rumble to a Sony pad.
struct Rumble {
    control: HANDLE,
    kind: playstation::Kind,
    bluetooth: bool,
    new_vibration: bool,
    output_len: usize,
    sequence: u8,
    sent: (u16, u16),
    /// The error of the last write, 0 for none.
    last_error: u32,
}

impl Rumble {
    fn send(&mut self, low: u16, high: u16) {
        let mut report = playstation::rumble_report(
            self.kind,
            self.bluetooth,
            low,
            high,
            self.sequence,
            self.new_vibration,
        );
        self.sequence = (self.sequence + 1) & 0x0F;
        // The device wants its own report size.
        report.resize(self.output_len.max(report.len()), 0);
        report.truncate(self.output_len.max(1));
        self.last_error = write(self.control, &report);
        self.sent = (low, high);
    }

    /// Sends what the title wants if it changed.
    fn update(&mut self) {
        let wanted = *lock(&RUMBLE);
        if wanted != self.sent {
            self.send(wanted.0, wanted.1);
        }
    }
}

/// Reads one pad until it fails (unplugged, switched off); whether it could be
/// opened and read at all.
fn read_pad(found: &Found) -> bool {
    let read = open(&found.path, GENERIC_READ, FILE_FLAG_OVERLAPPED);
    if read == INVALID_HANDLE_VALUE {
        return false;
    }
    // Feature reports and writes: a second, synchronous handle.
    let mut control = open(&found.path, GENERIC_READ | GENERIC_WRITE, 0);
    if control == INVALID_HANDLE_VALUE {
        control = open(&found.path, GENERIC_WRITE, 0);
    }
    let close_all = |read: HANDLE, control: HANDLE| {
        // SAFETY: our handles.
        unsafe {
            CloseHandle(read);
            if control != INVALID_HANDLE_VALUE {
                CloseHandle(control);
            }
        }
    };
    let Some((size, output_len)) = report_sizes(read).filter(|&(input, _)| input > 0) else {
        close_all(read, control);
        return false;
    };
    let mut rumble = None;
    let decoder = match found.driver {
        Driver::Sony(kind) => {
            if control != INVALID_HANDLE_VALUE {
                // Over Bluetooth a pad sends a short report, and ignores
                // output, until a title reads its calibration feature report.
                if found.bluetooth || kind == playstation::Kind::DualSense {
                    let _ = feature(control, 0x05, 41);
                }
                // Classic vibration: felt clearly stronger on a DualSense
                // (firmware that takes the improved emulation too) than the
                // improved emulation was.
                let new_vibration = false;
                if output_len > 0 {
                    rumble = Some(Rumble {
                        control,
                        kind,
                        bluetooth: found.bluetooth,
                        new_vibration,
                        output_len,
                        sequence: 0,
                        sent: (0, 0),
                        last_error: 0,
                    });
                }
            }
            Decoder::Sony(kind)
        }
        Driver::SwitchPro => {
            // Over USB the pad waits for a handshake; over Bluetooth it
            // already sends its simple report.
            if !found.bluetooth && control != INVALID_HANDLE_VALUE {
                for mut report in nintendo::usb_handshake() {
                    report.resize(output_len.max(report.len()), 0);
                    write(control, &report);
                    std::thread::sleep(Duration::from_millis(30));
                }
            }
            Decoder::SwitchPro
        }
        Driver::Generic => match GenericPad::new(read) {
            Some(pad) => Decoder::Generic(pad),
            None => {
                close_all(read, control);
                return false;
            }
        },
    };
    tracing::info!(
        driver = ?found.driver,
        vendor = found.vendor,
        product = found.product,
        bluetooth = found.bluetooth,
        report_bytes = size,
        rumble = rumble.is_some(),
        "HID pad connected"
    );
    let mut buf = vec![0u8; size];
    let mut state: Option<PadState> = None;
    let mut first = true;
    // SAFETY: an event of our own, closed below.
    let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
    'reading: loop {
        let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
        overlapped.hEvent = event;
        // SAFETY: the buffer and `overlapped` outlive the read: it is waited
        // for or cancelled and waited for before they go out of scope.
        let started = unsafe {
            ResetEvent(event);
            ReadFile(
                read,
                buf.as_mut_ptr().cast(),
                size as u32,
                std::ptr::null_mut(),
                &mut overlapped,
            )
        };
        // SAFETY: a plain query.
        if started == 0 && unsafe { GetLastError() } != ERROR_IO_PENDING {
            break;
        }
        loop {
            // SAFETY: our event.
            match unsafe { WaitForSingleObject(event, WAIT_MS) } {
                WAIT_OBJECT_0 => break,
                WAIT_TIMEOUT => {
                    if let Some(r) = rumble.as_mut() {
                        r.update();
                    }
                }
                _ => {
                    cancel(read, &mut overlapped);
                    break 'reading;
                }
            }
        }
        let mut got = 0u32;
        // SAFETY: the finished read of this handle.
        if unsafe { GetOverlappedResult(read, &overlapped, &mut got, 0) } == 0 {
            break;
        }
        let report = &buf[..got as usize];
        if first {
            tracing::info!(
                id = report.first().copied().unwrap_or(0),
                bytes = got,
                "HID pad first report"
            );
            first = false;
        }
        let now = decoder.after(report, state);
        if state.is_some() && now.is_none() {
            tracing::info!("HID adapter: its pad left");
        }
        state = now;
        *lock(&LATEST) = state.map(|pad| (pad, (found.vendor, found.product)));
        if let Some(r) = rumble.as_mut() {
            r.update();
        }
    }
    // Leave the motors still, then let go.
    if let Some(r) = rumble.as_mut()
        && r.sent != (0, 0)
    {
        r.send(0, 0);
    }
    tracing::info!("HID pad disconnected");
    // SAFETY: our event.
    unsafe { CloseHandle(event) };
    close_all(read, control);
    true
}

/// Cancels the read of `overlapped` and waits for it to finish, so its buffer
/// can go.
fn cancel(handle: HANDLE, overlapped: &mut OVERLAPPED) {
    let mut got = 0u32;
    // SAFETY: the pending read of this handle.
    unsafe {
        CancelIoEx(handle, overlapped);
        GetOverlappedResult(handle, overlapped, &mut got, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_come_out_of_usb_and_bluetooth_paths() {
        assert_eq!(
            ids_of(r"\\?\hid#vid_054c&pid_0ce6#7&1#{4d1e55b2}"),
            Some((0x054C, 0x0CE6))
        );
        assert_eq!(
            ids_of(r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&05c4#a"),
            Some((0x054C, 0x05C4))
        );
        assert_eq!(ids_of(r"\\?\hid#nothing"), None);
    }

    /// Hardware check, run by hand with a pad connected and its buttons
    /// pressed: `cargo test -p victory-hle generic_mapping_of_a_real_pad -- --ignored --nocapture`.
    /// Reads the first gamepad through the generic mapping and prints what it
    /// sees for about three seconds.
    #[test]
    #[ignore = "needs a pad"]
    fn generic_mapping_of_a_real_pad() {
        let mut skip = std::collections::HashSet::new();
        // SAFETY: set before any other thread reads it; run on its own.
        unsafe { std::env::set_var("VICTORY_HID_GENERIC", "1") };
        let found = devices(&mut skip)
            .into_iter()
            .next()
            .expect("no gamepad connected");
        let handle = open(&found.path, GENERIC_READ, 0);
        assert_ne!(handle, INVALID_HANDLE_VALUE);
        let pad = GenericPad::new(handle).expect("no layout");
        println!(
            "axes: {:?}",
            pad.values
                .iter()
                .map(|v| (v.usage, v.min, v.max))
                .collect::<Vec<_>>()
        );
        let (size, _) = report_sizes(handle).unwrap();
        let mut buf = vec![0u8; size];
        let end = std::time::Instant::now() + Duration::from_secs(3);
        while std::time::Instant::now() < end {
            let mut got = 0u32;
            // SAFETY: a synchronous read into our buffer.
            let ok = unsafe {
                ReadFile(
                    handle,
                    buf.as_mut_ptr().cast(),
                    size as u32,
                    &mut got,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 {
                println!("read failed: {}", unsafe { GetLastError() });
                break;
            }
            if got > 0 && buf[0] != 0 {
                println!("report {:#x} len {got}", buf[0]);
            }
            {
                if let Some(p) = pad.read(&buf[..got as usize]) {
                    println!("{p:?}");
                }
            }
        }
    }

    /// Hardware check: vibrates the connected PlayStation pad for about a
    /// second (strong motor, then weak, then both). Run by hand and feel it:
    /// `cargo test -p victory-hle rumble_of_a_real_pad -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs a PlayStation pad"]
    fn rumble_of_a_real_pad() {
        let mut skip = std::collections::HashSet::new();
        let all = devices(&mut skip);
        println!(
            "{:?}",
            all.iter()
                .map(|f| (f.driver, f.vendor, f.product))
                .collect::<Vec<_>>()
        );
        let found = all
            .into_iter()
            .find(|f| matches!(f.driver, Driver::Sony(_)))
            .expect("no PlayStation pad connected");
        let Driver::Sony(kind) = found.driver else {
            unreachable!()
        };
        let control = open(&found.path, GENERIC_READ | GENERIC_WRITE, 0);
        assert_ne!(control, INVALID_HANDLE_VALUE, "cannot open for writing");
        let (_, output_len) = report_sizes(control).unwrap();
        if found.bluetooth || kind == playstation::Kind::DualSense {
            let _ = feature(control, 0x05, 41);
        }
        let new_vibration = false;
        println!(
            "{kind:?} bluetooth={} output_len={output_len} new_vibration={new_vibration}",
            found.bluetooth
        );
        let mut rumble = Rumble {
            control,
            kind,
            bluetooth: found.bluetooth,
            new_vibration,
            output_len,
            sequence: 0,
            sent: (0, 0),
            last_error: 0,
        };
        for (what, low, high) in [
            ("strong (left)", 0xFFFF, 0),
            ("weak (right)", 0, 0xFFFF),
            ("both", 0xFFFF, 0xFFFF),
        ] {
            println!("{what}");
            rumble.send(low, high);
            println!("write error: {}", rumble.last_error);
            std::thread::sleep(Duration::from_millis(700));
            rumble.send(0, 0);
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    #[test]
    fn bluetooth_paths_are_told_apart() {
        assert!(is_bluetooth(
            r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&0ce6#a"
        ));
        assert!(!is_bluetooth(r"\\?\hid#vid_054c&pid_0ce6#7&1"));
    }
}
