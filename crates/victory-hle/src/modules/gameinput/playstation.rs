//! DualSense (PS5) and DualShock 4 (PS4) pads, read straight from HID over USB
//! or Bluetooth, so no XInput translator (DS4Windows, Steam Input) is needed.
//! (Xbox pads already come through XInput.) A reader thread keeps the newest
//! reading of the first pad it finds; `gameinput` lists it in the first host
//! slot XInput leaves empty.
//!
//! Reports (`data` below is the report with its id byte, and for the
//! Bluetooth 0x31 report its extra header byte, removed):
//! - USB `0x01` (64 bytes) and Bluetooth `0x31` (78 bytes): left stick x, y,
//!   right stick x, y, L2, R2, counter, then three button bytes.
//! - Bluetooth `0x01`, the simple report a pad sends until a title asks for
//!   its calibration (we never do): left stick x, y, right stick x, y, three
//!   button bytes, L2, R2.
//! - DualShock 4: the same simple layout in its USB `0x01` (64 bytes) and its
//!   Bluetooth `0x01` (10 bytes) and `0x11` (78 bytes, two header bytes after
//!   the id) reports.
//!
//! Sticks are 0 to 255 with 128 in the middle and up at 0.

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

/// The pads we read, by product id: DualSense, DualSense Edge; DualShock 4
/// (two revisions and the USB adapter).
const PRODUCTS: [(&str, Kind); 5] = [
    ("0ce6", Kind::DualSense),
    ("0df2", Kind::DualSense),
    ("05c4", Kind::DualShock4),
    ("09cc", Kind::DualShock4),
    ("0ba0", Kind::DualShock4),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Kind {
    DualSense,
    DualShock4,
}

/// The report's three button bytes, as the pad lays them out.
struct Buttons {
    face: u8,
    shoulder: u8,
    system: u8,
}

fn stick(v: u8) -> f32 {
    ((f32::from(v) - 128.0) / 127.0).clamp(-1.0, 1.0)
}

fn decode(sticks: [u8; 4], triggers: [u8; 2], b: &Buttons) -> PadState {
    // Hat in the low nibble: 0 up, then clockwise by 45 degrees; 8 is centred.
    let hat = match b.face & 0x0F {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => DOWN | RIGHT,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => UP | LEFT,
        _ => 0,
    };
    let table = [
        (b.face, 0x10, X), // square
        (b.face, 0x20, A), // cross
        (b.face, 0x40, B), // circle
        (b.face, 0x80, Y), // triangle
        (b.shoulder, 0x01, LEFT_SHOULDER),
        (b.shoulder, 0x02, RIGHT_SHOULDER),
        (b.shoulder, 0x10, VIEW), // create
        (b.shoulder, 0x20, MENU), // options
        (b.shoulder, 0x40, LEFT_STICK),
        (b.shoulder, 0x80, RIGHT_STICK),
    ];
    let buttons = table
        .iter()
        .filter(|(byte, mask, _)| byte & mask != 0)
        .fold(hat, |all, (_, _, bit)| all | bit);
    let _ = b.system; // PS and touchpad buttons: the title has no gamepad bit for them
    PadState {
        buttons,
        left_trigger: f32::from(triggers[0]) / 255.0,
        right_trigger: f32::from(triggers[1]) / 255.0,
        left_x: stick(sticks[0]),
        left_y: -stick(sticks[1]),
        right_x: stick(sticks[2]),
        right_y: -stick(sticks[3]),
    }
}

/// Whether the report is a DualShock 4 USB wireless adapter saying that no pad
/// is connected to it. The adapter stays enumerated and keeps sending the
/// 64-byte `0x01` report, mostly zeros, with bit 2 of byte 31 (the id counted
/// as byte 0) set; the bit is always clear for a pad on a cable.
#[cfg_attr(not(windows), allow(dead_code))]
fn gone(kind: Kind, report: &[u8]) -> bool {
    kind == Kind::DualShock4 && report.len() == 64 && report[0] == 0x01 && report[31] & 0x04 != 0
}

/// The reading to publish after `report`, given the one published before: the
/// new state, none when the adapter says its pad left, and the old one for a
/// report that says nothing about the pad.
#[cfg_attr(not(windows), allow(dead_code))]
pub(super) fn after(kind: Kind, report: &[u8], current: Option<PadState>) -> Option<PadState> {
    if gone(kind, report) {
        return None;
    }
    parse(kind, report).or(current)
}

/// A reading from one input report; `None` for a report that is no pad state
/// (feedback, a short or unknown one, an adapter with no pad).
pub(super) fn parse(kind: Kind, report: &[u8]) -> Option<PadState> {
    if gone(kind, report) {
        return None;
    }
    let (&id, rest) = report.split_first()?;
    match (kind, id) {
        // DualSense over USB: the data follows the id.
        (Kind::DualSense, 0x01) if report.len() == 64 => full(rest),
        // DualSense over Bluetooth, full: a header byte follows the id.
        (Kind::DualSense, 0x31) if report.len() >= 12 => full(&rest[1..]),
        // The simple layout: a DualSense over Bluetooth, a DualShock 4 over
        // USB or Bluetooth (the read may be padded to the longest report).
        (_, 0x01) if report.len() >= 10 => Some(simple(rest)),
        // DualShock 4 over Bluetooth, full: two header bytes follow the id.
        (Kind::DualShock4, 0x11) if report.len() >= 14 => Some(simple(&rest[2..])),
        _ => None,
    }
}

fn simple(d: &[u8]) -> PadState {
    decode(
        [d[0], d[1], d[2], d[3]],
        [d[7], d[8]],
        &Buttons {
            face: d[4],
            shoulder: d[5],
            system: d[6],
        },
    )
}

fn full(data: &[u8]) -> Option<PadState> {
    let d = data.get(..10)?;
    Some(decode(
        [d[0], d[1], d[2], d[3]],
        [d[4], d[5]],
        &Buttons {
            face: d[7],
            shoulder: d[8],
            system: d[9],
        },
    ))
}

/// The kind of pad a HID device's vendor and product id name, if it is one we
/// read.
pub(super) fn kind_of(vendor: u16, product: u16) -> Option<Kind> {
    if vendor != SONY {
        return None;
    }
    PRODUCTS
        .iter()
        .find(|(id, _)| u16::from_str_radix(id, 16) == Ok(product))
        .map(|&(_, kind)| kind)
}

/// Sony's USB vendor id.
pub(super) const SONY: u16 = 0x054C;

/// CRC-32 (the zlib one) of `data`, continuing from `crc` (0 to start).
fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut crc = !crc;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}

/// The signature of a Bluetooth output report: the CRC-32 of the seed byte
/// 0xA2 and every byte before the last four.
fn output_crc(report: &[u8]) -> u32 {
    crc32_update(crc32_update(0, &[0xA2]), &report[..report.len() - 4])
}

/// Motor speeds as the pad takes them: 8 bits, from the title's 16.
fn motor(speed: u16) -> u8 {
    (speed >> 8) as u8
}

/// An output report that sets the pad's rumble motors (`low`: the strong
/// one on the left, `high`: the weak one on the right) and nothing else.
/// Layouts after the Linux `hid-playstation` driver.
///
/// - `bluetooth`: the Bluetooth framing (report id and header, CRC-32 at
///   the end) instead of the USB one.
/// - `sequence`: the DualSense's Bluetooth sequence number, 0 to 15, one
///   more each report.
/// - `new_vibration`: a DualSense of firmware update 2.21 or later (and the
///   Edge) takes the improved rumble emulation; older ones the classic bit.
pub(super) fn rumble_report(
    kind: Kind,
    bluetooth: bool,
    low: u16,
    high: u16,
    sequence: u8,
    new_vibration: bool,
) -> Vec<u8> {
    let (left, right) = (motor(low), motor(high));
    match (kind, bluetooth) {
        (Kind::DualSense, false) => {
            let mut r = vec![0u8; 63];
            r[0] = 0x02;
            dualsense_common(&mut r[1..48], left, right, new_vibration);
            r
        }
        (Kind::DualSense, true) => {
            let mut r = vec![0u8; 78];
            r[0] = 0x31;
            r[1] = (sequence & 0x0F) << 4;
            r[2] = 0x10;
            dualsense_common(&mut r[3..50], left, right, new_vibration);
            let crc = output_crc(&r);
            r[74..].copy_from_slice(&crc.to_le_bytes());
            r
        }
        (Kind::DualShock4, false) => {
            let mut r = vec![0u8; 32];
            r[0] = 0x05;
            dualshock4_common(&mut r[1..11], left, right);
            r
        }
        (Kind::DualShock4, true) => {
            let mut r = vec![0u8; 78];
            r[0] = 0x11;
            // The report carries HID data and a CRC.
            r[1] = 0x80 | 0x40;
            dualshock4_common(&mut r[3..13], left, right);
            let crc = output_crc(&r);
            r[74..].copy_from_slice(&crc.to_le_bytes());
            r
        }
    }
}

/// The 47 bytes both DualSense output reports share. Only rumble is set.
fn dualsense_common(c: &mut [u8], left: u8, right: u8, new_vibration: bool) {
    // Select classic rumble style haptics and enable it.
    c[0] = 0x02;
    if new_vibration {
        c[38] = 0x04;
    } else {
        c[0] |= 0x01;
    }
    c[2] = right;
    c[3] = left;
}

/// The 10 bytes both DualShock 4 output reports share. Only rumble is set;
/// some third-party pads want the lightbar in the same report.
fn dualshock4_common(c: &mut [u8], left: u8, right: u8) {
    c[0] = 0x01;
    c[3] = right;
    c[4] = left;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usb(sticks: [u8; 4], triggers: [u8; 2], face: u8, shoulder: u8) -> Vec<u8> {
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&sticks);
        r[5..7].copy_from_slice(&triggers);
        r[8] = face;
        r[9] = shoulder;
        r
    }

    #[test]
    fn centred_pad_reads_neutral() {
        let pad = parse(Kind::DualSense, &usb([128; 4], [0, 0], 0x08, 0)).unwrap();
        assert_eq!(pad, PadState::default());
    }

    #[test]
    fn usb_buttons_sticks_and_triggers() {
        // cross + circle, L1 + options, hat up-right; left stick left, right stick up
        let pad = parse(
            Kind::DualSense,
            &usb([0, 128, 128, 0], [255, 51], 0x61, 0x21),
        )
        .unwrap();
        assert_eq!(pad.buttons, A | B | UP | RIGHT | LEFT_SHOULDER | MENU);
        assert_eq!(pad.left_x, -1.0);
        assert_eq!(pad.left_y, 0.0);
        assert_eq!(pad.right_y, 1.0);
        assert_eq!(pad.left_trigger, 1.0);
        assert!((pad.right_trigger - 0.2).abs() < 1e-6);
    }

    #[test]
    fn bluetooth_full_report_has_a_header_byte() {
        let mut r = vec![0u8; 78];
        r[0] = 0x31;
        r[2..6].copy_from_slice(&[255, 128, 128, 255]);
        r[6..8].copy_from_slice(&[10, 20]);
        r[9] = 0x08 | 0x80; // triangle, hat centred
        r[10] = 0x10 | 0x80; // create + R3
        let pad = parse(Kind::DualSense, &r).unwrap();
        assert_eq!(pad.buttons, Y | VIEW | RIGHT_STICK);
        assert_eq!(pad.left_x, 1.0);
        assert_eq!(pad.right_y, -1.0);
        assert!((pad.left_trigger - 10.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn bluetooth_simple_report_with_and_without_padding() {
        let mut r = vec![0u8; 10];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&[128, 0, 128, 128]);
        r[5] = 0x08 | 0x10; // square, hat centred
        r[6] = 0x02; // R1
        r[8] = 255;
        let pad = parse(Kind::DualSense, &r).unwrap();
        assert_eq!(pad.buttons, X | RIGHT_SHOULDER);
        assert_eq!(pad.left_y, 1.0);
        assert_eq!(pad.left_trigger, 1.0);
        r.resize(78, 0);
        assert_eq!(parse(Kind::DualSense, &r).unwrap(), pad);
    }

    #[test]
    fn every_hat_position() {
        let want = [
            UP,
            UP | RIGHT,
            RIGHT,
            DOWN | RIGHT,
            DOWN,
            DOWN | LEFT,
            LEFT,
            UP | LEFT,
            0,
        ];
        for (hat, buttons) in want.into_iter().enumerate() {
            assert_eq!(
                parse(Kind::DualSense, &usb([128; 4], [0, 0], hat as u8, 0))
                    .unwrap()
                    .buttons,
                buttons
            );
        }
    }

    #[test]
    fn other_reports_are_no_pad_state() {
        assert_eq!(parse(Kind::DualSense, &[]), None);
        assert_eq!(parse(Kind::DualSense, &[0x02; 78]), None);
        assert_eq!(parse(Kind::DualSense, &[0x01; 5]), None);
    }

    #[test]
    fn dualshock_4_reports() {
        // USB: 64 bytes, the simple layout. Square + options, hat down, L2 half.
        let mut r = vec![0u8; 64];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&[128, 128, 255, 128]);
        r[5] = 0x04 | 0x10;
        r[6] = 0x20;
        r[8] = 128;
        let usb = parse(Kind::DualShock4, &r).unwrap();
        assert_eq!(usb.buttons, X | DOWN | MENU);
        assert_eq!(usb.right_x, 1.0);
        assert!((usb.left_trigger - 128.0 / 255.0).abs() < 1e-6);
        // Bluetooth full: two header bytes after the id, then the same.
        let mut b = vec![0u8; 78];
        b[0] = 0x11;
        b[3..7].copy_from_slice(&[128, 128, 255, 128]);
        b[7] = 0x04 | 0x10;
        b[8] = 0x20;
        b[10] = 128;
        assert_eq!(parse(Kind::DualShock4, &b).unwrap(), usb);
        // The same 64 bytes read as a DualSense are another layout.
        assert_ne!(parse(Kind::DualSense, &r).unwrap(), usb);
    }

    #[test]
    fn ids_name_the_pad() {
        assert_eq!(kind_of(0x054C, 0x0CE6), Some(Kind::DualSense));
        assert_eq!(kind_of(0x054C, 0x0DF2), Some(Kind::DualSense));
        assert_eq!(kind_of(0x054C, 0x05C4), Some(Kind::DualShock4));
        assert_eq!(kind_of(0x054C, 0x09CC), Some(Kind::DualShock4));
        assert_eq!(kind_of(0x054C, 0x0BA0), Some(Kind::DualShock4));
        assert_eq!(kind_of(0x054C, 0x0000), None);
        assert_eq!(kind_of(0x046D, 0x0CE6), None);
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32_update(0, b"123456789"), 0xCBF4_3926);
        // Chunks continue each other.
        assert_eq!(crc32_update(crc32_update(0, b"1234"), b"56789"), 0xCBF4_3926);
    }

    #[test]
    fn motors_scale_to_eight_bits() {
        assert_eq!((motor(0), motor(0x00FF), motor(0x0100), motor(0xFFFF)), (0, 0, 1, 255));
    }

    /// The signature a Bluetooth report carries is the one its other bytes give.
    fn signed(r: &[u8]) -> bool {
        r[r.len() - 4..] == output_crc(r).to_le_bytes()
    }

    #[test]
    fn dualsense_usb_rumble_report() {
        let r = rumble_report(Kind::DualSense, false, 0xFF00, 0x8000, 0, true);
        assert_eq!(r.len(), 63);
        assert_eq!(r[0], 0x02);
        // Haptics select in flag 0, improved vibration in flag 2; right motor
        // before left in the common block.
        assert_eq!((r[1], r[2], r[3], r[4]), (0x02, 0, 0x80, 0xFF));
        assert_eq!(r[1 + 38], 0x04);
        // Nothing else is touched: no lightbar, LED, or audio flag.
        assert!(r[5..39].iter().chain(&r[40..]).all(|&b| b == 0));
        let old = rumble_report(Kind::DualSense, false, 0, 0, 0, false);
        assert_eq!((old[1], old[1 + 38]), (0x03, 0));
    }

    #[test]
    fn dualsense_bluetooth_rumble_report() {
        let r = rumble_report(Kind::DualSense, true, 0x4000, 0xC000, 5, true);
        assert_eq!(r.len(), 78);
        assert_eq!((r[0], r[1], r[2]), (0x31, 0x50, 0x10));
        assert_eq!((r[3], r[5], r[6]), (0x02, 0xC0, 0x40));
        assert_eq!(r[3 + 38], 0x04);
        assert!(signed(&r));
        // The sequence number is the high nibble and wraps.
        assert_eq!(rumble_report(Kind::DualSense, true, 0, 0, 17, true)[1], 0x10);
    }

    #[test]
    fn dualshock_4_rumble_reports() {
        let usb = rumble_report(Kind::DualShock4, false, 0xFF00, 0x2000, 0, false);
        assert_eq!(usb.len(), 32);
        assert_eq!((usb[0], usb[1], usb[4], usb[5]), (0x05, 0x01, 0x20, 0xFF));
        let bt = rumble_report(Kind::DualShock4, true, 0xFF00, 0x2000, 0, false);
        assert_eq!(bt.len(), 78);
        assert_eq!((bt[0], bt[1], bt[3], bt[6], bt[7]), (0x11, 0xC0, 0x01, 0x20, 0xFF));
        assert!(signed(&bt));
    }
}
