//! The Nintendo Switch Pro Controller, read from HID. UNTESTED on hardware:
//! the layouts follow community documentation of the controller's reports, and
//! the stick centre is nominal (no factory calibration is read from the pad).
//!
//! - Over Bluetooth the pad sends the simple report `0x3F` until a title sets
//!   another mode: id, two button bytes, a hat, four 16-bit sticks.
//! - Over USB it stays silent until a handshake (`0x80 0x02`, `0x80 0x04`);
//!   then a subcommand sets the full report `0x30`: id, timer, battery,
//!   three button bytes, two 12-bit sticks packed in three bytes each.
//!
//! The face buttons keep their position, not their name: Nintendo's B (bottom)
//! is Xbox A, Nintendo's A (right) is Xbox B, and likewise X and Y.

#![cfg_attr(not(windows), allow(dead_code))]

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

pub(super) const NINTENDO: u16 = 0x057E;
const PRO_CONTROLLER: u16 = 0x2009;

/// Whether the ids name a Pro Controller.
pub(super) fn is_pro(vendor: u16, product: u16) -> bool {
    vendor == NINTENDO && product == PRO_CONTROLLER
}

/// Output reports that wake a USB pad, in order, and ask for the full report.
pub(super) fn usb_handshake() -> Vec<Vec<u8>> {
    // Neutral rumble data, then subcommand 0x03 (input report mode) = 0x30.
    let mode = vec![
        0x01, 0x00, 0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40, 0x03, 0x30,
    ];
    vec![vec![0x80, 0x02], vec![0x80, 0x04], mode]
}

fn bits(byte: u8, table: &[(u8, u32)]) -> u32 {
    table
        .iter()
        .filter(|(mask, _)| byte & mask != 0)
        .fold(0, |acc, (_, button)| acc | button)
}

fn unit(v: f32) -> f32 {
    v.clamp(-1.0, 1.0)
}

/// A reading from an input report; `None` for any other report.
pub(super) fn parse(report: &[u8]) -> Option<PadState> {
    match (report.first()?, report.len()) {
        (0x3F, 12..) => Some(simple(report)),
        (0x30 | 0x21 | 0x31, 12..) => Some(full(report)),
        _ => None,
    }
}

fn simple(r: &[u8]) -> PadState {
    let mut buttons = bits(
        r[1],
        &[
            (0x01, A),
            (0x02, B),
            (0x04, X),
            (0x08, Y),
            (0x10, LEFT_SHOULDER),
            (0x20, RIGHT_SHOULDER),
        ],
    ) | bits(
        r[2],
        &[
            (0x01, VIEW),
            (0x02, MENU),
            (0x04, LEFT_STICK),
            (0x08, RIGHT_STICK),
        ],
    );
    buttons |= match r[3] {
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
    let axis = |at: usize| {
        let v = u16::from_le_bytes([r[at], r[at + 1]]);
        unit((f32::from(v) - 32768.0) / 32768.0)
    };
    PadState {
        buttons,
        left_trigger: if r[1] & 0x40 != 0 { 1.0 } else { 0.0 },
        right_trigger: if r[1] & 0x80 != 0 { 1.0 } else { 0.0 },
        left_x: axis(4),
        left_y: axis(6),
        right_x: axis(8),
        right_y: axis(10),
    }
}

fn full(r: &[u8]) -> PadState {
    // Right, shared and left button bytes.
    let (right, shared, left) = (r[3], r[4], r[5]);
    let buttons = bits(
        right,
        &[
            (0x01, X),
            (0x02, Y),
            (0x04, A),
            (0x08, B),
            (0x40, RIGHT_SHOULDER),
        ],
    ) | bits(
        shared,
        &[
            (0x01, VIEW),
            (0x02, MENU),
            (0x04, RIGHT_STICK),
            (0x08, LEFT_STICK),
        ],
    ) | bits(
        left,
        &[
            (0x01, DOWN),
            (0x02, UP),
            (0x04, RIGHT),
            (0x08, LEFT),
            (0x40, LEFT_SHOULDER),
        ],
    );
    // 12-bit sticks packed in three bytes; the centre is about 2048 and the
    // reach about 1500.
    let low = |lo: u8, hi: u8| u16::from(lo) | (u16::from(hi & 0x0F) << 8);
    let high = |lo: u8, hi: u8| (u16::from(lo) >> 4) | (u16::from(hi) << 4);
    let norm = |v: u16| unit((f32::from(v) - 2048.0) / 1500.0);
    PadState {
        buttons,
        left_trigger: if left & 0x80 != 0 { 1.0 } else { 0.0 },
        right_trigger: if right & 0x80 != 0 { 1.0 } else { 0.0 },
        left_x: norm(low(r[6], r[7])),
        left_y: norm(high(r[7], r[8])),
        right_x: norm(low(r[9], r[10])),
        right_y: norm(high(r[10], r[11])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_name_the_pro_controller() {
        assert!(is_pro(0x057E, 0x2009));
        assert!(!is_pro(0x057E, 0x2006));
        assert!(!is_pro(0x054C, 0x2009));
    }

    #[test]
    fn simple_report() {
        let mut r = [0u8; 12];
        r[0] = 0x3F;
        r[1] = 0x01 | 0x80; // Nintendo B (bottom) and ZR
        r[2] = 0x02; // plus
        r[3] = 8; // hat centred
        for at in [4, 6, 8, 10] {
            r[at + 1] = 0x80; // 32768
        }
        r[4] = 0xFF; // left stick x just under full right
        r[5] = 0xFF;
        let p = parse(&r).unwrap();
        assert_eq!(p.buttons, A | MENU);
        assert_eq!(p.right_trigger, 1.0);
        assert!(p.left_x > 0.99);
        assert_eq!((p.left_y, p.right_x), (0.0, 0.0));
        r[3] = 2;
        assert_eq!(parse(&r).unwrap().buttons & RIGHT, RIGHT);
    }

    #[test]
    fn full_report() {
        let mut r = [0u8; 13];
        r[0] = 0x30;
        r[3] = 0x08 | 0x40; // A (right, Xbox B) and R
        r[5] = 0x02 | 0x80; // d-pad up and ZL
        // Left stick: x 2048 + 1500, y 2048 - 1500; right stick centred.
        let (x, y) = (3548u16, 548u16);
        r[6] = x as u8;
        r[7] = ((x >> 8) as u8) | ((y as u8 & 0x0F) << 4);
        r[8] = (y >> 4) as u8;
        // Right stick x = y = 2048 (0x800).
        r[9] = 0x00;
        r[10] = 0x08;
        r[11] = 0x80;
        let p = parse(&r).unwrap();
        assert_eq!(p.buttons, B | RIGHT_SHOULDER | UP);
        assert_eq!(p.left_trigger, 1.0);
        assert!((p.left_x - 1.0).abs() < 1e-3 && (p.left_y + 1.0).abs() < 1e-3);
        assert!(p.right_x.abs() < 1e-3 && p.right_y.abs() < 1e-3);
    }

    #[test]
    fn other_reports_are_no_pad_state() {
        assert_eq!(parse(&[0x81, 0, 0]), None);
        assert_eq!(parse(&[0x3F, 0, 0]), None);
        assert_eq!(parse(&[]), None);
    }
}
