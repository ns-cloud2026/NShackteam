//! Generic HID gamepads and joysticks, mapped by guesswork: HID says which
//! usages a device has, not which one is the A button. The mapping is a pure
//! function of what the report held, so it is tested without a device.
//!
//! The guess, after the layout most cheap pads share:
//! - buttons 1 to 10: A, B, X, Y, left and right shoulder, view (back), menu
//!   (start), left and right stick click;
//! - the hat switch: the D-pad (0 up, clockwise, eight positions);
//! - X and Y: the left stick (Y grows downward in HID);
//! - Rx and Ry: the right stick, and then Z and Rz the triggers; with no Rx
//!   and Ry, Z and Rz are the right stick and there are no analogue triggers.

#![cfg_attr(not(windows), allow(dead_code))]

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

/// Generic Desktop usages.
pub(super) const X_AXIS: u16 = 0x30;
pub(super) const Y_AXIS: u16 = 0x31;
pub(super) const Z_AXIS: u16 = 0x32;
pub(super) const RX_AXIS: u16 = 0x33;
pub(super) const RY_AXIS: u16 = 0x34;
pub(super) const RZ_AXIS: u16 = 0x35;
pub(super) const HAT: u16 = 0x39;

/// One axis of a report: its usage, the value, and the logical range.
#[derive(Clone, Copy, Debug)]
pub(super) struct Axis {
    pub usage: u16,
    pub value: i64,
    pub min: i64,
    pub max: i64,
}

const BUTTONS: [u32; 10] = [
    A,
    B,
    X,
    Y,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    VIEW,
    MENU,
    LEFT_STICK,
    RIGHT_STICK,
];

/// `value` in `min..=max` as 0 to 1.
fn unit(a: &Axis) -> f32 {
    if a.max <= a.min {
        return 0.0;
    }
    (((a.value - a.min) as f32) / ((a.max - a.min) as f32)).clamp(0.0, 1.0)
}

/// `value` as -1 to 1; `down_is_positive` flips it, so up is positive.
fn stick(a: &Axis, down_is_positive: bool) -> f32 {
    let v = unit(a) * 2.0 - 1.0;
    // A small dead zone: cheap pads rest a little off centre.
    let v = if v.abs() < 0.08 { 0.0 } else { v };
    if down_is_positive { -v } else { v }
}

/// The D-pad of a hat value; nothing out of range (the released state).
fn hat(value: i64, min: i64, max: i64) -> u32 {
    let pos = value - min;
    if value > max || !(0..8).contains(&pos) {
        return 0;
    }
    [
        UP,
        UP | RIGHT,
        RIGHT,
        DOWN | RIGHT,
        DOWN,
        DOWN | LEFT,
        LEFT,
        UP | LEFT,
    ][pos as usize]
}

/// A reading from the pressed button usages (page 9, 1 based), the hat
/// (value, min, max) if there is one, and the axes.
pub(super) fn map(pressed: &[u16], hat_value: Option<(i64, i64, i64)>, axes: &[Axis]) -> PadState {
    let mut pad = PadState::default();
    for &n in pressed {
        if let Some(&button) = BUTTONS.get(usize::from(n).wrapping_sub(1)) {
            pad.buttons |= button;
        }
    }
    if let Some((v, min, max)) = hat_value {
        pad.buttons |= hat(v, min, max);
    }
    let find = |usage| axes.iter().find(|a| a.usage == usage);
    if let (Some(x), Some(y)) = (find(X_AXIS), find(Y_AXIS)) {
        pad.left_x = stick(x, false);
        pad.left_y = stick(y, true);
    }
    if let (Some(rx), Some(ry)) = (find(RX_AXIS), find(RY_AXIS)) {
        pad.right_x = stick(rx, false);
        pad.right_y = stick(ry, true);
        pad.left_trigger = find(Z_AXIS).map_or(0.0, unit);
        pad.right_trigger = find(RZ_AXIS).map_or(0.0, unit);
    } else if let (Some(z), Some(rz)) = (find(Z_AXIS), find(RZ_AXIS)) {
        pad.right_x = stick(z, false);
        pad.right_y = stick(rz, true);
    }
    pad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(usage: u16, value: i64) -> Axis {
        Axis {
            usage,
            value,
            min: 0,
            max: 255,
        }
    }

    #[test]
    fn buttons_follow_the_guessed_table() {
        let p = map(&[1, 2, 5, 8, 10, 11, 0], None, &[]);
        assert_eq!(p.buttons, A | B | LEFT_SHOULDER | MENU | RIGHT_STICK);
    }

    #[test]
    fn the_hat_is_the_dpad() {
        assert_eq!(map(&[], Some((0, 0, 7)), &[]).buttons, UP);
        assert_eq!(map(&[], Some((3, 0, 7)), &[]).buttons, DOWN | RIGHT);
        assert_eq!(map(&[], Some((7, 0, 7)), &[]).buttons, UP | LEFT);
        // One based hats, and the released value outside the range.
        assert_eq!(map(&[], Some((1, 1, 8)), &[]).buttons, UP);
        assert_eq!(map(&[], Some((8, 0, 7)), &[]).buttons, 0);
        assert_eq!(map(&[], Some((15, 0, 7)), &[]).buttons, 0);
    }

    #[test]
    fn sticks_flip_y_and_ignore_a_resting_offset() {
        let p = map(&[], None, &[axis(X_AXIS, 255), axis(Y_AXIS, 0)]);
        assert_eq!((p.left_x, p.left_y), (1.0, 1.0));
        let p = map(&[], None, &[axis(X_AXIS, 130), axis(Y_AXIS, 124)]);
        assert_eq!((p.left_x, p.left_y), (0.0, 0.0));
    }

    #[test]
    fn rx_ry_make_z_rz_the_triggers() {
        let axes = [
            axis(RX_AXIS, 255),
            axis(RY_AXIS, 128),
            axis(Z_AXIS, 255),
            axis(RZ_AXIS, 0),
        ];
        let p = map(&[], None, &axes);
        assert_eq!(
            (p.right_x, p.left_trigger, p.right_trigger),
            (1.0, 1.0, 0.0)
        );
        // Without Rx and Ry, Z and Rz are the right stick.
        let p = map(&[], None, &[axis(Z_AXIS, 255), axis(RZ_AXIS, 255)]);
        assert_eq!((p.right_x, p.right_y, p.left_trigger), (1.0, -1.0, 0.0));
    }
}
