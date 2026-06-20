//! Path and flags unpacking.

use crate::{OctreePath, PathAndFlags};

/// Unpack path and flags from node metadata.
///
/// The `path_and_flags` field encodes:
/// - Lower 2 bits: Level - 1 (so level is 1-4)
/// - Next 3*level bits: Octant path digits (0-7)
/// - Remaining bits: Flags
///
/// # Arguments
///
/// * `path_and_flags` - The packed value from `NodeMetadata`
///
/// # Example
///
/// For a node at path "012" (level 3):
/// - Level bits: 2 (since level - 1 = 2)
/// - Path bits: 0, 1, 2 (3 bits each)
/// - Flags: remaining bits
#[must_use]
pub fn unpack_path_and_flags(path_and_flags: u32) -> PathAndFlags {
    // Lower 2 bits encode level - 1.
    let level = 1 + (path_and_flags & 3) as usize;
    let mut remaining = path_and_flags >> 2;

    // Extract `level` octants directly into an OctreePath (3 bits each).
    let mut path = OctreePath::ROOT;
    for _ in 0..level {
        let digit = (remaining & 7) as u8;
        path = path.push(digit);
        remaining >>= 3;
    }

    // Remaining bits are flags.
    let flags = remaining;

    PathAndFlags { path, flags, level }
}

#[cfg(test)]
mod tests {
    // The packed test values group their binary digits by the documented bit
    // layout (path digits in threes, the level in twos), not in uniform nibbles.
    #![allow(clippy::unusual_byte_groupings)]

    use super::*;

    fn p(s: &str) -> OctreePath {
        OctreePath::parse(s).unwrap()
    }

    #[test]
    fn test_unpack_path_and_flags_level1() {
        // Level 1: level - 1 = 0, so lower 2 bits = 0.
        // Path digit: 5 (binary: 101).
        // Flags: 0.
        // Packed: 0b101_00 = 20.
        let result = unpack_path_and_flags(0b101_00);

        assert_eq!(result.level, 1);
        assert_eq!(result.path, p("5"));
        assert_eq!(result.flags, 0);
    }

    #[test]
    fn test_unpack_path_and_flags_level2() {
        // Level 2: level - 1 = 1, so lower 2 bits = 1.
        // Path digits: 3 (first), 7 (second).
        // Packed: 0b111_011_01 = path[0]=3, path[1]=7, level-1=1.
        let packed = 0b111_011_01;
        let result = unpack_path_and_flags(packed);

        assert_eq!(result.level, 2);
        assert_eq!(result.path, p("37"));
        assert_eq!(result.flags, 0);
    }

    #[test]
    fn test_unpack_path_and_flags_level4() {
        // Level 4: level - 1 = 3, so lower 2 bits = 3.
        // Path digits: 0, 1, 2, 3.
        // Packed: 0b011_010_001_000_11 = 0x36B.
        let packed = 0b011_010_001_000_11;
        let result = unpack_path_and_flags(packed);

        assert_eq!(result.level, 4);
        assert_eq!(result.path, p("0123"));
        assert_eq!(result.flags, 0);
    }

    #[test]
    fn test_unpack_path_and_flags_with_flags() {
        // Level 1, path = 0, flags = 5 — flags occupy bits 5+, path 0 contributes
        // nothing.
        let packed = 5 << 5;
        let result = unpack_path_and_flags(packed);

        assert_eq!(result.level, 1);
        assert_eq!(result.path, p("0"));
        assert_eq!(result.flags, 5);
    }

    #[test]
    fn test_unpack_path_and_flags_complex() {
        // Level 3, path = "764", flags = 42.
        // level - 1 = 2.
        // Packed bits: flags(42) | path[2](4) | path[1](6) | path[0](7) | level-1(2).
        // = 42 << 11 | 4 << 8 | 6 << 5 | 7 << 2 | 2.
        let packed = (42 << 11) | (4 << 8) | (6 << 5) | (7 << 2) | 2;
        let result = unpack_path_and_flags(packed);

        assert_eq!(result.level, 3);
        assert_eq!(result.path, p("764"));
        assert_eq!(result.flags, 42);
    }
}
