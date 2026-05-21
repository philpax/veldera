//! Packed octree path representation.
//!
//! Octree paths in the rocktree format are sequences of octant indices
//! (`0`–`7`) describing a walk from the planetoid root down to a node.
//! Encoding them as `String` was costing the LoD streaming hot path
//! heavily — every BFS visit allocated a new path string, every
//! `potential_nodes.insert` cloned one, every HashMap lookup hashed one.
//!
//! [`OctreePath`] packs the same information into 3 bits per octant
//! within a single `u64`, plus a separate depth byte. The whole value
//! is 16 bytes and `Copy`, so paths can be passed by value through the
//! BFS without allocation. The `Display` impl renders the canonical
//! string form when one is needed (HTTP URLs, debug output).
//!
//! ## Layout
//!
//! Octant `n` (0-indexed) occupies bits `n*3..n*3+3` of `bits`. The
//! first octant is at the least-significant position. Up to
//! [`OctreePath::MAX_DEPTH`] (= 21) octants fit before running out of
//! the 63 usable bits. The rocktree tree itself only goes to depth 20,
//! so the cap is comfortable.

use std::fmt;

/// A path through the octree, encoded as packed octant indices.
///
/// `Copy` and 16 bytes wide. Equality and hashing compare the canonical
/// representation, so two paths with the same octants and depth are
/// always identical regardless of how they were constructed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct OctreePath {
    bits: u64,
    depth: u8,
}

impl OctreePath {
    /// The root path, with zero octants.
    pub const ROOT: Self = Self { bits: 0, depth: 0 };

    /// Maximum number of octants representable. 21 × 3 = 63 bits, the
    /// most that fit alongside an unused sign bit in a `u64`.
    pub const MAX_DEPTH: usize = 21;

    /// Number of octants in this path.
    #[inline]
    #[must_use]
    pub const fn depth(self) -> usize {
        self.depth as usize
    }

    /// True if this is the root path (zero octants).
    #[inline]
    #[must_use]
    pub const fn is_root(self) -> bool {
        self.depth == 0
    }

    /// Append an octant (`0`–`7`) to the path and return the new path.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `octant >= 8` or if the resulting path would exceed
    /// [`MAX_DEPTH`](Self::MAX_DEPTH). Release builds produce a
    /// well-defined but garbled path on overflow rather than panicking,
    /// matching the cost model the rest of the LoD hot path expects.
    #[inline]
    #[must_use]
    pub fn push(self, octant: u8) -> Self {
        debug_assert!(octant < 8, "octant must be 0-7, got {octant}");
        debug_assert!(
            (self.depth as usize) < Self::MAX_DEPTH,
            "octree path overflow"
        );
        let shift = u64::from(self.depth) * 3;
        Self {
            bits: self.bits | (u64::from(octant) << shift),
            depth: self.depth + 1,
        }
    }

    /// Octant at the given level (0-indexed). Returns `None` if `level`
    /// is past the end of the path.
    #[inline]
    #[must_use]
    pub fn octant_at(self, level: usize) -> Option<u8> {
        if level >= self.depth as usize {
            return None;
        }
        Some(((self.bits >> (level * 3)) & 0b111) as u8)
    }

    /// Parent path (one level shallower), or `None` if at root.
    #[inline]
    #[must_use]
    pub fn parent(self) -> Option<Self> {
        if self.depth == 0 {
            return None;
        }
        let new_depth = self.depth - 1;
        let shift = u64::from(new_depth) * 3;
        Some(Self {
            bits: self.bits & !(0b111u64 << shift),
            depth: new_depth,
        })
    }

    /// Truncate to the first `n` octants. If `n >= depth()`, returns
    /// `self` unchanged.
    #[inline]
    #[must_use]
    pub fn truncated(self, n: usize) -> Self {
        if n >= self.depth as usize {
            return self;
        }
        let bits = self.bits & ((1u64 << (n * 3)) - 1);
        Self {
            bits,
            depth: n as u8,
        }
    }

    /// Extract the last `n` octants as a path of depth `n`. Returns
    /// `None` if `n > depth()`.
    #[inline]
    #[must_use]
    pub fn tail(self, n: usize) -> Option<Self> {
        if n > self.depth as usize {
            return None;
        }
        if n == 0 {
            return Some(Self::ROOT);
        }
        let start = self.depth as usize - n;
        let shift = start as u64 * 3;
        let mask = (1u64 << (n * 3)) - 1;
        Some(Self {
            bits: (self.bits >> shift) & mask,
            depth: n as u8,
        })
    }

    /// True if `prefix` is a prefix of `self`. The root is a prefix of
    /// every path.
    #[inline]
    #[must_use]
    pub fn starts_with(self, prefix: Self) -> bool {
        if prefix.depth > self.depth {
            return false;
        }
        if prefix.depth == 0 {
            return true;
        }
        let mask = (1u64 << (u64::from(prefix.depth) * 3)) - 1;
        (self.bits & mask) == prefix.bits
    }

    /// If `self` starts with `prefix`, return the remainder (the octants
    /// after `prefix`). Used in BFS code to find a node's
    /// relative-within-bulk path.
    #[inline]
    #[must_use]
    pub fn strip_prefix(self, prefix: Self) -> Option<Self> {
        if !self.starts_with(prefix) {
            return None;
        }
        let shift = u64::from(prefix.depth) * 3;
        Some(Self {
            bits: self.bits >> shift,
            depth: self.depth - prefix.depth,
        })
    }

    /// Concatenate two paths: `self` followed by `other`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if the combined depth would exceed [`MAX_DEPTH`](Self::MAX_DEPTH).
    #[inline]
    #[must_use]
    pub fn extend(self, other: Self) -> Self {
        debug_assert!(self.depth as usize + other.depth as usize <= Self::MAX_DEPTH);
        let shift = u64::from(self.depth) * 3;
        Self {
            bits: self.bits | (other.bits << shift),
            depth: self.depth + other.depth,
        }
    }

    /// Iterate octants in order from first to last.
    pub fn octants(self) -> impl Iterator<Item = u8> {
        (0..self.depth as usize).map(move |i| ((self.bits >> (i * 3)) & 0b111) as u8)
    }

    /// Parse a path from a string of octant digits (`"0".."7"`).
    ///
    /// # Errors
    ///
    /// Returns [`ParseOctreePathError`] if the string contains a
    /// non-octant character or exceeds [`MAX_DEPTH`](Self::MAX_DEPTH)
    /// digits.
    pub fn parse(s: &str) -> Result<Self, ParseOctreePathError> {
        if s.len() > Self::MAX_DEPTH {
            return Err(ParseOctreePathError::TooLong);
        }
        let mut path = Self::ROOT;
        for c in s.chars() {
            match c {
                '0'..='7' => path = path.push(c as u8 - b'0'),
                _ => return Err(ParseOctreePathError::InvalidDigit(c)),
            }
        }
        Ok(path)
    }
}

impl fmt::Display for OctreePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::fmt::Write;
        for i in 0..self.depth as usize {
            let octant = ((self.bits >> (i * 3)) & 0b111) as u8;
            f.write_char((b'0' + octant) as char)?;
        }
        Ok(())
    }
}

impl fmt::Debug for OctreePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OctreePath(\"{self}\")")
    }
}

/// Error returned by [`OctreePath::parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseOctreePathError {
    /// Path string is longer than [`OctreePath::MAX_DEPTH`].
    TooLong,
    /// Path string contains a character outside `0`–`7`.
    InvalidDigit(char),
}

impl fmt::Display for ParseOctreePathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLong => f.write_str("octree path exceeds maximum depth"),
            Self::InvalidDigit(c) => write!(f, "invalid octant digit '{c}'"),
        }
    }
}

impl std::error::Error for ParseOctreePathError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_is_empty() {
        let p = OctreePath::ROOT;
        assert!(p.is_root());
        assert_eq!(p.depth(), 0);
        assert_eq!(p.to_string(), "");
    }

    #[test]
    fn push_extends_path() {
        let p = OctreePath::ROOT.push(0).push(1).push(2).push(3);
        assert_eq!(p.depth(), 4);
        assert_eq!(p.to_string(), "0123");
        assert_eq!(p.octant_at(0), Some(0));
        assert_eq!(p.octant_at(1), Some(1));
        assert_eq!(p.octant_at(2), Some(2));
        assert_eq!(p.octant_at(3), Some(3));
        assert_eq!(p.octant_at(4), None);
    }

    #[test]
    fn parse_and_display_round_trip() {
        for s in ["", "0", "07", "0123456701234567"] {
            let p = OctreePath::parse(s).unwrap();
            assert_eq!(p.to_string(), s);
            assert_eq!(p.depth(), s.len());
        }
    }

    #[test]
    fn parse_rejects_bad_digits() {
        assert!(OctreePath::parse("8").is_err());
        assert!(OctreePath::parse("a").is_err());
        assert!(OctreePath::parse("01x").is_err());
    }

    #[test]
    fn parse_rejects_too_long() {
        let too_long = "0".repeat(OctreePath::MAX_DEPTH + 1);
        assert!(matches!(
            OctreePath::parse(&too_long),
            Err(ParseOctreePathError::TooLong)
        ));
    }

    #[test]
    fn parent_drops_last_octant() {
        let p = OctreePath::parse("0123").unwrap();
        assert_eq!(p.parent().unwrap().to_string(), "012");
        assert_eq!(p.parent().unwrap().parent().unwrap().to_string(), "01");
    }

    #[test]
    fn parent_of_root_is_none() {
        assert!(OctreePath::ROOT.parent().is_none());
    }

    #[test]
    fn starts_with_handles_root_and_partial() {
        let p = OctreePath::parse("0123").unwrap();
        assert!(p.starts_with(OctreePath::ROOT));
        assert!(p.starts_with(OctreePath::parse("01").unwrap()));
        assert!(p.starts_with(p));
        assert!(!p.starts_with(OctreePath::parse("02").unwrap()));
        assert!(!p.starts_with(OctreePath::parse("01234").unwrap()));
    }

    #[test]
    fn strip_prefix_returns_relative() {
        let p = OctreePath::parse("0123").unwrap();
        let prefix = OctreePath::parse("01").unwrap();
        assert_eq!(p.strip_prefix(prefix).unwrap().to_string(), "23");
        assert_eq!(p.strip_prefix(OctreePath::ROOT).unwrap(), p);
        assert!(p.strip_prefix(OctreePath::parse("12").unwrap()).is_none());
    }

    #[test]
    fn tail_keeps_last_n() {
        let p = OctreePath::parse("01234567").unwrap();
        assert_eq!(p.tail(0).unwrap(), OctreePath::ROOT);
        assert_eq!(p.tail(4).unwrap().to_string(), "4567");
        assert_eq!(p.tail(8).unwrap().to_string(), "01234567");
        assert!(p.tail(9).is_none());
    }

    #[test]
    fn truncated_keeps_first_n() {
        let p = OctreePath::parse("0123").unwrap();
        assert_eq!(p.truncated(0).to_string(), "");
        assert_eq!(p.truncated(2).to_string(), "01");
        assert_eq!(p.truncated(4).to_string(), "0123");
        // n >= depth is a no-op.
        assert_eq!(p.truncated(99).to_string(), "0123");
    }

    #[test]
    fn extend_concatenates() {
        let a = OctreePath::parse("01").unwrap();
        let b = OctreePath::parse("234").unwrap();
        assert_eq!(a.extend(b).to_string(), "01234");
        assert_eq!(OctreePath::ROOT.extend(a), a);
        assert_eq!(a.extend(OctreePath::ROOT), a);
    }

    #[test]
    fn octants_iterates_in_order() {
        let p = OctreePath::parse("0123").unwrap();
        let v: Vec<u8> = p.octants().collect();
        assert_eq!(v, vec![0, 1, 2, 3]);
    }

    #[test]
    fn max_depth_path_round_trips() {
        let s = "0".repeat(OctreePath::MAX_DEPTH);
        let p = OctreePath::parse(&s).unwrap();
        assert_eq!(p.depth(), OctreePath::MAX_DEPTH);
        assert_eq!(p.to_string(), s);
    }

    #[test]
    fn is_copy_and_small() {
        // Catches a regression if the layout grows unexpectedly.
        assert!(std::mem::size_of::<OctreePath>() <= 16);
    }
}
