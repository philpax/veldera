//! The Mixamo humanoid skeleton as a typed [`Bone`] enum.
//!
//! Every bone-name decision in the body module — animation masking, ragdoll
//! topology, the point-pose arm chain — works over [`Bone`] rather than raw
//! strings. The only string boundary is reading a skinned-mesh `Name`:
//! [`Bone::from_name`] parses it once, and [`Bone`]'s [`Display`] renders the
//! canonical stem back out when a string is genuinely needed (entity names,
//! log lines). A future rig swap re-points the parsing and rendering here and
//! nothing downstream changes.
//!
//! [`Display`]: std::fmt::Display

use std::fmt;

// ----------------------------------------------------------------------------
// Bone identity
// ----------------------------------------------------------------------------

/// Which side of the body a paired bone belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side {
    Left,
    Right,
}

/// A finger of the hand. Mixamo names the phalanges `…Hand{Finger}{1..4}`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Finger {
    Thumb,
    Index,
    Middle,
    Ring,
    Pinky,
}

/// A bone in the Mixamo humanoid skeleton, identified by its stem (the part of
/// the bone name after the `mixamorig*:` prefix Mixamo assigns per upload).
///
/// Paired bones carry a [`Side`]; finger phalanges carry a [`Side`], a
/// [`Finger`], and a 1-based `segment` (1 = proximal … 4 = tip marker).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Bone {
    /// Pelvis — root of the skeleton. Its translation is the only one our
    /// converter preserves (the rest is stripped to remove root motion).
    Hips,
    Spine,
    Spine1,
    Spine2,
    /// Neck. The ragdoll pins this upright as the torso anchor (see
    /// `ragdoll`), keeping the head — which has no rigid body — upright through
    /// transform propagation.
    Neck,
    /// Base of the skull. Animation rotates it with the spine; first-person
    /// hides its mesh by zeroing the scale.
    Head,
    /// Top-of-skull marker bone. With [`Head`](Bone::Head) it defines the
    /// bind-pose head height used to place the eye.
    HeadTopEnd,
    Shoulder(Side),
    Arm(Side),
    ForeArm(Side),
    Hand(Side),
    HandFinger {
        side: Side,
        finger: Finger,
        /// Phalange index, 1 (proximal) through 4 (tip marker).
        segment: u8,
    },
    UpLeg(Side),
    Leg(Side),
    Foot(Side),
    ToeBase(Side),
    ToeEnd(Side),
}

// ----------------------------------------------------------------------------
// Animation mask groups
// ----------------------------------------------------------------------------

/// Animation mask bit for upper-body bones (Spine and above: torso, neck, head,
/// shoulders, arms, hands, fingers). A clip with this bit set in its `mask`
/// field skips upper-body bones entirely.
pub const UPPER_BODY_MASK: u64 = 1 << 0;
/// Animation mask bit for lower-body bones (Hips and below: pelvis, legs, feet,
/// toes). A clip with this bit set in its `mask` field skips lower-body bones.
pub const LOWER_BODY_MASK: u64 = 1 << 1;

// ----------------------------------------------------------------------------
// Classification and conversion
// ----------------------------------------------------------------------------

impl Bone {
    /// Parse a full skinned-mesh bone name (including any `mixamorig*:` prefix)
    /// into a [`Bone`], or `None` if it isn't a recognised skeleton bone (mesh
    /// nodes, accessory bones, non-Mixamo rigs).
    pub fn from_name(name: &str) -> Option<Bone> {
        Self::from_stem(bone_stem(name))
    }

    /// Parse a bone *stem* (the name with its `mixamorig*:` prefix already
    /// stripped) into a [`Bone`].
    pub fn from_stem(stem: &str) -> Option<Bone> {
        use Bone::*;
        match stem {
            "Hips" => return Some(Hips),
            "Spine" => return Some(Spine),
            "Spine1" => return Some(Spine1),
            "Spine2" => return Some(Spine2),
            "Neck" => return Some(Neck),
            "Head" => return Some(Head),
            "HeadTop_End" => return Some(HeadTopEnd),
            _ => {}
        }

        let (side, rest) = Side::split_prefix(stem)?;
        Some(match rest {
            "Shoulder" => Shoulder(side),
            "Arm" => Arm(side),
            "ForeArm" => ForeArm(side),
            "Hand" => Hand(side),
            "UpLeg" => UpLeg(side),
            "Leg" => Leg(side),
            "Foot" => Foot(side),
            "ToeBase" => ToeBase(side),
            "Toe_End" => ToeEnd(side),
            _ => {
                // The only remaining shape is a finger phalange,
                // `Hand{Finger}{segment}`.
                let (finger, digits) = Finger::split_prefix(rest.strip_prefix("Hand")?)?;
                let segment: u8 = digits.parse().ok()?;
                if !(1..=4).contains(&segment) {
                    return None;
                }
                HandFinger {
                    side,
                    finger,
                    segment,
                }
            }
        })
    }

    /// The animation mask group this bone belongs to: [`LOWER_BODY_MASK`] for
    /// Hips and below, [`UPPER_BODY_MASK`] for Spine and above.
    pub fn mask_group(self) -> u64 {
        use Bone::*;
        match self {
            Hips | UpLeg(_) | Leg(_) | Foot(_) | ToeBase(_) | ToeEnd(_) => LOWER_BODY_MASK,
            Spine
            | Spine1
            | Spine2
            | Neck
            | Head
            | HeadTopEnd
            | Shoulder(_)
            | Arm(_)
            | ForeArm(_)
            | Hand(_)
            | HandFinger { .. } => UPPER_BODY_MASK,
        }
    }
}

impl fmt::Display for Bone {
    /// Render the canonical Mixamo stem (without the `mixamorig*:` prefix).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Bone::*;
        match self {
            Hips => f.write_str("Hips"),
            Spine => f.write_str("Spine"),
            Spine1 => f.write_str("Spine1"),
            Spine2 => f.write_str("Spine2"),
            Neck => f.write_str("Neck"),
            Head => f.write_str("Head"),
            HeadTopEnd => f.write_str("HeadTop_End"),
            Shoulder(s) => write!(f, "{}Shoulder", s.prefix()),
            Arm(s) => write!(f, "{}Arm", s.prefix()),
            ForeArm(s) => write!(f, "{}ForeArm", s.prefix()),
            Hand(s) => write!(f, "{}Hand", s.prefix()),
            HandFinger {
                side,
                finger,
                segment,
            } => write!(f, "{}Hand{}{segment}", side.prefix(), finger.label()),
            UpLeg(s) => write!(f, "{}UpLeg", s.prefix()),
            Leg(s) => write!(f, "{}Leg", s.prefix()),
            Foot(s) => write!(f, "{}Foot", s.prefix()),
            ToeBase(s) => write!(f, "{}ToeBase", s.prefix()),
            ToeEnd(s) => write!(f, "{}Toe_End", s.prefix()),
        }
    }
}

impl Side {
    /// The Mixamo name prefix for this side (`Left`/`Right`).
    pub fn prefix(self) -> &'static str {
        match self {
            Side::Left => "Left",
            Side::Right => "Right",
        }
    }

    /// Split a `Left`/`Right` prefix off a stem, returning the side and the
    /// remainder.
    fn split_prefix(stem: &str) -> Option<(Side, &str)> {
        if let Some(rest) = stem.strip_prefix("Left") {
            Some((Side::Left, rest))
        } else {
            stem.strip_prefix("Right").map(|rest| (Side::Right, rest))
        }
    }
}

impl Finger {
    /// The Mixamo name fragment for this finger (`Thumb`, `Index`, …).
    fn label(self) -> &'static str {
        match self {
            Finger::Thumb => "Thumb",
            Finger::Index => "Index",
            Finger::Middle => "Middle",
            Finger::Ring => "Ring",
            Finger::Pinky => "Pinky",
        }
    }

    /// Split a finger label off the front of a stem fragment, returning the
    /// finger and the remainder (the segment digits).
    fn split_prefix(fragment: &str) -> Option<(Finger, &str)> {
        const FINGERS: [Finger; 5] = [
            Finger::Thumb,
            Finger::Index,
            Finger::Middle,
            Finger::Ring,
            Finger::Pinky,
        ];
        FINGERS.into_iter().find_map(|finger| {
            fragment
                .strip_prefix(finger.label())
                .map(|rest| (finger, rest))
        })
    }
}

/// Strip the `mixamorig*:` prefix from a bone name, returning the input
/// unchanged if no colon is present (non-Mixamo names).
pub fn bone_stem(name: &str) -> &str {
    match name.rfind(':') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every stem in the standard Mixamo humanoid skeleton (the `leonard.glb`
    /// rig): the landmark bones plus all side/finger/segment combinations.
    fn all_skeleton_stems() -> Vec<String> {
        let mut stems: Vec<String> = [
            "Hips",
            "Spine",
            "Spine1",
            "Spine2",
            "Neck",
            "Head",
            "HeadTop_End",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        for side in ["Left", "Right"] {
            for part in [
                "Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot", "ToeBase", "Toe_End",
            ] {
                stems.push(format!("{side}{part}"));
            }
            for finger in ["Thumb", "Index", "Middle", "Ring", "Pinky"] {
                for segment in 1..=4 {
                    stems.push(format!("{side}Hand{finger}{segment}"));
                }
            }
        }
        stems
    }

    #[test]
    fn every_skeleton_bone_round_trips_through_stem() {
        for stem in all_skeleton_stems() {
            let bone =
                Bone::from_stem(&stem).unwrap_or_else(|| panic!("{stem} should parse to a Bone"));
            assert_eq!(
                bone.to_string(),
                stem,
                "{stem} should render back to itself"
            );
        }
    }

    #[test]
    fn from_name_strips_the_mixamorig_prefix() {
        assert_eq!(
            Bone::from_name("mixamorig9:RightForeArm"),
            Some(Bone::ForeArm(Side::Right))
        );
        assert_eq!(
            Bone::from_name("mixamorig:RightHandIndex2"),
            Some(Bone::HandFinger {
                side: Side::Right,
                finger: Finger::Index,
                segment: 2,
            })
        );
    }

    #[test]
    fn mask_groups_split_at_the_hips() {
        // Hips and below are lower body; Spine and above (including fingers)
        // are upper body. Fingers and toes are the cases the old substring
        // matcher leaned on, so check them explicitly.
        for stem in [
            "Hips",
            "LeftUpLeg",
            "RightLeg",
            "LeftFoot",
            "RightToeBase",
            "LeftToe_End",
        ] {
            assert_eq!(
                Bone::from_stem(stem).unwrap().mask_group(),
                LOWER_BODY_MASK,
                "{stem}"
            );
        }
        for stem in [
            "Spine",
            "Neck",
            "Head",
            "HeadTop_End",
            "RightHand",
            "LeftHandPinky3",
        ] {
            assert_eq!(
                Bone::from_stem(stem).unwrap().mask_group(),
                UPPER_BODY_MASK,
                "{stem}"
            );
        }
    }

    #[test]
    fn non_skeleton_names_do_not_parse() {
        // Mesh nodes, accessory bones, and malformed finger names.
        for name in [
            "Ch31_Body",
            "character",
            "LeftHandIndex0",
            "LeftHandIndex5",
            "RightHandToe1",
            "Spine9",
        ] {
            assert_eq!(Bone::from_stem(name), None, "{name} should not be a Bone");
        }
    }
}
