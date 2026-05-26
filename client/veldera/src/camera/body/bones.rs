//! Mixamo bone-name constants and classification helpers.
//!
//! Every bone-name match in the body module routes through these
//! constants and helpers so a future rig swap (different naming
//! convention, different skeleton) touches a single file.

// ----------------------------------------------------------------------------
// Bone stem names. The "stem" is the part of each bone name after the
// `mixamorig*:` prefix Mixamo assigns per upload.
// ----------------------------------------------------------------------------

/// Pelvis bone — root of the lower-body chain. Hips translation is the
/// only one our converter preserves (everything else gets stripped to
/// remove root motion).
pub const BONE_HIPS: &str = "Hips";
/// Base of the skull. Animation rotates this with the spine; in
/// first-person we hide its mesh by zeroing the scale.
pub const BONE_HEAD: &str = "Head";
/// Top-of-skull marker bone. Together with [`BONE_HEAD`] it defines
/// the head's bind-pose vertical extent (head-height heuristic for
/// the eye position).
pub const BONE_HEAD_TOP_END: &str = "HeadTop_End";
/// Right-arm chain bones used by the point-IK system.
pub const BONE_RIGHT_ARM: &str = "RightArm";
pub const BONE_RIGHT_FORE_ARM: &str = "RightForeArm";
pub const BONE_RIGHT_HAND: &str = "RightHand";
/// Neck bone. Used by the upper-body mask classifier.
pub const BONE_NECK: &str = "Neck";

/// Suffix patterns that mark a bone as part of the lower body
/// (everything from the hips down — pelvis, thighs, calves, feet, toes).
const LOWER_BODY_BONE_SUFFIXES: &[&str] = &["UpLeg", "Leg", "Foot", "ToeBase", "Toe_End"];

/// Suffix patterns that mark a bone as part of the upper body (arms
/// + hands, not counting fingers which are matched by substring below).
const UPPER_BODY_BONE_SUFFIXES: &[&str] = &["Shoulder", "Arm", "Hand"];

/// Substring patterns for finger bones (Mixamo names them
/// `…HandThumb1`, `…HandIndex2`, etc.).
const FINGER_BONE_PATTERNS: &[&str] = &[
    "HandThumb",
    "HandIndex",
    "HandMiddle",
    "HandRing",
    "HandPinky",
];

/// Bone-stem prefix for the spine column (`Spine`, `Spine1`, `Spine2`).
const SPINE_BONE_PREFIX: &str = "Spine";

// ----------------------------------------------------------------------------
// Mask group bits. Animation clips use these in their `mask` field to
// say "don't affect this group of bones"; the runtime populates the
// graph's `mask_groups` map to say "this bone belongs to this group".
// ----------------------------------------------------------------------------

/// Animation mask bit for upper-body bones (Spine and above: torso,
/// neck, head, shoulders, arms, hands). A clip with this bit set in its
/// `mask` field skips upper-body bones entirely.
pub const UPPER_BODY_MASK: u64 = 1 << 0;
/// Animation mask bit for lower-body bones (Hips and below: pelvis,
/// legs, feet, toes). A clip with this bit set in its `mask` field skips
/// lower-body bones entirely.
pub const LOWER_BODY_MASK: u64 = 1 << 1;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Strip the `mixamorig*:` prefix from a bone name. Falls through to
/// returning the input unchanged if no colon is present (non-Mixamo
/// names).
pub fn bone_stem(name: &str) -> &str {
    match name.rfind(':') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

/// Classify a Mixamo bone stem into a mask group. Hips and below
/// returns [`LOWER_BODY_MASK`]; Spine and above returns
/// [`UPPER_BODY_MASK`]; unknown bones return `0` (animated by every
/// clip regardless of mask).
pub fn bone_mask_group(stem: &str) -> u64 {
    if stem == BONE_HIPS || LOWER_BODY_BONE_SUFFIXES.iter().any(|s| stem.ends_with(s)) {
        return LOWER_BODY_MASK;
    }
    if stem.starts_with(SPINE_BONE_PREFIX)
        || stem == BONE_NECK
        || stem == BONE_HEAD
        || stem == BONE_HEAD_TOP_END
        || UPPER_BODY_BONE_SUFFIXES.iter().any(|s| stem.ends_with(s))
        || FINGER_BONE_PATTERNS.iter().any(|p| stem.contains(p))
    {
        return UPPER_BODY_MASK;
    }
    0
}
