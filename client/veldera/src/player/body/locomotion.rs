//! Locomotion blend: pick clip weights each tick from the controller's
//! velocity, ground state, and crouch ratio, then apply them to the
//! `AnimationPlayer`.
//!
//! Standing forward + strafe use the locomotion pack (hands by side).
//! Backward and crouching use the rifle-8-way pack masked to the lower
//! body, with `locomotion/idle` layered as an upper-body-only node so
//! the arms stay hands-down rather than holding an invisible rifle.

use std::{collections::HashMap, f32::consts::PI};

use avian3d::prelude::*;
use bevy::{animation::graph::AnimationNodeIndex, prelude::*, reflect::TypePath};
use serde::Deserialize;

use super::{BodyAssets, BodyVisual};
use crate::{
    player::{FpsController, FpsPlayerConfig, LogicalPlayer},
    world::{coords::RadialFrame, floating_origin::WorldPosition},
};

/// Hot-reloadable locomotion-blend tuning, loaded from
/// `assets/config/game/player/body/locomotion.toml`. All speeds are metres/second.
#[derive(Default, Asset, Resource, TypePath, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocomotionConfig {
    /// Speeds below this are "idle" — no directional clip contributes. Above it
    /// we blend in walking/running.
    pub deadzone_m_s: f32,
    /// Reference horizontal speed for the `locomotion/walking` clip; crossfaded
    /// from idle below and to running above.
    pub walk_ref_m_s: f32,
    /// Reference horizontal speed for the `locomotion/running` clip; anything
    /// faster stays pinned to running.
    pub run_ref_m_s: f32,
    /// Vertical speed above which the body switches to the airborne pose. Uses
    /// vertical velocity rather than `ground_tick` (which drops for a single
    /// tick when cresting uneven ground and would spam the jump-loop pose).
    pub airborne_vertical_m_s: f32,
}

/// Eight-way direction labels — match the clip names from the Mixamo
/// "Rifle 8-Way Locomotion Pack". Ordered counter-clockwise starting at
/// forward, so `index * 45°` is the body-local heading of each.
const DIRECTION_NAMES: [&str; 8] = [
    "forward",        // +Z
    "forward left",   // +Z, -X
    "left",           // -X
    "backward left",  // -Z, -X
    "backward",       // -Z
    "backward right", // -Z, +X
    "right",          // +X
    "forward right",  // +Z, +X
];

// ============================================================================
// Per-tick driver system
// ============================================================================

/// Recompute every relevant clip's weight from the controller state and
/// drive `AnimationPlayer` accordingly. The blend tree is conceptually:
///
/// ```text
///     ┌── idle (1−speed)  ────────────┐
///     │                                ├── (1 − crouch) ── standing output
/// gait├── walk (8-way blend) ─────────┤
///     │                                │
///     ├── run  (8-way blend) ─────────┤
///     │                                │
///     ├── sprint (8-way blend) ───────┘
///     │
///     └── airborne: jump loop @ 1
///
///     ┌── idle crouching (1−speed) ───┐
/// gait├── walk crouching (8-way) ─────┴── crouch_amount  ── crouching output
/// ```
pub(super) fn update_locomotion_blend(
    config: Res<FpsPlayerConfig>,
    loco: Res<LocomotionConfig>,
    body_assets: Res<BodyAssets>,
    logical_query: Query<
        (&FpsController, &LinearVelocity, &Transform, &WorldPosition),
        With<LogicalPlayer>,
    >,
    body_query: Query<&BodyVisual>,
    mut player_query: Query<&mut AnimationPlayer>,
) {
    if body_assets.animation_nodes.is_empty() {
        return;
    }

    for body in &body_query {
        let Some(player_entity) = body.animation_player else {
            continue;
        };
        let Ok(mut player) = player_query.get_mut(player_entity) else {
            continue;
        };
        let Ok((controller, velocity, _xform, world_pos)) = logical_query.get(body.logical_entity)
        else {
            continue;
        };

        let frame = RadialFrame::from_ecef_position(world_pos.position);
        let local_up = frame.up;
        let forward =
            (frame.north * controller.yaw.cos() - frame.east * controller.yaw.sin()).normalize();
        let right = local_up.cross(forward).normalize();

        let vertical_speed = velocity.0.dot(local_up);
        let horizontal_vel = velocity.0 - local_up * vertical_speed;
        let fwd_speed = horizontal_vel.dot(forward);
        let side_speed = horizontal_vel.dot(right);
        let speed = (fwd_speed * fwd_speed + side_speed * side_speed).sqrt();

        let airborne = vertical_speed.abs() > loco.airborne_vertical_m_s;
        let crouch_amount = if controller.upright_height > controller.crouch_height {
            ((controller.upright_height - controller.height)
                / (controller.upright_height - controller.crouch_height))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let targets = compute_locomotion_weights(
            &loco,
            speed,
            fwd_speed,
            side_speed,
            airborne,
            crouch_amount,
        );

        apply_locomotion_weights(
            &mut player,
            &body_assets.animation_nodes,
            body_assets.idle_upper_body_node,
            &targets,
        );

        let _ = config;
    }
}

// ============================================================================
// Pure functions: compute target weights from controller state
// ============================================================================

/// Target weights for one tick, decomposed into named clips plus the
/// special "upper-body idle" node that's layered on top during crouch.
struct LocomotionTargets {
    /// Named clip weights, keyed by the `pack/stem` glTF animation name.
    clips: HashMap<String, f32>,
    /// Weight for the masked `locomotion/idle` node that supplies the
    /// upper-body pose during crouch (zero when not crouching).
    idle_upper_body: f32,
}

fn compute_locomotion_weights(
    cfg: &LocomotionConfig,
    speed: f32,
    fwd_speed: f32,
    side_speed: f32,
    airborne: bool,
    crouch_amount: f32,
) -> LocomotionTargets {
    let mut clips: HashMap<String, f32> = HashMap::new();

    if airborne {
        clips.insert("locomotion/jump".to_string(), 1.0);
        return LocomotionTargets {
            clips,
            idle_upper_body: 0.0,
        };
    }

    let standing_w = (1.0 - crouch_amount).clamp(0.0, 1.0);
    let crouching_w = crouch_amount.clamp(0.0, 1.0);

    let mut idle_upper_body = 0.0;
    if standing_w > 0.0 {
        idle_upper_body +=
            write_standing_weights(cfg, &mut clips, speed, fwd_speed, side_speed, standing_w);
    }
    if crouching_w > 0.0 {
        write_crouching_weights(cfg, &mut clips, speed, fwd_speed, side_speed, crouching_w);
        idle_upper_body += crouching_w;
    }

    LocomotionTargets {
        clips,
        idle_upper_body,
    }
}

/// Locomotion-pack 3-gait blend (idle / walking / running) summing to 1.
fn locomotion_gait_blend(cfg: &LocomotionConfig, speed: f32) -> (f32, f32, f32) {
    if speed <= cfg.deadzone_m_s {
        return (1.0, 0.0, 0.0);
    }
    if speed < cfg.walk_ref_m_s {
        let t =
            ((speed - cfg.deadzone_m_s) / (cfg.walk_ref_m_s - cfg.deadzone_m_s)).clamp(0.0, 1.0);
        return (1.0 - t, t, 0.0);
    }
    if speed < cfg.run_ref_m_s {
        let t = ((speed - cfg.walk_ref_m_s) / (cfg.run_ref_m_s - cfg.walk_ref_m_s)).clamp(0.0, 1.0);
        return (0.0, 1.0 - t, t);
    }
    (0.0, 0.0, 1.0)
}

/// Returns the additional `idle_upper_body` weight contribution from
/// the rifle-pack backward clips used in this standing tick.
fn write_standing_weights(
    cfg: &LocomotionConfig,
    clips: &mut HashMap<String, f32>,
    speed: f32,
    fwd_speed: f32,
    side_speed: f32,
    standing_w: f32,
) -> f32 {
    let (idle_g, walk_g, run_g) = locomotion_gait_blend(cfg, speed);

    if speed <= cfg.deadzone_m_s {
        add_weight(clips, "locomotion/idle", standing_w);
        return 0.0;
    }

    // Signed forward axis [-1, 1]: positive is "into the locomotion
    // pack's forward-locomotion territory"; negative drives the rifle
    // pack's backward clips for the lower body.
    let dir_fwd_signed = fwd_speed / speed;
    let dir_fwd = dir_fwd_signed.max(0.0);
    let dir_back = (-dir_fwd_signed).max(0.0);
    let dir_side = (side_speed / speed).abs();
    let side_name = if side_speed < 0.0 { "left" } else { "right" };

    // Forward + strafe: locomotion pack.
    if dir_fwd > 0.0 {
        if walk_g > 0.0 {
            add_weight(clips, "locomotion/walking", standing_w * walk_g * dir_fwd);
        }
        if run_g > 0.0 {
            add_weight(clips, "locomotion/running", standing_w * run_g * dir_fwd);
        }
    }
    if dir_side > 0.0 {
        if walk_g > 0.0 {
            add_weight(
                clips,
                &format!("locomotion/{side_name} strafe walking"),
                standing_w * walk_g * dir_side,
            );
        }
        if run_g > 0.0 {
            add_weight(
                clips,
                &format!("locomotion/{side_name} strafe"),
                standing_w * run_g * dir_side,
            );
        }
    }

    // Backward: rifle pack. Split between pure-backward and diagonal
    // backward by the side-axis magnitude. Cardinal-side movement
    // (dir_side = 1 with dir_back = 0) bypasses this entirely.
    if dir_back > 0.0 {
        let back_pure = (dir_back * (1.0 - dir_side)).max(0.0);
        let back_side = dir_back * dir_side;
        if walk_g > 0.0 {
            if back_pure > 0.0 {
                add_weight(
                    clips,
                    "rifle-8-way/walk backward",
                    standing_w * walk_g * back_pure,
                );
            }
            if back_side > 0.0 {
                add_weight(
                    clips,
                    &format!("rifle-8-way/walk backward {side_name}"),
                    standing_w * walk_g * back_side,
                );
            }
        }
        if run_g > 0.0 {
            if back_pure > 0.0 {
                add_weight(
                    clips,
                    "rifle-8-way/run backward",
                    standing_w * run_g * back_pure,
                );
            }
            if back_side > 0.0 {
                add_weight(
                    clips,
                    &format!("rifle-8-way/run backward {side_name}"),
                    standing_w * run_g * back_side,
                );
            }
        }
    }

    // Idle takes the gait-idle bucket plus any "leftover" weight that
    // didn't go to a directional clip (e.g. diagonals beyond unit
    // magnitude).
    let consumed = dir_fwd + dir_back + dir_side;
    let leftover = (1.0 - consumed).max(0.0);
    add_weight(
        clips,
        "locomotion/idle",
        standing_w * (idle_g + (walk_g + run_g) * leftover),
    );

    // Upper-body idle weight matches how much of the standing lower
    // body comes from rifle clips this tick; both fade in together.
    standing_w * dir_back * (walk_g + run_g)
}

fn write_crouching_weights(
    cfg: &LocomotionConfig,
    clips: &mut HashMap<String, f32>,
    speed: f32,
    fwd_speed: f32,
    side_speed: f32,
    crouching_w: f32,
) {
    if speed <= cfg.deadzone_m_s {
        add_weight(clips, "rifle-8-way/idle crouching", crouching_w);
        return;
    }
    // 8-way blend across the rifle-pack crouching clips. The clip mask
    // (set at graph-build time) limits these to the lower body; the
    // upper body comes from the layered idle-upper node in
    // `LocomotionTargets::idle_upper_body`.
    for (dir_idx, dir_w) in direction_8way_blend(fwd_speed, side_speed) {
        let dir = DIRECTION_NAMES[dir_idx];
        add_weight(
            clips,
            &format!("rifle-8-way/walk crouching {dir}"),
            crouching_w * dir_w,
        );
    }
}

/// Map body-local velocity to up to two adjacent 8-way directions with
/// barycentric-style weights. Returns `(direction_index, weight)` pairs
/// whose weights sum to 1.
fn direction_8way_blend(fwd_speed: f32, side_speed: f32) -> [(usize, f32); 2] {
    // atan2(-side, fwd):
    //   fwd>0, side=0  → 0          (forward)
    //   fwd=0, side<0  → +π/2       (left)
    //   fwd<0, side=0  → +π         (backward)
    //   fwd=0, side>0  → -π/2       (right, wraps to 3π/2 below)
    //
    // We use -side so positive theta rotates counter-clockwise, which
    // matches the `DIRECTION_NAMES` ordering (forward, forward-left, …).
    let theta = (-side_speed).atan2(fwd_speed);
    // Normalise to [0, 2π) then scale so each direction occupies 1 unit.
    let normalised = (theta.rem_euclid(2.0 * PI)) / (PI / 4.0);
    let lower = normalised.floor() as usize % 8;
    let upper = (lower + 1) % 8;
    let t = normalised - normalised.floor();
    [(lower, 1.0 - t), (upper, t)]
}

fn add_weight(weights: &mut HashMap<String, f32>, name: &str, w: f32) {
    *weights.entry(name.to_string()).or_insert(0.0) += w;
}

// ============================================================================
// Apply: write the computed weights into the AnimationPlayer
// ============================================================================

/// Walk every clip in the graph and push its target weight into the
/// player, plus the special idle-upper-body node used in crouching.
///
/// We only call `play()` the first time a clip needs a non-zero weight
/// — afterwards we mutate the existing `ActiveAnimation` directly. Every
/// frame `play()` would re-invoke `.repeat()` and reset `completions`,
/// which can confuse Bevy's animation tick.
fn apply_locomotion_weights(
    player: &mut AnimationPlayer,
    nodes: &HashMap<String, AnimationNodeIndex>,
    idle_upper_node: Option<AnimationNodeIndex>,
    targets: &LocomotionTargets,
) {
    for (name, node) in nodes {
        let weight = targets.clips.get(name).copied().unwrap_or(0.0);
        set_node_weight(player, *node, weight);
    }
    if let Some(node) = idle_upper_node {
        set_node_weight(player, node, targets.idle_upper_body);
    }
}

fn set_node_weight(player: &mut AnimationPlayer, node: AnimationNodeIndex, weight: f32) {
    match player.animation_mut(node) {
        Some(active) => {
            active.set_weight(weight);
        }
        None => {
            if weight > 0.0 {
                player.play(node).set_weight(weight).repeat();
            }
        }
    }
}
