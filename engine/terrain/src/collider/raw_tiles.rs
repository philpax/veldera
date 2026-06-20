//! The raw-tiles terrain-collider reconcile: main's pre-branch synchronous
//! build.
//!
//! Used only when the raw-tiles algorithm is selected (see
//! [`crate::collider::COLLIDER`]). Each displayed tile gets a plain
//! octant-masked trimesh plus boundary skirts, built synchronously on the main
//! thread ([`veldera_physics::terrain::create_terrain_collider`]) — no fusion,
//! simplification, carving, or roads. The on-the-ground collision behaviour is
//! exactly as it was before the `roads` branch.
//!
//! Unlike the streaming-selection algorithms, the raw-tiles path has no separate
//! WYSIWYG mirror: the banded octree walk in [`crate::lod`] selects the whole
//! near field itself (colliding the displayed mesh via the innermost-band
//! descent), and this reconcile drives the spawned colliders toward
//! [`LodState::physics_target_paths`].

use avian3d::prelude::*;
use bevy::prelude::*;
use rocktree_decode::OctreePath;

use veldera_geo::floating_origin::{FloatingOriginCamera, WorldPosition};
use veldera_physics::{
    GameLayer, PhysicsState, PhysicsStreamingConfig, TerrainCollider,
    terrain::create_terrain_collider,
};

use crate::{
    collider::viz::reconcile_collider_wireframes,
    lod::{ColliderReconcile, LodState, poll_lod_node_tasks},
};

/// Register the raw-tiles reconcile and the shared per-entity wireframe overlay.
/// Called from [`crate::lod::LodPlugin::build`] when
/// [`crate::collider::COLLIDER`] selects the raw-tiles algorithm.
pub(crate) fn register(app: &mut App) {
    app.add_systems(
        Update,
        update_physics_colliders
            .in_set(ColliderReconcile)
            .after(poll_lod_node_tasks),
    )
    .add_systems(
        Update,
        reconcile_collider_wireframes.after(ColliderReconcile),
    );
}

/// Update physics colliders to match the physics selection's current target
/// (synchronous build; see the module docs).
fn update_physics_colliders(
    mut commands: Commands,
    time: Res<Time>,
    mut lod_state: ResMut<LodState>,
    physics_state: Res<PhysicsState>,
    streaming: Res<PhysicsStreamingConfig>,
    camera_query: Query<&FloatingOriginCamera>,
) {
    let Ok(camera) = camera_query.single() else {
        return;
    };

    // Spawn relative to the origin-shift bookkeeping, not the live camera:
    // the camera advances every frame (interpolated sub-tick motion included)
    // while physics positions are only re-based when a shift is applied.
    // Using the live camera bakes the difference into the collider as a
    // permanent offset — centimetres while walking, metres while falling fast.
    let camera_pos = physics_state
        .origin_camera_position()
        .unwrap_or(camera.position);
    let target_paths = lod_state.physics_target_paths.clone();
    let now = time.elapsed_secs_f64();

    // Track when each path entered the target set; the timestamp resets the
    // moment a path drops out, so re-selections start a fresh wait.
    lod_state
        .collider_candidate_since
        .retain(|p, _| target_paths.contains_key(p));
    for path in target_paths.keys() {
        lod_state
            .collider_candidate_since
            .entry(*path)
            .or_insert(now);
    }

    // Collect spawns and rebuilds: paths with no entity, or whose live
    // entity was built with a different mask. Deepest first, so children
    // are live before any parent rebuild masks their octants out; nearest
    // first within a depth so the ground under the player wins the build
    // budget.
    let mut pending: Vec<(OctreePath, u8, f64)> = target_paths
        .iter()
        .filter(|(path, mask)| match lod_state.physics_colliders.get(path) {
            None => true,
            Some((_, built_mask)) => built_mask != *mask,
        })
        .filter_map(|(path, mask)| {
            // BFS-selected paths whose data hasn't fully loaded yet are
            // skipped; we'll catch them in a later frame.
            let node_data = lod_state.node_data.get(path)?;
            let distance = (node_data.world_position - camera_pos).length();
            Some((*path, *mask, distance))
        })
        .collect();
    pending.sort_by(|a, b| {
        std::cmp::Reverse(a.0.depth())
            .cmp(&std::cmp::Reverse(b.0.depth()))
            .then(a.2.total_cmp(&b.2))
    });

    // Trimesh construction is the expensive part of collider streaming;
    // capping builds per frame bounds the frame cost during fast flight
    // when the band boundaries sweep the world.
    let max_builds = match streaming.max_collider_builds_per_frame {
        0 => usize::MAX,
        n => n,
    };
    let mut builds = 0usize;

    for (path, target_mask, _) in pending {
        if builds >= max_builds {
            break;
        }

        // Spawn-persistence gate for brand-new paths: selections must
        // survive a config-set dwell time before paying a trimesh build, so
        // selections that flicker during fast movement never build at all.
        // Regions with no live coverage bypass the gate — first coverage is
        // never delayed. Mask rebuilds of live entities skip the gate too:
        // they refine existing coverage and the despawn rules depend on
        // them converging.
        if !lod_state.physics_colliders.contains_key(&path) {
            let since = lod_state
                .collider_candidate_since
                .get(&path)
                .copied()
                .unwrap_or(now);
            let waited = now - since >= streaming.collider_spawn_persistence_secs;
            if !waited && lod_state.collider_region_covered(path) {
                continue;
            }
        }

        let Some(node_data) = lod_state.node_data.get(&path) else {
            continue;
        };

        // Only mask out octants that actually have live collider coverage
        // below: a committed child whose build failed (or is still pending)
        // must not leave a hole in the parent. Extra coverage from an unmasked
        // octant overlaps the late child briefly — jitter, not a fall.
        let mask = target_mask & lod_state.live_descendant_bits(path);

        match lod_state.physics_colliders.get(&path) {
            Some((_, built_mask)) if *built_mask == mask => continue,
            _ => {}
        }

        builds += 1;

        // Radial down at the node; the direction varies negligibly across a
        // single tile, so one vector serves the whole collider's skirts.
        let down = (-node_data.world_position.normalize()).as_vec3();
        let Some(collider) = create_terrain_collider(
            &node_data.meshes,
            &node_data.transform,
            streaming.min_collider_triangle_height as f32,
            down,
            streaming.collider_skirt_depth as f32,
            mask,
        ) else {
            tracing::debug!("Skipping invalid mesh for physics collider: '{}'", path);
            continue;
        };

        // Camera-relative position so the floating origin shift keeps it
        // in f32 range.
        let relative_pos = node_data.world_position - camera_pos;
        let physics_pos = Vec3::new(
            relative_pos.x as f32,
            relative_pos.y as f32,
            relative_pos.z as f32,
        );

        let entity = commands
            .spawn((
                RigidBody::Static,
                collider,
                Position(physics_pos),
                // Rotation is identity since rotation is baked into the
                // collider vertices.
                Rotation::default(),
                // Transform is needed for Avian's debug rendering (it
                // reads GlobalTransform).
                Transform::from_translation(physics_pos),
                WorldPosition::from_dvec3(node_data.world_position),
                TerrainCollider {
                    path,
                    octant_mask: mask,
                },
                CollisionLayers::new(
                    [GameLayer::Ground],
                    [GameLayer::Ground, GameLayer::Vehicle, GameLayer::Ragdoll],
                ),
            ))
            .id();

        // Replace any previous entity for this path (mask rebuild) in the
        // same frame, so the swap is atomic from physics's point of view.
        if let Some((old_entity, _)) = lod_state.physics_colliders.insert(path, (entity, mask)) {
            commands.entity(old_entity).despawn();
        }
        tracing::debug!(
            "Created physics collider for node '{}' (depth {}, mask {:#04x})",
            path,
            path.depth(),
            mask,
        );
    }

    // Despawn colliders no longer in the target set — but only once every
    // overlapping target path (ancestor or descendant — the replacement
    // coverage for this region) is live with its current mask, so a
    // deferred or failed replacement build never leaves the region bare.
    let obsolete: Vec<OctreePath> = lod_state
        .physics_colliders
        .keys()
        .filter(|p| !target_paths.contains_key(*p))
        .copied()
        .collect();

    for path in obsolete {
        let replacements_live = target_paths.iter().all(|(t, m)| {
            let overlaps = t.starts_with(path) || path.starts_with(*t);
            !overlaps
                || lod_state
                    .physics_colliders
                    .get(t)
                    .is_some_and(|(_, built)| built == m)
        });
        if !replacements_live {
            continue;
        }
        if let Some((entity, _)) = lod_state.physics_colliders.remove(&path) {
            commands.entity(entity).despawn();
            tracing::debug!("Removed physics collider for node '{}'", path);
        }
    }
}
