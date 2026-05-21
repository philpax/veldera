//! GPU pixel inspector for the cloud raymarch.
//!
//! Hover a debug cursor over the cloud and the raymarch shader writes
//! that pixel's intermediate raymarch state — `cam_proj`, `t_start`,
//! `t_end`, chord-grid sample indices, integrated transmittance, etc.
//! — into a single small storage buffer. Bevy's
//! [`bevy::render::gpu_readback::GpuReadbackPlugin`] streams the
//! buffer back to the CPU each frame, and the client UI reads the
//! result from [`CloudInspectLatest`] to render the values as text.
//!
//! Intended use is diagnostic: watch a value tick deterministically
//! as you nudge the camera, see exactly which intermediate steps.
//! Always cheap — one pixel per frame in the shader; one async copy
//! on the GPU; a single Vec<u8> event on the CPU. No stall.

use bevy::{
    app::{App, Plugin, Startup},
    asset::{Assets, RenderAssetUsages},
    ecs::{
        prelude::{Commands, Component, Res, ResMut},
        resource::Resource,
        system::SystemParam,
    },
    math::{Vec2, Vec3},
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{BufferUsages, ShaderType},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
    },
};

/// Per-pixel raymarch state captured at the inspect cursor. The
/// shader fills this in on the raymarch pass for the single pixel
/// indicated by [`CloudInspectCursor`]; the rest of the buffer
/// retains its previous frame's content (since only one pixel
/// writes).
///
/// Layout matches the WGSL `CloudInspectData` struct in
/// `cloud_raymarch.wgsl`. `Vec3` first so the std430 padding sits
/// in its trailing slot rather than between scalars.
#[derive(ShaderType, Default, Debug, Clone, Copy)]
pub struct CloudInspectData {
    /// Un-jittered world position the first sample landed at (or
    /// zero if the ray missed the cloud shell).
    pub first_hit_pos: Vec3,
    /// Camera projection along the ray direction
    /// (`dot(cam_world, ray_dir)`).
    pub cam_proj: f32,
    /// Cloud-shell entry distance from the camera.
    pub t_start: f32,
    /// Cloud-shell exit distance from the camera.
    pub t_end: f32,
    /// `t_end − t_start`; how much of the ray actually traverses the
    /// cloud shell.
    pub chord_length: f32,
    /// First world-snap grid index whose cell overlaps the chord.
    pub k_first: i32,
    /// Last world-snap grid index whose cell overlaps the chord.
    pub k_last: i32,
    /// Loop iterations actually executed (may be less than
    /// `k_last − k_first + 1` if the transmittance early-out fired).
    pub iter_count: u32,
    /// Theoretical maximum loop iterations.
    pub max_iter: u32,
    /// Final transmittance after the full raymarch.
    pub transmittance: f32,
    /// `1 − transmittance`, the integrated cloud opacity.
    pub opacity: f32,
    /// First `t` value at which density crossed the `1e-7`
    /// threshold. Zero if the ray missed the shell or only crossed
    /// empty cells.
    pub first_hit_t: f32,
    /// Density at the first-hit sample.
    pub first_hit_density: f32,
}

/// Inspect-cursor input, set by the client UI from egui's pointer
/// position. Lives in the main world; mirrored to the render world
/// each frame via [`ExtractResource`] so the cloud uniform-prep can
/// read it.
#[derive(Resource, ExtractResource, Default, Debug, Clone, Copy)]
pub struct CloudInspectCursor {
    /// Normalised window UV (0..1 in both axes). Conversion to a
    /// raymarch buffer pixel happens shader-side via
    /// `vec2<i32>(cursor * buffer_size)`, which sidesteps the
    /// physical-vs-logical-pixel mismatch on HiDPI displays. Outside
    /// the [0, 1] range when the cursor is off-window.
    pub cursor: Vec2,
    /// `false` when the cursor is outside the cloud render area
    /// (e.g. hovering over an egui panel). The shader gates its
    /// write on this so the inspect buffer's last frame's values
    /// stay visible while you mouse over a panel.
    pub active: bool,
    /// When `true`, the cursor is pinned to the screen centre
    /// `(0.5, 0.5)` regardless of mouse position, and `active` is
    /// forced on. Lets the user vary just camera pose (no mouse
    /// motion) while watching the same notional pixel's values
    /// change — much easier than chasing a cursor that drifts as
    /// you orbit. Toggled from the inspector UI.
    pub lock_to_centre: bool,
}

/// Handle to the `ShaderStorageBuffer` the shader writes the
/// inspect data into. Held in a resource so both the render-world
/// bind-group prep and the main-world readback observer can reach
/// the asset by name.
#[derive(Resource, ExtractResource, Clone)]
pub struct CloudInspectBuffer(pub bevy::asset::Handle<ShaderStorageBuffer>);

/// Latest readback data. Written by the [`Readback`] observer once
/// per frame (modulo the standard 1–2 frame GPU→CPU latency); read
/// by the UI to format values as text.
///
/// `None` until the first readback completes.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct CloudInspectLatest(pub Option<CloudInspectData>);

/// Marker on the entity that carries the [`Readback`] component, so
/// we can find / despawn it (e.g. to pause readbacks while the
/// inspector panel is closed — not currently wired up but cheap to
/// add).
#[derive(Component)]
pub struct CloudInspectReadbackEntity;

/// Bundles the resources `prepare_cloud_bind_groups` needs to wire
/// the inspect storage buffer into the raymarch bind group. Exists
/// purely to keep that function's parameter count under Bevy's
/// 16-param `IntoSystem` limit.
#[derive(SystemParam)]
pub struct CloudInspectBindParams<'w> {
    pub handle: Option<Res<'w, CloudInspectBuffer>>,
    pub assets: Res<'w, RenderAssets<GpuShaderStorageBuffer>>,
}

impl CloudInspectBindParams<'_> {
    /// Resolve the GPU storage buffer the shader writes its inspect
    /// data into, or `None` if the asset isn't uploaded yet (first
    /// frame). Caller should skip binding-group creation in that
    /// case — `prepare_cloud_bind_groups` runs every frame, so the
    /// buffer is available the frame after upload completes.
    pub fn resolve(&self) -> Option<&GpuShaderStorageBuffer> {
        self.handle.as_ref().and_then(|h| self.assets.get(&h.0))
    }
}

/// Wires up the inspect cursor + readback + observer. Called from
/// [`crate::CloudsPlanetPlugin::build`].
pub(super) struct CloudInspectPlugin;

impl Plugin for CloudInspectPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CloudInspectCursor>()
            .init_resource::<CloudInspectLatest>()
            .add_plugins(ExtractResourcePlugin::<CloudInspectCursor>::default())
            .add_plugins(ExtractResourcePlugin::<CloudInspectBuffer>::default())
            .add_systems(Startup, setup_inspect_buffer);
    }
}

fn setup_inspect_buffer(mut commands: Commands, mut buffers: ResMut<Assets<ShaderStorageBuffer>>) {
    let size = CloudInspectData::min_size().get() as usize;
    let mut buffer = ShaderStorageBuffer::with_size(size, RenderAssetUsages::RENDER_WORLD);
    // `COPY_SRC` is required for the readback machinery to copy our
    // storage buffer into a CPU-mapped staging buffer; the default
    // `STORAGE` usage isn't sufficient on its own.
    buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
    let handle = buffers.add(buffer);

    commands.insert_resource(CloudInspectBuffer(handle.clone()));
    commands
        .spawn((CloudInspectReadbackEntity, Readback::buffer(handle)))
        .observe(
            |event: bevy::ecs::observer::On<ReadbackComplete>,
             mut latest: ResMut<CloudInspectLatest>| {
                let data: CloudInspectData = event.to_shader_type();
                latest.0 = Some(data);
            },
        );
}
