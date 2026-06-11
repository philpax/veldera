//! Split the Sketchfab "Generic passenger car pack" glb into per-car glbs.
//!
//! The pack stores ten cars laid out in a showroom grid: each car body and
//! its four wheels are sibling nodes under `RootNode`, with placement baked
//! into node matrices and geometry in millimetres. Wheel node names are
//! reused across cars (`Wheel_A` serves both the sedan and the wagon), so
//! wheels are grouped to bodies by proximity, not by name.
//!
//! For each car this tool:
//! - bakes node matrices into world-space geometry and scales mm → m,
//! - re-frames the car so +X is right, +Y is up, -Z is forward, with the
//!   origin on the ground at the centroid of the four wheel centres,
//! - emits a `body` node plus `wheel_fl`/`wheel_fr`/`wheel_rl`/`wheel_rr`
//!   nodes, each wheel pivoted at its own axle so the game can spin and
//!   steer it,
//! - subsets the pack's materials/textures/images to those the car uses.
//!
//! ```text
//! split-car-pack <input.glb> <output_dir> [--views <dir>]
//! ```
//!
//! `--views` additionally renders side-view silhouettes (forward to the
//! right) for visually verifying each car's forward axis.

mod buffer;
mod glb;

use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
};

use glam::{Mat3, Mat4, Vec3};
use gltf_json::{
    Accessor, Image, Index, Material, Mesh, Node, Root, Scene,
    accessor::{ComponentType, GenericComponentType, Type},
    validation::{Checked, USize64},
};

use buffer::BufferBuilder;

/// Millimetres (the pack's `RootNode` space) to metres.
const MM_TO_M: f32 = 1e-3;

/// Car slugs whose heuristic forward direction (larger overhang = rear) is
/// wrong and must be flipped. Verified against the `--views` silhouettes.
const FORWARD_FLIP_OVERRIDES: &[&str] = &["coupe", "hatchback", "sport"];

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (input, output_dir, views_dir) = match args.as_slice() {
        [_, input, output] => (input, output, None),
        [_, input, output, flag, views] if flag == "--views" => {
            (input, output, Some(PathBuf::from(views)))
        }
        _ => {
            eprintln!("usage: split-car-pack <input.glb> <output_dir> [--views <dir>]");
            std::process::exit(2);
        }
    };
    split(
        Path::new(input),
        Path::new(output_dir),
        views_dir.as_deref(),
    )
}

fn split(input: &Path, output_dir: &Path, views_dir: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let gltf::Gltf { document, blob } = gltf::Gltf::open(input)?;
    let blob = blob.ok_or("input glb has no binary chunk")?;
    let src_root = document.clone().into_json();

    let root_node = document
        .nodes()
        .find(|n| n.name() == Some("RootNode"))
        .ok_or("no RootNode in input")?;

    // Bake every top-level group (a car body or a wheel) into world-space
    // geometry in metres.
    let mut bodies: Vec<BakedGroup> = Vec::new();
    let mut wheels: Vec<BakedGroup> = Vec::new();
    let scale = Mat4::from_scale(Vec3::splat(MM_TO_M));
    for child in root_node.children() {
        let name = child.name().unwrap_or("").to_string();
        let lower = name.to_lowercase();
        let kind = if lower.contains("body") {
            GroupKind::Body
        } else if lower.starts_with("wheel") {
            GroupKind::Wheel
        } else {
            println!("skipping non-car node: {name}");
            continue;
        };
        let mut prims = Vec::new();
        bake_node(&child, scale, &blob, &mut prims)?;
        let group = BakedGroup::new(name, prims)?;
        match kind {
            GroupKind::Body => bodies.push(group),
            GroupKind::Wheel => wheels.push(group),
        }
    }

    // Group wheels to their nearest body in the showroom-grid plane.
    let mut wheels_by_body: Vec<Vec<BakedGroup>> = (0..bodies.len()).map(|_| Vec::new()).collect();
    for wheel in wheels {
        let (closest, _) = bodies
            .iter()
            .enumerate()
            .map(|(i, b)| (i, xz_distance(b.center, wheel.center)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or("no bodies found")?;
        wheels_by_body[closest].push(wheel);
    }

    std::fs::create_dir_all(output_dir)?;
    if let Some(dir) = views_dir {
        std::fs::create_dir_all(dir)?;
    }

    println!(
        "{:<10} {:>6} {:>6} {:>6} {:>9} {:>6} {:>7} {:>7}",
        "car", "len", "width", "height", "wheelbase", "track", "wheel-r", "wheel-w"
    );
    for (body, car_wheels) in bodies.iter().zip(&wheels_by_body) {
        let slug = slug_for_body(&body.name);
        if car_wheels.len() != 4 {
            return Err(format!(
                "car '{slug}' has {} wheels, expected 4 (wheel grouping failed)",
                car_wheels.len()
            )
            .into());
        }
        let car = orient_car(&slug, body, car_wheels)?;
        if let Some(dir) = views_dir {
            render_side_view(&car, &dir.join(format!("{slug}.png")))?;
        }
        let out_path = output_dir.join(format!("{slug}.glb"));
        emit_car(&car, &src_root, &blob, &out_path)?;
        println!(
            "{:<10} {:>6.2} {:>6.2} {:>6.2} {:>9.2} {:>6.2} {:>7.3} {:>7.3}",
            slug,
            car.size.z,
            car.size.x,
            car.size.y,
            car.wheelbase,
            car.track,
            car.wheel_radius,
            car.wheel_width
        );
    }
    Ok(())
}

// ============================================================================
// Baking: node hierarchies to world-space geometry
// ============================================================================

enum GroupKind {
    Body,
    Wheel,
}

/// One source primitive with its transform baked into the vertex data.
struct BakedPrim {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    /// Material index into the *source* document.
    material: usize,
}

/// A top-level pack node (car body or wheel) baked to world space.
struct BakedGroup {
    name: String,
    prims: Vec<BakedPrim>,
    min: Vec3,
    center: Vec3,
}

impl BakedGroup {
    fn new(name: String, prims: Vec<BakedPrim>) -> Result<Self, Box<dyn Error>> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for prim in &prims {
            for &p in &prim.positions {
                min = min.min(p);
                max = max.max(p);
            }
        }
        if !min.is_finite() {
            return Err(format!("group '{name}' has no geometry").into());
        }
        Ok(Self {
            name,
            prims,
            min,
            center: (min + max) / 2.0,
        })
    }
}

/// Recursively bake a node subtree's mesh primitives into world space.
fn bake_node(
    node: &gltf::Node,
    parent: Mat4,
    blob: &[u8],
    out: &mut Vec<BakedPrim>,
) -> Result<(), Box<dyn Error>> {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        let normal_matrix = Mat3::from_mat4(world).inverse().transpose();
        let flip_winding = world.determinant() < 0.0;
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                return Err(format!("non-triangle primitive in {:?}", node.name()).into());
            }
            let reader = prim.reader(|_| Some(blob));
            let positions: Vec<Vec3> = reader
                .read_positions()
                .ok_or("primitive without positions")?
                .map(|p| world.transform_point3(Vec3::from(p)))
                .collect();
            let normals: Vec<Vec3> = match reader.read_normals() {
                Some(iter) => iter
                    .map(|n| (normal_matrix * Vec3::from(n)).normalize_or_zero())
                    .collect(),
                None => vec![Vec3::Y; positions.len()],
            };
            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(iter) => iter.into_f32().collect(),
                None => vec![[0.0, 0.0]; positions.len()],
            };
            let mut indices: Vec<u32> = match reader.read_indices() {
                Some(iter) => iter.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            if flip_winding {
                for tri in indices.chunks_exact_mut(3) {
                    tri.swap(1, 2);
                }
            }
            let material = prim
                .material()
                .index()
                .ok_or("primitive uses the default material")?;
            out.push(BakedPrim {
                positions,
                normals,
                uvs,
                indices,
                material,
            });
        }
    }
    for child in node.children() {
        bake_node(&child, world, blob, out)?;
    }
    Ok(())
}

fn xz_distance(a: Vec3, b: Vec3) -> f32 {
    let d = a - b;
    (d.x * d.x + d.z * d.z).sqrt()
}

/// "Compact Body" → "compact", "minivan body" → "minivan", "SUV Body" → "suv".
fn slug_for_body(name: &str) -> String {
    name.to_lowercase()
        .replace("body", "")
        .trim()
        .replace(' ', "_")
}

// ============================================================================
// Car framing: orientation, origin, wheel slots
// ============================================================================

/// A fully-framed car in car-local space (+X right, +Y up, -Z forward,
/// origin on the ground under the wheel centroid), ready to emit.
struct Car {
    slug: String,
    body_prims: Vec<BakedPrim>,
    /// fl, fr, rl, rr order. Each wheel's geometry is recentred on its axle.
    wheel_prims: Vec<Vec<BakedPrim>>,
    /// fl, fr, rl, rr axle positions in car space.
    wheel_positions: [Vec3; 4],
    size: Vec3,
    wheelbase: f32,
    track: f32,
    wheel_radius: f32,
    wheel_width: f32,
}

const WHEEL_SLOT_NAMES: [&str; 4] = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"];

/// Determine the car's frame and transform body and wheels into it.
fn orient_car(slug: &str, body: &BakedGroup, wheels: &[BakedGroup]) -> Result<Car, Box<dyn Error>> {
    let centers: Vec<Vec3> = wheels.iter().map(|w| w.center).collect();
    let centroid = centers.iter().sum::<Vec3>() / 4.0;

    // Pair the four wheels into two axles: of the three possible pairings,
    // the one with the smallest within-pair distances pairs left with right
    // (track width is smaller than wheelbase).
    let pairings = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]];
    let axle_pairs = pairings
        .iter()
        .min_by(|a, b| {
            let dist = |pairs: &[(usize, usize); 2]| {
                pairs
                    .iter()
                    .map(|&(i, j)| centers[i].distance(centers[j]))
                    .sum::<f32>()
            };
            dist(a).total_cmp(&dist(b))
        })
        .expect("pairings is non-empty");
    let axle_mids = axle_pairs.map(|(i, j)| (centers[i] + centers[j]) / 2.0);
    let long_axis = {
        let mut d = axle_mids[0] - axle_mids[1];
        d.y = 0.0;
        d.normalize()
    };

    // The end with the larger body overhang past its axle is usually the
    // rear; FORWARD_FLIP_OVERRIDES corrects cars where this guess is wrong
    // (verified visually via --views).
    let mut max_pos = 0.0f32;
    let mut max_neg = 0.0f32;
    for prim in &body.prims {
        for &p in &prim.positions {
            let proj = (p - centroid).dot(long_axis);
            max_pos = max_pos.max(proj);
            max_neg = max_neg.max(-proj);
        }
    }
    let mut forward = if max_pos > max_neg {
        -long_axis
    } else {
        long_axis
    };
    if FORWARD_FLIP_OVERRIDES.contains(&slug) {
        forward = -forward;
    }

    // Car-space basis: +X right, +Y up, -Z forward. The origin sits on the
    // ground (lowest wheel point) under the wheel centroid.
    let up = Vec3::Y;
    let right = forward.cross(up).normalize();
    let rotation = Mat3::from_cols(right, up, -forward).transpose();
    let ground_y = wheels.iter().map(|w| w.min.y).fold(f32::INFINITY, f32::min);
    let origin = Vec3::new(centroid.x, ground_y, centroid.z);
    let to_car = |p: Vec3| rotation * (p - origin);

    let transform_prims = |prims: &[BakedPrim], offset: Vec3| -> Vec<BakedPrim> {
        prims
            .iter()
            .map(|prim| BakedPrim {
                positions: prim.positions.iter().map(|&p| to_car(p) - offset).collect(),
                normals: prim.normals.iter().map(|&n| rotation * n).collect(),
                uvs: prim.uvs.clone(),
                indices: prim.indices.clone(),
                material: prim.material,
            })
            .collect()
    };

    let body_prims = transform_prims(&body.prims, Vec3::ZERO);

    // Assign wheels to fl/fr/rl/rr slots by their car-space centres.
    let mut slots: [Option<usize>; 4] = [None; 4];
    for (i, w) in wheels.iter().enumerate() {
        let c = to_car(w.center);
        let slot = match (c.z < 0.0, c.x < 0.0) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        };
        if slots[slot].replace(i).is_some() {
            return Err(format!(
                "car '{slug}': two wheels landed in slot {}",
                WHEEL_SLOT_NAMES[slot]
            )
            .into());
        }
    }

    let mut wheel_prims = Vec::with_capacity(4);
    let mut wheel_positions = [Vec3::ZERO; 4];
    let mut wheel_radius = 0.0f32;
    let mut wheel_width = 0.0f32;
    for (slot, idx) in slots.iter().enumerate() {
        let idx = idx.expect("four wheels fill four distinct slots");
        let wheel = &wheels[idx];
        // Recompute the axle centre from car-space bounds (the rotated AABB
        // centre is not the AABB centre of the rotated points).
        let transformed = transform_prims(&wheel.prims, Vec3::ZERO);
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for prim in &transformed {
            for &p in &prim.positions {
                min = min.min(p);
                max = max.max(p);
            }
        }
        let axle = (min + max) / 2.0;
        wheel_positions[slot] = axle;
        wheel_radius += (max.y - min.y) / 8.0;
        wheel_width += (max.x - min.x) / 4.0;
        wheel_prims.push(
            transformed
                .into_iter()
                .map(|prim| BakedPrim {
                    positions: prim.positions.iter().map(|&p| p - axle).collect(),
                    ..prim
                })
                .collect(),
        );
    }

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for prim in &body_prims {
        for &p in &prim.positions {
            min = min.min(p);
            max = max.max(p);
        }
    }

    Ok(Car {
        slug: slug.to_string(),
        body_prims,
        wheel_prims,
        wheel_positions,
        size: max - min,
        wheelbase: (wheel_positions[0].z - wheel_positions[2].z).abs(),
        track: (wheel_positions[0].x - wheel_positions[1].x).abs(),
        wheel_radius,
        wheel_width,
    })
}

// ============================================================================
// Emission
// ============================================================================

/// Emit one car as a self-contained glb with a subset of the pack's
/// materials, textures, and images.
fn emit_car(
    car: &Car,
    src_root: &Root,
    src_blob: &[u8],
    out_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut root = Root {
        asset: gltf_json::Asset {
            copyright: Some(
                "\"Generic passenger car pack\" by Comrade1280 (Sketchfab), CC BY 4.0".to_string(),
            ),
            extensions: None,
            extras: Default::default(),
            generator: Some("veldera split-car-pack".to_string()),
            min_version: None,
            version: "2.0".to_string(),
        },
        ..Default::default()
    };
    let mut buffer = BufferBuilder::new();
    // Samplers are copied wholesale (the pack has one); texture sampler
    // indices stay valid.
    root.samplers = src_root.samplers.clone();

    let mut remap = MaterialRemap::default();

    let body_mesh = push_mesh(
        &mut root,
        &mut buffer,
        &car.body_prims,
        src_root,
        src_blob,
        &mut remap,
    )?;
    let body_node = push_node(&mut root, "body", body_mesh, None);

    let mut children = vec![body_node];
    for (slot, prims) in car.wheel_prims.iter().enumerate() {
        let mesh = push_mesh(
            &mut root,
            &mut buffer,
            prims,
            src_root,
            src_blob,
            &mut remap,
        )?;
        children.push(push_node(
            &mut root,
            WHEEL_SLOT_NAMES[slot],
            mesh,
            Some(car.wheel_positions[slot]),
        ));
    }

    let root_index = root.nodes.len() as u32;
    root.nodes.push(Node {
        camera: None,
        children: Some(children),
        extensions: None,
        extras: Default::default(),
        matrix: None,
        mesh: None,
        name: Some(car.slug.clone()),
        rotation: None,
        scale: None,
        translation: None,
        skin: None,
        weights: None,
    });
    root.scenes.push(Scene {
        extensions: None,
        extras: Default::default(),
        name: Some("Scene".to_string()),
        nodes: vec![Index::new(root_index)],
    });
    root.scene = Some(Index::new(0));

    root.buffers.push(gltf_json::Buffer {
        byte_length: USize64(buffer.data.len() as u64),
        extensions: None,
        extras: Default::default(),
        name: None,
        uri: None,
    });
    root.buffer_views = buffer.views;

    let json = serde_json::to_vec(&root)?;
    glb::write_glb(out_path, &json, &buffer.data)?;
    Ok(())
}

/// Source-document index → output-document index maps for the material
/// dependency chain (materials → textures → images).
#[derive(Default)]
struct MaterialRemap {
    materials: BTreeMap<usize, u32>,
    textures: BTreeMap<usize, u32>,
    images: BTreeMap<usize, u32>,
}

/// Copy a source material (and its texture/image dependencies) into the
/// output document, returning the new material index.
fn ensure_material(
    src_index: usize,
    root: &mut Root,
    buffer: &mut BufferBuilder,
    src_root: &Root,
    src_blob: &[u8],
    remap: &mut MaterialRemap,
) -> Result<u32, Box<dyn Error>> {
    if let Some(&idx) = remap.materials.get(&src_index) {
        return Ok(idx);
    }
    let mut material: Material = src_root
        .materials
        .get(src_index)
        .ok_or("material index out of range")?
        .clone();

    let mut remap_texture =
        |index: Index<gltf_json::Texture>| -> Result<Index<gltf_json::Texture>, Box<dyn Error>> {
            let src_tex = index.value();
            if let Some(&idx) = remap.textures.get(&src_tex) {
                return Ok(Index::new(idx));
            }
            let mut texture = src_root
                .textures
                .get(src_tex)
                .ok_or("texture index out of range")?
                .clone();
            let src_img = texture.source.value();
            let new_img = if let Some(&idx) = remap.images.get(&src_img) {
                idx
            } else {
                let image = src_root.images.get(src_img).ok_or("image out of range")?;
                let view_index = image
                    .buffer_view
                    .ok_or("image without buffer view (external URI?)")?
                    .value();
                let view = &src_root.buffer_views[view_index];
                let offset = view.byte_offset.map_or(0, |o| o.0 as usize);
                let length = view.byte_length.0 as usize;
                let bytes = src_blob
                    .get(offset..offset + length)
                    .ok_or("image buffer view out of range")?;
                let new_view = buffer.add_image(bytes);
                let idx = root.images.len() as u32;
                root.images.push(Image {
                    buffer_view: Some(new_view),
                    mime_type: image.mime_type.clone(),
                    name: image.name.clone(),
                    uri: None,
                    extensions: None,
                    extras: Default::default(),
                });
                remap.images.insert(src_img, idx);
                idx
            };
            texture.source = Index::new(new_img);
            let idx = root.textures.len() as u32;
            root.textures.push(texture);
            remap.textures.insert(src_tex, idx);
            Ok(Index::new(idx))
        };

    if let Some(info) = &mut material.pbr_metallic_roughness.base_color_texture {
        info.index = remap_texture(info.index)?;
    }
    if let Some(info) = &mut material.pbr_metallic_roughness.metallic_roughness_texture {
        info.index = remap_texture(info.index)?;
    }
    if let Some(info) = &mut material.normal_texture {
        info.index = remap_texture(info.index)?;
    }
    if let Some(info) = &mut material.occlusion_texture {
        info.index = remap_texture(info.index)?;
    }
    if let Some(info) = &mut material.emissive_texture {
        info.index = remap_texture(info.index)?;
    }

    let idx = root.materials.len() as u32;
    root.materials.push(material);
    remap.materials.insert(src_index, idx);
    Ok(idx)
}

/// Emit a mesh from baked primitives, returning its index.
fn push_mesh(
    root: &mut Root,
    buffer: &mut BufferBuilder,
    prims: &[BakedPrim],
    src_root: &Root,
    src_blob: &[u8],
    remap: &mut MaterialRemap,
) -> Result<u32, Box<dyn Error>> {
    let mut primitives = Vec::with_capacity(prims.len());
    for prim in prims {
        let material = ensure_material(prim.material, root, buffer, src_root, src_blob, remap)?;

        let pos_bytes: Vec<u8> = prim
            .positions
            .iter()
            .flat_map(|v| v.to_array().into_iter().flat_map(f32::to_le_bytes))
            .collect();
        let pos_view = buffer.add_array(&pos_bytes, 12);
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for &p in &prim.positions {
            min = min.min(p);
            max = max.max(p);
        }
        let pos_accessor = push_accessor(
            root,
            pos_view,
            prim.positions.len(),
            Type::Vec3,
            Some((min, max)),
        );

        let normal_bytes: Vec<u8> = prim
            .normals
            .iter()
            .flat_map(|v| v.to_array().into_iter().flat_map(f32::to_le_bytes))
            .collect();
        let normal_view = buffer.add_array(&normal_bytes, 12);
        let normal_accessor =
            push_accessor(root, normal_view, prim.normals.len(), Type::Vec3, None);

        let uv_bytes: Vec<u8> = prim
            .uvs
            .iter()
            .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
            .collect();
        let uv_view = buffer.add_array(&uv_bytes, 8);
        let uv_accessor = push_accessor(root, uv_view, prim.uvs.len(), Type::Vec2, None);

        let index_bytes: Vec<u8> = prim.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        let index_view = buffer.add_indices(&index_bytes);
        let index_accessor =
            push_accessor(root, index_view, prim.indices.len(), Type::Scalar, None);

        let mut attributes = BTreeMap::new();
        attributes.insert(
            Checked::Valid(gltf_json::mesh::Semantic::Positions),
            Index::new(pos_accessor),
        );
        attributes.insert(
            Checked::Valid(gltf_json::mesh::Semantic::Normals),
            Index::new(normal_accessor),
        );
        attributes.insert(
            Checked::Valid(gltf_json::mesh::Semantic::TexCoords(0)),
            Index::new(uv_accessor),
        );
        primitives.push(gltf_json::mesh::Primitive {
            attributes,
            extensions: None,
            extras: Default::default(),
            indices: Some(Index::new(index_accessor)),
            material: Some(Index::new(material)),
            mode: Checked::Valid(gltf_json::mesh::Mode::Triangles),
            targets: None,
        });
    }
    let idx = root.meshes.len() as u32;
    root.meshes.push(Mesh {
        extensions: None,
        extras: Default::default(),
        name: None,
        primitives,
        weights: None,
    });
    Ok(idx)
}

fn push_accessor(
    root: &mut Root,
    view: Index<gltf_json::buffer::View>,
    count: usize,
    type_: Type,
    min_max: Option<(Vec3, Vec3)>,
) -> u32 {
    let component = match type_ {
        Type::Scalar => ComponentType::U32,
        _ => ComponentType::F32,
    };
    let idx = root.accessors.len() as u32;
    root.accessors.push(Accessor {
        buffer_view: Some(view),
        byte_offset: Some(USize64(0)),
        count: USize64(count as u64),
        component_type: Checked::Valid(GenericComponentType(component)),
        extensions: None,
        extras: Default::default(),
        type_: Checked::Valid(type_),
        min: min_max.map(|(min, _)| serde_json::json!([min.x, min.y, min.z])),
        max: min_max.map(|(_, max)| serde_json::json!([max.x, max.y, max.z])),
        name: None,
        normalized: false,
        sparse: None,
    });
    idx
}

fn push_node(root: &mut Root, name: &str, mesh: u32, translation: Option<Vec3>) -> Index<Node> {
    let idx = root.nodes.len() as u32;
    root.nodes.push(Node {
        camera: None,
        children: None,
        extensions: None,
        extras: Default::default(),
        matrix: None,
        mesh: Some(Index::new(mesh)),
        name: Some(name.to_string()),
        rotation: None,
        scale: None,
        translation: translation.map(|t| t.to_array()),
        skin: None,
        weights: None,
    });
    Index::new(idx)
}

// ============================================================================
// Debug side views
// ============================================================================

/// Render a side-view point silhouette (forward to the right): body in grey,
/// front wheels in red, rear wheels in blue.
fn render_side_view(car: &Car, path: &Path) -> Result<(), Box<dyn Error>> {
    const WIDTH: u32 = 900;
    const HEIGHT: u32 = 360;
    const MARGIN: f32 = 20.0;

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut sample = |prims: &[BakedPrim], offset: Vec3| {
        for prim in prims {
            for &p in &prim.positions {
                min = min.min(p + offset);
                max = max.max(p + offset);
            }
        }
    };
    sample(&car.body_prims, Vec3::ZERO);
    for (prims, &pos) in car.wheel_prims.iter().zip(&car.wheel_positions) {
        sample(prims, pos);
    }

    let span_z = (max.z - min.z).max(1e-3);
    let span_y = (max.y - min.y).max(1e-3);
    let scale =
        ((WIDTH as f32 - 2.0 * MARGIN) / span_z).min((HEIGHT as f32 - 2.0 * MARGIN) / span_y);

    let mut img = image::RgbaImage::from_pixel(WIDTH, HEIGHT, image::Rgba([255, 255, 255, 255]));
    let mut splat = |prims: &[BakedPrim], offset: Vec3, color: [u8; 4]| {
        for prim in prims {
            // Forward is -Z; map it to +X on the image so the car faces right.
            let project = |p: Vec3| {
                let p = p + offset;
                (
                    (max.z - p.z) * scale + MARGIN,
                    (max.y - p.y) * scale + MARGIN,
                )
            };
            for tri in prim.indices.chunks_exact(3) {
                for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                    let (x0, y0) = project(prim.positions[tri[i] as usize]);
                    let (x1, y1) = project(prim.positions[tri[j] as usize]);
                    let steps = (x1 - x0).abs().max((y1 - y0).abs()).ceil().max(1.0) as u32;
                    for s in 0..=steps {
                        let t = s as f32 / steps as f32;
                        let (px, py) = ((x0 + (x1 - x0) * t) as u32, (y0 + (y1 - y0) * t) as u32);
                        if px < WIDTH && py < HEIGHT {
                            img.put_pixel(px, py, image::Rgba(color));
                        }
                    }
                }
            }
        }
    };
    splat(&car.body_prims, Vec3::ZERO, [90, 90, 90, 255]);
    for (slot, (prims, &pos)) in car.wheel_prims.iter().zip(&car.wheel_positions).enumerate() {
        let color = if slot < 2 {
            [220, 40, 40, 255]
        } else {
            [40, 40, 220, 255]
        };
        splat(prims, pos, color);
    }
    img.save(path)?;
    Ok(())
}
