//! FBX discovery and ufbx load options.

use std::{error::Error, path::Path};

/// Recursively walk the input directory for FBX files. Returns
/// `(name, path)` pairs where `name` is the file's path relative to the
/// input root with the `.fbx` extension stripped, e.g.
/// `locomotion/idle` or `Ch31_nonPBR`. Forward slashes regardless of OS.
pub(crate) fn list_fbx_files(
    dir: &Path,
) -> Result<Vec<(String, std::path::PathBuf)>, Box<dyn Error>> {
    let mut out = Vec::new();
    collect_fbx_recursive(dir, dir, &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn collect_fbx_recursive(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, std::path::PathBuf)>,
) -> Result<(), Box<dyn Error>> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_fbx_recursive(root, &path, out)?;
        } else if path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|e| e.eq_ignore_ascii_case("fbx"))
        {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let name = rel
                .with_extension("")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.push((name, path));
        }
    }
    Ok(())
}

pub(crate) fn load_opts() -> ufbx::LoadOpts<'static> {
    ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        geometry_transform_handling: ufbx::GeometryTransformHandling::ModifyGeometryNoFallback,
        pivot_handling: ufbx::PivotHandling::AdjustToPivot,
        generate_missing_normals: true,
        ..Default::default()
    }
}
