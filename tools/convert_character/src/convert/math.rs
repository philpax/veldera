//! Small math helpers for converting ufbx types into glTF-ready arrays.

const MIXAMO_PREFIX_SEPARATOR: char = ':';

pub(crate) fn vec3_to_f32(v: ufbx::Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

pub(crate) fn quat_to_f32(q: ufbx::Quat) -> [f32; 4] {
    [q.x as f32, q.y as f32, q.z as f32, q.w as f32]
}

pub(crate) fn matrix_to_f32_col_major(m: ufbx::Matrix) -> [f32; 16] {
    [
        m.m00 as f32,
        m.m10 as f32,
        m.m20 as f32,
        0.0,
        m.m01 as f32,
        m.m11 as f32,
        m.m21 as f32,
        0.0,
        m.m02 as f32,
        m.m12 as f32,
        m.m22 as f32,
        0.0,
        m.m03 as f32,
        m.m13 as f32,
        m.m23 as f32,
        1.0,
    ]
}

pub(crate) fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

pub(crate) fn invert_affine(m: ufbx::Matrix) -> ufbx::Matrix {
    // Closed-form inverse of an affine 3x4 (treating the missing row as
    // [0, 0, 0, 1]). Avoids pulling in a math crate just for this.
    let a = m.m00;
    let b = m.m01;
    let c = m.m02;
    let d = m.m03;
    let e = m.m10;
    let f = m.m11;
    let g = m.m12;
    let h = m.m13;
    let i = m.m20;
    let j = m.m21;
    let k = m.m22;
    let l = m.m23;

    let det = a * (f * k - g * j) - b * (e * k - g * i) + c * (e * j - f * i);
    if det.abs() < 1e-12 {
        return m;
    }
    let inv_det = 1.0 / det;

    let r00 = (f * k - g * j) * inv_det;
    let r01 = -(b * k - c * j) * inv_det;
    let r02 = (b * g - c * f) * inv_det;
    let r10 = -(e * k - g * i) * inv_det;
    let r11 = (a * k - c * i) * inv_det;
    let r12 = -(a * g - c * e) * inv_det;
    let r20 = (e * j - f * i) * inv_det;
    let r21 = -(a * j - b * i) * inv_det;
    let r22 = (a * f - b * e) * inv_det;

    let r03 = -(r00 * d + r01 * h + r02 * l);
    let r13 = -(r10 * d + r11 * h + r12 * l);
    let r23 = -(r20 * d + r21 * h + r22 * l);

    ufbx::Matrix {
        m00: r00,
        m01: r01,
        m02: r02,
        m03: r03,
        m10: r10,
        m11: r11,
        m12: r12,
        m13: r13,
        m20: r20,
        m21: r21,
        m22: r22,
        m23: r23,
    }
}

pub(crate) fn aabb_min(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut m = [f32::INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            if p[i] < m[i] {
                m[i] = p[i];
            }
        }
    }
    m
}

pub(crate) fn aabb_max(positions: &[[f32; 3]]) -> [f32; 3] {
    let mut m = [f32::NEG_INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            if p[i] > m[i] {
                m[i] = p[i];
            }
        }
    }
    m
}

/// The bone name after the last Mixamo namespace separator (`mixamorig:Hips`
/// → `Hips`), used to retarget animation tracks to base joints by name.
pub(crate) fn bone_stem(name: &str) -> &str {
    match name.rfind(MIXAMO_PREFIX_SEPARATOR) {
        Some(i) => &name[i + MIXAMO_PREFIX_SEPARATOR.len_utf8()..],
        None => name,
    }
}
