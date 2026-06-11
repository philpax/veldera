//! Pack a glTF JSON document and binary buffer into a single .glb file.
//!
//! See the glTF 2.0 spec §3.4 "GLB File Format Specification".

use std::io::Write;

const GLB_MAGIC: u32 = 0x46546c67; // "glTF"
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4e4f534a; // "JSON"
const CHUNK_BIN: u32 = 0x004e4942; // "BIN\0"

pub fn write_glb(path: &std::path::Path, json: &[u8], bin: &[u8]) -> std::io::Result<()> {
    let json_padded_len = align_up(json.len(), 4);
    let bin_padded_len = align_up(bin.len(), 4);
    let total_len = 12 + 8 + json_padded_len + 8 + bin_padded_len;

    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    w.write_all(&GLB_MAGIC.to_le_bytes())?;
    w.write_all(&GLB_VERSION.to_le_bytes())?;
    w.write_all(&(total_len as u32).to_le_bytes())?;

    w.write_all(&(json_padded_len as u32).to_le_bytes())?;
    w.write_all(&CHUNK_JSON.to_le_bytes())?;
    w.write_all(json)?;
    for _ in 0..(json_padded_len - json.len()) {
        w.write_all(b" ")?;
    }

    w.write_all(&(bin_padded_len as u32).to_le_bytes())?;
    w.write_all(&CHUNK_BIN.to_le_bytes())?;
    w.write_all(bin)?;
    for _ in 0..(bin_padded_len - bin.len()) {
        w.write_all(b"\0")?;
    }
    w.flush()?;
    Ok(())
}

fn align_up(n: usize, alignment: usize) -> usize {
    (n + alignment - 1) & !(alignment - 1)
}
