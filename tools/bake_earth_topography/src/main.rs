//! Bakes the ETOPO 2022 ice-surface elevation dataset into the runtime
//! asset used by the cloud renderer's Earth-aware weather model.
//!
//! Pipeline:
//!
//! 1. Download `ETOPO_2022_v1_60s_N90W180_surface.tif` from NOAA
//!    (~900 MB GeoTIFF, float32 elevation in metres) into
//!    `source_assets/`. Cached across runs.
//! 2. Decode as a 21600×10800 grid of float elevations
//!    (negative = bathymetry, positive = land surface, includes the
//!    ~3 km thick Greenland and Antarctic ice sheets — relevant for
//!    clouds, which orographically lift over the ice surface).
//! 3. Box-average downsample to 2048×1024.
//! 4. Map elevation \[-500, 9000\] m → \[0, 255\] and write a single-
//!    channel PNG to `client/veldera/assets/world/earth_topography.png`.
//!
//! Output channel encoding (currently single channel — RGBA versions
//! can extend in place once the runtime starts consuming derived
//! data like land-mask thresholds or biome hints):
//!
//! - **0**: ≤ -500 m (open ocean).
//! - **~13**: sea level (0 m). Soft coastline threshold for the cloud
//!   shader to derive a land mask from.
//! - **255**: ~9000 m (Himalayas, Greenland ice cap, Andes cordillera).
//!
//! Run: `cargo run --release -p bake-earth-topography`.

use std::error::Error;
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter, Write, copy};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SOURCE_URL: &str = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif";
const SOURCE_RELATIVE: &str = "source_assets/ETOPO_2022_v1_60s_N90W180_surface.tif";
const OUTPUT_RELATIVE: &str = "client/veldera/assets/world/earth_topography.png";

const OUTPUT_W: u32 = 2048;
const OUTPUT_H: u32 = 1024;
const ELEVATION_MIN_M: f32 = -500.0;
const ELEVATION_MAX_M: f32 = 9000.0;

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = find_workspace_root()?;
    let source_path = project_root.join(SOURCE_RELATIVE);
    let output_path = project_root.join(OUTPUT_RELATIVE);

    if let Some(parent) = source_path.parent() {
        create_dir_all(parent)?;
    }
    if let Some(parent) = output_path.parent() {
        create_dir_all(parent)?;
    }

    if source_path.exists() {
        println!("Reusing cached source: {}", source_path.display());
    } else {
        println!("Downloading source GeoTIFF from {SOURCE_URL}");
        download(SOURCE_URL, &source_path)?;
    }

    let start = Instant::now();
    println!("Decoding GeoTIFF (this allocates ~900 MB; please be patient)...");
    let (src_w, src_h, src) = read_float_tiff(&source_path)?;
    println!(
        "  loaded {src_w}×{src_h} float grid in {:.1}s",
        start.elapsed().as_secs_f32()
    );

    let start = Instant::now();
    println!("Downsampling to {OUTPUT_W}×{OUTPUT_H} via box average...");
    let resampled = downsample_average(&src, src_w, src_h, OUTPUT_W, OUTPUT_H);
    println!("  done in {:.1}s", start.elapsed().as_secs_f32());

    println!(
        "Quantising to 8-bit, elevation range [{ELEVATION_MIN_M} m, {ELEVATION_MAX_M} m]..."
    );
    let bytes: Vec<u8> = resampled
        .iter()
        .map(|&v| {
            let t = (v - ELEVATION_MIN_M) / (ELEVATION_MAX_M - ELEVATION_MIN_M);
            (t.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();

    println!("Writing {}", output_path.display());
    write_grayscale_png(&output_path, OUTPUT_W, OUTPUT_H, &bytes)?;

    let nonzero = bytes.iter().filter(|&&b| b > 0).count();
    let max = bytes.iter().copied().max().unwrap_or(0);
    let sum: u64 = bytes.iter().map(|&b| u64::from(b)).sum();
    let mean = sum as f32 / bytes.len() as f32;
    println!(
        "Stats: {} px ({:.1}% land), max={}, mean={:.1}",
        nonzero,
        100.0 * nonzero as f32 / bytes.len() as f32,
        max,
        mean,
    );

    println!("Done.");
    Ok(())
}

/// Walk up from this crate's manifest dir to find the workspace root —
/// the directory whose `Cargo.toml` contains `[workspace]`.
fn find_workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for candidate in start.ancestors() {
        let manifest = candidate.join("Cargo.toml");
        if let Ok(text) = std::fs::read_to_string(&manifest)
            && text.contains("[workspace]")
        {
            return Ok(candidate.to_path_buf());
        }
    }
    Err("could not locate workspace root from this binary's manifest dir".into())
}

fn download(url: &str, dest: &Path) -> Result<(), Box<dyn Error>> {
    let mut response = reqwest::blocking::get(url)?.error_for_status()?;
    if let Some(total) = response.content_length() {
        println!("  size: {} MB", total / (1024 * 1024));
    }
    // Write to a `.part` file and rename on success so an interrupted
    // download doesn't leave a half-finished file at the cached path.
    let tmp = dest.with_extension("part");
    let mut file = BufWriter::new(File::create(&tmp)?);
    copy(&mut response, &mut file)?;
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

/// Read a TIFF as a flat row-major `Vec<f32>`. ETOPO 60s is float32 in
/// metres; we promote any integer formats to f32 in case a different
/// resolution is ever swapped in.
fn read_float_tiff(path: &Path) -> Result<(u32, u32, Vec<f32>), Box<dyn Error>> {
    let file = BufReader::new(File::open(path)?);
    // ETOPO 60s decodes to ~900 MB of float32 in one allocation; well
    // past the tiff crate's default decompression-bomb guard. We trust
    // the NOAA source, so opt out of the limits entirely.
    let mut decoder = tiff::decoder::Decoder::new(file)?
        .with_limits(tiff::decoder::Limits::unlimited());
    let (w, h) = decoder.dimensions()?;
    let image = decoder.read_image()?;
    use tiff::decoder::DecodingResult::*;
    let data = match image {
        F32(v) => v,
        F64(v) => v.into_iter().map(|x| x as f32).collect(),
        I16(v) => v.into_iter().map(f32::from).collect(),
        I32(v) => v.into_iter().map(|x| x as f32).collect(),
        U16(v) => v.into_iter().map(f32::from).collect(),
        other => {
            return Err(format!(
                "unsupported TIFF pixel format ({}); expected float32",
                describe_decoding_result(&other),
            )
            .into());
        }
    };
    Ok((w, h, data))
}

fn describe_decoding_result(d: &tiff::decoder::DecodingResult) -> &'static str {
    use tiff::decoder::DecodingResult::*;
    match d {
        U8(_) => "U8",
        U16(_) => "U16",
        U32(_) => "U32",
        U64(_) => "U64",
        I8(_) => "I8",
        I16(_) => "I16",
        I32(_) => "I32",
        I64(_) => "I64",
        F16(_) => "F16",
        F32(_) => "F32",
        F64(_) => "F64",
    }
}

/// Box-average downsample. Each output pixel is the mean of every source
/// pixel whose centre falls inside its footprint. Slightly more involved
/// than a fixed N×N average because the ratio isn't an integer
/// (21600 → 2048 is ×10.55), but cheap enough (~22M out × ~100 in samples
/// = ~2 billion adds, a few seconds in release mode).
fn downsample_average(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; (dst_w as usize) * (dst_h as usize)];
    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;
    for y in 0..dst_h {
        let y0 = (y as f32 * sy).floor() as u32;
        let y1 = ((y + 1) as f32 * sy).ceil() as u32;
        let y1 = y1.min(src_h);
        for x in 0..dst_w {
            let x0 = (x as f32 * sx).floor() as u32;
            let x1 = ((x + 1) as f32 * sx).ceil() as u32;
            let x1 = x1.min(src_w);
            let mut sum = 0.0_f32;
            let mut count = 0u32;
            for sy_i in y0..y1 {
                let row = (sy_i as usize) * (src_w as usize);
                for sx_i in x0..x1 {
                    sum += src[row + sx_i as usize];
                    count += 1;
                }
            }
            if count > 0 {
                out[(y as usize) * (dst_w as usize) + (x as usize)] = sum / count as f32;
            }
        }
    }
    out
}

fn write_grayscale_png(
    path: &Path,
    w: u32,
    h: u32,
    data: &[u8],
) -> Result<(), Box<dyn Error>> {
    let file = BufWriter::new(File::create(path)?);
    let mut encoder = png::Encoder::new(file, w, h);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(data)?;
    Ok(())
}
