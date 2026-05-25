//! Convert a directory of Mixamo FBX files into a single skinned glTF binary.
//!
//! ```text
//! convert-character <input_dir> <output.glb>
//! ```

mod buffer;
mod convert;
mod glb;

use std::{error::Error, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: convert-character <input_dir> <output.glb>");
        std::process::exit(2);
    }
    let input_dir = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let started = Instant::now();
    convert::convert(input_dir, output_path)?;
    println!("Done in {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}
