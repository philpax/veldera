//! Protobuf code generator for rocktree types.
//!
//! Run with: `cargo run -p rocktree-proto --bin generate`

use std::path::PathBuf;
use std::{env, fs, io};

fn main() -> io::Result<()> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let proto_path = workspace_root.join("proto/rocktree.proto");
    let out_dir = manifest_dir.join("src/generated");

    println!("Generating protobuf code from: {}", proto_path.display());
    println!("Output directory: {}", out_dir.display());

    // Ensure output directory exists.
    fs::create_dir_all(&out_dir)?;

    // Configure prost to generate code.
    prost_build::Config::new()
        .out_dir(&out_dir)
        .compile_protos(&[&proto_path], &[workspace_root.join("proto")])?;

    // The proto package is `geo_globetrotter_proto_rocktree`, so prost generates
    // `geo_globetrotter_proto_rocktree.rs`. We'll include it directly in mod.rs.
    let generated_file = out_dir.join("geo_globetrotter_proto_rocktree.rs");
    if generated_file.exists() {
        // Read the generated content.
        let content = fs::read_to_string(&generated_file)?;

        // Write mod.rs that includes the generated types directly.
        // Add lint allows for generated code.
        let mod_content = format!(
            "// Generated protobuf types. Do not edit manually.\n\
             // Regenerate with: cargo run -p rocktree-proto --bin generate\n\n\
             #![allow(clippy::doc_markdown)]\n\
             #![allow(clippy::must_use_candidate)]\n\n\
             {content}"
        );
        fs::write(out_dir.join("mod.rs"), mod_content)?;
        fs::remove_file(&generated_file)?;

        println!("Successfully generated protobuf types!");
    } else {
        eprintln!("Warning: Expected generated file not found: {generated_file:?}");
        eprintln!("Available files in output directory:");
        for entry in fs::read_dir(&out_dir)? {
            eprintln!("  {:?}", entry?.path());
        }
    }

    Ok(())
}
