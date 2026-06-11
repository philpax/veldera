//! Binary buffer accumulator for the glTF .glb output.
//!
//! Accumulates raw bytes for a single glTF binary buffer. Each "add_*"
//! method writes a typed slice, registers a `BufferView` for it, and
//! returns the view's index. Accessor creation is the caller's job
//! (so the same view can be re-used by multiple accessors, e.g. an
//! interleaved attribute stream).

use gltf_json::{
    Index,
    buffer::{Target, View},
    validation::USize64,
};
use std::io::Write;

pub struct BufferBuilder {
    pub data: Vec<u8>,
    pub views: Vec<View>,
}

impl BufferBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            views: Vec::new(),
        }
    }

    /// Append encoded image bytes (PNG/JPEG). No target; image data lives
    /// in its own buffer view per the glTF spec.
    pub fn add_image(&mut self, bytes: &[u8]) -> Index<View> {
        self.add_view(bytes, None, None)
    }

    /// Append a tightly-packed typed slice for use as a vertex attribute.
    pub fn add_array(&mut self, bytes: &[u8], stride: u32) -> Index<View> {
        self.add_view(bytes, Some(Target::ArrayBuffer), Some(stride))
    }

    /// Append index data.
    pub fn add_indices(&mut self, bytes: &[u8]) -> Index<View> {
        self.add_view(bytes, Some(Target::ElementArrayBuffer), None)
    }

    fn add_view(
        &mut self,
        bytes: &[u8],
        target: Option<Target>,
        stride: Option<u32>,
    ) -> Index<View> {
        self.align_to(4);
        let offset = self.data.len();
        self.data.write_all(bytes).expect("Vec write is infallible");
        let view = View {
            buffer: Index::new(0),
            byte_length: USize64(bytes.len() as u64),
            byte_offset: Some(USize64(offset as u64)),
            byte_stride: stride.map(|s| gltf_json::buffer::Stride(s as usize)),
            extensions: None,
            extras: Default::default(),
            name: None,
            target: target.map(gltf_json::validation::Checked::Valid),
        };
        let idx = self.views.len() as u32;
        self.views.push(view);
        Index::new(idx)
    }

    fn align_to(&mut self, alignment: usize) {
        while !self.data.len().is_multiple_of(alignment) {
            self.data.push(0);
        }
    }
}
