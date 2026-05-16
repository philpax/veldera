// Cloud composite into the HDR view target.
//
// Reads the half-resolution raymarch buffer, bilinear-upsamples to full
// resolution, and outputs `(inscattering, transmittance)` as the fragment
// colour. The pipeline blend mode is configured so:
//
//   dst = src * 1 + dst * src.a
//
// where `src.a` is the cloud transmittance: `dst` is dimmed by the cloud's
// opacity and the cloud's inscattering is added on top.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

#import bevy_pbr_clouds_planet::types::CloudUniform;

@group(0) @binding(0) var<uniform> cloud: CloudUniform;
@group(0) @binding(1) var cloud_raymarch_in: texture_2d<f32>;
@group(0) @binding(2) var cloud_sampler: sampler;

@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSampleLevel(cloud_raymarch_in, cloud_sampler, in.uv, 0.0);
    return vec4(sample.rgb, sample.a);
}
