//! HTTP client for fetching Google Earth mesh data.
//!
//! This module provides the main `Client` type for downloading planetoid metadata,
//! bulk metadata, and node data from Google Earth's servers.

use crate::cache::{Cache, NoCache};
use crate::error::{Error, Result};
use crate::types::{
    BulkMetadata, BulkRequest, Mesh, Node, NodeMetadata, NodeRequest, Planetoid, TextureFormat,
};
use glam::{DMat4, Vec3};
use prost::Message;
use rocktree_decode::OrientedBoundingBox;
use rocktree_proto as proto;
use std::sync::Arc;

/// Base URL for Google Earth's rocktree API.
const BASE_URL: &str = "https://kh.google.com/rt/earth/";

/// HTTP client for fetching Google Earth mesh data.
///
/// The client handles HTTP requests, caching, and protobuf decoding. It is
/// designed to be runtime-agnostic and works with any async executor.
///
/// # Example
///
/// ```ignore
/// let client = Client::new();
/// let planetoid = client.fetch_planetoid().await?;
/// ```
pub struct Client<C: Cache = NoCache> {
    http: reqwest::Client,
    cache: Arc<C>,
    base_url: String,
}

impl Client<NoCache> {
    /// Create a new client with default settings and no caching.
    #[must_use]
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            cache: Arc::new(NoCache),
            base_url: BASE_URL.to_string(),
        }
    }
}

impl Default for Client<NoCache> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Cache> Client<C> {
    /// Create a new client with a custom cache.
    #[must_use]
    pub fn with_cache(cache: C) -> Self {
        Self {
            http: reqwest::Client::new(),
            cache: Arc::new(cache),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Create a new client with a custom HTTP client and cache.
    #[must_use]
    pub fn with_http_and_cache(http: reqwest::Client, cache: C) -> Self {
        Self {
            http,
            cache: Arc::new(cache),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Set a custom base URL for testing.
    #[must_use]
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Fetch the root planetoid metadata.
    ///
    /// This returns information about the planet including radius and the
    /// epoch for the root bulk metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the response cannot be decoded.
    pub async fn fetch_planetoid(&self) -> Result<Planetoid> {
        let url = format!("{}PlanetoidMetadata", self.base_url);
        let data = self.fetch_bytes(&url).await?;

        let proto = proto::PlanetoidMetadata::decode(data.as_slice()).map_err(|e| {
            Error::Protobuf {
                context: "planetoid metadata",
                message: e.to_string(),
            }
        })?;

        let root_epoch = proto
            .root_node_metadata
            .as_ref()
            .map_or(0, |r| r.epoch.unwrap_or(0));

        Ok(Planetoid {
            radius: f64::from(proto.radius.unwrap_or(0.0)),
            root_epoch,
        })
    }

    /// Fetch bulk metadata for a given path and epoch.
    ///
    /// Bulk metadata contains information about a region of the octree,
    /// including node metadata and child bulk paths.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the response cannot be decoded.
    pub async fn fetch_bulk(&self, request: &BulkRequest) -> Result<BulkMetadata> {
        let url = format!(
            "{}BulkMetadata/pb=!1m2!1s{}!2u{}",
            self.base_url, request.path, request.epoch
        );
        let data = self.fetch_bytes(&url).await?;

        let proto =
            proto::BulkMetadata::decode(data.as_slice()).map_err(|e| Error::Protobuf {
                context: "bulk metadata",
                message: e.to_string(),
            })?;

        Self::decode_bulk_metadata(&request.path, &proto)
    }

    /// Fetch node data for a given request.
    ///
    /// Node data contains the actual mesh geometry and textures for rendering.
    ///
    /// # Errors
    ///
    /// Returns an error if the HTTP request fails or the response cannot be decoded.
    pub async fn fetch_node(&self, request: &NodeRequest) -> Result<Node> {
        let url = if let Some(imagery_epoch) = request.imagery_epoch {
            format!(
                "{}NodeData/pb=!1m2!1s{}!2u{}!2e{}!3u{}!4b0",
                self.base_url,
                request.path,
                request.epoch,
                request.texture_format,
                imagery_epoch
            )
        } else {
            format!(
                "{}NodeData/pb=!1m2!1s{}!2u{}!2e{}!4b0",
                self.base_url, request.path, request.epoch, request.texture_format
            )
        };
        let data = self.fetch_bytes(&url).await?;

        let proto = proto::NodeData::decode(data.as_slice()).map_err(|e| Error::Protobuf {
            context: "node data",
            message: e.to_string(),
        })?;

        Self::decode_node_data(&request.path, &proto)
    }

    /// Fetch raw bytes from a URL, using cache if available.
    async fn fetch_bytes(&self, url: &str) -> Result<Vec<u8>> {
        // Check cache first.
        if let Some(data) = self.cache.get(url).await? {
            tracing::debug!(url, "cache hit");
            return Ok(data);
        }

        tracing::debug!(url, "fetching");

        // Fetch from network.
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Http {
                url: url.to_string(),
                message: e.to_string(),
            })?;

        let status = response.status();
        if !status.is_success() {
            return Err(Error::HttpStatus {
                url: url.to_string(),
                status: status.as_u16(),
            });
        }

        let data = response.bytes().await.map_err(|e| Error::Http {
            url: url.to_string(),
            message: e.to_string(),
        })?;
        let data = data.to_vec();

        // Store in cache.
        self.cache.put(url, data.clone()).await?;

        Ok(data)
    }

    /// Decode bulk metadata from protobuf.
    fn decode_bulk_metadata(base_path: &str, proto: &proto::BulkMetadata) -> Result<BulkMetadata> {
        // Flags from the proto definition.
        const NODATA: u32 = 8;
        const LEAF: u32 = 4;
        const USE_IMAGERY_EPOCH: u32 = 16;

        // head_node_center is Vec<f64>, convert to Vec3.
        let head_node_center = if proto.head_node_center.len() >= 3 {
            #[allow(clippy::cast_possible_truncation)]
            Vec3::new(
                proto.head_node_center[0] as f32,
                proto.head_node_center[1] as f32,
                proto.head_node_center[2] as f32,
            )
        } else {
            Vec3::ZERO
        };

        let head_epoch = proto
            .head_node_key
            .as_ref()
            .and_then(|k| k.epoch)
            .unwrap_or(0);

        let meters_per_texel: Vec<f32> = proto.meters_per_texel.clone();
        #[allow(clippy::cast_possible_wrap)]
        let default_texture_format = proto.default_available_texture_formats.unwrap_or(0) as i32;
        let default_imagery_epoch = proto.default_imagery_epoch;

        let mut nodes = Vec::new();
        let mut child_bulk_paths = Vec::new();

        for node_meta in &proto.node_metadata {
            let path_and_flags = node_meta.path_and_flags.unwrap_or(0);
            let pf = rocktree_decode::unpack_path_and_flags(path_and_flags);

            let full_path = format!("{base_path}{}", pf.path);

            let has_data = (pf.flags & NODATA) == 0;
            let is_leaf = (pf.flags & LEAF) != 0;
            let use_imagery_epoch = (pf.flags & USE_IMAGERY_EPOCH) != 0;

            // Check for child bulk (4-char paths that aren't leaves).
            if pf.path.len() == 4 && !is_leaf {
                let epoch = node_meta.bulk_metadata_epoch.unwrap_or(head_epoch);
                child_bulk_paths.push(pf.path.clone());
                // Store epoch info separately if needed.
                let _ = epoch; // Epoch is used when creating BulkRequest for child.
            }

            // Skip nodes without OBB if they have data or aren't leaves.
            let has_obb = node_meta.oriented_bounding_box.is_some();
            if (has_data || !is_leaf) && !has_obb {
                continue;
            }

            if (has_data || !is_leaf) && has_obb {
                let meters_per_texel_value = node_meta.meters_per_texel.unwrap_or_else(|| {
                    if pf.level > 0 && (pf.level - 1) < meters_per_texel.len() {
                        meters_per_texel[pf.level - 1]
                    } else {
                        1.0
                    }
                });

                let obb_data = node_meta.oriented_bounding_box.as_ref().unwrap();
                let obb = rocktree_decode::unpack_obb(
                    obb_data,
                    head_node_center,
                    meters_per_texel_value,
                )?;

                let epoch = node_meta.epoch.unwrap_or(head_epoch);

                #[allow(clippy::cast_possible_wrap)]
                let texture_format = node_meta
                    .available_texture_formats
                    .map_or(default_texture_format, |f| f as i32);

                let imagery_epoch = if use_imagery_epoch {
                    node_meta.imagery_epoch.or(default_imagery_epoch)
                } else {
                    None
                };

                nodes.push(NodeMetadata {
                    path: full_path,
                    meters_per_texel: meters_per_texel_value,
                    obb,
                    has_data,
                    epoch,
                    texture_format: select_texture_format(texture_format),
                    imagery_epoch,
                });
            }
        }

        Ok(BulkMetadata {
            path: base_path.to_string(),
            head_node_center,
            meters_per_texel,
            nodes,
            child_bulk_paths,
            epoch: head_epoch,
        })
    }

    /// Decode node data from protobuf.
    fn decode_node_data(path: &str, proto: &proto::NodeData) -> Result<Node> {
        let matrix_data: &[f64] = &proto.matrix_globe_from_mesh;
        let matrix_globe_from_mesh = if matrix_data.len() == 16 {
            DMat4::from_cols_array(matrix_data.try_into().unwrap_or(&[0.0; 16]))
        } else {
            DMat4::IDENTITY
        };

        let mut meshes = Vec::new();

        for mesh_proto in &proto.meshes {
            let mesh = Self::decode_mesh(mesh_proto)?;
            meshes.push(mesh);
        }

        // Get OBB from first mesh if available (or create a default).
        let obb = OrientedBoundingBox {
            center: glam::DVec3::ZERO,
            extents: glam::DVec3::ONE,
            orientation: glam::DMat3::IDENTITY,
        };

        Ok(Node {
            path: path.to_string(),
            matrix_globe_from_mesh,
            meters_per_texel: 1.0, // Will be set from metadata.
            obb,
            meshes,
        })
    }

    /// Decode a mesh from protobuf.
    fn decode_mesh(proto: &proto::Mesh) -> Result<Mesh> {
        // Unpack vertices.
        let vertices_data = proto.vertices.as_deref().unwrap_or(&[]);
        let mut vertices = rocktree_decode::unpack_vertices(vertices_data)?;

        // Unpack indices.
        let indices_data = proto.indices.as_deref().unwrap_or(&[]);
        let indices = rocktree_decode::unpack_indices(indices_data)?;

        // Unpack texture coordinates.
        let texcoords_data = proto.texture_coordinates.as_deref().unwrap_or(&[]);
        let uv_transform = if !texcoords_data.is_empty() && !vertices.is_empty() {
            rocktree_decode::unpack_tex_coords(texcoords_data, &mut vertices)?
        } else {
            rocktree_decode::UvTransform::default()
        };

        // Apply explicit UV offset/scale if provided.
        let uv_transform = if proto.uv_offset_and_scale.len() == 4 {
            rocktree_decode::UvTransform {
                offset: glam::Vec2::new(
                    proto.uv_offset_and_scale[0],
                    proto.uv_offset_and_scale[1],
                ),
                scale: glam::Vec2::new(
                    proto.uv_offset_and_scale[2],
                    proto.uv_offset_and_scale[3],
                ),
            }
        } else {
            // Flip V coordinate.
            rocktree_decode::UvTransform {
                offset: glam::Vec2::new(
                    uv_transform.offset.x,
                    uv_transform.offset.y - 1.0 / uv_transform.scale.y,
                ),
                scale: glam::Vec2::new(uv_transform.scale.x, -uv_transform.scale.y),
            }
        };

        // Unpack octant masks and get layer bounds.
        let octant_data = proto.layer_and_octant_counts.as_deref().unwrap_or(&[]);
        let layer_bounds = if !octant_data.is_empty() && !indices.is_empty() && !vertices.is_empty()
        {
            rocktree_decode::unpack_octant_mask_and_layer_bounds(
                octant_data,
                &indices,
                &mut vertices,
            )?
        } else {
            [indices.len(); 10]
        };

        // Truncate indices to layer 3 bound (visible geometry).
        let visible_index_count = layer_bounds[3].min(indices.len());
        let indices: Vec<u16> = indices.into_iter().take(visible_index_count).collect();

        // Decode texture.
        let (texture_data, texture_format, texture_width, texture_height) =
            Self::decode_texture(proto)?;

        Ok(Mesh {
            vertices,
            indices,
            uv_transform,
            texture_data,
            texture_format,
            texture_width,
            texture_height,
        })
    }

    /// Decode texture data from a mesh.
    fn decode_texture(mesh: &proto::Mesh) -> Result<(Vec<u8>, TextureFormat, u32, u32)> {
        let textures = &mesh.texture;
        if textures.is_empty() {
            return Err(Error::InvalidData {
                context: "mesh texture",
                detail: "no textures found".to_string(),
            });
        }

        let texture = &textures[0];
        if texture.data.is_empty() {
            return Err(Error::InvalidData {
                context: "mesh texture",
                detail: "no texture data found".to_string(),
            });
        }

        let tex_data = &texture.data[0];
        let width = texture.width.unwrap_or(256);
        let height = texture.height.unwrap_or(256);

        let format = texture.format.unwrap_or(proto::texture::Format::Jpg as i32);
        match format {
            f if f == proto::texture::Format::Jpg as i32 => {
                let decoded = rocktree_decode::texture::decode_jpeg_to_rgba(tex_data)?;
                Ok((decoded.data, TextureFormat::Rgb, width, height))
            }
            f if f == proto::texture::Format::CrnDxt1 as i32 => {
                let decoded = rocktree_decode::texture::decode_crn_to_rgba(tex_data)?;
                // Return as RGBA since we fully decode CRN.
                Ok((
                    decoded.data,
                    TextureFormat::Rgba,
                    decoded.width,
                    decoded.height,
                ))
            }
            other => Err(Error::InvalidData {
                context: "texture format",
                detail: format!("unsupported format: {other}"),
            }),
        }
    }
}

/// Select the best texture format from available formats bitmask.
fn select_texture_format(available: i32) -> i32 {
    // Preference order: CRN_DXT1 (6), JPG (1).
    const CRN_DXT1: i32 = 6;
    const JPG: i32 = 1;

    let supported = [CRN_DXT1, JPG];

    for format in supported {
        // Format availability is encoded as (1 << (format - 1)).
        if available & (1 << (format - 1)) != 0 {
            return format;
        }
    }

    // Default to first supported.
    supported[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_texture_format_prefers_crn() {
        // Both available.
        let both = (1 << (6 - 1)) | (1 << (1 - 1));
        assert_eq!(select_texture_format(both), 6);
    }

    #[test]
    fn test_select_texture_format_jpg_only() {
        let jpg_only = 1 << (1 - 1);
        assert_eq!(select_texture_format(jpg_only), 1);
    }

    #[test]
    fn test_select_texture_format_none_available() {
        // Returns default (CRN_DXT1).
        assert_eq!(select_texture_format(0), 6);
    }

    #[test]
    fn test_client_default() {
        let client = Client::new();
        assert!(client.base_url.starts_with("https://"));
    }
}
