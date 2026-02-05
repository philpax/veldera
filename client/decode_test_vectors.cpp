// Standalone test vector decoder for verification against Rust implementation.
//
// Reads raw protobuf files and outputs decoded results as JSON.
// Build: g++ -std=c++17 -I. -Ieigen -Iproto decode_test_vectors.cpp proto/rocktree.pb.cc `pkg-config --cflags --libs protobuf` -o decode_test_vectors
// Run:   ./decode_test_vectors ../rust/test_vectors

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>

// Include Eigen for matrix types.
#include <Eigen/Dense>
using namespace Eigen;

// Include generated protobuf headers.
#include "proto/rocktree.pb.h"

// Use the protobuf namespace.
using namespace geo_globetrotter_proto_rocktree;

// Include decoder functions.
#include "rocktree_decoder.h"

// Simple JSON output helpers.
void json_start_object(std::ostream& out) { out << "{"; }
void json_end_object(std::ostream& out) { out << "}"; }
void json_start_array(std::ostream& out) { out << "["; }
void json_end_array(std::ostream& out) { out << "]"; }
void json_key(std::ostream& out, const char* key) { out << "\"" << key << "\": "; }
void json_string(std::ostream& out, const std::string& s) { out << "\"" << s << "\""; }
void json_number(std::ostream& out, double n) { out << std::setprecision(17) << n; }
void json_int(std::ostream& out, int64_t n) { out << n; }
void json_comma(std::ostream& out) { out << ", "; }
void json_newline(std::ostream& out) { out << "\n"; }

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to create file: " + path);
    }
    file << content;
}

void decode_bulk_metadata(const std::string& input_path, const std::string& output_path) {
    std::string data = read_file(input_path);

    BulkMetadata bulk;
    if (!bulk.ParseFromString(data)) {
        throw std::runtime_error("Failed to parse BulkMetadata from: " + input_path);
    }

    std::ostringstream out;
    json_start_object(out);
    json_newline(out);

    // head_node_center
    json_key(out, "head_node_center");
    json_start_array(out);
    for (int i = 0; i < bulk.head_node_center_size(); i++) {
        if (i > 0) json_comma(out);
        json_number(out, bulk.head_node_center(i));
    }
    json_end_array(out);
    json_comma(out);
    json_newline(out);

    // meters_per_texel
    json_key(out, "meters_per_texel");
    json_start_array(out);
    for (int i = 0; i < bulk.meters_per_texel_size(); i++) {
        if (i > 0) json_comma(out);
        json_number(out, bulk.meters_per_texel(i));
    }
    json_end_array(out);
    json_comma(out);
    json_newline(out);

    // epoch
    auto epoch = bulk.has_head_node_key() ? bulk.head_node_key().epoch() : 0;
    json_key(out, "epoch");
    json_int(out, epoch);
    json_comma(out);
    json_newline(out);

    // node_metadata - count and decode paths
    json_key(out, "node_count");
    json_int(out, bulk.node_metadata_size());
    json_comma(out);
    json_newline(out);

    // Decode all node metadata paths
    json_key(out, "node_paths");
    json_start_array(out);
    json_newline(out);
    bool first_node = true;
    for (const auto& node_meta : bulk.node_metadata()) {
        auto pf = unpackPathAndFlags(node_meta);
        if (!first_node) {
            json_comma(out);
            json_newline(out);
        }
        first_node = false;

        json_start_object(out);
        json_key(out, "path");
        json_string(out, std::string(pf.path));
        json_comma(out);
        json_key(out, "level");
        json_int(out, pf.level);
        json_comma(out);
        json_key(out, "flags");
        json_int(out, pf.flags);
        json_end_object(out);
    }
    json_newline(out);
    json_end_array(out);
    json_newline(out);

    json_end_object(out);
    json_newline(out);

    write_file(output_path, out.str());
    std::cout << "  Decoded bulk metadata to: " << output_path << std::endl;
}

void decode_node_data(const std::string& input_path, const std::string& output_path) {
    std::string data = read_file(input_path);

    NodeData node_data;
    if (!node_data.ParseFromString(data)) {
        throw std::runtime_error("Failed to parse NodeData from: " + input_path);
    }

    std::ostringstream out;
    json_start_object(out);
    json_newline(out);

    // mesh_count
    json_key(out, "mesh_count");
    json_int(out, node_data.meshes_size());
    json_comma(out);
    json_newline(out);

    // meshes
    json_key(out, "meshes");
    json_start_array(out);
    json_newline(out);

    bool first_mesh = true;
    int mesh_idx = 0;
    for (const auto& mesh : node_data.meshes()) {
        if (!first_mesh) {
            json_comma(out);
            json_newline(out);
        }
        first_mesh = false;

        json_start_object(out);
        json_newline(out);

        // Decode vertices
        auto vertices = unpackVertices(mesh.vertices());
        auto vtx = (vertex_t*)vertices.data();
        auto vertex_count = vertices.size() / sizeof(vertex_t);

        // Decode indices
        auto indices = unpackIndices(mesh.indices());

        // Decode texture coordinates
        Vector2f uv_offset, uv_scale;
        unpackTexCoords(mesh.texture_coordinates(), vertices.data(), vertices.size(), uv_offset, uv_scale);

        // Apply explicit UV offset/scale if provided
        if (mesh.uv_offset_and_scale_size() == 4) {
            uv_offset[0] = mesh.uv_offset_and_scale(0);
            uv_offset[1] = mesh.uv_offset_and_scale(1);
            uv_scale[0] = mesh.uv_offset_and_scale(2);
            uv_scale[1] = mesh.uv_offset_and_scale(3);
        } else {
            uv_offset[1] -= 1.0f / uv_scale[1];
            uv_scale[1] *= -1.0f;
        }

        // Decode octant masks and get layer bounds
        int layer_bounds[10];
        unpackOctantMaskAndOctantCountsAndLayerBounds(
            mesh.layer_and_octant_counts(),
            indices.data(), indices.size(),
            vertices.data(), vertices.size(),
            layer_bounds
        );

        // Get texture dimensions
        int tex_width = 256, tex_height = 256;
        if (mesh.texture_size() > 0) {
            tex_width = mesh.texture(0).width();
            tex_height = mesh.texture(0).height();
        }

        // Output mesh data
        json_key(out, "index");
        json_int(out, mesh_idx++);
        json_comma(out);
        json_newline(out);

        json_key(out, "vertex_count");
        json_int(out, vertex_count);
        json_comma(out);
        json_newline(out);

        // Original indices count (before layer truncation)
        json_key(out, "original_index_count");
        json_int(out, indices.size());
        json_comma(out);
        json_newline(out);

        // Index count after layer 3 truncation
        json_key(out, "index_count");
        json_int(out, layer_bounds[3]);
        json_comma(out);
        json_newline(out);

        json_key(out, "texture_width");
        json_int(out, tex_width);
        json_comma(out);
        json_newline(out);

        json_key(out, "texture_height");
        json_int(out, tex_height);
        json_comma(out);
        json_newline(out);

        json_key(out, "uv_offset");
        json_start_array(out);
        json_number(out, uv_offset[0]);
        json_comma(out);
        json_number(out, uv_offset[1]);
        json_end_array(out);
        json_comma(out);
        json_newline(out);

        json_key(out, "uv_scale");
        json_start_array(out);
        json_number(out, uv_scale[0]);
        json_comma(out);
        json_number(out, uv_scale[1]);
        json_end_array(out);
        json_comma(out);
        json_newline(out);

        // Layer bounds
        json_key(out, "layer_bounds");
        json_start_array(out);
        for (int i = 0; i < 10; i++) {
            if (i > 0) json_comma(out);
            json_int(out, layer_bounds[i]);
        }
        json_end_array(out);
        json_comma(out);
        json_newline(out);

        // First few vertices
        json_key(out, "first_vertices");
        json_start_array(out);
        json_newline(out);
        int max_vertices = std::min((size_t)5, vertex_count);
        for (int i = 0; i < max_vertices; i++) {
            if (i > 0) {
                json_comma(out);
                json_newline(out);
            }
            json_start_object(out);
            json_key(out, "x");
            json_int(out, vtx[i].x);
            json_comma(out);
            json_key(out, "y");
            json_int(out, vtx[i].y);
            json_comma(out);
            json_key(out, "z");
            json_int(out, vtx[i].z);
            json_comma(out);
            json_key(out, "w");
            json_int(out, vtx[i].w);
            json_comma(out);
            json_key(out, "u");
            json_int(out, vtx[i].u);
            json_comma(out);
            json_key(out, "v");
            json_int(out, vtx[i].v);
            json_end_object(out);
        }
        json_newline(out);
        json_end_array(out);
        json_comma(out);
        json_newline(out);

        // First few indices
        json_key(out, "first_indices");
        json_start_array(out);
        int max_indices = std::min((size_t)20, indices.size());
        for (int i = 0; i < max_indices; i++) {
            if (i > 0) json_comma(out);
            json_int(out, indices[i]);
        }
        json_end_array(out);
        json_newline(out);

        json_end_object(out);
    }

    json_newline(out);
    json_end_array(out);
    json_newline(out);

    json_end_object(out);
    json_newline(out);

    write_file(output_path, out.str());
    std::cout << "  Decoded node data to: " << output_path << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_vectors_dir>" << std::endl;
        return 1;
    }

    std::string dir = argv[1];

    std::cout << "Decoding test vectors from: " << dir << std::endl;
    std::cout << std::endl;

    try {
        // Decode bulk metadata
        std::cout << "1. Decoding bulk metadata..." << std::endl;
        decode_bulk_metadata(dir + "/bulk_root.pb", dir + "/bulk_root_cpp.json");

        // Decode node data files
        std::cout << "\n2. Decoding node data..." << std::endl;
        std::vector<std::string> nodes = {"024", "03", "134"};
        for (const auto& node : nodes) {
            std::string input = dir + "/node_" + node + ".pb";
            std::string output = dir + "/node_" + node + "_cpp.json";
            std::cout << "   Processing node '" << node << "'..." << std::endl;
            decode_node_data(input, output);
        }

        std::cout << "\nDone! Compare *_cpp.json files with Rust output." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
