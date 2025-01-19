#pragma once

#include "raytracingHlslCompat.h"
#include "tiny_obj_loader.h"
#include <iostream>
#include <vector>
#include <string>


struct Material {
    std::string name;
    float ambient[3];
    float diffuse[3];
    float specular[3];
    float shininess;
    std::string diffuseTexture;
};

struct Mesh {
    std::string name;                    // Shape/mesh name from OBJ
    std::vector<Vertex> vertices;
    std::vector<Index> indices;
	UINT32 vertexOffset;            // Offset in the vertex buffer
	UINT32 indexOffset;             // Offset in the index buffer
    int materialId;                      // Single material ID for the entire mesh
};

class ModelLoader {
public:
    bool LoadModel(const std::string& objPath) {
        std::string err;
        
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string mtl_base_dir = "./assets/"; // Path to material files

        bool ret = tinyobj::LoadObj(
            &attrib, &shapes, &materials,
            &err,
            objPath.c_str(),
            mtl_base_dir.c_str());

        if (!err.empty()) {
            std::cerr << "TinyObjReader error: " << err << std::endl;
            return false;
        }

        if (!ret) {
            return false;
        }

        // Load materials
        ProcessMaterials(materials);

        // Load geometry
        ProcessGeometry(attrib, shapes, materials);

        return true;
    }

    std::vector<Mesh>& GetMeshes()  { return meshes; }
    std::vector<Material>& GetMaterials() { return loadedMaterials; }

private:
    void ProcessMaterials(const std::vector<tinyobj::material_t>& materials) {
        loadedMaterials.reserve(materials.size());
        
        for (const auto& mat : materials) {
            Material material;
            
            // Copy material properties
            for (int i = 0; i < 3; i++) {
                material.ambient[i] = mat.ambient[i];
                material.diffuse[i] = mat.diffuse[i];
                material.specular[i] = mat.specular[i];
            }
            
            material.shininess = mat.shininess;
            material.diffuseTexture = mat.diffuse_texname;
            material.name = mat.name;
            
            loadedMaterials.push_back(material);
        }
    }

    void ProcessGeometry(const tinyobj::attrib_t& attrib,
                        const std::vector<tinyobj::shape_t>& shapes,
                        const std::vector<tinyobj::material_t>& materials) {
        // Process each shape
        for (const auto& shape : shapes) {
            // Map to store separate meshes for each material ID
            std::unordered_map<int, Mesh> materialMeshes;
            
            size_t index_offset = 0;
            
            // Loop over faces
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                int fv = shape.mesh.num_face_vertices[f];
                int materialId = shape.mesh.material_ids[f];
                
                // Initialize new mesh for this material if it doesn't exist
                if (materialMeshes.find(materialId) == materialMeshes.end()) {
                    materialMeshes[materialId].name = shape.name + "_mat" + std::to_string(materialId);
                    materialMeshes[materialId].materialId = materialId;
                }
                
                Mesh& currentMesh = materialMeshes[materialId];

                // Loop over vertices in the face
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    
                    Vertex vertex{};

                    // Position
                    vertex.position.x = attrib.vertices[3 * idx.vertex_index + 0];
                    vertex.position.y = attrib.vertices[3 * idx.vertex_index + 1];
                    vertex.position.z = attrib.vertices[3 * idx.vertex_index + 2];

                    // Normal
                    if (idx.normal_index >= 0) {
                        vertex.normal.x = attrib.normals[3 * idx.normal_index + 0];
                        vertex.normal.y = attrib.normals[3 * idx.normal_index + 1];
                        vertex.normal.z = attrib.normals[3 * idx.normal_index + 2];
                    }

                    // Texture coordinates
                    if (idx.texcoord_index >= 0) {
                        vertex.texCoord.x = attrib.texcoords[2 * idx.texcoord_index + 0];
                        vertex.texCoord.y = attrib.texcoords[2 * idx.texcoord_index + 1];
                    }

                    currentMesh.vertices.push_back(vertex);
                    currentMesh.indices.push_back(currentMesh.vertices.size() - 1);
                }
                
                index_offset += fv;
            }
            
            // Add all material meshes to the final mesh list
            for (const auto& [materialId, mesh] : materialMeshes) {
                meshes.push_back(mesh);
            }
        }
    }

    std::vector<Mesh> meshes;
    std::vector<Material> loadedMaterials;
};