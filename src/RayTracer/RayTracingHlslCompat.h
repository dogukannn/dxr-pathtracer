//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#ifndef RAYTRACINGHLSLCOMPAT_H
#define RAYTRACINGHLSLCOMPAT_H


//define PAD_VAR for 16 byte alignment

#define PAD_VAR UINT CONCAT(padding_, __COUNTER__);

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b


#ifdef HLSL
#include "HlslCompat.h"
#else
using namespace DirectX;

// Shader will use byte encoding to access indices.
typedef UINT16 Index;
#endif

struct SceneConstantBuffer
{
    XMMATRIX projectionToWorld;
    XMVECTOR cameraPosition;
    XMVECTOR lightPosition;
    XMVECTOR lightAmbientColor;
    XMVECTOR lightDiffuseColor;
    XMVECTOR random_floats;

	float accumulative_frame_count;
    UINT LightCount;
    PAD_VAR;
    PAD_VAR;

    UINT LightMeshIndices[16];
};

struct CubeConstantBuffer
{
    XMFLOAT4 albedo;
};

struct Vertex
{
    XMFLOAT3 position;
    XMFLOAT3 normal;
	XMFLOAT2 texCoord;
};

enum BRDFType : UINT
{
    Phong = 0,
	BlinnPhong = 1,
};

struct InstanceData
{
    XMFLOAT3 color;
    float exponent;

	XMFLOAT3 kd;
	UINT brdfType;

    XMFLOAT3 ks;
	UINT vertexOffset;

	UINT indexOffset;
	UINT cdfOffset;
	UINT triangleCount;
    UINT is_emissive;

    float totalArea;
    XMFLOAT3 emission;
};


#endif // RAYTRACINGHLSLCOMPAT_H