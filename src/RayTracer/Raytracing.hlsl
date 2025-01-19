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

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#define HLSL
#include "RaytracingHlslCompat.h"

static float pi = 3.14159265359;

RaytracingAccelerationStructure Scene : register(t0, space0);
RWTexture2D<float4> RenderTarget : register(u0);
ByteAddressBuffer Indices : register(t1, space0);
StructuredBuffer<Vertex> Vertices : register(t2, space0);
ByteAddressBuffer InstanceDatas : register(t3, space0);
ByteAddressBuffer CDFBuffer : register(t4, space0);

ConstantBuffer<SceneConstantBuffer> g_sceneCB : register(b0);
ConstantBuffer<CubeConstantBuffer> g_cubeCB : register(b1);

float hash(float3 p) {
    return frac(sin(dot(p, float3(12.9898, 78.233, 45.164))) * 43758.5453123);
}

float2 random2D(float3 seed) {
    return float2(hash(seed), hash(seed + float3(1.0, 0.0, 0.0)));
}


float3 CreateNonColinearVector(float3 normal) {
    // Choose a vector that is not colinear with the normal
    float3 nonColinearVector;
    if (abs(normal.x) < abs(normal.y) && abs(normal.x) < abs(normal.z)) {
        nonColinearVector = float3(1.0, 0.0, 0.0);
    } else if (abs(normal.y) < abs(normal.z)) {
        nonColinearVector = float3(0.0, 1.0, 0.0);
    } else {
        nonColinearVector = float3(0.0, 0.0, 1.0);
    }
    return nonColinearVector;
}

float3 randomInHemisphere(float3 normal, float3 seed) {
    // Generate two random numbers
    //float2 rand = random2D(seed);
    //float phi = rand.x * 2.0 * 3.14159265; // Random angle around the normal
    //float cosTheta = sqrt(rand.y);         // Bias toward the normal
    //float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    //// Convert to Cartesian coordinates (local space)
    //float3 tangent = normalize(abs(normal.y) > 0.99 ? float3(1, 0, 0) : cross(float3(0, 1, 0), normal));
    //float3 bitangent = cross(normal, tangent);
    //float3 randomDir = tangent * cos(phi) * sinTheta +
    //                   bitangent * sin(phi) * sinTheta +
    //                   normal * cosTheta;
    //return normalize(randomDir);

	float2 rand = random2D(seed);
	float r1 = rand.x;
	float r2 = rand.y;

	float3 r = normal;
	float3 rp = CreateNonColinearVector(r);
	float3 u = normalize(cross(r, rp));
	float3 v = normalize(cross(r, u));

	float3 uu = u * sqrt(r2) * cos(2.0f * pi * r1);
	float3 vv = v * sqrt(r2) * sin(2.0f * pi * r1);
	float3 nn = r * sqrt(1.0f - r2);

	return normalize(uu + vv + nn);
}


// Load three 16 bit indices from a byte addressed buffer.
uint3 Load3x16BitIndices(uint offsetBytes)
{
    uint3 indices;

    // ByteAdressBuffer loads must be aligned at a 4 byte boundary.
    // Since we need to read three 16 bit indices: { 0, 1, 2 } 
    // aligned at a 4 byte boundary as: { 0 1 } { 2 0 } { 1 2 } { 0 1 } ...
    // we will load 8 bytes (~ 4 indices { a b | c d }) to handle two possible index triplet layouts,
    // based on first index's offsetBytes being aligned at the 4 byte boundary or not:
    //  Aligned:     { 0 1 | 2 - }
    //  Not aligned: { - 0 | 1 2 }
    const uint dwordAlignedOffset = offsetBytes & ~3;    
    const uint2 four16BitIndices = Indices.Load2(dwordAlignedOffset);
 
    // Aligned: { 0 1 | 2 - } => retrieve first three 16bit indices
    if (dwordAlignedOffset == offsetBytes)
    {
        indices.x = four16BitIndices.x & 0xffff;
        indices.y = (four16BitIndices.x >> 16) & 0xffff;
        indices.z = four16BitIndices.y & 0xffff;
    }
    else // Not aligned: { - 0 | 1 2 } => retrieve last three 16bit indices
    {
        indices.x = (four16BitIndices.x >> 16) & 0xffff;
        indices.y = four16BitIndices.y & 0xffff;
        indices.z = (four16BitIndices.y >> 16) & 0xffff;
    }

    return indices;
}

typedef BuiltInTriangleIntersectionAttributes MyAttributes;
struct RayPayload
{
    float4 color;
    int recursion_depth;
    UINT is_shadow_ray;
    UINT is_indirect_ray;
};

// Retrieve hit world position.
float3 HitWorldPosition()
{
    return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
}

// Retrieve attribute at a hit position interpolated from vertex attributes using the hit's barycentrics.
float3 HitAttribute(float3 vertexAttribute[3], BuiltInTriangleIntersectionAttributes attr)
{
    return vertexAttribute[0] +
        attr.barycentrics.x * (vertexAttribute[1] - vertexAttribute[0]) +
        attr.barycentrics.y * (vertexAttribute[2] - vertexAttribute[0]);
}

inline void GenerateCameraRay(uint2 index, float2 sampleOffset, out float3 origin, out float3 direction)
{
    float2 xy = index + sampleOffset; // Adjust pixel position by sample offset.
    float2 screenPos = xy / DispatchRaysDimensions().xy * 2.0 - 1.0;

    // Invert Y for DirectX-style coordinates.
    screenPos.y = -screenPos.y;

    // Unproject the pixel coordinate into a ray.
    float4 world = mul(float4(screenPos, 0, 1), g_sceneCB.projectionToWorld);

    world.xyz /= world.w;
    origin = g_sceneCB.cameraPosition.xyz;
    direction = normalize(world.xyz - origin);
}

// Diffuse lighting calculation.
float4 CalculateDiffuseLighting(float3 hitPosition, float3 normal)
{
    float3 pixelToLight = normalize(g_sceneCB.lightPosition.xyz - hitPosition);

    // Diffuse contribution.
    float fNDotL = max(0.0f, dot(pixelToLight, normal));

    InstanceData i = InstanceDatas.Load<InstanceData>(sizeof(InstanceData) * InstanceID());

    return float4(i.color, 1.0f) * g_sceneCB.lightDiffuseColor * fNDotL;
}

float3 SamplePointOnMesh(in InstanceData mesh, in float3 seed)
{
    uint triangleCount = mesh.triangleCount;
	
	float2 rand = random2D(seed);
    float cdf_random = hash(rand.x * rand.y * 20123424.0f);

    for(uint i = 0; i < triangleCount; i++)
    {
        float cdf = CDFBuffer.Load <float> ((mesh.cdfOffset * 4) + i * 4);
        if(cdf_random <= cdf)
        {
    // Get the base index of the triangle's first 16 bit index.
    //uint indexSizeInBytes = 2;
    //uint indicesPerTriangle = 3;
    //uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    //uint baseIndex = (i.indexOffset * indexSizeInBytes) + PrimitiveIndex() * triangleIndexStride;


            //uint index = i * 3;
            uint3 indices = Load3x16BitIndices((mesh.indexOffset * 2) + i * 3 * 2);
            float3 v0 = Vertices[indices.x + mesh.vertexOffset].position;
            float3 v1 = Vertices[indices.y + mesh.vertexOffset].position;
            float3 v2 = Vertices[indices.z + mesh.vertexOffset].position;

            return v0 * (1.0f - sqrt(rand.x)) + v1 * (sqrt(rand.x) * (1.0f - rand.y)) + v2 * (rand.y * sqrt(rand.x));
        }
    }

    return float3(0, 0, 0);
}


float3 SampleLights(InstanceData mesh, float3 p, float3 n, float3 r_direction, float3 seed)
{
    seed += (p + n + r_direction) * 1229.0f;
    float3 res = float3(0, 0, 0);
    for (int i = 0; i < g_sceneCB.LightCount; i++)
    {
	    InstanceData light = InstanceDatas.Load<InstanceData>(sizeof(InstanceData) * g_sceneCB.LightMeshIndices[i]);

        float3 sampled_point = SamplePointOnMesh(light, seed);
        float A = light.totalArea;
        float3 intensity = light.emission;

        float3 wi = normalize(sampled_point - p);

        //detect shadow

        RayDesc shadowRay;
        shadowRay.Origin = p + n * 0.1f;
        shadowRay.Direction = wi;
        shadowRay.TMin = 0.001;
        shadowRay.TMax = 1e6;

        RayPayload shadowPayload;
        shadowPayload.color = float4(0, 0, 0, 0);
        shadowPayload.recursion_depth = 1;
        shadowPayload.is_shadow_ray = 1;
        TraceRay(Scene, RAY_FLAG_NONE, ~0, 0, 1, 0, shadowRay, shadowPayload);

        if(shadowPayload.color.x > 0.0f)
        {
            continue;
        }

        float3 w0 = -normalize(r_direction);
        float cost = max(0.0, dot(wi, n));

        float3 kd = mesh.color;
        //float3 ks = float3(0.0, 09.0, 1.0);
        //float3 ks = i.color;
        float3 ks = 0.0f;
        float shininess = 200.0f;

        float3 reflected = reflect(-wi, n);
        float cosa = max(0.0, dot(w0, reflected));
        float pw = pow(cosa, shininess);

        float3 brdf = (kd * (1.0f / pi)) + (ks * ((shininess + 2.0f) / (2.0f * pi)) * (pw));
        float3 radiance = (intensity * cost * A) / dot(sampled_point - p, sampled_point - p);
        res += brdf * radiance * cost;
    }

    return res;
}

[shader("raygeneration")]
void MyRaygenShader()
{
    float3 finalColor = float3(0, 0, 0);
    uint sampleCount = 2; // Define the number of samples per pixel.

    // Loop over the number of samples per pixel.
    for (uint i = 0; i < sampleCount; ++i)
    {
        // Generate a random sample offset within the pixel.
        //float2 sampleOffset = float2(
        //    (i % 2) * 0.5 + 0.25, // Example stratified sampling pattern.
       //     (i / 2) * 0.5 + 0.25
       // );
        //generate sample offset using g_sceneCB.random_floats
        float2 sampleOffset = float2(
            g_sceneCB.random_floats[i],
            g_sceneCB.random_floats[i + 1]
        );

        float3 rayDir;
        float3 origin;

        // Generate a ray for the current sample.
        GenerateCameraRay(DispatchRaysIndex().xy, sampleOffset, origin, rayDir);

        // Trace the ray.
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = rayDir;
        ray.TMin = 0.001; // Avoid aliasing issues.
        ray.TMax = 10000.0; // Set maximum ray extent.

        RayPayload payload = { float4(0, 0, 0, 0), 6, 0, 0};
        payload.recursion_depth = 6;
        payload.is_shadow_ray = 0;
        payload.is_indirect_ray = 0;

        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, ray, payload);

        // Accumulate the result for this sample.
        finalColor += payload.color.xyz;
    }

    // Average the accumulated color.
    finalColor /= sampleCount;

    //finalColor = clamp(finalColor, 0.0f, 1.0f);

    // Write the raytraced color to the output texture.
    float4 before = RenderTarget[DispatchRaysIndex().xy] * (g_sceneCB.accumulative_frame_count);
    RenderTarget[DispatchRaysIndex().xy] = (before + float4(finalColor, 1.0f)) / (g_sceneCB.accumulative_frame_count + 1.0f);
    RenderTarget[DispatchRaysIndex().xy].a = 1.0f;
}

[shader("closesthit")]
void MyClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
    if (payload.is_shadow_ray == 1)
    {
        InstanceData i = InstanceDatas.Load < InstanceData > (sizeof(InstanceData) * InstanceID());
        if(i.is_emissive > 0)
        {
            payload.color = float4(0,0,0,0);
            return;
        }
        payload.color = float4(1.0f, 1.0f, 1.0f, 1.0f);
        return;
    }

    payload.recursion_depth--;
    if(payload.recursion_depth <= 0)
    {
        return;
    }
    //adjust color from red to blue according to the instanceID(), max is 8
    //payload.color = float4(InstanceID() / 8.0f, 0.0f, 1.0f - InstanceID() / 8.0f, 0.0f);
    //return;
//
//
    //payload.color = float4(1.0f, 0.0f, 0.0f, 0.0f);
    //return;

    InstanceData i = InstanceDatas.Load < InstanceData > (sizeof(InstanceData) * InstanceID());

    float3 hitPosition = HitWorldPosition();

    // Get the base index of the triangle's first 16 bit index.
    uint indexSizeInBytes = 2;
    uint indicesPerTriangle = 3;
    uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    uint baseIndex = (i.indexOffset * indexSizeInBytes) + PrimitiveIndex() * triangleIndexStride;

    // adjust color from red to blue according to the index offset max is 72
    //payload.color = float4(i.indexOffset / 72.0f, 0.0f, 1.0f - i.indexOffset / 72.0f, 0.0f);
    // Load up 3 16 bit indices for the triangle.
    const uint3 indices = Load3x16BitIndices(baseIndex);


    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 vertexNormals[3] = { 
        Vertices[i.vertexOffset + indices[0]].normal, 
        Vertices[i.vertexOffset + indices[1]].normal, 
        Vertices[i.vertexOffset + indices[2]].normal 
    };


    float3 vertexPositions[3] = { 
        Vertices[i.vertexOffset + indices[0]].position, 
        Vertices[i.vertexOffset + indices[1]].position, 
        Vertices[i.vertexOffset + indices[2]].position 
    };

    // Compute the triangle's normal.
    // This is redundant and done for illustration purposes 
    // as all the per-vertex normals are the same and match triangle's normal in this sample. 
    float3 triangleNormal = HitAttribute(vertexNormals, attr);
    float3 trianglePosition = HitAttribute(vertexPositions, attr);

    //payload.color = float4(hitPosition, 1.0f);  // Initialize payload color
    //return;

    // Emissive contribution
    if (i.is_emissive > 0) {
        if(payload.is_indirect_ray > 0)
        {
            return;
        }
        payload.color += float4(i.emission, 1.0f); // Directly add emissive color to payload
        return;  // Stop further processing since the surface emits light
    }

    // Phong BRDF constants
    const float shininess = 200.0f;  // Shininess for specular highlight
    const float3 baseColor = i.color;  // Base color from instance data


    float3 accumulatedColor = 0;
    //nee sample
    //float3 SampleLights(InstanceData mesh, float3 p, float3 n, float3 r_direction, float3 seed)
    float3 nee_light_sample = SampleLights(i, hitPosition, triangleNormal, WorldRayDirection(), (hitPosition * g_sceneCB.random_floats[0] + triangleNormal) * g_sceneCB.random_floats[3]);

    // Generate three random directions for scattering
    float3 scatterDirections[5];
    for (int j = 0; j < 5; j++) {
        scatterDirections[j] = randomInHemisphere(triangleNormal, float3(hitPosition + triangleNormal * j * 4.33253f));
    }

    // Trace each scattered ray and accumulate radiance
    for (int j = 0; j < 2; j++) {
        RayDesc scatterRay;
        scatterRay.Origin = hitPosition + triangleNormal * 0.001;  // Offset to avoid self-intersection
        scatterRay.Direction = scatterDirections[j];
        scatterRay.TMin = 0.001;
        scatterRay.TMax = 1e6;

        RayPayload scatterPayload;
        scatterPayload.color = float4(0, 0, 0, 0);  // Initialize scatter payload color
        scatterPayload.recursion_depth = payload.recursion_depth;
        scatterPayload.is_shadow_ray = 0;
        scatterPayload.is_indirect_ray = 1;
        TraceRay(Scene, RAY_FLAG_NONE, ~0, 0, 1, 0, scatterRay, scatterPayload);

        float3 wi = normalize(scatterRay.Direction);
        float3 intensity = scatterPayload.color.xyz;
        float3 w0 = -normalize(WorldRayDirection());
        float cost = max(0.0, dot(wi, triangleNormal));

        float3 kd = i.color;
        //float3 ks = float3(0.0, 09.0, 1.0);
        //float3 ks = i.color;
        float3 ks = 0.0f;

        float3 reflected = reflect(-wi, triangleNormal);
        float cosa = max(0.0, dot(w0, reflected));
        float pw = pow(cosa, shininess);

        float3 brdf = (kd * (1.0f / pi)) + (ks * ((shininess + 2.0f) / (2.0f * pi)) * (pw));

        // Add contribution from this ray (Phong BRDF scaling)
        //float3 diffuse = baseColor / 3.14159265;  // Lambertian diffuse term
        //float3 specular = pow(max(dot(normalize(scatterRay.Direction), triangleNormal), 0.0), shininess);
        //float3 brdf = diffuse + specular;

        //accumulatedColor += (intensity * brdf * (3.14)) / 5.0; // Divide by 3 to average contributions
        accumulatedColor += (intensity * brdf * pi); // Divide by 3 to average contributions
        //accumulatedColor += (intensity * kd); // Divide by 3 to average contributions

        //accumulatedColor += scatterPayload.color * brdf * (6.28) / 5.0; // Divide by 3 to average contributions
    }

    payload.color += float4(accumulatedColor / 2.0f, 1.0f) + float4(nee_light_sample, 0.0f); // Add accumulated color to the payload
    //payload.color += float4(accumulatedColor / 2.0f, 1.0f); // Add accumulated color to the payload
    return;

    if (InstanceID() == 1)
    {
	    //mirror material
        float3 normal = normalize(triangleNormal);
        float3 incident = normalize(WorldRayDirection());
        float3 reflection = reflect(incident, normal);
        float3 reflectionOrigin = hitPosition + normal * 0.001f;
        //scatter reflection ray a bit
        //float nois = noise(0, 1);

        reflection += float3(0.0f, 0.2f, 0.0f) * 0.1f;
        RayDesc reflectionRay;
        reflectionRay.Origin = reflectionOrigin;
        reflectionRay.Direction = reflection;
        reflectionRay.TMin = 0.001f;
        reflectionRay.TMax = 10000.0f;
        payload.recursion_depth--;
        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, reflectionRay, payload);
        //color = payload.color * 0.5f;
    }

    //payload.color = color;
}

[shader("closesthit")]
void MySecondClosestHitShader(inout RayPayload payload, in MyAttributes attr)
{
    payload.color = float4(1.0f, 0.0f, 0.0f, 0.0f);
    return;
    float3 hitPosition = HitWorldPosition();

    // Get the base index of the triangle's first 16 bit index.
    uint indexSizeInBytes = 2;
    uint indicesPerTriangle = 3;
    uint triangleIndexStride = indicesPerTriangle * indexSizeInBytes;
    uint baseIndex = PrimitiveIndex() * triangleIndexStride;

    // Load up 3 16 bit indices for the triangle.
    const uint3 indices = Load3x16BitIndices(baseIndex);


    // Retrieve corresponding vertex normals for the triangle vertices.
    float3 vertexNormals[3] = { 
        Vertices[indices[0]].normal, 
        Vertices[indices[1]].normal, 
        Vertices[indices[2]].normal 
    };

    // Compute the triangle's normal.
    // This is redundant and done for illustration purposes 
    // as all the per-vertex normals are the same and match triangle's normal in this sample. 
    float3 triangleNormal = HitAttribute(vertexNormals, attr);

    float4 diffuseColor = CalculateDiffuseLighting(hitPosition, triangleNormal);
    float4 color = g_sceneCB.lightAmbientColor + diffuseColor;


    InstanceData i = InstanceDatas.Load < InstanceData > (sizeof(InstanceData) * InstanceID());

    if (InstanceID() == 1)
    {
	    //mirror material
        float3 normal = normalize(triangleNormal);
        float3 incident = normalize(WorldRayDirection());
        float3 reflection = reflect(incident, normal);
        float3 reflectionOrigin = hitPosition + normal * 0.001f;
        //scatter reflection ray a bit
        //float nois = noise(0, 1);

        reflection += float3(0.0f, 0.2f, 0.0f) * 0.1f;
        RayDesc reflectionRay;
        reflectionRay.Origin = reflectionOrigin;
        reflectionRay.Direction = reflection;
        reflectionRay.TMin = 0.001f;
        reflectionRay.TMax = 10000.0f;
        payload.recursion_depth--;
        TraceRay(Scene, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 1, 0, reflectionRay, payload);
        color = payload.color * 0.5f;
    }

    payload.color = color;
}



[shader("miss")]
void MyMissShader(inout RayPayload payload)
{
    float4 background = float4(0.0f, 0.0f, 0.0f, 0.0f);
    payload.color = background;
}

#endif // RAYTRACING_HLSL