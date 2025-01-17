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

#pragma once

#include "AssetLoader.h"
#include "DXSample.h"
#include "StepTimer.h"
#include "RaytracingHlslCompat.h"

#define MAX_RECURSION_DEPTH 6

namespace GlobalRootSignatureParams {
    enum Value {
        OutputViewSlot = 0,
        AccelerationStructureSlot,
        SceneConstantSlot,
        VertexBuffersSlot,
        InstanceDataBuffersSlot,
        CDFBuffersSlot,
        Count 
    };
}

namespace LocalRootSignatureParams {
    enum Value {
        CubeConstantSlot = 0,
        Count 
    };
}

//struct for handling mouse drag
struct MouseDrag
{
	bool active;
	INT x;
	INT y;
};

class D3D12RaytracingSimpleLighting : public DXSample
{
public:
    D3D12RaytracingSimpleLighting(UINT width, UINT height, std::wstring name);

    // IDeviceNotify
    virtual void OnDeviceLost() override;
    virtual void OnDeviceRestored() override;

    // Messages
    virtual void OnInit();
    virtual void OnUpdate();
    virtual void OnRender();
    virtual void OnSizeChanged(UINT width, UINT height, bool minimized);
    virtual void OnDestroy();
	virtual void OnKeyDown(UINT8 key) override;
    virtual void OnMouseMove(UINT x, UINT y) override;
    virtual void OnLeftButtonDown(UINT x, UINT y) override;
	virtual void OnLeftButtonUp(UINT x, UINT y) override;
    virtual IDXGISwapChain* GetSwapchain() { return m_deviceResources->GetSwapChain(); }

private:
    static const UINT FrameCount = 2;

    // We'll allocate space for several of these and they will need to be padded for alignment.
    static_assert(sizeof(SceneConstantBuffer) <= D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, "Checking the size here.");

    union AlignedSceneConstantBuffer
    {
        SceneConstantBuffer constants;
        uint8_t alignmentPadding[D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT];
    };
    AlignedSceneConstantBuffer*  m_mappedConstantData;
    ComPtr<ID3D12Resource>       m_perFrameConstants;

    // DirectX Raytracing (DXR) attributes
    ComPtr<ID3D12Device5> m_dxrDevice;
    ComPtr<ID3D12GraphicsCommandList5> m_dxrCommandList;
    ComPtr<ID3D12StateObject> m_dxrStateObject;

    // Root signatures
    ComPtr<ID3D12RootSignature> m_raytracingGlobalRootSignature;
    ComPtr<ID3D12RootSignature> m_raytracingLocalRootSignature;
    ComPtr<ID3D12RootSignature> m_raytracingSecondLocalRootSignature;

    // Descriptors
    ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;
    UINT m_descriptorsAllocated;
    UINT m_descriptorSize;

    
    // Raytracing scene
    SceneConstantBuffer m_sceneCB[FrameCount];
    CubeConstantBuffer m_cubeCB;

    // Geometry
    ModelLoader loader;

    // Mouse drag
	MouseDrag m_mouseDrag;
	XMVECTOR m_currentVelocity = XMVectorZero();
	float m_acceleration = 20.0f;      // Units per second squared
	float m_deceleration = 10.0f;      // Units per second squared
	float m_deltaTime = 0.016f;        // Assuming 60 FPS, update this with actual frame time
	bool m_rightMouseDown = false;


    struct D3DBuffer
    {
        ComPtr<ID3D12Resource> resource;
        D3D12_CPU_DESCRIPTOR_HANDLE cpuDescriptorHandle;
        D3D12_GPU_DESCRIPTOR_HANDLE gpuDescriptorHandle;
    };

    //std::vector<D3DBuffer> m_index_buffers;
    //std::vector<D3DBuffer> m_vertex_buffers;

    D3DBuffer m_indexBuffer;
    D3DBuffer m_vertexBuffer;
    D3DBuffer m_cdfBuffer;

	D3DBuffer m_instanceDataBuffer;

    // Acceleration structure
	std::vector<ComPtr<ID3D12Resource>> m_bottomLevelAccelerationStructures;
    ComPtr<ID3D12Resource> m_topLevelAccelerationStructure;

    // Raytracing output
    ComPtr<ID3D12Resource> m_raytracingOutput;
    D3D12_GPU_DESCRIPTOR_HANDLE m_raytracingOutputResourceUAVGpuDescriptor;
    UINT m_raytracingOutputResourceUAVDescriptorHeapIndex;

    // Shader tables
    static const wchar_t* c_hitGroupName;
    static const wchar_t* c_secondHitGroupName;
    static const wchar_t* c_raygenShaderName;
    static const wchar_t* c_closestHitShaderName;
    static const wchar_t* c_secondClosestHitShaderName;
    static const wchar_t* c_missShaderName;
    ComPtr<ID3D12Resource> m_missShaderTable;
    ComPtr<ID3D12Resource> m_hitGroupShaderTable;
    ComPtr<ID3D12Resource> m_rayGenShaderTable;
    
    // Application state
    StepTimer m_timer;
    float m_curRotationAngleRad;
    XMVECTOR m_eye;
    XMVECTOR m_at;
    XMVECTOR m_up;

    void UpdateCameraMatrices();
    void InitializeScene();
    void RecreateD3D();
    void DoRaytracing();
    void CreateConstantBuffers();
    void CreateDeviceDependentResources();
    void CreateWindowSizeDependentResources();
    void ReleaseDeviceDependentResources();
    void ReleaseWindowSizeDependentResources();
    void CreateRaytracingInterfaces();
    void SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig);
    void CreateRootSignatures();
    void CreateLocalRootSignatureSubobjects(CD3DX12_STATE_OBJECT_DESC* raytracingPipeline);
    void CreateRaytracingPipelineStateObject();
    void CreateDescriptorHeap();
    void CreateRaytracingOutputResource();
    void BuildGeometry();
    void BuildAccelerationStructures();
    void BuildShaderTables();
    void UpdateForSizeChange(UINT clientWidth, UINT clientHeight);
    void CopyRaytracingOutputToBackbuffer();
    void CalculateFrameStats();
    UINT AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse = UINT_MAX);
    UINT CreateBufferSRV(D3DBuffer* buffer, UINT numElements, UINT elementSize);
};
