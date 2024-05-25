#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>    // includes for helper CUDA functions
//#include <helper_math.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "defines.h"
#include "tables.h"
#include "MarchingCubesCuda.h"
#include "Hash.cuh"

#define GPU_THREAD_NUM 1024
uint* g_puTriTable = 0;
uint* g_puEdgeTable = 0;		
uint* g_puNumVrtsTable = 0;


// textures containing look-up tables
cudaTextureObject_t triTex;
cudaTextureObject_t numVertsTex;

// volume data
cudaTextureObject_t volumeTex;

__device__ uint3 calcGridIdxU(uint i, uint3 ngrid)
{
	uint3 gridPos;
	uint w = i % (ngrid.x*ngrid.y);
	gridPos.x = w%ngrid.x;
	gridPos.y = w / ngrid.x;
	gridPos.z = i / (ngrid.x*ngrid.y);
	return gridPos;
}

__device__ float sampleVolume2(float *data, uint3 p, uint3 gridSize)
{
	p.x = min(p.x, gridSize.x - 1);
	p.y = min(p.y, gridSize.y - 1);
	p.z = min(p.z, gridSize.z - 1);
	uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
	return data[i];
}

void CuScan(unsigned int* dScanData, unsigned int* dData, unsigned int num)
{
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(dData),
		thrust::device_ptr<unsigned int>(dData + num),
		thrust::device_ptr<unsigned int>(dScanData));
}

__global__
void CompactVoxels(uint *compacted, uint *occupied, uint *occupiedScan, uint num)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (occupied[i] && (i < num)) {
		compacted[occupiedScan[i]] = i;
	}
}

__global__
void ClassifyVoxel2(cudaTextureObject_t numVertsTex, uint* voxelCubeIdx, uint *voxelVerts, uint *voxelTris, uint *voxelOccupied, float *volume,
uint3 ngrids, uint nvoxels, float3 voxel_h, float threshold)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (i < nvoxels) {
		uint3 gridPos = calcGridIdxU(i, ngrids);

#if SAMPLE_VOLUME
		float field[8];
		field[0] = sampleVolume2(volume, gridPos, ngrids);
		field[1] = sampleVolume2(volume, gridPos + make_uint3(1, 0, 0), ngrids);
		field[2] = sampleVolume2(volume, gridPos + make_uint3(1, 1, 0), ngrids);
		field[3] = sampleVolume2(volume, gridPos + make_uint3(0, 1, 0), ngrids);
		field[4] = sampleVolume2(volume, gridPos + make_uint3(0, 0, 1), ngrids);
		field[5] = sampleVolume2(volume, gridPos + make_uint3(1, 0, 1), ngrids);
		field[6] = sampleVolume2(volume, gridPos + make_uint3(1, 1, 1), ngrids);
		field[7] = sampleVolume2(volume, gridPos + make_uint3(0, 1, 1), ngrids);
#else
		float3 p;
		p.x = -1.0f + (gridPos.x*voxel_h.x);
		p.y = -1.0f + (gridPos.y*voxel_h.y);
		p.z = -1.0f + (gridPos.z*voxel_h.z);

		float field[8];
		field[0] = fieldFunc(p);
		field[1] = fieldFunc(p + make_float3(voxel_h.x, 0, 0));
		field[2] = fieldFunc(p + make_float3(voxel_h.x, voxel_h.y, 0));
		field[3] = fieldFunc(p + make_float3(0, voxel_h.y, 0));
		field[4] = fieldFunc(p + make_float3(0, 0, voxel_h.z));
		field[5] = fieldFunc(p + make_float3(voxel_h.x, 0, voxel_h.z));
		field[6] = fieldFunc(p + make_float3(voxel_h.x, voxel_h.y, voxel_h.z));
		field[7] = fieldFunc(p + make_float3(0, voxel_h.y, voxel_h.z));
#endif

		uint cubeindex;
		cubeindex = uint(field[0] < threshold);
		cubeindex += uint(field[1] < threshold) * 2;
		cubeindex += uint(field[2] < threshold) * 4;
		cubeindex += uint(field[3] < threshold) * 8;
		cubeindex += uint(field[4] < threshold) * 16;
		cubeindex += uint(field[5] < threshold) * 32;
		cubeindex += uint(field[6] < threshold) * 64;
		cubeindex += uint(field[7] < threshold) * 128;

		uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

		voxelCubeIdx[i] = cubeindex;	
		voxelVerts[i] = numVerts;		
		voxelTris[i] = numVerts / 3;		

#if SKIP_EMPTY_VOXELS
		voxelOccupied[i] = (numVerts > 0);	
#endif
	}
}

__device__
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);

}

__global__
void CalVertexEdge(float4* edgeVrts, uint *edgeOccupied, float *volume, uint3 dir,
uint3 edgeSize, uint3 ngrids, uint nvoxels, uint nedge,
float3 voxel_h, float3 voxel_min, float threshold)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
	if (i >= nedge) return;

	uint3 gridPos = calcGridIdxU(i, edgeSize);

	float3 p;
	p.x = voxel_min.x + (gridPos.x*voxel_h.x);
	p.y = voxel_min.y + (gridPos.y*voxel_h.y);
	p.z = voxel_min.z + (gridPos.z*voxel_h.z);

	// calculate cell vertex positions
	float3 v[2];
	v[0] = p;
	v[1] = p + make_float3(dir.x*voxel_h.x, dir.y*voxel_h.y, dir.z*voxel_h.z);

	// read field values at neighbouring grid vertices
#if SAMPLE_VOLUME
	float field[2];
	field[0] = sampleVolume2(volume, gridPos, ngrids);
	field[1] = sampleVolume2(volume, gridPos + dir, ngrids);

	uint cubeindex;
	cubeindex = uint(field[0] < threshold);
	cubeindex += uint(field[1] < threshold) * 2;
#else
	// evaluate field values
	float4 field[2];
	field[0] = fieldFunc4(p);
	field[1] = fieldFunc4(p + make_float3(dir.x*voxelSize.x, dir.y*voxelSize.y, dir.z*voxelSize.z));

	uint cubeindex;
	cubeindex = uint(field[0].w < isoValue);
	cubeindex += uint(field[1].w < isoValue) * 2;
#endif


	if (cubeindex == 1 || cubeindex == 2){
		float3 vertex, normal = make_float3(0.0f, 5.0f, 0.0f);

#if SAMPLE_VOLUME
		vertex = vertexInterp(threshold, v[0], v[1], field[0], field[1]);
#else
		vertexInterp2(isoValue, v[0], v[1], field[0], field[1], vertex, normal);
#endif

		edgeVrts[i] = make_float4(vertex.x, vertex.y, vertex.z, 1.0f);
		edgeOccupied[i] = 1;
	}
	else{
		//edgeVrts[i] = make_float4(0.0f, 5.0f, 0.0f, 1.0f);
		edgeOccupied[i] = 0;
	}
}

__global__
void CompactEdges(float4 *compactedVrts, uint *occupied, uint *occupiedScan, float4 *vrts, uint num)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (occupied[i] && (i < num)) {
		compactedVrts[occupiedScan[i]] = vrts[i];
	}
}

__device__
uint calcGridIdx3(uint3 p, uint3 ngrid)
{
	p.x = min(p.x, ngrid.x - 1);
	p.y = min(p.y, ngrid.y - 1);
	p.z = min(p.z, ngrid.z - 1);
	return (p.z*ngrid.x*ngrid.y) + (p.y*ngrid.x) + p.x;
}

// calculate triangle normal
__device__
float3 calcNormal(float4 v0, float4 v1, float4 v2)
{
	float3 edge0 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
	float3 edge1 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
	// note - it's faster to perform normalization in vertex shader rather than here
	return Cross(edge0, edge1);
}

__global__
void GenerateTriangles3(cudaTextureObject_t triTex, cudaTextureObject_t numVertsTex, uint3 *vertIdx, uint *voxelTrisScan, uint *edgeOccupiedScan, uint *voxelCubeIdx,
uint3 edgeSizeX, uint3 edgeSizeY, uint3 edgeSizeZ, uint3 edgeNum,
uint *compactedVoxelArray, uint3 ngrids, uint activeVoxels,
float3 voxelSize, float isoValue, uint maxVerts, uint numMesh, float4 *dCompactedEdgeVrts)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (idx > activeVoxels - 1){
		return;
		//idx = activeVoxels-1;
	}

#if SKIP_EMPTY_VOXELS
	uint voxel = compactedVoxelArray[idx];
#else
	uint voxel = idx;
#endif

	uint3 gpos = calcGridIdxU(voxel, ngrids);

	uint cubeindex = voxelCubeIdx[voxel];

#if USE_SHARED
	__shared__ uint vertlist[12 * NTHREADS];
	vertlist[12 * threadIdx.x + 0] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z), edgeSizeX)];
	vertlist[12 * threadIdx.x + 1] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y, gpos.z), edgeSizeY) + edgeNum.x];
	vertlist[12 * threadIdx.x + 2] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y + 1, gpos.z), edgeSizeX)];
	vertlist[12 * threadIdx.x + 3] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z), edgeSizeY) + edgeNum.x];
	vertlist[12 * threadIdx.x + 4] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z + 1), edgeSizeX)];
	vertlist[12 * threadIdx.x + 5] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y, gpos.z + 1), edgeSizeY) + edgeNum.x];
	vertlist[12 * threadIdx.x + 6] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y + 1, gpos.z + 1), edgeSizeX)];
	vertlist[12 * threadIdx.x + 7] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z + 1), edgeSizeY) + edgeNum.x];
	vertlist[12 * threadIdx.x + 8] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
	vertlist[12 * threadIdx.x + 9] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
	vertlist[12 * threadIdx.x + 10] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y + 1, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
	vertlist[12 * threadIdx.x + 11] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y + 1, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
	__syncthreads();
#else
	uint vertlist[12];
	vertlist[0] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z), edgeSizeX)];
	vertlist[2] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y + 1, gpos.z), edgeSizeX)];
	vertlist[4] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z + 1), edgeSizeX)];
	vertlist[6] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y + 1, gpos.z + 1), edgeSizeX)];

	vertlist[1] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y, gpos.z), edgeSizeY) + edgeNum.x];
	vertlist[3] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z), edgeSizeY) + edgeNum.x];
	vertlist[5] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y, gpos.z + 1), edgeSizeY) + edgeNum.x];
	vertlist[7] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z + 1), edgeSizeY) + edgeNum.x];

	vertlist[8] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
	vertlist[9] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
	vertlist[10] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x + 1, gpos.y + 1, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
	vertlist[11] = edgeOccupiedScan[calcGridIdx3(make_uint3(gpos.x, gpos.y + 1, gpos.z), edgeSizeZ) + edgeNum.x + edgeNum.y];
#endif

	// output triangle
	uint numTri = tex1Dfetch<uint>(numVertsTex, cubeindex) / 3;

	for (int i = 0; i < numTri; ++i){
		uint index = voxelTrisScan[voxel] + i;

		uint vidx[3];
		uint edge[3];
		edge[0] = tex1Dfetch<uint>(triTex, (cubeindex * 16) + 3 * i);
		edge[1] = tex1Dfetch<uint>(triTex, (cubeindex * 16) + 3 * i + 1);
		edge[2] = tex1Dfetch<uint>(triTex, (cubeindex * 16) + 3 * i + 2);

#if USE_SHARED
		vidx[0] = min(vertlist[12 * threadIdx.x + edge[0]], maxVerts - 1);
		vidx[2] = min(vertlist[12 * threadIdx.x + edge[1]], maxVerts - 1);
		vidx[1] = min(vertlist[12 * threadIdx.x + edge[2]], maxVerts - 1);
#else
		vidx[0] = min(vertlist[edge[0]], maxVerts - 1);
		vidx[2] = min(vertlist[edge[1]], maxVerts - 1);
		vidx[1] = min(vertlist[edge[2]], maxVerts - 1);
#endif
		if (index < numMesh){
			vertIdx[index] = make_uint3(vidx[2], vidx[1], vidx[0]);
		}
	}
}

__global__
void CalVertexNormalKernel(float4 *vrts, uint3 *tris, float3 *nrms, int nvrts, int ntris)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;
	if (idx >= ntris) return;

	uint3 id = tris[idx];

	float3 normal;
	normal = Cross(make_float3(vrts[id.y]) - make_float3(vrts[id.x]), make_float3(vrts[id.z]) - make_float3(vrts[id.x]));

	//normal = normalize(normal);

#ifdef RX_USE_ATOMIC_FUNC
	atomicFloatAdd(&nrms[id.x].x, normal.x);
	atomicFloatAdd(&nrms[id.x].y, normal.y);
	atomicFloatAdd(&nrms[id.x].z, normal.z);
	atomicFloatAdd(&nrms[id.y].x, normal.x);
	atomicFloatAdd(&nrms[id.y].y, normal.y);
	atomicFloatAdd(&nrms[id.y].z, normal.z);
	atomicFloatAdd(&nrms[id.z].x, normal.x);
	atomicFloatAdd(&nrms[id.z].y, normal.y);
	atomicFloatAdd(&nrms[id.z].z, normal.z);
#else
	nrms[id.x] += normal;
	__threadfence();
	nrms[id.y] += normal;
	__threadfence();
	nrms[id.z] += normal;
#endif
}

__global__
void NormalizeKernel(float3 *v, int nv)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint idx = __mul24(blockId, blockDim.x) + threadIdx.x;
	if (idx >= nv) return;

	Normalize(v[idx]);
	
}

void InitTableMC(void)
{
	cudaMalloc((void**)&g_puEdgeTable, 256 * sizeof(uint));
	cudaMemcpy(g_puEdgeTable, edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMalloc((void**)&g_puTriTable, 256 * 16 * sizeof(uint));
	cudaMemcpy(g_puTriTable, triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice);

	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = g_puTriTable;
	texRes.res.linear.sizeInBytes = 256 * 16 * sizeof(uint);
	texRes.res.linear.desc = channelDesc;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&triTex, &texRes, &texDescr, NULL);

	cudaMalloc((void**)&g_puNumVrtsTable, 256 * sizeof(uint));
	cudaMemcpy(g_puNumVrtsTable, numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice);

	memset(&texRes, 0, sizeof(cudaResourceDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = g_puNumVrtsTable;
	texRes.res.linear.sizeInBytes = 256 * sizeof(uint);
	texRes.res.linear.desc = channelDesc;

	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&numVertsTex, &texRes, &texDescr, NULL);
}

void ClearTableMC(void)
{
	cudaFree(g_puEdgeTable);
	cudaFree(g_puTriTable);
	cudaFree(g_puNumVrtsTable);
}

void CuMCCalTriNum(float *dVolume, uint *dVoxBit, uint *dVoxVNum, uint *dVoxVNumScan,
	uint *dVoxTNum, uint *dVoxTNumScan, uint *dVoxOcc, uint *dVoxOccScan, uint *dCompactedVox,
	uint3 grid_size, uint num_voxels, float3 grid_width, float threshold,
	uint &num_active_voxels, uint &nvrts, uint &ntris)
{
	uint lval, lsval;

	int threads = GPU_THREAD_NUM;
	dim3 grid((num_voxels + threads - 1) / threads, 1, 1);
	
	// get around maximum grid size of 65535 in each dimension	
	//if (grid.x > 65535){
	//	grid.y = (grid.x + 32768 - 1) / 32768;
	//	grid.x = 32768;
	//}

	ClassifyVoxel2 << <grid, threads >> >(numVertsTex, dVoxBit, dVoxVNum, dVoxTNum, dVoxOcc, dVolume, grid_size, num_voxels, grid_width, threshold);
	num_active_voxels = num_voxels;


	CuScan(dVoxOccScan, dVoxOcc, num_voxels);

	cudaMemcpy((void*)&lval, (void*)(dVoxOcc + num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)&lsval, (void*)(dVoxOccScan + num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	num_active_voxels = lval + lsval;

	if (!num_active_voxels){
		nvrts = 0; ntris = 0;
		return;
	}

	CompactVoxels << <grid, threads >> >(dCompactedVox, dVoxOcc, dVoxOccScan, num_voxels);
	CuScan(dVoxVNumScan, dVoxVNum, num_voxels);
	CuScan(dVoxTNumScan, dVoxTNum, num_voxels);
	cudaMemcpy((void*)&lval, (void*)(dVoxVNum + num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)&lsval, (void*)(dVoxVNumScan + num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	nvrts = lval + lsval;

	cudaMemcpy((void*)&lval, (void*)(dVoxTNum + num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)&lsval, (void*)(dVoxTNumScan + num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	ntris = lval + lsval;
}

void CuMCCalEdgeVrts(float *dVolume, float *dEdgeVrts, float *dCompactedEdgeVrts,
	uint *dEdgeOcc, uint *dEdgeOccScan, uint3 edge_size[3], uint num_edge[4],
	uint3 grid_size, uint num_voxels, float3 grid_width, float3 grid_min, float threshold,
	uint &nvrts)
{
	uint lval, lsval;

	uint3 dir[3];
	dir[0] = make_uint3(1, 0, 0);
	dir[1] = make_uint3(0, 1, 0);
	dir[2] = make_uint3(0, 0, 1);

	uint cpos = 0;
	int threads = GPU_THREAD_NUM;
	dim3 grid;
	for (int i = 0; i < 3; ++i){
		grid = dim3((num_edge[i] + threads - 1) / threads, 1, 1);
		//if (grid.x > 65535){
		//	grid.y = (grid.x + 32768 - 1) / 32768;
		//	grid.x = 32768;
		//}
		CalVertexEdge << <grid, threads >> >(((float4*)dEdgeVrts) + cpos, dEdgeOcc + cpos,
			dVolume, dir[i], edge_size[i], grid_size,
			num_voxels, num_edge[i], grid_width, grid_min, threshold);

		cpos += num_edge[i];
	}
	cudaThreadSynchronize();

	CuScan(dEdgeOccScan, dEdgeOcc, num_edge[3]);

	cudaMemcpy((void*)&lval, (void*)(dEdgeOcc + num_edge[3] - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)&lsval, (void*)(dEdgeOccScan + num_edge[3] - 1), sizeof(uint), cudaMemcpyDeviceToHost);
	nvrts = lval + lsval;

	if (nvrts == 0){
		return;
	}

	grid = dim3((num_edge[3] + threads - 1) / threads, 1, 1);
	//if (grid.x > 65535){
	//	grid.y = (grid.x + 32768 - 1) / 32768;
	//	grid.x = 32768;
	//}

	// compact edge vertex array
	CompactEdges << <grid, threads >> >((float4*)dCompactedEdgeVrts, dEdgeOcc, dEdgeOccScan,
		(float4*)dEdgeVrts, num_edge[3]);
	cudaThreadSynchronize();
}

void CuMCCalTri(uint *dTris, uint *dVoxBit, uint *dVoxTNumScan, uint *dCompactedVox,
	uint *dEdgeOccScan, uint3 edge_size[3], uint num_edge[4],
	uint3 grid_size, uint num_voxels, float3 grid_width, float threshold,
	uint num_active_voxels, uint nvrts, uint ntris, float *dCompactedEdgeVrts)
{
	int threads = NTHREADS;
	dim3 grid((num_active_voxels + threads - 1) / threads, 1, 1);
	//if (grid.x > 65535){
	//	grid.y = (grid.x + 32768 - 1) / 32768;
	//	grid.x = 32768;
	//}

	uint3 numEdge = make_uint3(num_edge[0], num_edge[1], num_edge[2]);
	GenerateTriangles3 << <grid, threads >> >(triTex, numVertsTex, (uint3*)dTris, dVoxTNumScan, dEdgeOccScan, dVoxBit,
		edge_size[0], edge_size[1], edge_size[2], numEdge, dCompactedVox, grid_size,
		num_active_voxels, grid_width, threshold, nvrts, ntris, (float4*)dCompactedEdgeVrts);
	cudaThreadSynchronize();
}

void CuMCCalNrm(float *dNrms, uint *dTris, float *dCompactedEdgeVrts, uint nvrts, uint ntris)
{
	cudaMemset((void*)dNrms, 0, sizeof(float3)*nvrts);

	int threads = NTHREADS;
	dim3 grid((ntris + threads - 1) / threads, 1, 1);
	//if (grid.x > 65535){
	//	grid.y = (grid.x + 32768 - 1) / 32768;
	//	grid.x = 32768;
	//}

	CalVertexNormalKernel << <grid, threads >> >((float4*)dCompactedEdgeVrts, (uint3*)dTris, (float3*)dNrms, nvrts, ntris);
	cudaThreadSynchronize();

	grid = dim3((nvrts + threads - 1) / threads, 1, 1);
	//if (grid.x > 65535){
	//	grid.y = (grid.x + 32768 - 1) / 32768;
	//	grid.x = 32768;
	//}

	NormalizeKernel << <grid, threads >> >((float3*)dNrms, nvrts);
	cudaThreadSynchronize();
}


__global__
void CopyToTotalParticles1(REAL3* totalParticles, uint* d_TotalType, REAL3* particles, uint*d_Type, uint numParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	totalParticles[idx] = particles[idx];
	d_TotalType[idx] = d_Type[idx];

}

__global__
void CopyToTotalParticles2(REAL3* totalParticles, uint* d_TotalType, REAL3* particles, uint numParticles, uint numFlipParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	uint index = idx + numFlipParticles;
	totalParticles[index] = particles[idx];
	d_TotalType[index] = FLUID;
}

void CopyToTotalParticles_kernel(FLIP3D_Cuda* _fluid, SurfaceTurbulence* _turbulence, REAL3* d_TotalParticles, uint* d_Type, uint _numTotalParticles)
{
	CopyToTotalParticles1 << < divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_TotalParticles, d_Type, _fluid->d_CurPos(), _fluid->d_Type(), _fluid->_numParticles);

	CopyToTotalParticles2 << < divup(_turbulence->_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_TotalParticles, d_Type, _turbulence->d_DisplayPos(), _turbulence->_numFineParticles, _fluid->_numParticles);
}

void CalculateHash_kernel(uint* d_GridHash, uint* d_GridIdx, REAL3* d_TotalParticles, uint res, uint _numTotalParticles)
{
	CalculateHash_D << <divup(_numTotalParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_GridHash, d_GridIdx, d_TotalParticles, res, _numTotalParticles);
}

void SortParticle_kernel(uint* d_GridHash, uint* d_GridIdx, uint _numTotalParticles)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(d_GridHash),
		thrust::device_ptr<uint>(d_GridHash + _numTotalParticles),
		thrust::device_ptr<uint>(d_GridIdx));
}

void ComputeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = divup(n, numThreads);
}

void FindCellStart_kernel(uint* d_GridHash, uint* d_CellStart, uint* d_CellEnd,uint _numTotalParticles)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numTotalParticles, 128, numBlocks, numThreads);

	uint smemSize = sizeof(uint) * (numThreads + 1);
	FindCellStart_D << <numBlocks, numThreads, smemSize >> >
		(d_GridHash, d_CellStart, d_CellEnd, _numTotalParticles);
}

__device__ float hypotLength(float3 p)
{
	return (float)hypot(hypot((double)p.x, (double)p.y), (double)p.z);
}

__global__ void ComputeLevelSetKernel( REAL3* gridPosition, REAL3* particles, uint* d_TotalType, REAL* levelSet, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numParticles, uint3 res, uint hashRes, float3 pos0, float3 gridLength)
{
	uint id = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint index = __mul24(id, blockDim.x) + threadIdx.x;

	uint3 gridIndex = calcGridIdxU(index, res);

	if (gridIndex.x < res.x && gridIndex.y < res.y && gridIndex.z < res.z) {
		float3 pos;
		pos.x = (float)gridIndex.x / (float)res.x;
		pos.y = (float)gridIndex.y / (float)res.y;
		pos.z = (float)gridIndex.z / (float)res.z;

		REAL cellSize = 1.0 / (REAL)hashRes;
		int3 gridPos = calcGridPos(pos, cellSize);
		
		float r = 0.0f;
		double wsum = 0.0f;
		float density = 0.5f;
		float sdf = 0.0f;
		//float radius = 1.5f * density / 400.0f;
		float radius = 1.5f * density / 180.0f;
		float h = 4.0f * radius;
		REAL3 avgPos = make_REAL3(0.0f, 0.0f, 0.0f);
		int key = (res.x * res.y) * gridIndex.z + res.x * gridIndex.y + gridIndex.x;

		int cnt = 0;
		int width = 2;
		FOR_NEIGHBOR(width) {
			int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
			uint neighHash = calcGridHash(neighborPos, hashRes);
			uint startIdx = cellStart[neighHash];

			if (startIdx != 0xffffffff)
			{
				uint endIdx = cellEnd[neighHash];
				for (uint i = startIdx; i < endIdx; i++)
				{
					uint sortedIdx = gridIdx[i];
					REAL3 pos2 = particles[sortedIdx];
					uint type2 = d_TotalType[sortedIdx];
					REAL3 relPos = pos - pos2;
					if (Length(relPos) > width * cellSize)
						continue;
					if (type2 == WALL) {
						//float dist = hypotLength(relPos);
						//if (dist < density / res.x) {
						//	sdf = 4.5f * density / res.x;
						//	if (gridIndex.x == 0 || gridIndex.x == res.x - 1 || gridIndex.y == 0 || gridIndex.y == res.y - 1 || gridIndex.z == 0 || gridIndex.z == res.z - 1) {
						//		sdf = fmaxf(sdf, 0.01f);
						//	}
						//	levelSet[key] = -sdf;

						//	REAL x = (float)gridIndex.x / (float)res.x;
						//	REAL y = (float)gridIndex.y / (float)res.y;
						//	REAL z = (float)gridIndex.z / (float)res.z;
						//	gridPosition[key] = make_REAL3(x, y, z);
						//	return;
						//}
						continue;
					}
					float lengthSquared = (relPos.x * relPos.x + relPos.y * relPos.y + relPos.z * relPos.z);
					float w = fmax(1.0f - lengthSquared / (h * h), 0.0f);
					r += radius * w;
					avgPos += pos2 *w ;
					wsum += (double)w;
					cnt++;
				}
			}
		}END_FOR;

		//printf("wsum %f avgPos: %f %f %f Pos: %f %f %f Radius%f\n", wsum, avgPos.x, avgPos.y, avgPos.z, pos.x, pos.y, pos.z, r);
		//printf("wsum: %f\n", wsum);
		//printf("avgPos: %f %f %f\n", avgPos.x, avgPos.y, avgPos.z);
		//printf("Pos: %f %f %f \n", pos.x, pos.y, pos.z);
		//printf("Radius %f cnt %d\n", r/(float)cnt, cnt);

		if (wsum) {
			r /= wsum;
			avgPos /= wsum;
			sdf = fabs(Length(avgPos - pos)) - r;
		}
		else {
			sdf = 1.0f;
		}

		if (gridIndex.x == 0 || gridIndex.x == res.x - 1 || gridIndex.y == 0 || gridIndex.y == res.y - 1 || gridIndex.z == 0 || gridIndex.z == res.z - 1) {
			sdf = fmaxf(sdf, 0.01f);
		}
		levelSet[key] = -sdf;

		REAL x = (float)gridIndex.x / (float)res.x;
		REAL y = (float)gridIndex.y / (float)res.y;
		REAL z = (float)gridIndex.z / (float)res.z;
		gridPosition[key] = make_REAL3(x, y, z);

		//// Sphere equation
		//levelSet[key] = -(pow(pos.x - 0.5, 2.0) + pow(pos.y - 0.5, 2.0) + pow(pos.z - 0.5, 2.0) - 0.2 * 0.2);
	}
}

void ComputeLevelset_kernel(REAL3* gridPosition, REAL3* particles, uint* d_TotalType, REAL* levelSet, uint* d_GridIdx, uint* d_CellStart, uint* d_CellEnd, uint _numTotalParticles, uint3 res, uint hashRes)
{
	REAL3 pos0 = make_float3(0.0f, 0.0f, 0.0f);
	REAL3 length = make_float3(1.0f, 1.0f, 1.0f);

	int numCells = res.x * res.y * res.z;
	int threads = 256;
	dim3 grid((numCells + threads - 1) / threads, 1, 1);

	ComputeLevelSetKernel << <grid, threads >> > (gridPosition, particles, d_TotalType, levelSet, d_GridIdx, d_CellStart, d_CellEnd, _numTotalParticles, res, hashRes, pos0, length);
	cudaThreadSynchronize();
}

