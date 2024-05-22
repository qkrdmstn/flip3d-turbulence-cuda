#ifndef __MARCHING_CUBES_CUDA_H__
#define __MARCHING_CUBES_CUDA_H__

#pragma once
#include <vector>
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/PrefixArray.h"
#include "Vec3.h"
#include "FLIP3D_Cuda.h"
#include "SurfaceTurbulence.h"
using namespace std;

void InitTableMC(void);
void ClearTableMC(void);
void CuMCCalNrm(float *dNrms, uint *dTris, float *dCompactedEdgeVrts, uint nvrts, uint ntris);
void CuMCCalTriNum(float *dVolume, uint *dVoxBit, uint *dVoxVNum, uint *dVoxVNumScan, 
					uint *dVoxTNum, uint *dVoxTNumScan, uint *dVoxOcc, uint *dVoxOccScan, uint *dCompactedVox, uint3 grid_size, 
					uint num_voxels, float3 grid_width, float threshold, uint &num_active_voxels, uint &nvrts, uint &ntris);

void CuMCCalEdgeVrts(float *dVolume, float *dEdgeVrts, float *dCompactedEdgeVrts, uint *dEdgeOcc, uint *dEdgeOccScan, uint3 edge_size[3], 
	uint num_edge[4], uint3 grid_size, uint num_voxels, float3 grid_width, float3 grid_min, float threshold, uint &nvrts);

void CuMCCalTri(uint *dTris, uint *dVoxBit, uint *dVoxTNumScan, uint *dCompactedVox, uint *dEdgeOccScan, uint3 edge_size[3], uint num_edge[4],
	uint3 grid_size, uint num_voxels, float3 grid_width, float threshold,uint num_active_voxels, uint nvrts, uint ntris, float *dCompactedEdgeVrts);

void CopyToTotalParticles_kernel(FLIP3D_Cuda* _fluid, SurfaceTurbulence* _turbulence, REAL3* d_TotalParticles, uint* d_Type, uint _numTotalParticles);


void CalculateHash_kernel(uint* d_GridHash, uint* d_GridIdx, REAL3* d_TotalParticles, uint res, uint _numTotalParticles);

void SortParticle_kernel(uint* d_GridHash, uint* d_GridIdx, uint _numTotalParticles);

void FindCellStart_kernel(uint* d_GridHash, uint* d_CellStart, uint* d_CellEnd, uint _numTotalParticles);

void ComputeLevelset_kernel(REAL3* gridPosition, REAL3* particles, uint* d_TotalType, REAL* levelSet, uint* d_GridIdx, uint* d_CellStart, uint* d_CellEnd, uint _numTotalParticles, uint3 res, uint hashRes);

class MarchingCubes_CUDA
{
public:
	float3	h;
	float3	_voxelMin;
	float3	_voxelMax;
	float4	*_4VertexPos;
	uint3	_gridSize;
	uint3	_gridSizeMask;
	uint3	_gridSizeShift;
	uint3	_edgeSize[3];
	uint3	*_3TriangleIndex;
	uint	_numVoxel;
	uint	_maxVertices;
	uint	_numActiveVoxel;
	uint	_numVertex;
	uint	_numTriangle;
	uint	_numEdge[4];
	uint	*_voxelCubeIndex;
	uint	*_edgeOccupied;
	uint	*_edgeOccupiedScan;
	uint	*_indexArray;
	uint	*_scan;
	uint	*_numVoxelTriangle;
	uint	*_numVoxelTriangleScan;
	uint	*_voxelVertices;
	uint	*_voxelVerticesScan;
	uint	*_compactedVoxelArray;
	uint	*_voxelOccupied;
	uint	*_voxelOccupiedScan;
	float	*_volume;
	float	*_noise;
	float	*_edgeVertices;
	float	*_vertexNormals;
	float	*_compactedEdgeVertices;
	int		_vertexStore;
	bool	_set;
	bool	_validSurface;	

public:
	MarchingCubes_CUDA(void);
	~MarchingCubes_CUDA(void);

public:
	void	free(void);
	void	init(FLIP3D_Cuda* fluid, SurfaceTurbulence* turbulence, int rx, int ry, int rz);
	void	surfaceRecon(float threshold);
	void	copyCPU(vector<vec3> &vertices, vector<vec3> &normals, vector<int> &faces);

public:
	FLIP3D_Cuda* _fluid;
	SurfaceTurbulence* _turbulence;

public:
	REAL3* d_TotalParticles;
	uint* d_Type;
	uint _numTotalParticles;

	//temp Visualize
	REAL3* d_gridPosition;
	REAL3* h_gridPosition;
	REAL* h_level;

public:
	vector<vec3> h_Vertices;
	vector<vec3> h_VertexNormals;
	vector<int> h_Faces;

public: //hash
	uint* d_GridHash;
	uint* d_GridIdx;
	uint* d_CellStart;
	uint* d_CellEnd;

	uint hashRes;

public:
	void	MarchingCubes(void);
	void	SetHashTable_kernel(void);
	void	Smoothing(void);
	void	renderSurface(void);
};

#endif
