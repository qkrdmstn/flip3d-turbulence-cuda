#ifndef __HASH__CUH__
#define __HASH__CUH__

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <cuda_runtime.h>
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "thrust/sort.h"

#define AIR		0
#define FLUID	1
#define WALL	2

#define FOR_NEIGHBOR(n) for( int dz=-n; dz<=n; dz++ ) for( int dy=-n; dy<=n; dy++ ) for( int dx=-n; dx<=n; dx++ ) {
#define FOR_NEIGHBOR_WALL(n) for( int dz=-n; dz<=n-1; dz++ ) for( int dy=-n; dy<=n-1; dy++ ) for( int dx=-n; dx<=n-1; dx++ ) {
#define END_FOR }

__device__ static int3 calcGridPos(REAL3 pos, REAL cellSize)
{
	int3 intPos = make_int3(floorf(pos.x / cellSize), floorf(pos.y / cellSize), floorf(pos.z / cellSize));
	return intPos;
}

__device__ static uint calcGridHash(int3 pos, uint gridRes)
{
	pos.x = pos.x & (gridRes - 1);  // wrap grid, assumes size is power of 2
	pos.y = pos.y & (gridRes - 1);
	pos.z = pos.z & (gridRes - 1);
	return __umul24(__umul24(pos.z, gridRes), gridRes) + __umul24(pos.y, gridRes) + pos.x;
}

__device__ static  uint GetNumFluidParticleAt(uint i, uint j, uint k, REAL3* pos, uint* type, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes)
{
	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(i + 0.5, j + 0.5, k + 0.5) * cellSize;

	uint cnt = 0;
	int3 gridPos = make_int3(i, j, k);
	uint neighHash = calcGridHash(gridPos, gridRes);
	uint startIdx = cellStart[neighHash];
	if (startIdx != 0xffffffff) {
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];

			REAL3 dist = pos[sortedIdx] - centerPos;
			REAL d2 = LengthSquared(dist);
			if (d2 > cellSize * cellSize)
				continue;

			if (type[sortedIdx] == FLUID)
				cnt++;
		}
	}
	return cnt;
}

__global__  void CalculateHash_D(uint* gridHash, uint* gridIdx, REAL3* pos, uint gridRes, uint numParticles);

__global__  void FindCellStart_D(uint* gridHash, uint* cellStart, uint* cellEnd, uint numParticles);

#endif 