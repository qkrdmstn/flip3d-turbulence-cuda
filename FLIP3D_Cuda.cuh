#include "FLIP3D_Cuda.h"
#include<stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define FOR_NEIGHBOR(n) for( int z=-n; z<=n; z++ ) for( int y=-n; y<=n; y++ ) for( int x=-n; x<=n; x++ ) {
#define END_FOR }

__device__ REAL SmoothKernel(REAL r2, REAL h)
{
	return max(1.0 - r2 / (h * h), 0.0);
}

__device__ REAL SharpKernel(REAL r2, REAL h)
{
	return max(h * h / max(r2, 1.0e-5f) - 1.0, 0.0);
}

__device__ int3 calcGridPos(REAL3 pos, REAL cellSize)
{
	int3 intPos = make_int3(floorf(pos.x / cellSize), floorf(pos.y / cellSize), floorf(pos.z / cellSize));
	return intPos;
}

__device__ uint calcGridHash(int3 pos, uint gridRes)
{
	pos.x = pos.x &
		(gridRes - 1);  // wrap grid, assumes size is power of 2
	pos.y = pos.y & (gridRes - 1);
	pos.z = pos.z & (gridRes - 1);
	return __umul24(__umul24(pos.z, gridRes), gridRes) +
		__umul24(pos.y, gridRes) + pos.x;

}

__global__ void ComputeDensity_D(REAL3* pos, uint* type, REAL* dens, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, uint numParticles, REAL densVal, REAL maxDens, BOOL* flag)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	if (type[idx] == WALL) {
		dens[idx] = 1.0;
		return;
	}

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);
	uint hash = calcGridHash(gridPos, gridRes);

	REAL wsum = 0.0;
	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (sortedIdx != idx)
				{
					if (type[sortedIdx] == WALL)
						continue;
					
					REAL3 dist = pos[sortedIdx] - pos[idx];


					REAL d2 = LengthSquared(dist);
					REAL w = mass[sortedIdx] * SmoothKernel(d2, 4.0 * densVal / gridRes);
					wsum += w;
				}
			}
		}
	} END_FOR;
	dens[idx] = wsum / maxDens;
}

__global__ void CompExternlaForce_D(REAL3* pos, REAL3* vel, REAL3 gravity, REAL3 ext, uint numParticles, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	REAL3 v = vel[idx];
	v += gravity * dt;
	v += ext * dt;
	vel[idx] = v;
}

__global__ void TrasnferToGrid_D(VolumeCollection volumes, REAL3* pos, REAL3* vel, uint* type, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, uint numParticles)
{
	int3 gridPos;
	gridPos.x = blockIdx.x * blockDim.x + threadIdx.x;
	gridPos.y = blockIdx.y * blockDim.y + threadIdx.y;
	gridPos.z = blockIdx.z * blockDim.z + threadIdx.z;

	if (gridPos.x >= gridRes || gridPos.y >= gridRes || gridPos.z >= gridRes) return;

	int cellCount = (gridRes) * (gridRes) * (gridRes);
	REAL cellPhysicalSize = 1.0 / gridRes;

	REAL3 xVelocityPos = make_REAL3(gridPos.x, (gridPos.y + 0.5), (gridPos.z + 0.5)) * cellPhysicalSize;
	REAL3 yVelocityPos = make_REAL3((gridPos.x + 0.5), gridPos.y, (gridPos.z + 0.5)) * cellPhysicalSize;
	REAL3 zVelocityPos = make_REAL3((gridPos.x + 0.5), (gridPos.y + 0.5), gridPos.z) * cellPhysicalSize;

	REAL4 velocity = make_REAL4(0, 0, 0, 0);
	REAL4 weight = make_REAL4(0, 0, 0, 0);

	FOR_NEIGHBOR(2) {

		int3 neighbourPos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];

				if (type[sortedIdx] == WALL)
					continue;

				REAL3 neighborPos = pos[sortedIdx];
				REAL3 neighborVel = vel[sortedIdx];

				REAL weightX = mass[sortedIdx] * SharpKernel(LengthSquared(neighborPos - xVelocityPos), 1.4);
				REAL weightY = mass[sortedIdx] * SharpKernel(LengthSquared(neighborPos - yVelocityPos), 1.4);
				REAL weightZ = mass[sortedIdx] * SharpKernel(LengthSquared(neighborPos - zVelocityPos), 1.4);

				velocity.x += weightX * neighborVel.x;
				velocity.y += weightY * neighborVel.y;
				velocity.z += weightZ * neighborVel.z;

				weight.x += weightX;
				weight.y += weightY;
				weight.z += weightZ;
			}
		}
	} END_FOR;
	velocity.x = weight.x ? velocity.x / weight.x : 0.0;
	velocity.y = weight.y ? velocity.y / weight.y : 0.0;
	velocity.z = weight.z ? velocity.z / weight.z : 0.0;

	volumes.vel.writeSurface<REAL4>(velocity, gridPos.x, gridPos.y, gridPos.z);
	volumes.velSave.writeSurface<REAL4>(velocity, gridPos.x, gridPos.y, gridPos.z);
}

__device__ REAL LevelSet(int3 gridPos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	int3 cellPos = make_int3(gridPos.x, gridPos.y, gridPos.z);
	uint neighHash = calcGridHash(cellPos, gridRes);
	uint startIdx = cellStart[neighHash];

	REAL accm = 0.0;
	if (startIdx != 0xffffffff)
	{
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];
			if (type[sortedIdx] == FLUID)
				accm += dens[sortedIdx];
			else
				return 1.0;
		}
	}
	REAL n0 = 1.0 / (densVal * densVal * densVal);
	return 0.2 * n0 - accm;
}

__global__ void MarkWater_D(VolumeCollection volumes, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	int3 gridPos;
	gridPos.x = blockIdx.x * blockDim.x + threadIdx.x;
	gridPos.y = blockIdx.y * blockDim.y + threadIdx.y;
	gridPos.z = blockIdx.z * blockDim.z + threadIdx.z;

	if (gridPos.x >= gridRes || gridPos.y >= gridRes || gridPos.z >= gridRes) return;

	uint c = CONTENT_AIR;

	int3 cellPos = make_int3(gridPos.x, gridPos.y, gridPos.z);
	uint neighHash = calcGridHash(cellPos, gridRes);
	uint startIdx = cellStart[neighHash];

	if (startIdx != 0xffffffff)
	{
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];
			if (type[sortedIdx] == WALL)
				c = CONTENT_WALL;
		}
		if (c != CONTENT_WALL)
		{
			REAL levelSet = LevelSet(gridPos, type, dens, gridHash, gridIdx, cellStart, cellEnd, densVal, gridRes);
			if (levelSet < 0.0)
				c = CONTENT_FLUID;
			else
				c = CONTENT_AIR;
		}
	}
	volumes.content.writeSurface<uint>(c, gridPos.x, gridPos.y, gridPos.z);
}

__device__ int WallCheck(uint type)
{
	if (type == CONTENT_WALL)
		return 1.0;
	return -1.0;
}

__global__ void EnforceBoundary_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;
	
	REAL4 velocity = volumes.vel.readSurface<REAL4>(x, y, z);

	if (x == 0 || x == gridRes)
		velocity.x = 0.0;
	if (y == 0 || y == gridRes)
		velocity.y = 0.0;
	if (z == 0 || z == gridRes)
		velocity.z = 0.0;

	if (x < gridRes && x>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x - 1, y, z)) < 0)
		velocity.x = 0.0;
	if (y < gridRes && y>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y - 1, z)) < 0)
		velocity.y = 0.0;
	if (z < gridRes && z>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y, z - 1)) < 0)
		velocity.z = 0.0;

	volumes.vel.writeSurface<REAL4>(velocity, x, y, z);
}

__device__ void FlipDiv(VolumeCollection volumes, int3 gridPos)
{
	REAL div = volumes.divergence.readSurface<REAL>(gridPos.x, gridPos.y, gridPos.z);
	volumes.divergence.writeSurface<REAL>(-1 * div, gridPos.x, gridPos.y, gridPos.z);
}

__device__ REAL A_ref(VolumeCollection volumes, int3 gridPos, int3 gridPos2, uint gridRes)
{
	uint content = volumes.content.readSurface<uint>(gridPos.x, gridPos.y, gridPos.z);
	uint content2 = volumes.content.readSurface<uint>(gridPos2.x, gridPos2.y, gridPos2.z);

	if (gridPos.x<0 || gridPos.x>gridRes - 1 || gridPos.y<0 || gridPos.y>gridRes - 1 || gridPos.z<0 || gridPos.z>gridRes - 1 || content != FLUID) return 0.0;
	if (gridPos2.x<0 || gridPos2.x>gridRes - 1 || gridPos2.y<0 || gridPos2.y>gridRes - 1 || gridPos2.z<0 || gridPos2.z>gridRes - 1 || content2 != FLUID) return 0.0;
	return -1.0;
}

__device__ REAL P_ref(VolumeCollection volumes, int3 gridPos, uint gridRes)
{
	uint content = volumes.content.readSurface<uint>(gridPos.x, gridPos.y, gridPos.z);
	if (gridPos.x<0 || gridPos.x>gridRes - 1 || gridPos.y<0 || gridPos.y>gridRes - 1 || gridPos.z<0 || gridPos.z>gridRes - 1 || content != CONTENT_FLUID) return 0.0;
	return volumes.pre.readSurface<REAL>(gridPos.x, gridPos.y, gridPos.z);
}

__device__ REAL A_diag(VolumeCollection volumes, int3 gridPos, uint gridRes)
{
	REAL diag = 6.0;
	uint content = volumes.content.readSurface<uint>(gridPos.x, gridPos.y, gridPos.z);
	REAL levelSet = volumes.levelSet.readSurface<REAL>(gridPos.x, gridPos.y, gridPos.z);
	if (content != CONTENT_FLUID) return diag;
	
	int3 pos[6] = { make_int3(gridPos.x - 1, gridPos.y, gridPos.z),  make_int3(gridPos.x + 1, gridPos.y, gridPos.z), make_int3(gridPos.x, gridPos.y - 1, gridPos.z), 
					make_int3(gridPos.x, gridPos.y + 1, gridPos.z), make_int3(gridPos.x, gridPos.y, gridPos.z - 1), make_int3(gridPos.x, gridPos.y, gridPos.z + 1) };

	for (int i = 0; i < 6; i++)
	{
		uint content2 = volumes.content.readSurface<uint>(pos[i].x, pos[i].y, pos[i].z);
		if (pos[i].x<0 || pos[i].x>gridRes - 1 || pos[i].y<0 || pos[i].y>gridRes - 1 || pos[i].z<0 || pos[i].z>gridRes - 1 || content2 == CONTENT_WALL) diag -= 1.0;
		else if (content2 == CONTENT_AIR)
		{
			REAL levelSet2 = volumes.levelSet.readSurface<REAL>(pos[i].x, pos[i].y, pos[i].z);
			diag -= levelSet2 / min(1.0e-6f, levelSet);
		}
	}
	return diag;
}

__device__ void BulidPreconditioner(VolumeCollection volumes, int3 gridPos, uint gridRes)
{
	REAL a = 0.25;
	if (volumes.content.readSurface<uint>(gridPos.x, gridPos.y, gridPos.z) == CONTENT_FLUID)
	{
		int3 leftPos = make_int3(gridPos.x - 1, gridPos.y, gridPos.z);
		int3 bottomPos = make_int3(gridPos.x, gridPos.y - 1, gridPos.z);
		int3 backPos = make_int3(gridPos.x, gridPos.y, gridPos.z - 1);

		REAL left = A_ref(volumes, leftPos, gridPos, gridRes) * P_ref(volumes, leftPos, gridRes);
		REAL bottom = A_ref(volumes, bottomPos, gridPos, gridRes) * P_ref(volumes, bottomPos, gridRes);
		REAL back = A_ref(volumes, backPos, gridPos, gridRes) * P_ref(volumes, backPos, gridRes);
		REAL diag = A_diag(volumes, gridPos, gridRes);
		REAL e = diag - (left * left) - (bottom * bottom) - (back * back);
		if (e < a * diag) e = diag;

		volumes.pre.writeSurface<REAL>(1.0 / sqrtf(e), gridPos.x, gridPos.y, gridPos.z);
	}
}

__device__ void ConjugateGradient()
{

}


__global__ void Project_D(VolumeCollection volumes, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	//Compute Divergence
	REAL cellSize = 1.0 / gridRes;
	if (volumes.content.readSurface<uint>(x, y, z) == CONTENT_FLUID)
	{
		REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
		REAL4 rightVel = volumes.vel.readSurface<REAL4>(x + 1, y, z);
		REAL4 upVel = volumes.vel.readSurface<REAL4>(x, y + 1, z);
		REAL4 frontVel = volumes.vel.readSurface<REAL4>(x, y, z + 1);

		REAL div = ((rightVel.x - curVel.x) + (upVel.y - curVel.y) + (frontVel.z - curVel.z)) / cellSize;
		volumes.divergence.writeSurface<REAL>(div, x, y, z);
	}

	//Compute LevelSet
	int3 gridPos = make_int3(x, y, z);
	REAL levelSet = LevelSet(gridPos, type, dens, gridHash, gridIdx, cellStart, cellEnd, densVal, gridRes);
	volumes.levelSet.writeSurface<REAL>(levelSet, x, y, z);

	//Solve Pressure
	FlipDiv(volumes, gridPos);
	BulidPreconditioner(volumes, gridPos, gridRes);
	ConjugateGradient();
}


__global__ void TrasnferToParticle()
{

}

__global__ void SolvePICFLIP_D()
{

}

__global__ void AdvecParticle_D(REAL3* beforePos, REAL3* curPos, REAL3* vel, uint* type, uint numParticles, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID)
		return;

	beforePos[idx] = curPos[idx];
	curPos[idx] += dt * vel[idx];
}

__global__ void CalculateHash_D(uint* gridHash, uint* gridIdx, REAL3* pos, uint gridRes, uint numParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numParticles)
		return;

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);
	uint hash = calcGridHash(gridPos, gridRes);

	gridHash[idx] = hash;
	gridIdx[idx] = idx;
}

__global__ void FindCellStart_D(uint* gridHash, uint* cellStart, uint* cellEnd, uint numParticles)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];  // blockSize + 1 elements
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	//if (idx >= numParticles)
	//	return;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (idx < numParticles) {
		hash = gridHash[idx];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (idx > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridHash[idx - 1];
		}
	}

	cg::sync(cta);

	if (idx < numParticles) {

		if (idx == 0 || hash != sharedHash[threadIdx.x]) {
			cellStart[hash] = idx;

			if (idx > 0) cellEnd[sharedHash[threadIdx.x]] = idx;
		}

		if (idx == numParticles - 1) {
			cellEnd[hash] = idx + 1;
		}
	}
}