#include "FLIP3D_Cuda.h"
#include <cmath>
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

					//if (idx == 1234)
					//{
					//	flag[sortedIdx] = true;
					//	flag[idx] = true;
					//}

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

__global__ void ComputeDivergence_D(VolumeCollection volumes, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes)
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
}

__global__ void ComputeLevelSet_D(VolumeCollection volumes, int3 gridPos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	REAL levelSet = LevelSet(gridPos, type, dens, gridHash, gridIdx, cellStart, cellEnd, densVal, gridRes);

	volumes.levelSet.writeSurface<REAL>(levelSet, x, y, z);
}

__global__ void SolvePressureJacobi_D(VolumeCollection volumes, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	if (volumes.content.readSurface<int>(x, y, z) == CONTENT_AIR) {
		volumes.press.writeSurface<float>(0.f, x, y, z);
		return;
	}

	int3 cellPos = make_int3(x, y, z);
	uint neighHash = calcGridHash(cellPos, gridRes);
	uint startIdx = cellStart[neighHash];

	REAL densSum = 0.0;
	uint cnt = 0;
	if (startIdx != 0xffffffff)
	{
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];
			densSum += dens[sortedIdx];
			cnt++;
		}
	}

	REAL thisDensity = densSum / (REAL)cnt;
	REAL RHS = -volumes.divergence.readSurface<REAL>(x, y, z) * thisDensity;

	REAL newPress = 0;
	REAL centerCoeff = 6;

	newPress += volumes.press.readTexture<REAL>(x + 1, y, z);
	newPress += volumes.press.readTexture<REAL>(x - 1, y, z);
	newPress += volumes.press.readTexture<REAL>(x, y + 1, z);
	newPress += volumes.press.readTexture<REAL>(x, y - 1, z);
	newPress += volumes.press.readTexture<REAL>(x, y, z + 1);
	newPress += volumes.press.readTexture<REAL>(x, y, z - 1);

	newPress += RHS;
	newPress /= centerCoeff;
	
	volumes.press.writeSurface<REAL>(newPress, x, y, z);
}

__global__ void ComputeVelocityWithPress_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	REAL cellSize = 1.0 / gridRes;

	REAL levelSet = volumes.levelSet.readSurface<REAL>(x, y, z);
	REAL levelSetX = volumes.levelSet.readSurface<REAL>(x - 1, y, z);
	REAL levelSetY = volumes.levelSet.readSurface<REAL>(x, y - 1, z);
	REAL levelSetZ = volumes.levelSet.readSurface<REAL>(x, y, z - 1);

	REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
	REAL4 newVel = make_REAL4(0, 0, 0, 0);
	uint4 hasVelocity = make_uint4(0, 0, 0, 0);

	if (x < gridRes && x>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressX = volumes.press.readSurface<REAL>(x - 1, y, z);
		if (levelSet * levelSetX < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(0.001f, levelSetX) * volumes.press.readSurface<REAL>(x - 1, y, z);
			pressX = levelSetX < 0.0 ? volumes.press.readSurface<REAL>(x - 1, y, z) : levelSet / min(0.00001f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		newVel.x = curVel.x - ((press - pressX) / cellSize);
		hasVelocity.x = true;
	}

	if (y < gridRes && y>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressY = volumes.press.readSurface<REAL>(x, y - 1, z);
		if (levelSet * levelSetY < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(0.001f, levelSetY) * volumes.press.readSurface<REAL>(x, y - 1 , z);
			pressY = levelSetY < 0.0 ? volumes.press.readSurface<REAL>(x, y - 1, z) : levelSet / min(0.00001f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		newVel.y = curVel.y - ((press - pressY) / cellSize);
		hasVelocity.y = true;
	}

	if (z < gridRes && z>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressZ = volumes.press.readSurface<REAL>(x, y, z - 1);
		if (levelSet * levelSetZ < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(0.001f, levelSetZ) * volumes.press.readSurface<REAL>(x, y, z - 1);
			pressZ = levelSetZ < 0.0 ? volumes.press.readSurface<REAL>(x, y, z - 1) : levelSet / min(0.00001f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		newVel.z = curVel.z - ((press - pressZ) / cellSize);
		hasVelocity.z = true;
	}

	volumes.hasVel.writeSurface<uint4>(hasVelocity, x, y, z);
	volumes.vel.writeSurface<REAL4>(newVel, x, y, z);
}

__global__ void ExtrapolateVelocity_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	const float epsilon = -1 + 1e-6;

	uint4 thisHasVelocity = volumes.hasVel.readSurface<uint4>(x, y, z);
	REAL4 thisNewVelocity = volumes.vel.readSurface<REAL4>(x, y, z);

	if (!thisHasVelocity.x) {
		float sumNeighborX = 0;
		int neighborXCount = 0;
		if (x > 0) {

			uint4 leftHasVelocity = volumes.hasVel.readSurface<uint4>(x - 1, y, z);
			REAL4 leftNewVelocity = volumes.vel.readSurface<REAL4>(x - 1, y, z);
			if (leftHasVelocity.x && leftNewVelocity.x > epsilon) {
				sumNeighborX += leftNewVelocity.x;
				neighborXCount++;
			}
		}
		if (y > 0) {

			uint4 downHasVelocity = volumes.hasVel.readSurface<uint4>(x, y - 1, z);
			REAL4 downNewVelocity = volumes.vel.readSurface<REAL4>(x, y - 1, z);
			if (downHasVelocity.x && downNewVelocity.y > epsilon) {
				sumNeighborX += downNewVelocity.x;
				neighborXCount++;
			}
		}
		if (z > 0) {

			uint4 backHasVelocity = volumes.hasVel.readSurface<uint4>(x, y, z - 1);
			REAL4 backNewVelocity = volumes.vel.readSurface<REAL4>(x, y, z - 1);
			if (backHasVelocity.x && backNewVelocity.z > epsilon) {
				sumNeighborX += backNewVelocity.x;
				neighborXCount++;
			}
		}
		if (x < gridRes - 1) {

			uint4 rightHasVelocity = volumes.hasVel.readSurface<uint4>(x + 1, y, z);
			REAL4 rightNewVelocity = volumes.vel.readSurface<REAL4>(x + 1, y, z);
			if (rightHasVelocity.x && rightNewVelocity.x < -epsilon) {
				sumNeighborX += rightNewVelocity.x;
				neighborXCount++;
			}
		}
		if (y < gridRes - 1) {

			uint4 upHasVelocity = volumes.hasVel.readSurface<uint4>(x, y + 1, z);
			REAL4 newVelocity = volumes.vel.readSurface<REAL4>(x, y + 1, z);
			if (upHasVelocity.x && newVelocity.y < -epsilon) {
				sumNeighborX += newVelocity.x;
				neighborXCount++;
			}
		}
		if (z < gridRes - 1) {

			uint4 frontHasVelocity = volumes.hasVel.readSurface<uint4>(x, y, z + 1);
			REAL4 frontNewVelocity = volumes.vel.readSurface<REAL4>(x, y, z + 1);
			if (frontHasVelocity.x && frontNewVelocity.z < -epsilon) {
				sumNeighborX += frontNewVelocity.x;
				neighborXCount++;
			}
		}

		if (neighborXCount > 0) {
			thisNewVelocity.x = sumNeighborX / (float)neighborXCount;
			thisHasVelocity.x = true;
		}
	}

	if (!thisHasVelocity.y) {
		float sumNeighborY = 0;
		int neighborYCount = 0;
		if (x > 0) {

			uint4 leftHasVelocity = volumes.hasVel.readSurface<uint4>(x - 1, y, z);
			REAL4 leftNewVelocity = volumes.vel.readSurface<REAL4>(x - 1, y, z);
			if (leftHasVelocity.y && leftNewVelocity.x > epsilon) {
				sumNeighborY += leftNewVelocity.y;
				neighborYCount++;
			}
		}
		if (y > 0) {

			uint4 downHasVelocity = volumes.hasVel.readSurface<uint4>(x, y - 1, z);
			REAL4 downNewVelocity = volumes.vel.readSurface<REAL4>(x, y - 1, z);
			if (downHasVelocity.y && downNewVelocity.y > epsilon) {
				sumNeighborY += downNewVelocity.y;
				neighborYCount++;
			}
		}
		if (z > 0) {

			uint4 backHasVelocity = volumes.hasVel.readSurface<uint4>(x, y, z - 1);
			REAL4 backNewVelocity = volumes.vel.readSurface<REAL4>(x, y, z - 1);
			if (backHasVelocity.y && backNewVelocity.z > epsilon) {
				sumNeighborY += backNewVelocity.y;
				neighborYCount++;
			}
		}
		if (x < gridRes - 1) {

			uint4 rightHasVelocity = volumes.hasVel.readSurface<uint4>(x + 1, y, z);
			REAL4 rightNewVelocity = volumes.vel.readSurface<REAL4>(x + 1, y, z);
			if (rightHasVelocity.y && rightNewVelocity.x < -epsilon) {
				sumNeighborY += rightNewVelocity.y;
				neighborYCount++;
			}
		}
		if (y < gridRes - 1) {

			uint4 upHasVelocity = volumes.hasVel.readSurface<uint4>(x, y + 1, z);
			REAL4 newVelocity = volumes.vel.readSurface<REAL4>(x, y + 1, z);
			if (upHasVelocity.y && newVelocity.y < -epsilon) {
				sumNeighborY += newVelocity.y;
				neighborYCount++;
			}
		}
		if (z < gridRes - 1) {

			uint4 frontHasVelocity = volumes.hasVel.readSurface<uint4>(x, y, z + 1);
			REAL4 frontNewVelocity = volumes.vel.readSurface<REAL4>(x, y, z + 1);
			if (frontHasVelocity.y && frontNewVelocity.z < -epsilon) {
				sumNeighborY += frontNewVelocity.y;
				neighborYCount++;
			}
		}
		if (neighborYCount > 0) {
			thisNewVelocity.y = sumNeighborY / (float)neighborYCount;
			thisHasVelocity.y = true;

		}

	}

	if (!thisHasVelocity.z) {
		float sumNeighborZ = 0;
		int neighborZCount = 0;
		if (x > 0) {

			uint4 leftHasVelocity = volumes.hasVel.readSurface<uint4>(x - 1, y, z);
			REAL4 leftNewVelocity = volumes.vel.readSurface<REAL4>(x - 1, y, z);
			if (leftHasVelocity.z && leftNewVelocity.x > epsilon) {
				sumNeighborZ += leftNewVelocity.z;
				neighborZCount++;
			}
		}
		if (y > 0) {

			uint4 downHasVelocity = volumes.hasVel.readSurface<uint4>(x, y - 1, z);
			REAL4 downNewVelocity = volumes.vel.readSurface<REAL4>(x, y - 1, z);
			if (downHasVelocity.z && downNewVelocity.y > epsilon) {
				sumNeighborZ += downNewVelocity.z;
				neighborZCount++;
			}
		}
		if (z > 0) {

			uint4 backHasVelocity = volumes.hasVel.readSurface<uint4>(x, y, z - 1);
			REAL4 backNewVelocity = volumes.vel.readSurface<REAL4>(x, y, z - 1);
			if (backHasVelocity.z && backNewVelocity.z > epsilon) {
				sumNeighborZ += backNewVelocity.z;
				neighborZCount++;
			}
		}
		if (x < gridRes - 1) {

			uint4 rightHasVelocity = volumes.hasVel.readSurface<uint4>(x + 1, y, z);
			REAL4 rightNewVelocity = volumes.vel.readSurface<REAL4>(x + 1, y, z);
			if (rightHasVelocity.z && rightNewVelocity.x < -epsilon) {
				sumNeighborZ += rightNewVelocity.z;
				neighborZCount++;
			}
		}
		if (y < gridRes - 1) {

			uint4 upHasVelocity = volumes.hasVel.readSurface<uint4>(x, y + 1, z);
			REAL4 newVelocity = volumes.vel.readSurface<REAL4>(x, y + 1, z);
			if (upHasVelocity.z && newVelocity.y < -epsilon) {
				sumNeighborZ += newVelocity.z;
				neighborZCount++;
			}
		}
		if (z < gridRes - 1) {

			uint4 frontHasVelocity = volumes.hasVel.readSurface<uint4>(x, y, z + 1);
			REAL4 frontNewVelocity = volumes.vel.readSurface<REAL4>(x, y, z + 1);
			if (frontHasVelocity.z && frontNewVelocity.z < -epsilon) {
				sumNeighborZ += frontNewVelocity.z;
				neighborZCount++;
			}
		}
		if (neighborZCount > 0) {
			thisNewVelocity.z = sumNeighborZ / (float)neighborZCount;
			thisHasVelocity.z = true;
		}
	}

	volumes.hasVel.writeSurface<uint4>(thisHasVelocity, x, y, z);
	volumes.vel.writeSurface<REAL4>(thisNewVelocity, x, y, z);
}

__global__ void SubtarctGrid_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	REAL4 beforeVel = volumes.velSave.readSurface<REAL4>(x, y, z);
	REAL4 afterVel = volumes.vel.readSurface<REAL4>(x, y, z);

	REAL subVelX = afterVel.x - beforeVel.x;
	REAL subVelY = afterVel.y - beforeVel.y;
	REAL subVelZ = afterVel.z - beforeVel.z;
	REAL4 newVel = make_float4(subVelX, subVelY, subVelZ, 0.0);

	volumes.velSave.writeSurface<REAL4>(newVel, x, y, z);
}



__device__
inline REAL3 lerp(REAL3 v0, REAL3 v1, REAL t) {
	return (1 - t) * v0 + t * v1;
}

__device__
REAL3 GetLerpValueAtPoint(VolumeData data, REAL x, REAL y, REAL z, uint gridRes)
{
	x = max(min(x, gridRes - 1.f), 0.f);
	y = max(min(y, gridRes - 1.f), 0.f);
	z = max(min(z, gridRes - 1.f), 0.f);

	uint i = floor(x);
	uint j = floor(y);
	uint k = floor(z);

	REAL tx = x - (REAL)i;
	REAL ty = y - (REAL)j;
	REAL tz = z - (REAL)k;
	
	REAL3 a = make_REAL3(data.readSurface<REAL4>(i, j, k).x, data.readSurface<REAL4>(i, j, k).y, data.readSurface<REAL4>(i, j, k).z);
	REAL3 b = make_REAL3(data.readSurface<REAL4>(i, j, k + 1).x, data.readSurface<REAL4>(i, j, k + 1).y, data.readSurface<REAL4>(i, j, k + 1).z);
	REAL3 x0y0 =
		lerp(a, b, tz);

	REAL3 c = make_REAL3(data.readSurface<REAL4>(i, j + 1, k).x, data.readSurface<REAL4>(i, j + 1, k).y, data.readSurface<REAL4>(i, j + 1, k).z);
	REAL3 d = make_REAL3(data.readSurface<REAL4>(i, j + 1, k + 1).x, data.readSurface<REAL4>(i, j + 1, k + 1).y, data.readSurface<REAL4>(i, j + 1, k + 1).z);
	REAL3 x0y1 =
		lerp(c, d, tz);

	REAL3 e = make_REAL3(data.readSurface<REAL4>(i + 1, j, k).x, data.readSurface<REAL4>(i + 1, j, k).y, data.readSurface<REAL4>(i + 1, j, k).z);
	REAL3 f = make_REAL3(data.readSurface<REAL4>(i + 1, j, k + 1).x, data.readSurface<REAL4>(i + 1, j, k + 1).y, data.readSurface<REAL4>(i + 1, j, k + 1).z);
	REAL3 x1y0 =
		lerp(e, f, tz);

	REAL3 g = make_REAL3(data.readSurface<REAL4>(i + 1, j + 1, k).x, data.readSurface<REAL4>(i + 1, j + 1, k).y, data.readSurface<REAL4>(i + 1, j + 1, k).z);
	REAL3 h = make_REAL3(data.readSurface<REAL4>(i + 1, j + 1, k + 1).x, data.readSurface<REAL4>(i + 1, j + 1, k + 1).y, data.readSurface<REAL4>(i + 1, j + 1, k + 1).z);
	REAL3 x1y1 =
		lerp(g, h, tz);

	REAL3 x0 = lerp(x0y0, x0y1, ty);
	REAL3 x1 = lerp(x1y0, x1y1, ty);

	REAL3 result = lerp(x0, x1, tx);
	return result;
}

__device__
REAL3 GetPointSaveVelocity(REAL3 physicalPos, REAL cellPhysicalSize, uint gridRes, VolumeCollection volume) {
	REAL x = physicalPos.x / cellPhysicalSize;
	REAL y = physicalPos.y / cellPhysicalSize;
	REAL z = physicalPos.z / cellPhysicalSize;

	REAL3 result;

	result.x = GetLerpValueAtPoint(volume.velSave, x, y - 0.5, z - 0.5, gridRes).x;
	result.y = GetLerpValueAtPoint(volume.velSave, x - 0.5, y, z - 0.5, gridRes).y;
	result.z = GetLerpValueAtPoint(volume.velSave, x - 0.5, y - 0.5, z, gridRes).z;

	return result;
}

__device__
REAL3 GetPointAfterVelocity(REAL3 physicalPos, REAL cellPhysicalSize, uint gridRes, VolumeCollection volume) {
	REAL x = physicalPos.x / cellPhysicalSize;
	REAL y = physicalPos.y / cellPhysicalSize;
	REAL z = physicalPos.z / cellPhysicalSize;

	REAL3 result;

	result.x = GetLerpValueAtPoint(volume.vel, x, y - 0.5, z - 0.5, gridRes).x;
	result.y = GetLerpValueAtPoint(volume.vel, x - 0.5, y, z - 0.5, gridRes).y;
	result.z = GetLerpValueAtPoint(volume.vel, x - 0.5, y - 0.5, z, gridRes).z;

	return result;
}

__global__ void TrasnferToParticle_D(VolumeCollection volumes, uint gridRes, REAL3* pos, REAL3* vel, uint numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles) return;

	REAL cellSize = 1.0 / gridRes;
	REAL3 position = pos[index];
	REAL3 oldGridVel = GetPointSaveVelocity(position, cellSize, gridRes, volumes);
	REAL3 newGridVel = GetPointAfterVelocity(position, cellSize, gridRes, volumes);
	REAL3 velChange = newGridVel - oldGridVel;

	REAL3 FLIP = vel[index] + velChange;
	REAL3 PIC = newGridVel;

	REAL FLIPCoeff = 0.95;
	vel[index] = FLIPCoeff * FLIP + (1.0 - FLIPCoeff) * PIC;
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

	curPos[idx].x = max((REAL)(1.0/32), min((REAL)(1.0 - 1.0/32), curPos[idx].x));
	curPos[idx].y = max((REAL)(1.0/32), min((REAL)(1.0 - 1.0/32), curPos[idx].y));
	curPos[idx].z = max((REAL)(1.0/32), min((REAL)(1.0 - 1.0/32), curPos[idx].z));
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