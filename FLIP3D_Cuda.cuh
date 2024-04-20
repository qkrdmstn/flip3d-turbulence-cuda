#include "FLIP3D_Cuda.h"
#include <cmath>
#include<stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
__device__ uint s = 6;
#define FOR_NEIGHBOR(n) for( int dz=-n; dz<=n; dz++ ) for( int dy=-n; dy<=n; dy++ ) for( int dx=-n; dx<=n; dx++ ) {
#define FOR_NEIGHBOR_WALL(n) for( int dz=-n; dz<=n-1; dz++ ) for( int dy=-n; dy<=n-1; dy++ ) for( int dx=-n; dx<=n-1; dx++ ) {
#define END_FOR }

__device__ REAL SmoothKernel(REAL r2, REAL h)
{
	return max(1.0 - r2 / (h * h), 0.0);
}

__device__ REAL SharpKernel(REAL r2, REAL h)
{
	return max(h * h / max(r2, 0.00001f) - 1.0, 0.0);
}

__device__ int3 calcGridPos(REAL3 pos, REAL cellSize)
{
	int3 intPos = make_int3(floorf(pos.x / cellSize), floorf(pos.y / cellSize), floorf(pos.z / cellSize));
	return intPos;
}

__device__ uint calcGridHash(int3 pos, uint gridRes)
{
	pos.x = pos.x & (gridRes - 1);  // wrap grid, assumes size is power of 2
	pos.y = pos.y & (gridRes - 1);
	pos.z = pos.z & (gridRes - 1);
	return __umul24(__umul24(pos.z, gridRes), gridRes) +
		__umul24(pos.y, gridRes) + pos.x;

}

__device__ REAL LevelSet(int3 gridPos, REAL3* pos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	uint neighHash = calcGridHash(gridPos, gridRes);
	uint startIdx = cellStart[neighHash];

	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(gridPos.x + 0.5, gridPos.y + 0.5, gridPos.z + 0.5) * cellSize;

	REAL accm = 0.0;
	if (startIdx != 0xffffffff)
	{
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];

			REAL3 dist = pos[sortedIdx] - centerPos;
			REAL d2 = LengthSquared(dist);
			if (d2 > cellSize * cellSize)
				continue;

			if (type[sortedIdx] == FLUID)
				accm += dens[sortedIdx];
			else
				return 1.0;
		}
	}
	REAL n0 = 1.0 / (densVal * densVal * densVal);
	return 0.2 * n0 - accm;
}

__global__ void ResetCell_D(VolumeCollection volumes, uint gridRes) {
	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	volumes.content.writeSurface<uint>(CONTENT_AIR, x, y, z);

	volumes.hasVel.writeSurface<uint4>(make_uint4(0, 0, 0, 0), x, y, z);
	volumes.vel.writeSurface<REAL4>(make_REAL4(0, 0, 0, 0), x, y, z);
	volumes.velSave.writeSurface<REAL4>(make_REAL4(0, 0, 0, 0), x, y, z);
	volumes.density.writeSurface<REAL>(0, x, y, z);
	//volumes.press.writeSurface<REAL>(0.0, x, y, z);
}

__global__ void ComputeParticleDensity_D(REAL3* pos, uint* type, REAL* dens, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, uint numParticles, REAL densVal, REAL maxDens, BOOL* flag)
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

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
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
	
	REAL3 xVelocityPos = make_REAL3(gridPos.x, (gridPos.y + 0.5), (gridPos.z + 0.5));
	REAL3 yVelocityPos = make_REAL3((gridPos.x + 0.5), gridPos.y, (gridPos.z + 0.5));
	REAL3 zVelocityPos = make_REAL3((gridPos.x + 0.5), (gridPos.y + 0.5), gridPos.z);

	REAL4 velocity = make_REAL4(0, 0, 0, 0);
	REAL4 weight = make_REAL4(0, 0, 0, 0);

	FOR_NEIGHBOR_WALL(3) {

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);

		//if (gridPos.x < 0 || gridPos.x > gridRes - 1 || gridPos.y < 0 || gridPos.y > gridRes - 1 || gridPos.z < 0 || gridPos.z > gridRes - 1)
		//	continue;
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
				REAL x = max(0.0f, min((REAL)gridRes, gridRes * neighborPos.x));
				REAL y = max(0.0f, min((REAL)gridRes, gridRes * neighborPos.y));
				REAL z = max(0.0f, min((REAL)gridRes, gridRes * neighborPos.z));
				REAL3 pos = make_REAL3(x, y, z);

				REAL3 neighborVel = vel[sortedIdx];
				REAL neighborMass = mass[sortedIdx];
				
				REAL weightX = neighborMass * SharpKernel(LengthSquared(pos - xVelocityPos), 1.4);
				REAL weightY = neighborMass * SharpKernel(LengthSquared(pos - yVelocityPos), 1.4);
				REAL weightZ = neighborMass * SharpKernel(LengthSquared(pos - zVelocityPos), 1.4);

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


__global__ void MarkWater_D(VolumeCollection volumes, REAL3* pos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	int3 gridPos;
	gridPos.x = blockIdx.x * blockDim.x + threadIdx.x;
	gridPos.y = blockIdx.y * blockDim.y + threadIdx.y;
	gridPos.z = blockIdx.z * blockDim.z + threadIdx.z;

	if (gridPos.x >= gridRes || gridPos.y >= gridRes || gridPos.z >= gridRes) return;

	volumes.content.writeSurface<uint>(CONTENT_AIR, gridPos.x, gridPos.y, gridPos.z);

	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(gridPos.x + 0.5, gridPos.y + 0.5, gridPos.z + 0.5) * cellSize;

	uint neighHash = calcGridHash(gridPos, gridRes);
	uint startIdx = cellStart[neighHash];
	if (startIdx != 0xffffffff)
	{
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];
			
			REAL3 dist = pos[sortedIdx] - centerPos;
			REAL d2 = LengthSquared(dist);
			if (d2 > cellSize * cellSize)
				continue;
			if (type[sortedIdx] == WALL) {
				volumes.content.writeSurface<uint>(CONTENT_WALL, gridPos.x, gridPos.y, gridPos.z);
			}
		}
		if (volumes.content.readSurface<uint>(gridPos.x, gridPos.y, gridPos.z) != CONTENT_WALL)
		{
			REAL levelSet = LevelSet(gridPos, pos, type, dens, gridHash, gridIdx, cellStart, cellEnd, densVal, gridRes);

			if (levelSet < 0.0)
				volumes.content.writeSurface<uint>(CONTENT_FLUID, gridPos.x, gridPos.y, gridPos.z);
			else
				volumes.content.writeSurface<uint>(CONTENT_AIR, gridPos.x, gridPos.y, gridPos.z);
		}
	}
}

__device__ REAL WallCheck(uint type)
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
	{
		velocity.x = 0.0;
		
	}
	if (y < gridRes && y>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y - 1, z)) < 0)
	{
		velocity.y = 0.0;
	}
	if (z < gridRes && z>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y, z - 1)) < 0)
	{
		velocity.z = 0.0;
	}

	volumes.vel.writeSurface<REAL4>(velocity, x, y, z);
}

__global__ void FixBoundaryX(VolumeCollection volumes, uint gridRes)
{
	uint index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index >= (gridRes + 1) * (gridRes + 1)) 
		return;
	//printf("%d/%d\n", index, (gridRes + 1) * (gridRes + 1));

	uint y = index / (gridRes + 1);
	uint z = index - y * (gridRes + 1);

	REAL4 newVel0 = volumes.vel.readSurface<REAL4>(0, y, z);
	newVel0.x = max(0.0f, newVel0.x);
	volumes.vel.writeSurface<REAL4>(newVel0, 0, y, z);

	REAL4 newVel1 = volumes.vel.readSurface<REAL4>(gridRes, y, z);
	newVel1.x = min(0.0f, newVel1.x);
	volumes.vel.writeSurface<REAL4>(newVel1, gridRes, y, z);

	volumes.content.writeSurface<uint>(CONTENT_WALL, gridRes, y, z);
}

__global__ void FixBoundaryY(VolumeCollection volumes, uint gridRes)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (gridRes + 1) * (gridRes + 1)) return;

	uint x = index / (gridRes + 1);
	uint z = index - x * (gridRes + 1);

	REAL4 newVel0 = volumes.vel.readSurface<REAL4>(x, 0, z);
	newVel0.y = max(0.0f, newVel0.y);
	volumes.vel.writeSurface<REAL4>(newVel0, x, 0, z);

	REAL4 newVel1 = volumes.vel.readSurface<REAL4>(x, gridRes, z);
	newVel1.y = min(0.0f, newVel1.y);
	volumes.vel.writeSurface<REAL4>(newVel1, x, gridRes, z);

	volumes.content.writeSurface<uint>(CONTENT_WALL, x, gridRes, z);
}

__global__ void FixBoundaryZ(VolumeCollection volumes, uint gridRes)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= (gridRes + 1) * (gridRes + 1)) return;

	uint x = index / (gridRes + 1);
	uint y = index - x * (gridRes + 1);

	REAL4 newVel0 = volumes.vel.readSurface<REAL4>(x, y, 0);
	newVel0.z = max(0.0f, newVel0.z);
	volumes.vel.writeSurface<REAL4>(newVel0, x, y, 0);

	REAL4 newVel1 = volumes.vel.readSurface<REAL4>(x, y, gridRes);
	newVel1.z = min(0.0f, newVel1.z);
	volumes.vel.writeSurface<REAL4>(newVel1, x, y, gridRes);

	volumes.content.writeSurface<uint>(CONTENT_WALL, x, y, gridRes);
}

__global__ void ComputeGridDensity_D(VolumeCollection volumes, REAL3* pos, uint* type, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, REAL maxDens, uint gridRes, BOOL* flag)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	if (volumes.content.readSurface<uint>(x,y,z) == CONTENT_WALL || volumes.content.readSurface<uint>(x, y, z) == CONTENT_AIR) {
		volumes.density.writeSurface<REAL>(1.0f, x, y, z);
		return;
	}

	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(x + 0.5, y + 0.5, z + 0.5) * cellSize;
	int3 gridPos = make_int3(x, y, z);
	uint hash = calcGridHash(gridPos, gridRes);

	REAL wsum = 0.0;
	uint cnt = 0;
	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
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

				REAL3 dist = pos[sortedIdx] - centerPos;
				REAL d2 = LengthSquared(dist);

				REAL w = mass[sortedIdx] * SmoothKernel(d2, 4.0 * densVal / gridRes);
				wsum += w;
				cnt++;
			}
		}
	} END_FOR;
	REAL dens = wsum / maxDens;
	//printf("dens cal: %f cnt %d\n", dens, cnt);
	volumes.density.writeSurface<REAL>(dens, x, y, z);
}

__global__ void ComputeDivergence_D(VolumeCollection volumes, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	//Compute Divergence
	REAL cellSize = 1.0 / gridRes;
//	if (volumes.content.readSurface<uint>(x, y, z) == CONTENT_FLUID)
	{
		REAL thisDensity = volumes.density.readTexture<REAL>(x, y, z);

		REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
		REAL4 rightVel = volumes.vel.readSurface<REAL4>(x + 1, y, z);
		REAL4 upVel = volumes.vel.readSurface<REAL4>(x, y + 1, z);
		REAL4 frontVel = volumes.vel.readSurface<REAL4>(x, y, z + 1);

		REAL div = ((rightVel.x - curVel.x) + (upVel.y - curVel.y) + (frontVel.z - curVel.z));
		volumes.divergence.writeSurface<REAL>(div, x, y, z);

		//if (x == s && y == s && z == s)
		//{
			//printf("div: %f \n", div);
			//printf("curVel: %f %f %f\n", curVel.x, curVel.y, curVel.z);
			//printf("rightVel: %f %f %f\n", rightVel.x, rightVel.y, rightVel.z);
			//printf("upVel: %f %f %f\n", upVel.x, upVel.y, upVel.z);
			//printf("frontVel: %f %f %f\n", frontVel.x, frontVel.y, frontVel.z);
		//}
	}
}

__global__ void ComputeLevelSet_D(VolumeCollection volumes, REAL3* pos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	int3 gridPos = make_int3(x, y, z);
	REAL levelSet = LevelSet(gridPos, pos, type, dens, gridHash, gridIdx, cellStart, cellEnd, densVal, gridRes);

	//printf("level: %f\n", levelSet);
	volumes.levelSet.writeSurface<REAL>(levelSet, x, y, z);
}

__global__ void SolvePressureJacobi_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;
	//if (x <= 0 || y <= 0 || z <= 0) return;

	uint thisContent = volumes.content.readSurface<uint>(x, y, z);
	if (thisContent == CONTENT_AIR) {
		volumes.press.writeSurface<REAL>(0.0, x, y, z);
		return;
	}

	REAL cellSize = 1.0 / gridRes;
	REAL thisDiv = volumes.divergence.readSurface<REAL>(x, y, z);
	REAL thisDensity = volumes.density.readTexture<REAL>(x, y, z);
	REAL RHS = -thisDiv * thisDensity;

	REAL newPress = 0.0;
	REAL centerCoeff = 6.0;

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


#if 1
	uint thisContent = volumes.content.readSurface<uint>(x, y, z);
	REAL thisPress = volumes.press.readSurface<REAL>(x, y, z);
	REAL thisDensity = volumes.density.readSurface<REAL>(x, y, z);

	REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
	REAL4 newVel = make_REAL4(0, 0, 0, 0);
	uint4 hasVelocity = make_uint4(0, 0, 0, 0);

	//if (thisContent == CONTENT_AIR) {
	//	volumes.vel.writeSurface<REAL4>(newVel, x, y, z);
	//	return;
	//}

	REAL densityUsed;
	if (x > 0) {
		REAL leftPress = volumes.press.readSurface<REAL>(x - 1, y, z);
		uint leftContent = volumes.content.readSurface<uint>(x - 1, y, z);

		if (thisContent == CONTENT_FLUID || leftContent == CONTENT_FLUID) {
			densityUsed = thisDensity;
			REAL leftDensity = volumes.density.readSurface<REAL>(x - 1, y, z);
			if (thisContent == CONTENT_FLUID && leftContent == CONTENT_FLUID) {
				densityUsed = (leftDensity + thisDensity) / 2.0;
			}
			else if (leftContent == CONTENT_FLUID) {
				densityUsed = leftDensity;
			}

			newVel.x = curVel.x - ((thisPress - leftPress) / densityUsed);
			hasVelocity.x = true;
		}
	}

	if (y > 0) {
		REAL downPress = volumes.press.readSurface<REAL>(x, y - 1, z);
		uint downContent = volumes.content.readSurface<uint>(x, y - 1, z);

		if (thisContent == CONTENT_FLUID || downContent == CONTENT_FLUID) {
			densityUsed = thisDensity;
			REAL downDensity = volumes.density.readSurface<REAL>(x, y - 1, z);
			if (thisContent == CONTENT_FLUID && downContent == CONTENT_FLUID) {
				densityUsed = (downDensity + thisDensity) / 2.0;
			}
			else if (downContent == CONTENT_FLUID) {
				densityUsed = downDensity;
			}
			newVel.y = curVel.y - ((thisPress - downPress) / densityUsed);
			hasVelocity.y = true;
		}

	}

	if (z > 0) {
		REAL backPress = volumes.press.readSurface<REAL>(x, y, z - 1);
		uint backContent = volumes.content.readSurface<uint>(x, y, z - 1);

		if (thisContent == CONTENT_FLUID || backContent == CONTENT_FLUID) {
			densityUsed = thisDensity;
			REAL backDensity = volumes.density.readSurface<REAL>(x, y, z - 1);
			if (thisContent == CONTENT_FLUID && backContent == CONTENT_FLUID) {
				densityUsed = (backDensity + thisDensity) / 2.0;
			}
			else if (backContent == CONTENT_FLUID) {
				densityUsed = backDensity;
			}
			newVel.z = curVel.z - ((thisPress - backPress) / densityUsed);
			hasVelocity.z = true;
		}
	}
#else
	REAL cellSize = 1.0 / gridRes;
	REAL dens = volumes.density.readSurface<REAL>(x, y, z);
	REAL levelSet = volumes.levelSet.readSurface<REAL>(x, y, z);
	uint thisContent = volumes.content.readSurface<uint>(x, y, z);

	REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
	REAL4 newVel = make_REAL4(0, 0, 0, 0);
	uint4 hasVelocity = make_uint4(0, 0, 0, 0);

	if (dens < 0.001 || thisContent == CONTENT_AIR) {
		volumes.vel.writeSurface<REAL4>(curVel, x, y, z);
		return;
	}

	if (x < gridRes && x>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressX = volumes.press.readSurface<REAL>(x - 1, y, z);
		REAL levelSetX = volumes.levelSet.readSurface<REAL>(x - 1, y, z);

		if (levelSet * levelSetX < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(0.001f, levelSetX) * volumes.press.readSurface<REAL>(x - 1, y, z);
			pressX = levelSetX < 0.0 ? volumes.press.readSurface<REAL>(x - 1, y, z) : levelSetX / min(0.00001f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		newVel.x = curVel.x - ((press - pressX) / dens);
		hasVelocity.x = true;
	}

	if (y < gridRes && y>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressY = volumes.press.readSurface<REAL>(x, y - 1, z);
		REAL levelSetY = volumes.levelSet.readSurface<REAL>(x, y - 1, z);

		if (levelSet * levelSetY < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(0.001f, levelSetY) * volumes.press.readSurface<REAL>(x, y - 1 , z);
			pressY = levelSetY < 0.0 ? volumes.press.readSurface<REAL>(x, y - 1, z) : levelSetY / min(0.00001f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		newVel.y = curVel.y - ((press - pressY) / dens);
		hasVelocity.y = true;
	}

	if (z < gridRes && z>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressZ = volumes.press.readSurface<REAL>(x, y, z - 1);
		REAL levelSetZ = volumes.levelSet.readSurface<REAL>(x, y, z - 1);

		if (levelSet * levelSetZ < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(0.001f, levelSetZ) * volumes.press.readSurface<REAL>(x, y, z - 1);
			pressZ = levelSetZ < 0.0 ? volumes.press.readSurface<REAL>(x, y, z - 1) : levelSetZ / min(0.00001f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		newVel.z = curVel.z - ((press - pressZ) / dens);
		hasVelocity.z = true;
	}


#endif

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

	//if (x == s && y == s && z == s ) {

	//	printf("SubtarctGrid_D: %f %f %f\n", newVel.x, newVel.y, newVel.z);
	//}
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
REAL3 GetPointSaveVelocity(REAL3 physicalPos,  uint gridRes, VolumeCollection volume) {
	REAL cellPhysicalSize = 1.0 / gridRes;

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
REAL3 GetPointAfterVelocity(REAL3 physicalPos, uint gridRes, VolumeCollection volume) {
	REAL cellPhysicalSize = 1.0 / gridRes;

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
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles) return;

	REAL3 position = pos[index];
	REAL3 oldGridVel = GetPointSaveVelocity(position, gridRes, volumes);
	REAL3 newGridVel = GetPointAfterVelocity(position, gridRes, volumes);

	REAL3 FLIP = vel[index] + oldGridVel;
	REAL3 PIC = newGridVel;

	REAL FLIPCoeff = 0.95;
	vel[index] = FLIPCoeff * FLIP + (1.0 - FLIPCoeff) * PIC;

	//if (index == 1234)
	//{
	//	printf("pos: %f %f %f\n", position.x, position.y, position.z);
	//	printf("old: %f %f %f\n", oldGridVel.x, oldGridVel.y, oldGridVel.z);
	//	printf("new: %f %f %f\n", newGridVel.x, newGridVel.y, newGridVel.z);
	//	printf("flip: %f %f %f\n", FLIP.x, FLIP.y, FLIP.z);
	//	printf("pic: %f %f %f\n", PIC.x, PIC.y, PIC.z);
	//	printf("final: %f %f %f\n", vel[index].x, vel[index].y, vel[index].z);
	//}
}

__global__ void AdvecParticle_D(VolumeCollection volumes, REAL3* beforePos, REAL3* curPos, REAL3* vel, uint* type, uint gridRes, uint numParticles, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID)
		return;

	beforePos[idx] = curPos[idx];

	REAL3 lerpVel = GetPointAfterVelocity(curPos[idx], gridRes, volumes);
	curPos[idx] += dt * lerpVel;
	//curPos[idx] += dt * vel[idx];

	REAL wallThick = 1.0 / gridRes;
	curPos[idx].x = max(wallThick, min((REAL)(1.0 - wallThick), curPos[idx].x));
	curPos[idx].y = max(wallThick, min((REAL)(1.0 - wallThick), curPos[idx].y));
	curPos[idx].z = max(wallThick, min((REAL)(1.0 - wallThick), curPos[idx].z));
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

__global__ void GridVisualize_D(VolumeCollection volumes, uint gridRes, REAL3* gridPos, REAL3* gridVel, REAL* gridPress, REAL* gridDens, REAL* gridLevelSet, REAL* gridDiv, uint* gridContent)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(x + 0.5, y + 0.5, z + 0.5) * cellSize;
	int index = x * (gridRes * gridRes) + y * (gridRes)+z;

	gridPos[index] = make_REAL3(centerPos.x, centerPos.y, centerPos.z);

	REAL4 vel = volumes.vel.readSurface<REAL4>(x, y, z);
	gridVel[index] = make_REAL3(vel.x, vel.y, vel.z);

	REAL press = volumes.press.readSurface<REAL>(x, y, z);
	gridPress[index] = press;

	REAL dens = volumes.density.readSurface<REAL>(x, y, z);
	gridDens[index] = dens;

	REAL levelSet = volumes.levelSet.readSurface<REAL>(x, y, z);
	gridLevelSet[index] = levelSet;

	REAL divergence = volumes.divergence.readSurface<REAL>(x, y, z);
	gridDiv[index] = divergence;
	
	uint content = volumes.content.readSurface<uint>(x, y, z);
	gridContent[index] = content;

	
}