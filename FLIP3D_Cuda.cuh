#ifndef __FLIP_CUDA_CUH__
#define __FLIP_CUDA_CUH__

#include "FLIP3D_Cuda.h"
#include <cmath>
#include<stdio.h>
#include "Hash.cuh"
#include "WeightKernels.cuh"



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

__global__ void ComputeWallParticleNormal_D(REAL3* pos, uint* type, REAL3* normal, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numParticles, uint gridRes)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	REAL cellSize = 1.0 / gridRes;
	REAL wallThick = 1.0 / gridRes;
	REAL3 position = pos[idx];

	REAL3 normalVector = make_REAL3(0.0f, 0.0f, 0.0f);
	if (type[idx] == WALL) {
		if (position.x <= 1.1 * wallThick) {
			normalVector.x = 1.0;
		}
		if (position.x >= 1.0 - 1.1 * wallThick) {
			normalVector.x = -1.0;
		}
		if (position.y <= 1.1 * wallThick) {
			normalVector.y = 1.0;
		}
		if (position.y >= 1.0 - 1.1 * wallThick) {
			normalVector.y = -1.0;
		}
		if (position.z <= 1.1 * wallThick) {
			normalVector.z = 1.0;
		}
		if (position.z >= 1.0 - 1.1 * wallThick) {
			normalVector.z = -1.0;
		}

		if (normalVector.x == 0.0f && normalVector.y == 0.0f && normalVector.z == 0.0f) {
			int3 gridPos = calcGridPos(position, cellSize);
			FOR_NEIGHBOR(3) {

				int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
				uint neighHash = calcGridHash(neighbourPos, gridRes);
				uint startIdx = cellStart[neighHash];

				if (startIdx != 0xffffffff)
				{
					uint endIdx = cellEnd[neighHash];
					for (uint i = startIdx; i < endIdx; i++)
					{
						uint sortedIdx = gridIdx[i];
						if (sortedIdx != idx && type[sortedIdx] == WALL) {
							REAL d = Length(position - pos[sortedIdx]);
							REAL w = 1.0 / d;
							normalVector += w * (position - pos[sortedIdx]) / d;
						}

					}
				}
			}END_FOR;
		}
	}

	Normalize(normalVector);
	normal[idx] = normalVector;
}

__global__ void ResetCell_D(VolumeCollection volumes, uint gridRes) {

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	volumes.content.writeSurface<uint>(CONTENT_AIR, x, y, z);

	volumes.hasVel.writeSurface<uint4>(make_uint4(0, 0, 0, 0), x, y, z);
	volumes.vel.writeSurface<REAL4>(make_REAL4(0, 0, 0, 0), x, y, z);
	volumes.velSave.writeSurface<REAL4>(make_REAL4(0, 0, 0, 0), x, y, z);

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
					//if (d2 > cellSize * cellSize)
					//	continue;

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

	if (gridPos.x > gridRes || gridPos.y > gridRes || gridPos.z > gridRes) return;

	int cellCount = (gridRes) * (gridRes) * (gridRes);
	REAL cellPhysicalSize = 1.0 / gridRes;
	


#if 0
	REAL3 xVelocityPos = make_REAL3(gridPos.x, (gridPos.y + 0.5), (gridPos.z + 0.5));
	REAL3 yVelocityPos = make_REAL3((gridPos.x + 0.5), gridPos.y, (gridPos.z + 0.5));
	REAL3 zVelocityPos = make_REAL3((gridPos.x + 0.5), (gridPos.y + 0.5), gridPos.z);

	REAL4 velocity = make_REAL4(0, 0, 0, 0);
	REAL4 weight = make_REAL4(0, 0, 0, 0);

	FOR_NEIGHBOR_WALL(2) {

		int3 neighborGridPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);

		//if (gridPos.x < 0 || gridPos.x > gridRes - 1 || gridPos.y < 0 || gridPos.y > gridRes - 1 || gridPos.z < 0 || gridPos.z > gridRes - 1)
		//	continue;
		uint neighHash = calcGridHash(neighborGridPos, gridRes);
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
#else

	REAL3 xVelocityPos = make_REAL3(gridPos.x, (gridPos.y + 0.5), (gridPos.z + 0.5)) * cellPhysicalSize;
	REAL3 yVelocityPos = make_REAL3((gridPos.x + 0.5), gridPos.y, (gridPos.z + 0.5)) * cellPhysicalSize;
	REAL3 zVelocityPos = make_REAL3((gridPos.x + 0.5), (gridPos.y + 0.5), gridPos.z) * cellPhysicalSize;

	REAL4 velocity = make_REAL4(0, 0, 0, 0);
	REAL4 weight = make_REAL4(0, 0, 0, 0);

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

				REAL3 pPosition = pos[sortedIdx];
				REAL3 pVelocity = vel[sortedIdx];
				REAL thisWeightX = trilinearHatKernel(pPosition - xVelocityPos, cellPhysicalSize);
				REAL thisWeightY = trilinearHatKernel(pPosition - yVelocityPos, cellPhysicalSize);
				REAL thisWeightZ = trilinearHatKernel(pPosition - zVelocityPos, cellPhysicalSize);

				velocity.x += thisWeightX * pVelocity.x;
				velocity.y += thisWeightY * pVelocity.y;
				velocity.z += thisWeightZ * pVelocity.z;

				weight.x += thisWeightX;
				weight.y += thisWeightY;
				weight.z += thisWeightZ;
			}
		}
	}END_FOR;
	velocity.x = weight.x ? velocity.x / weight.x : 0.0;
	velocity.y = weight.y ? velocity.y / weight.y : 0.0;
	velocity.z = weight.z ? velocity.z / weight.z : 0.0;
#endif
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
	if (x < gridRes && x>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x - 1, y, z)) < 0)
	{
		velocity.x = 0.0;
	}

	if (y == 0 || y == gridRes)
		velocity.y = 0.0;
	if (y < gridRes && y>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y - 1, z)) < 0)
	{
		velocity.y = 0.0;
	}

	if (z == 0 || z == gridRes)
		velocity.z = 0.0;
	if (z < gridRes && z>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y, z - 1)) < 0)
	{
		velocity.z = 0.0;
	}

	volumes.vel.writeSurface<REAL4>(velocity, x, y, z);
}

__global__ void ComputeGridDensity_D(VolumeCollection volumes, REAL3* pos, uint* type, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, REAL maxDens, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	if (volumes.density.readSurface<REAL>(x, y, z) == CONTENT_WALL) {
		volumes.density.writeSurface<REAL>(1.0f, x, y, z);
		return;
	}

	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(x + 0.5, y + 0.5, z + 0.5) * cellSize;
	int3 gridPos = make_int3(x, y, z);

	uint cnt = 0;
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

				if (type[sortedIdx] == WALL)
					continue;

				REAL3 dist = pos[sortedIdx] - centerPos;
				REAL d2 = LengthSquared(dist);
				if (d2 > cellSize * cellSize)
					continue;

				REAL w = mass[sortedIdx] * SmoothKernel(d2, 4.0 * densVal / gridRes);
				wsum += w;
			}
		}
	}END_FOR;
	REAL dens = wsum / maxDens;
	volumes.density.writeSurface<REAL>(dens, x, y, z);
}

__global__ void ComputeDivergence_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	//Compute Divergence
	REAL cellSize = 1.0 / gridRes;
	//if (volumes.content.readSurface<uint>(x, y, z) == CONTENT_FLUID)
	{
		REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
		REAL4 rightVel = volumes.vel.readSurface<REAL4>(x + 1, y, z);
		REAL4 upVel = volumes.vel.readSurface<REAL4>(x, y + 1, z);
		REAL4 frontVel = volumes.vel.readSurface<REAL4>(x, y, z + 1);

		REAL div = ((rightVel.x - curVel.x) + (upVel.y - curVel.y) + (frontVel.z - curVel.z));
		//REAL div = ((rightVel.x - curVel.x) + (upVel.y - curVel.y) + (frontVel.z - curVel.z)) / cellSize;

		//if (x == 15 && y == 2 && z == 15) {
		//	printf("X vel: %f %f\n", rightVel.x, curVel.x);
		//	printf("Y vel: %f %f\n", upVel.y , curVel.y);
		//	printf("Z vel: %f %f\n", frontVel.z , curVel.z);

		//	printf("div: %f\n", div);
		//}

		volumes.divergence.writeSurface<REAL>(div, x, y, z);
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

	volumes.levelSet.writeSurface<REAL>(levelSet, x, y, z);
}

__global__ void SolvePressureJacobi_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	uint thisContent = volumes.content.readSurface<uint>(x, y, z);
	if (thisContent == CONTENT_AIR || thisContent == CONTENT_WALL) {
		volumes.press.writeSurface<REAL>(0.0, x, y, z);
		return;
	}

	REAL cellSize = 1.0 / gridRes;
	REAL thisDiv = volumes.divergence.readSurface<REAL>(x, y, z);
	REAL thisDensity = volumes.density.readTexture<REAL>(x, y, z);
	REAL RHS = -thisDiv * thisDensity;

	REAL newPress = 0.0;
	REAL centerCoeff = 6.0;

	REAL rP = volumes.press.readTexture<REAL>(x + 1, y, z);
	REAL lP = volumes.press.readTexture<REAL>(x - 1, y, z);
	REAL uP = volumes.press.readTexture<REAL>(x, y + 1, z);
	REAL dP = volumes.press.readTexture<REAL>(x, y - 1, z);
	REAL fP = volumes.press.readTexture<REAL>(x, y, z + 1);
	REAL bP = volumes.press.readTexture<REAL>(x, y, z - 1);

	newPress += rP;
	newPress += lP;
	newPress += uP;
	newPress += dP;
	newPress += fP;
	newPress += bP;
	newPress += RHS;
	newPress /= centerCoeff;

	//if (x == 15 && y == 2 && z == 15) {
	//	printf("RHS: %f\n", RHS);
	//	printf("rP: %f\n", rP);
	//	printf("lP: %f\n", lP);
	//	printf("uP: %f\n", uP);
	//	printf("dP: %f\n", dP);
	//	printf("fP: %f\n", fP);
	//	printf("bP: %f\n", bP);
	//	printf("final: %f\n", newPress);
	//}
	volumes.press.writeSurface<REAL>(newPress, x, y, z);
}

__global__ void ComputeVelocityWithPress_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;
	uint thisContent = volumes.content.readSurface<uint>(x, y, z);
	REAL thisPress = volumes.press.readSurface<REAL>(x, y, z);
	REAL thisDensity = volumes.density.readSurface<REAL>(x, y, z);

	REAL cellSize = 1.0 / gridRes;
	REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
	REAL4 newv = make_REAL4(0,0,0,0);
	uint4 hasVelocity = make_uint4(0, 0, 0, 0);

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
			REAL uX = curVel.x - ((thisPress - leftPress)/ densityUsed);
			curVel.x = uX;
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
			REAL uY = curVel.y - ((thisPress - downPress) / densityUsed);
			curVel.y = uY;
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
			REAL uZ = curVel.z - ((thisPress - backPress )/ densityUsed);
			curVel.z = uZ;
			hasVelocity.z = true;
		}
	}

	volumes.hasVel.writeSurface<uint4>(hasVelocity, x, y, z);
	volumes.vel.writeSurface<REAL4>(curVel, x, y, z);
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

__device__ inline REAL3 lerp(REAL3 v0, REAL3 v1, REAL t) {
	return (1 - t) * v0 + t * v1;
}

__device__ REAL3 GetLerpValueAtPoint(VolumeData data, REAL x, REAL y, REAL z, uint gridRes)
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

__device__ REAL3 GetPointSaveVelocity(REAL3 physicalPos,  uint gridRes, VolumeCollection volume) {
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

__device__ REAL3 GetPointAfterVelocity(REAL3 physicalPos, uint gridRes, VolumeCollection volume) {
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
}

__global__ void ConstraintOuterWall_D(REAL3* pos, REAL3* vel, REAL3* normal, uint* type, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numParticles, REAL gridRes, REAL densVal)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID)
		return;

	REAL wallThick = 1.0 / gridRes;
	pos[idx].x = max(wallThick, min((REAL)(1.0 - wallThick), pos[idx].x));
	pos[idx].y = max(wallThick, min((REAL)(1.0 - wallThick), pos[idx].y));
	pos[idx].z = max(wallThick, min((REAL)(1.0 - wallThick), pos[idx].z));

#if 1
	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);

	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];
		REAL re = 1.5 * densVal / gridRes;

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (type[sortedIdx] == WALL) {
					REAL dist = Length(pos[idx] - pos[sortedIdx]);
					if (dist < re) {
						REAL3 normalVector = normal[sortedIdx];
						if (normalVector.x == 0.0f && normalVector.y == 0.0f && normalVector.z == 0.0f)
							normalVector = (pos[idx] - pos[sortedIdx]) / dist;

						pos[idx] += (re - dist) * normalVector;
						REAL dot = Dot(vel[idx], normalVector);
						vel[idx] -= dot * normalVector;
					}
				}
			}

		}

	}END_FOR;
#endif
}

__device__ REAL3 Resample(REAL3 curPos, REAL3 curVel, REAL3* pos, REAL3* vel, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, REAL re)
{
	REAL wsum = 0.0f;
	REAL3 save = make_REAL3(curVel.x, curVel.y, curVel.z);
	curVel = make_REAL3(0, 0, 0);


	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(curPos.x + dx, curPos.y + dy, curPos.z + dz);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];
		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				REAL3 dist = curPos - pos[sortedIdx];
				REAL d2 = LengthSquared(dist);
				REAL w = 1.0 * SharpKernel(d2, re);
				curVel += w * vel[sortedIdx];
				wsum += w;
			}
		}
	}END_FOR;

	if (wsum)
		curVel /= wsum;
	else
		curVel = save;

	return curVel;
}

__global__ void Correct_D(REAL3* pos, REAL3* vel, REAL3* normal, REAL* mass, uint* type, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, uint numParticles, REAL dt, REAL re, uint r1, uint r2, uint r3)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID)
		return;

	REAL springCoeff = 50.0f;
	REAL3 spring = make_REAL3(0, 0, 0);

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);

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
					REAL3 dist = pos[sortedIdx] - pos[idx];
					REAL d = Length(dist);
					REAL w = springCoeff * mass[sortedIdx] * SmoothKernel(d * d, re);
					if (d > 0.1 * re) {
						spring += w * (pos[idx] - pos[sortedIdx]) / d * re;
					}
					else {
						if (type[sortedIdx] == FLUID) {
							spring.x += 0.01 * re / dt * (r1 % 101) / 100.0;
							spring.y += 0.01 * re / dt * (r2 % 101) / 100.0;
							spring.z += 0.01 * re / dt * (r3 % 101) / 100.0;
						}
						else
						{
							spring.x += 0.05 * re / dt * normal[sortedIdx].x;
							spring.y += 0.05 * re / dt * normal[sortedIdx].y;
							spring.z += 0.05 * re / dt * normal[sortedIdx].z;
						}
					}
					
				}
			}
		}
	} END_FOR;
	REAL3 temp = pos[idx] + dt * spring;

	REAL3 temp2 = vel[idx];
	temp2 = Resample(temp, temp2, pos, vel, gridHash, gridIdx, cellStart, cellEnd, gridRes, re);

	pos[idx] = temp;
	vel[idx] = temp2;

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

#endif