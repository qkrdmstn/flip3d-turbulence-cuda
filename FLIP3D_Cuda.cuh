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

__global__ void ResetCell_D(VolumeCollection volumes, uint gridRes, float content) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	volumes.content.writeSurface<int>(CONTENT_AIR, x, y, z);

	volumes.hasVelocity.writeSurface<int4>(make_int4(0, 0, 0, 0), x, y, z);
	volumes.velocity.writeSurface<float4>(make_float4(0, 0, 0, 0), x, y, z);
	volumes.newVelocity.writeSurface<float4>(make_float4(0, 0, 0, 0), x, y, z);
}

__device__
float inline trilinearUnitKernel(float r) {
	r = abs(r);
	if (r > 1) return 0;
	return 1 - r;
}

__device__
float inline trilinearHatKernel(float3 r, float support) {
	return trilinearUnitKernel(r.x / support) * trilinearUnitKernel(r.y / support) * trilinearUnitKernel(r.z / support);
}

__global__  void transferToCellAccumPhase(VolumeCollection volumes, uint gridRes, float cellPhysicalSize, int* cellStart, int* cellEnd, Particle* particles, int particleCount) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	int cellCount = (gridRes) * (gridRes) * (gridRes);

	float3 xVelocityPos = make_float3(x, (y + 0.5), (z + 0.5)) * cellPhysicalSize;
	float3 yVelocityPos = make_float3((x + 0.5), y, (z + 0.5)) * cellPhysicalSize;
	float3 zVelocityPos = make_float3((x + 0.5), y + 0.5, z) * cellPhysicalSize;

	float4 thisVelocity = make_float4(0, 0, 0, 0);
	float4 weight = make_float4(0, 0, 0, 0);


	for (int dx = -1; dx <= 1; ++dx) {
		for (int dy = -1; dy <= 1; ++dy) {
			for (int dz = -1; dz <= 1; ++dz) {
				int cell = (x + dx) * gridRes * gridRes + (y + dy) * gridRes + z + dz;

				if (cell >= 0 && cell < cellCount) {

					for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
						if (j >= 0 && j < particleCount) {


							const Particle& p = particles[j];
							float3 pPosition = p.position;
							float3 pVelocity = p.velocity;
							float thisWeightX = trilinearHatKernel(pPosition - xVelocityPos, cellPhysicalSize);
							float thisWeightY = trilinearHatKernel(pPosition - yVelocityPos, cellPhysicalSize);
							float thisWeightZ = trilinearHatKernel(pPosition - zVelocityPos, cellPhysicalSize);

							thisVelocity.x += thisWeightX * pVelocity.x;
							thisVelocity.y += thisWeightY * pVelocity.y;
							thisVelocity.z += thisWeightZ * pVelocity.z;

							weight.x += thisWeightX;
							weight.y += thisWeightY;
							weight.z += thisWeightZ;
						}
					}
				}

			}
		}
	}

	volumes.velocityAccumWeight.writeSurface<float4>(weight, x, y, z);
	volumes.velocity.writeSurface<float4>(thisVelocity, x, y, z);

}


__global__  void transferToCellDividePhase(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, Particle* particles, int particleCount) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;

	int index = x * (sizeY * sizeZ) + y * (sizeZ)+z;



	float4 weight = volumes.velocityAccumWeight.readSurface<float4>(x, y, z);
	int4 hasVelocity = volumes.hasVelocity.readSurface<int4>(x, y, z);
	float4 velocity = volumes.velocity.readSurface<float4>(x, y, z);

	if (weight.x > 0) {
		velocity.x /= weight.x;
		hasVelocity.x = true;
	}

	if (weight.y > 0) {
		velocity.y /= weight.y;
		hasVelocity.y = true;
	}

	if (weight.z > 0) {
		velocity.z /= weight.z;
		hasVelocity.z = true;
	}

	volumes.velocity.writeSurface<float4>(velocity, x, y, z);
	volumes.newVelocity.writeSurface<float4>(velocity, x, y, z);
	volumes.hasVelocity.writeSurface<int4>(hasVelocity, x, y, z);



	for (int j = cellStart[index]; j <= cellEnd[index]; ++j) {
		if (j >= 0 && j < particleCount) {

			volumes.content.writeSurface<int>(CONTENT_FLUID, x, y, z);
			break;
		}
	}
}

__global__  void transferVolumeFractionsToCell(VolumeCollection volumes, int sizeX, int sizeY, int sizeZ, float cellPhysicalSize, int* cellStart, int* cellEnd, Particle* particles, int particleCount, float4 phaseDensities) {

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;


	if (x >= sizeX || y >= sizeY || z >= sizeZ) return;


	int cellCount = (sizeX) * (sizeY) * (sizeZ);

	float3 centerPos = make_float3(x + 0.5, y + 0.5, z + 0.5) * cellPhysicalSize;

	float4 fractions = make_float4(0, 0, 0, 0);
	float weight = 0;

	int cell = (x)*sizeY * sizeZ + (y)*sizeZ + z;

	if (cell >= 0 && cell < cellCount) {
		for (int j = cellStart[cell]; j <= cellEnd[cell]; ++j) {
			if (j >= 0 && j < particleCount) {


				const Particle& p = particles[j];
				float3 pPosition = p.position;
				float thisWeight = trilinearHatKernel(pPosition - centerPos, cellPhysicalSize);

				fractions += thisWeight * p.volumeFractions;
				weight += thisWeight;
			}
		}
	}

	if (weight > 0) {
		fractions /= weight;

		float fractionsSum = fractions.x + fractions.y + fractions.z + fractions.w;
		fractions /= fractionsSum;
	}

	float density = dot(phaseDensities, fractions);
	volumes.density.writeSurface<float>(density, x, y, z);

	volumes.volumeFractions.writeSurface<float4>(fractions, x, y, z);
	volumes.newVolumeFractions.writeSurface<float4>(fractions, x, y, z);

}
