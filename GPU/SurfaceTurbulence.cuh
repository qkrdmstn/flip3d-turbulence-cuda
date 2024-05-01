#ifndef __SURFACETURBULENCE_CUH__
#define __SURFACETURBULENCE_CUH__

#include "SurfaceTurbulence.h"
#include "Hash.cuh"
#include "WeightKernels.cuh"

#include <cmath>
#include<stdio.h>

__global__ void Initialize_D(REAL3* coarsePos, uint* coarseType, REAL3* finePos, uint* particleGridIdx, uint numCoarseParticles, uint* gridIdx, uint* cellStart, uint* cellEnd, uint baseRes, REAL fineScaleLen, REAL outerRadius, REAL innerRadius)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numCoarseParticles)
		return;

	for (int i = 0; i < PER_PARTICLE; i++) {
		REAL3 newPos = make_REAL3(-1, -1, -1);
		int index = idx * PER_PARTICLE + i;
		finePos[index] = newPos;
		particleGridIdx[index] = numCoarseParticles * PER_PARTICLE;
	}

	if (coarseType[idx] != FLUID)
		return;

	uint discretization = (uint)PI * (outerRadius + innerRadius) / fineScaleLen;
	REAL dtheta = 2.0 * fineScaleLen / (outerRadius + innerRadius);
	REAL outerRadius2 = outerRadius * outerRadius;

	BOOL nearSurface = false;
	REAL3 pos = coarsePos[idx];
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				
				REAL x = max(0.0f, min((REAL)baseRes, baseRes * pos.x));
				REAL y = max(0.0f, min((REAL)baseRes, baseRes * pos.y));
				REAL z = max(0.0f, min((REAL)baseRes, baseRes * pos.z));

				uint indexX = x + i;
				uint indexY = y + j;
				uint indexZ = z + k;
				if (indexX < 0 || indexY < 0 || indexZ < 0) continue;
				
				if (GetNumFluidParticleAt(indexX, indexY, indexZ, coarsePos, coarseType, gridIdx, cellStart, cellEnd, baseRes) == 0){
					nearSurface = true;
					break;
				}
			}
		}
	}

	int cnt = 0;
	if (nearSurface) {
		for (uint i = 0; i <= discretization / 2; ++i) {
			REAL discretization2 = REAL(floor(2.0 * PI * sin(i * dtheta) / dtheta) + 1);
			for (REAL phi = 0; phi < 2.0 * PI; phi += REAL(2.0 * PI / discretization2)) {
				REAL theta = i * dtheta;
				REAL3 normal = make_REAL3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
				REAL3 position = pos + normal * outerRadius;

				bool valid = true;
				int3 gridPos = calcGridPos(pos, 1.0 / baseRes);
				FOR_NEIGHBOR(2) {
					int3 neighGridPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);

					uint neighHash = calcGridHash(neighGridPos, baseRes);
					uint startIdx = cellStart[neighHash];
					if (startIdx != 0xffffffff) {
						uint endIdx = cellEnd[neighHash];
						for (uint i = startIdx; i < endIdx; i++)
						{
							uint sortedIdx = gridIdx[i];
							REAL3 neighborPos = coarsePos[sortedIdx];

							REAL cellSize = 1.0 / baseRes;
							REAL d2 = LengthSquared(position - neighborPos);
							//if (d2 > cellSize * cellSize * 4)
							//	continue;
							//if (coarseType[sortedIdx] == WALL)
							//	continue;

							if (idx != sortedIdx && d2 < outerRadius2) {
								valid = false;
								break;
							}
						}
					}
				}END_FOR;

				if (valid) {
					int index = idx * PER_PARTICLE + cnt;
					REAL3 newPos = make_REAL3(position.x, position.y, position.z);

					finePos[index] = newPos;
					particleGridIdx[index] = idx * PER_PARTICLE + cnt;
					cnt++;
				}
			}
		}
	}
}

__global__ void StateCheck_D(REAL3* finePos, uint* particleGridIdx, uint* stateData, uint numCoarseParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numCoarseParticles)
		return;

	for (int i = 0; i < PER_PARTICLE; i++)
	{
		int index = idx * PER_PARTICLE + i;
		if (finePos[index].x > 0 && finePos[index].y > 0 && finePos[index].z > 0) {
			stateData[index] = 1;
		}
		else {
			stateData[index] = 0;
		}
	}
}

__global__ void ComputeCoarseDens_D(REAL r, REAL3* coarsePos, uint* coarseType, REAL* coarseKernelDens, uint* gridIdx, uint* cellStart, uint* cellEnd, uint hashRes, uint numCoarseParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numCoarseParticles)
		return;

	REAL cellSize = 1.0 / hashRes;
	REAL3 pos1 = coarsePos[idx];
	int3 gridPos = calcGridPos(pos1, hashRes);

	FOR_NEIGHBOR(2) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, hashRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				
				REAL3 pos2 = coarsePos[sortedIdx];

				REAL d2 = LengthSquared(pos2 - pos1);
				if (d2 > cellSize * cellSize * 4)
					continue;

				if (coarseType[idx] != FLUID || coarseType[sortedIdx] != FLUID)
					continue;
				coarseKernelDens[idx] += DistKernel(pos1 - pos2, r);
			}
		}
	}END_FOR;
}

__device__ REAL NeighborWeight(REAL3* finePos, REAL3* coarsePos, uint* coarseType, REAL* coarseKernelDens, uint fineIdx, uint coarseIdx, uint* gridIdx, uint* cellStart, uint* cellEnd, uint coarseRes, REAL r)
{
	REAL3 fPos = finePos[fineIdx];
	REAL3 cPos = coarsePos[coarseIdx];
	REAL weight = DistKernel((fPos - cPos), r) / coarseKernelDens[coarseIdx];
	if (weight == 0) return 0;

	REAL weightSum = 0.0;
	int3 gridPos = calcGridPos(fPos, coarseRes);

	REAL3 displacement = make_REAL3(0, 0, 0);
	FOR_NEIGHBOR(2) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, coarseRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (coarseType[sortedIdx] != FLUID) continue;

				REAL3 pos = coarsePos[sortedIdx];
				if (LengthSquared(fPos - pos) > r * r) continue;
				weightSum += (DistKernel(fPos - pos, r) / coarseKernelDens[sortedIdx]);
			}

		}

	}END_FOR;

	if (weightSum > 0)
		return weight / weightSum;
	else
		return 0.0f;
}

__global__ void Advection_D(REAL3* finePos, REAL3* coarseCurPos, REAL3* coarseBeforePos, uint* coarseType, REAL* coarseKernelDens, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numFineParticles, REAL coarseScaleLen, uint coarseRes, BOOL* flag)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 2.0 * coarseScaleLen;

	REAL cellSize = 1.0 / coarseRes;
	REAL3 fPos = finePos[idx];
	int3 gridPos = calcGridPos(fPos, cellSize);

	REAL3 displacement = make_REAL3(0, 0, 0);
	FOR_NEIGHBOR(2) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, coarseRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];

				REAL3 curPos = coarseCurPos[sortedIdx];

				REAL d2 = LengthSquared(curPos - fPos);
				if (d2 > cellSize * cellSize * 4)
					continue;

				if (coarseType[sortedIdx] != FLUID)
					continue;

				REAL3 beforePos = coarseBeforePos[sortedIdx];
				displacement += (curPos - beforePos) * NeighborWeight(finePos, coarseCurPos, coarseType, coarseKernelDens, idx, sortedIdx, gridIdx, cellStart, cellEnd, coarseRes, r);

			}
		}
	}END_FOR;

	finePos[idx] += displacement;
}

#endif