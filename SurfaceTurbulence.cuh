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
	int3 gridPos = calcGridPos(pos1, cellSize);

	REAL kernelDens = 0.0f;
	int width = (r / cellSize) + 1;
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
				if (coarseType[idx] != FLUID || coarseType[sortedIdx] != FLUID)
					continue;
				
				REAL3 pos2 = coarsePos[sortedIdx];
				kernelDens += DistKernel(pos1 - pos2, r);
			}
		}
	}END_FOR;
	coarseKernelDens[idx] = kernelDens;
}

__global__ void ComputeFineDens_D(REAL r, REAL3* finePos, REAL* fineKernelDens, uint* gridIdx, uint* cellStart, uint* cellEnd, uint hashRes, uint numFineParticles, BOOL* flag)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL cellSize = 1.0 / hashRes;
	REAL3 pos1 = finePos[idx];
	int3 gridPos = calcGridPos(pos1, cellSize);

	REAL kernelDens = 0.0f;
	int width = (r / cellSize) + 1;
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
				REAL3 pos2 = finePos[sortedIdx];

				REAL d2 = LengthSquared(pos2 - pos1);
				if (d2 > r*r)
					continue;
				kernelDens += DistKernel(pos1 - pos2, r);
			}
		}
	}END_FOR;
	fineKernelDens[idx] = kernelDens;
}


__device__ REAL NeighborCoarseWeight(REAL r, REAL3* finePos, REAL3* coarsePos, uint* coarseType, REAL* coarseKernelDens, uint fineIdx, uint coarseIdx, uint* gridIdx, uint* cellStart, uint* cellEnd, uint coarseRes)
{
	REAL3 fPos = finePos[fineIdx];
	REAL3 cPos = coarsePos[coarseIdx];
	REAL weight = DistKernel((fPos - cPos), r) / coarseKernelDens[coarseIdx];
	if (weight == 0) return 0;

	REAL cellSize = 1.0 / coarseRes;
	REAL weightSum = 0.0;
	int3 gridPos = calcGridPos(fPos, cellSize);

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

__device__ REAL NeighborFineWeight(REAL r, REAL3* finePos, REAL* fineKernelDens, uint fineIdx1, uint fineIdx2, uint* gridIdx, uint* cellStart, uint* cellEnd, uint fineRes)
{
	REAL3 fPos1 = finePos[fineIdx1];
	REAL3 fPos2 = finePos[fineIdx2];
	REAL weight = DistKernel((fPos1 - fPos2), r) / fineKernelDens[fineIdx2];
	if (weight == 0) return 0;

	REAL cellSize = 1.0 / fineRes;
	REAL weightSum = 0.0;
	int3 gridPos = calcGridPos(fPos1, cellSize);

	REAL width = (r / cellSize) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, fineRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				REAL3 pos = finePos[sortedIdx];
				if (LengthSquared(fPos1 - pos) > r * r) continue;
				weightSum += (DistKernel(fPos1 - pos, r) / fineKernelDens[sortedIdx]);
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
				if (coarseType[sortedIdx] != FLUID)
					continue;

				REAL3 curPos = coarseCurPos[sortedIdx];

				REAL d2 = LengthSquared(curPos - fPos);
				if (d2 > cellSize * cellSize * 4)
					continue;

				REAL3 beforePos = coarseBeforePos[sortedIdx];
				displacement += (curPos - beforePos) * NeighborCoarseWeight(r, finePos, coarseCurPos, coarseType, coarseKernelDens, idx, sortedIdx, gridIdx, cellStart, cellEnd, coarseRes);
			}
		}
	}END_FOR;

	finePos[idx] += displacement;
	flag[idx] = false;
}


__device__ REAL MetaballDens(REAL dist, REAL coarseScaleLen)
{
	return exp(-2 * pow((dist / coarseScaleLen), 2));
}

__device__ REAL MetaballLevelSet(REAL3 finePos, REAL3* coarsePos, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL outerRadius, REAL innerRadius, uint coarseRes)
{
	REAL coarseScaleLen = 1.0 / coarseRes;
	REAL cellSize = coarseScaleLen;

	REAL R = outerRadius;
	REAL r = innerRadius;
	REAL u = (3.0 / 2.0) * R;
	REAL a = log(2.0 / (1.0 + MetaballDens(u, coarseScaleLen))) / (pow(u / 2.0, 2.0) - pow(r, 2.0));

	REAL3 fPos = finePos;
	int3 gridPos = calcGridPos(fPos, cellSize);
	
	REAL f = 0.0;
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
				REAL3 cPos = coarsePos[sortedIdx];

				REAL d2 = LengthSquared(fPos - cPos);
				if (d2 > cellSize * cellSize * 4)
					continue;

				f += exp(-a * d2);
			}
		}
	}END_FOR;
	if (f > 1.0f) f = 1.0f;

	f = (sqrt(-log(f) / a) - r) / (R - r);
	return f;
}

__device__ REAL3 MetaballConstraintGradient(REAL3 finePos, REAL3* coarsePos, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL outerRadius, REAL innerRadius, uint coarseRes, int width)
{
	REAL coarseScaleLen = 1.0 / coarseRes;
	REAL cellSize = coarseScaleLen;

	REAL R = outerRadius;
	REAL r = innerRadius;
	REAL u = (3.0 / 2.0) * R;
	REAL a = log(2.0 / (1.0 + MetaballDens(u, coarseScaleLen))) / (pow(u / 2.0, 2.0) - pow(r, 2.0));

	REAL3 fPos = finePos;
	int3 gridPos = calcGridPos(fPos, cellSize);

	REAL3 gradient = make_REAL3(0, 0, 0);
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, coarseRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				REAL3 cPos = coarsePos[sortedIdx];

				REAL d2 = LengthSquared(fPos - cPos);
				if (d2 > cellSize * cellSize * 4)
					continue;

				gradient += (fPos - cPos) * 2.0 * a * exp(-a * d2);
			}
		}
	}END_FOR;

	Normalize(gradient);
	return gradient;
}

__global__ void SurfaceConstraint_D(REAL3* finePos, REAL3* coarsePos, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL outerRadius, REAL innerRadius, uint coarseRes, uint numFineParticles, REAL3* normal)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL R = outerRadius;
	REAL r = innerRadius;

	REAL3 fPos = finePos[idx];
	REAL levelSet = MetaballLevelSet(fPos, coarsePos, gridIdx, cellStart, cellEnd, outerRadius, innerRadius, coarseRes);
	if (levelSet <= 1.0 && levelSet >= 0.0) return;

	REAL3 gradient = MetaballConstraintGradient(fPos, coarsePos, gridIdx, cellStart, cellEnd, outerRadius, innerRadius, coarseRes, 2); // Constraints Projection
	if (levelSet < 0.0) {
		fPos -= gradient * (R - r) * levelSet;
	}
	else if (levelSet > 1.0) {
		fPos -= gradient * (R - r) * (levelSet - 1);
	}

	finePos[idx] = fPos;
}

__global__ void ComputeSurfaceNormal_D(	REAL3* coarsePos, uint* coarseGridIdx, uint* coarseCellStart, uint* coarseCellEnd, uint coarseRes,
										REAL3* finePos, REAL* fineKernelDens, REAL3* surfaceNormal, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint fineRes, uint numFineParticles,
										REAL outerRadius, REAL innerRadius)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 1.0 / coarseRes;
	REAL3 fPos = finePos[idx];
	REAL3 gradient = MetaballConstraintGradient(fPos, coarsePos, coarseGridIdx, coarseCellStart, coarseCellEnd, outerRadius, innerRadius, coarseRes, 1);

	REAL3 n = gradient;
	REAL3 vx = make_REAL3(1, 0, 0);
	REAL3 vy = make_REAL3(0, 1, 0);
	REAL dotX = Dot(n, vx);
	REAL dotY = Dot(n, vy);
	REAL3 t1 = fabs(dotX) < fabs(dotY) ? Cross(n, vx) : Cross(n, vy);
	REAL3 t2 = Cross(n, t1);
	Normalize(t1);
	Normalize(t2);

	//// least-square plane fitting with tangent frame
	REAL sw = 0, swx = 0, swy = 0, swxy = 0, swx2 = 0, swy2 = 0, swxz = 0, swyz = 0, swz = 0;

	REAL3 pos1 = finePos[idx];
	int3 gridPos = calcGridPos(pos1, 1.0 / fineRes);
	int width = (r * fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				REAL3 pos2 = finePos[sortedIdx];

				REAL x = Dot((pos2 - pos1), t1);
				REAL y = Dot((pos2 - pos1), t2);
				REAL z = Dot((pos2 - pos1), n);
				REAL w = NeighborFineWeight(r, finePos, fineKernelDens, idx, sortedIdx, fineGridIdx, fineCellStart, fineCellEnd, fineRes);
				swx2 += w * x * x;
				swy2 += w * y * y;
				swxy += w * x * y;
				swxz += w * x * z;
				swyz += w * y * z;
				swx += w * x;
				swy += w * y;
				swz += w * z;
				sw += w;
			}
		}
	}END_FOR;
	REAL det = -sw * swxy * swxy + 2.0 * swx * swxy * swy - swx2 * swy * swy - swx * swx * swy2 + sw * swx2 * swy2;
	
	if (det == 0.0) surfaceNormal[idx] = make_REAL3(0, 0, 0);
	else {
		REAL3 abc = make_REAL3(
			swxz * (-swy * swy + sw * swy2) + swyz * (-sw * swxy + swx * swy) + swz * (swxy * swy - swx * swy2),
			swxz * (-sw * swxy + swx * swy) + swyz * (-swx * swx + sw * swx2) + swz * (swx * swxy - swx2 * swy),
			swxz * (swxy * swy - swx * swy2) + swyz * (swx * swxy - swx2 * swy) + swz * (-swxy * swxy + swx2 * swy2)
		) * (1.0 / det);

		REAL3 normal = (t1 * abc.x + t2 * abc.y - n);
		Normalize(normal);
		normal *= -1;

		if (Dot(normal, gradient) < 0.0) { normal = normal * -1; }
		surfaceNormal[idx] = normal;
	}
}

__global__ void SmoothNormal_D(REAL3* finePos, REAL* fineKernelDens,REAL3* surfaceTempNormal, REAL3* surfaceNormal, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint fineRes, uint numFineParticles, uint coarseRes)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 1.0 / coarseRes;
	REAL3 fPos = finePos[idx];

	REAL3 newNormal = make_REAL3(0,0,0);
	int3 gridPos = calcGridPos(fPos, 1.0 / fineRes);
	int width = (r * fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				newNormal += surfaceTempNormal[sortedIdx] * NeighborFineWeight(r, finePos, fineKernelDens, idx, sortedIdx, fineGridIdx, fineCellStart, fineCellEnd, fineRes);
			}
		}
	}END_FOR;
	Normalize(newNormal);
	surfaceNormal[idx] = newNormal;
}
#endif