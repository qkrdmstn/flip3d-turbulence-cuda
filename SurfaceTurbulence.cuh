#ifndef __SURFACETURBULENCE_CUH__
#define __SURFACETURBULENCE_CUH__

#include "SurfaceTurbulence.h"
#include "Hash.cuh"
#include "WeightKernels.cuh"

#include <cmath>
#include<stdio.h>

__global__ void Initialize_D(REAL3* coarsePos, uint* coarseType, REAL3* finePos, uint* particleGridIdx, uint numCoarseParticles, uint* gridIdx, uint* cellStart, uint* cellEnd, MaintenanceParam maintenanceParam)
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

	uint discretization = (uint)PI * (maintenanceParam._outerRadius + maintenanceParam._innerRadius) / maintenanceParam._fineScaleLen;
	REAL dtheta = 2.0 * maintenanceParam._fineScaleLen / (maintenanceParam._outerRadius + maintenanceParam._innerRadius);
	REAL outerRadius2 = maintenanceParam._outerRadius * maintenanceParam._outerRadius;

	//uint discretization = (uint)PI * (maintenanceParam._coarseScaleLen + (maintenanceParam._coarseScaleLen / 2.0)) / maintenanceParam._fineScaleLen;
	//REAL dtheta = 2.0 * maintenanceParam._fineScaleLen / (maintenanceParam._coarseScaleLen + (maintenanceParam._coarseScaleLen / 2.0));
	//REAL outerRadius2 = maintenanceParam._coarseScaleLen * maintenanceParam._coarseScaleLen;

	BOOL nearSurface = false;
	REAL3 pos = coarsePos[idx];
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				
				REAL x = max(0.0f, min((REAL)maintenanceParam._coarseRes, maintenanceParam._coarseRes * pos.x));
				REAL y = max(0.0f, min((REAL)maintenanceParam._coarseRes, maintenanceParam._coarseRes * pos.y));
				REAL z = max(0.0f, min((REAL)maintenanceParam._coarseRes, maintenanceParam._coarseRes * pos.z));

				uint indexX = x + i;
				uint indexY = y + j;
				uint indexZ = z + k;
				if (indexX < 0 || indexY < 0 || indexZ < 0) continue;
				
				if (GetNumParticleAt(indexX, indexY, indexZ, coarsePos, coarseType, gridIdx, cellStart, cellEnd, maintenanceParam._coarseRes) == 0){
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
				REAL3 position = pos + normal * maintenanceParam._outerRadius;

				bool valid = true;
				int3 gridPos = calcGridPos(pos, 1.0 / maintenanceParam._coarseRes);
				REAL r = 2 * maintenanceParam._outerRadius;
				int width = (r * maintenanceParam._coarseRes) + 1;
				FOR_NEIGHBOR(width) {
					int3 neighGridPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);

					uint neighHash = calcGridHash(neighGridPos, maintenanceParam._coarseRes);
					uint startIdx = cellStart[neighHash];
					if (startIdx != 0xffffffff) {
						uint endIdx = cellEnd[neighHash];
						for (uint i = startIdx; i < endIdx; i++)
						{
							uint sortedIdx = gridIdx[i];
							REAL3 neighborPos = coarsePos[sortedIdx];

							REAL cellSize = 1.0 / maintenanceParam._coarseRes;
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

__global__ void ComputeCoarseDens_D(REAL r, REAL3* coarsePos, uint* coarseType, REAL* coarseKernelDens, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numCoarseParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numCoarseParticles)
		return;
	if (coarseType[idx] != FLUID)
		return;

	REAL cellSize = 1.0 / maintenanceParam._coarseRes;
	REAL3 pos1 = coarsePos[idx];
	int3 gridPos = calcGridPos(pos1, cellSize);

	REAL kernelDens = 0.0f;
	int width = (r / cellSize) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._coarseRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (coarseType[sortedIdx] != FLUID)
					continue;
				
				REAL3 pos2 = coarsePos[sortedIdx];
				kernelDens += DistKernel(pos1 - pos2, r);
			}
		}
	}END_FOR;
	coarseKernelDens[idx] = kernelDens;
}

__global__ void ComputeFineDens_D(REAL r, REAL3* finePos, REAL* fineKernelDens, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numFineParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL cellSize = 1.0 / maintenanceParam._fineRes;
	REAL3 pos1 = finePos[idx];
	int3 gridPos = calcGridPos(pos1, cellSize);

	REAL kernelDens = 0.0f;
	int width = (r / cellSize) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
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


//Advection
__device__ REAL weightKernelAdvection(REAL3 distance, MaintenanceParam param) {
	if (Length(distance) > 2.f * param._coarseScaleLen) {
		return 0;
	}
	else {
		return DistKernel(distance, 2.f * param._coarseScaleLen);
	}
}

__device__ REAL NeighborCoarseWeight(REAL r, REAL3* finePos, REAL3* coarsePos, uint* coarseType, REAL* coarseKernelDens, uint fineIdx, uint coarseIdx, uint* gridIdx, uint* cellStart, uint* cellEnd, MaintenanceParam maintenanceParam)
{
	REAL3 fPos = finePos[fineIdx];
	REAL3 cPos = coarsePos[coarseIdx];
	REAL weight = DistKernel((fPos - cPos), r) / coarseKernelDens[coarseIdx];
	if (weight == 0) return 0;

	REAL cellSize = 1.0 / maintenanceParam._coarseRes;
	REAL weightSum = 0.0;
	int3 gridPos = calcGridPos(fPos, cellSize);

	int width = (r / cellSize) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._coarseRes);
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

__global__ void Advection_D(REAL3* finePos, REAL3* coarseCurPos, REAL3* coarseBeforePos, uint* coarseType, REAL* coarseKernelDens, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numFineParticles, BOOL* flag, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 2.0 * maintenanceParam._coarseScaleLen;

	REAL cellSize = 1.0 / maintenanceParam._coarseRes;
	REAL3 fPos = finePos[idx];
	int3 gridPos = calcGridPos(fPos, cellSize);

	REAL3 displacement = make_REAL3(0, 0, 0);
	REAL totalW = 0.0f;

	int width = (r / cellSize) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._coarseRes);
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
				if (d2 > r * r)
					continue;

				REAL3 beforePos = coarseBeforePos[sortedIdx];
				REAL w = weightKernelAdvection((beforePos - fPos), maintenanceParam);
				displacement += w * (curPos - beforePos);
				totalW += w;

				//displacement += (curPos - beforePos) * NeighborCoarseWeight(r, finePos, coarseCurPos, coarseType, coarseKernelDens, idx, sortedIdx, gridIdx, cellStart, cellEnd, maintenanceParam);
			}
		}
	}END_FOR;
	if (totalW != 0) displacement /= totalW;
	finePos[idx] += displacement;
	flag[idx] = false;
}


//Surface Constraints
__device__ REAL MetaballDens(REAL dist, REAL outerRadius)
{
	return exp(-2 * pow((dist / outerRadius), 2));
}

__device__ REAL MetaballLevelSet(REAL3 finePos, REAL3* coarsePos, uint* coarseType, uint* gridIdx, uint* cellStart, uint* cellEnd, MaintenanceParam maintenanceParam)
{
	REAL cellSize = 1.0 / maintenanceParam._coarseRes;

	REAL R = maintenanceParam._outerRadius;
	REAL r = maintenanceParam._innerRadius;
	REAL u = (3.0 / 2.0) * R;
	REAL a = log(2.0 / (1.0 + MetaballDens(u, maintenanceParam._outerRadius))) / (pow(u / 2.0, 2.0) - pow(r, 2.0));

	REAL3 fPos = finePos;
	int3 gridPos = calcGridPos(fPos, cellSize);
	
	REAL f = 0.0;

	REAL radius = 2 * maintenanceParam._outerRadius;
	int width = (radius / cellSize) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._coarseRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				REAL3 cPos = coarsePos[sortedIdx];

				REAL d2 = LengthSquared(fPos - cPos);
				if (d2 > radius * radius)
					continue;
				if (coarseType[sortedIdx] != FLUID)
					continue;
				f += exp(-a * d2);
			}
		}
	}END_FOR;
	if (f > 1.0f) f = 1.0f;

	f = (sqrt(-log(f) / a) - r) / (R - r);
	if (f > 10)
	{
		f = 10.0f;
	}
	return f;
}

__device__ REAL3 MetaballConstraintGradient(REAL3 finePos, REAL3* coarsePos, uint* coarseType, uint* gridIdx, uint* cellStart, uint* cellEnd, int width, MaintenanceParam maintenanceParam)
{
	REAL cellSize = 1.0 / maintenanceParam._coarseRes;

	REAL R = maintenanceParam._outerRadius;
	REAL r = maintenanceParam._innerRadius;
	REAL u = (3.0 / 2.0) * R;
	REAL a = log(2.0 / (1.0 + MetaballDens(u, maintenanceParam._outerRadius))) / (pow(u / 2.0, 2.0) - pow(r, 2.0));

	REAL3 fPos = finePos;
	int3 gridPos = calcGridPos(fPos, cellSize);

	REAL3 gradient = make_REAL3(0, 0, 0);
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._coarseRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				REAL3 cPos = coarsePos[sortedIdx];

				REAL d2 = LengthSquared(fPos - cPos);
				if (d2 > cellSize * cellSize * width * width)
					continue;
				if (coarseType[sortedIdx] != FLUID)
					continue;
				gradient += (fPos - cPos) * 2.0 * a * exp(-a * d2);
			}
		}
	}END_FOR;

	Normalize(gradient);
	return gradient;
}

__global__ void SurfaceConstraint_D(REAL3* finePos, REAL3* coarsePos, uint* coarseType, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numFineParticles, REAL3* normal, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL R = maintenanceParam._outerRadius;
	REAL r = maintenanceParam._innerRadius;

	REAL3 fPos = finePos[idx];
	REAL levelSet = MetaballLevelSet(fPos, coarsePos, coarseType, gridIdx, cellStart, cellEnd, maintenanceParam);
	if (levelSet <= 1.0 && levelSet >= 0.0) return;

	REAL radius = 2 * maintenanceParam._coarseScaleLen;
	int width = (radius * maintenanceParam._coarseRes) + 1;
	REAL3 gradient = MetaballConstraintGradient(fPos, coarsePos, coarseType, gridIdx, cellStart, cellEnd, width, maintenanceParam); // Constraints Projection
	if (levelSet < 0.0) {
		fPos -= gradient * (R - r) * levelSet;
	}
	else if (levelSet > 1.0) {
		fPos -= gradient * (R - r) * (levelSet - 1);
	}

	finePos[idx] = fPos;
}


//Normal Compute
__device__ REAL NeighborFineWeight(REAL r, REAL3* finePos, REAL* fineKernelDens, REAL* fineWeightSum, uint fineIdx1, uint fineIdx2)
{
	REAL3 fPos1 = finePos[fineIdx1];
	REAL3 fPos2 = finePos[fineIdx2];
	REAL weight = DistKernel((fPos1 - fPos2), r) / fineKernelDens[fineIdx2];
	if (weight == 0) return 0;

	REAL ws = fineWeightSum[fineIdx1];

	if (ws > 0)
		return weight / ws;
	else
		return 0.0f;
}

__global__ void ComputeFineNeighborWeightSum_D(REAL r, REAL3* finePos, REAL* fineKernelDens, REAL* fineWeightSum, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numFineParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL3 fPos1 = finePos[idx];
	REAL cellSize = 1.0 / maintenanceParam._fineRes;
	REAL weightSum = 0.0;
	int3 gridPos = calcGridPos(fPos1, cellSize);

	REAL width = (r / cellSize) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
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
	fineWeightSum[idx] = weightSum;
}

__global__ void ComputeSurfaceNormal_D(	REAL3* coarsePos, uint* coarseType, uint* coarseGridIdx, uint* coarseCellStart, uint* coarseCellEnd,
										REAL3* finePos, REAL* fineKernelDens, REAL* fineWeightSum, REAL3* surfaceNormal, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles,
										MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = maintenanceParam._coarseScaleLen;
	REAL3 fPos = finePos[idx];
	REAL3 gradient = MetaballConstraintGradient(fPos, coarsePos, coarseType, coarseGridIdx, coarseCellStart, coarseCellEnd, 2, maintenanceParam);

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
	int3 gridPos = calcGridPos(pos1, 1.0 / maintenanceParam._fineRes);
	int width = (r * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
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
				REAL w = NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
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
	

	if ((det <= DBL_EPSILON && det >= -DBL_EPSILON)) {
		surfaceNormal[idx] = make_REAL3(0, 0, 0);
	}
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

__global__ void SmoothNormal_D(REAL3* finePos, REAL* fineKernelDens, REAL* fineWeightSum, REAL3* surfaceTempNormal, REAL3* surfaceNormal, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles,  MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = maintenanceParam._coarseScaleLen;
	REAL3 fPos = finePos[idx];

	REAL3 newNormal = make_REAL3(0,0,0);
	int3 gridPos = calcGridPos(fPos, 1.0 / maintenanceParam._fineRes);
	int width = (r * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				newNormal += surfaceTempNormal[sortedIdx] * NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
			}
		}
	}END_FOR;
	Normalize(newNormal);
	surfaceNormal[idx] = newNormal;
}

//Normal Regularization
__global__ void CopyToTempPos_D(REAL3* finePos, REAL3* fineTempPos, uint numFineParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	fineTempPos[idx] = finePos[idx];
}

__global__ void CopyToPos_D(REAL3* finePos, REAL3* fineTempPos, uint numFineParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	finePos[idx] = fineTempPos[idx];
}

__global__ void NormalRegularization_D(REAL3* finePos, REAL3* fineTempPos, REAL3* surfaceNormal, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles,MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = maintenanceParam._coarseScaleLen;

	REAL3 pos = finePos[idx];
	REAL3 normal = surfaceNormal[idx];
	REAL3 displacement = make_REAL3(0, 0, 0);

	int3 gridPos = calcGridPos(pos, 1.0 / maintenanceParam._fineRes);
	int width = (r * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				REAL3 pos2 = finePos[sortedIdx];

				REAL3 dir = pos - pos2;
				REAL3 dn = normal * Dot(dir, normal);

				REAL3 crossVec = Cross(normal, -1 * dir);
				Normalize(crossVec);

				REAL3 projectedNormal = surfaceNormal[sortedIdx] - crossVec * Dot(crossVec, surfaceNormal[sortedIdx]);
				Normalize(projectedNormal);

				if (Dot(projectedNormal, normal) < 0.0 || abs(Dot(normal, (normal + projectedNormal)) < 1e-6)) continue;
				dn = -1 * normal * (Dot((normal + projectedNormal), dir) / (2 * Dot(normal, (normal + projectedNormal))));

				REAL w = NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
				displacement += dn * w;
			}
		}
	}END_FOR;

	fineTempPos[idx] += displacement;
}

__global__ void TangentRegularization_D(REAL3* finePos, REAL3* fineTempPos, REAL3* surfaceNormal, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 3 * maintenanceParam._fineScaleLen;

	REAL3 pos = finePos[idx];
	REAL3 normal = surfaceNormal[idx];
	Normalize(normal);

	REAL3 displacement = make_REAL3(0, 0, 0);

	int3 gridPos = calcGridPos(pos, 1.0 / maintenanceParam._fineRes);
	int width = (r * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				REAL3 pos2 = finePos[sortedIdx];

				REAL3 dir = pos - pos2;
				REAL3 dn = normal * Dot(dir, normal);
				REAL3 dt = dir - dn; //dir의 tangent 방향 성분
				Normalize(dt);

				REAL w = NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
				displacement += dt * w;
			}
		}
	}END_FOR;
	displacement *= 0.5 * maintenanceParam._fineScaleLen;
	fineTempPos[idx] += displacement;


}


//FineParticle Insert/Delete

__device__ BOOL IsInDomain(REAL3 pos)
{
	REAL x = pos.x;
	REAL y = pos.y;
	REAL z = pos.z;

	if (0.0 <= x && x <= 1.0 &&
		0.0 <= y && y <= 1.0 &&
		0.0 <= z && z <= 1.0)
		return true;
	else
		return false;
}

__global__ void InsertFineParticles_D(uint* secondParticleGridIdx, REAL3* finePos, REAL3* surfaceNormal, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles, uint numCoarseParticles,
	REAL* waveSeedAmplitude, REAL* waveH, REAL* waveDtH, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	int insertIdx = numFineParticles + idx;
	finePos[insertIdx] = make_REAL3(-1, -1, -1);
	secondParticleGridIdx[insertIdx] = numCoarseParticles * PER_PARTICLE;

	//tan
	REAL tangentRadius = 3.0 * maintenanceParam._fineScaleLen;
	REAL insertRadius = 1.0 * maintenanceParam._fineScaleLen;

	REAL3 pos = finePos[idx];
	REAL3 normal = surfaceNormal[idx];
	Normalize(normal);

	REAL3 displacementTangent = make_REAL3(0, 0, 0);

	int3 gridPos1 = calcGridPos(pos, 1.0 / maintenanceParam._fineRes);
	int width1 = (tangentRadius * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width1) {
		int3 neighborPos = make_int3(gridPos1.x + dx, gridPos1.y + dy, gridPos1.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				REAL3 pos2 = finePos[sortedIdx];

				REAL3 dir = pos - pos2;
				REAL3 dn = normal * Dot(dir, normal);
				REAL3 dt = dir - dn; //dir의 tangent 방향 성분
				Normalize(dt);

				REAL w = NeighborFineWeight(tangentRadius, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
				displacementTangent += dt * w;
			}
		}
	}END_FOR;
	Normalize(displacementTangent);
	REAL3 center = pos + (displacementTangent * insertRadius);
	if (!IsInDomain(center)) return;

	int cnt = 0;

	int3 gridPos2 = calcGridPos(center, 1.0 / maintenanceParam._fineRes);
	int width2 = ((insertRadius) * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width2) {
		int3 neighborPos = make_int3(gridPos2.x + dx, gridPos2.y + dy, gridPos2.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				REAL3 pos2 = finePos[sortedIdx];
				if (Length(pos2 - center) < insertRadius - 1e-6)
					cnt++;
			}
		}
	}END_FOR;

	if (cnt == 0)
	{
		//Insert
		finePos[insertIdx] = center;
		secondParticleGridIdx[insertIdx] = insertIdx;

		//wave init
		waveSeedAmplitude[insertIdx] = 0.0f;
		waveH[insertIdx] = 0.0f;
		waveDtH[insertIdx] = 0.0f;
	}
}

__global__ void DeleteFineParticles_D(uint* secondParticleGridIdx, REAL3* finePos, REAL3* surfaceNormal, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles, uint numCoarseParticles, REAL* waveSeedAmplitude, REAL* waveH, REAL* waveDtH, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	secondParticleGridIdx[idx] = idx;

	REAL deleteRadius = (0.67) * maintenanceParam._fineScaleLen;
	REAL3 pos = finePos[idx];

	int3 gridPos1 = calcGridPos(pos, 1.0 / maintenanceParam._fineRes);
	int width1 = (deleteRadius * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width1) {
		int3 neighborPos = make_int3(gridPos1.x + dx, gridPos1.y + dy, gridPos1.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				if (sortedIdx == idx) continue;
				
				REAL3 pos2 = finePos[sortedIdx];
				if (LengthSquared(pos2 - pos) <= deleteRadius * deleteRadius || !IsInDomain(pos)) {
					//delete
					secondParticleGridIdx[sortedIdx] = numCoarseParticles * PER_PARTICLE;
					finePos[sortedIdx] = make_REAL3(-1,-1,-1);

					waveSeedAmplitude[sortedIdx] = 0.0f;
					waveH[sortedIdx] = 0.0f;
					waveDtH[sortedIdx] = 0.0f;
				}
			}
		}
	}END_FOR;
}

__global__ void AdvectionDeleteFineParticles_D(uint* secondParticleGridIdx, REAL3* finePos, REAL3* coarsePos, uint* coarseType, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numFineParticles, uint numCoarseParticles, REAL* waveSeedAmplitude, REAL* waveH, REAL* waveDtH, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 2.0 * maintenanceParam._coarseScaleLen;
	REAL cellSize = 1.0/maintenanceParam._coarseRes;

	REAL3 fPos = finePos[idx];
	int3 gridPos = calcGridPos(fPos, cellSize);
	int cnt = 0;
	FOR_NEIGHBOR(2) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._coarseRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (coarseType[sortedIdx] != FLUID)
					continue;
				REAL3 cPos = coarsePos[sortedIdx];
				if (LengthSquared(cPos - fPos) <= r * r) {
					cnt++;
					break;
				}
			}
		}
	}END_FOR;

	if (cnt == 0)
	{
		//delete
		secondParticleGridIdx[idx] = numCoarseParticles * PER_PARTICLE;
		finePos[idx] = make_REAL3(-1, -1, -1);


		waveSeedAmplitude[idx] = 0.0f;
		waveH[idx] = 0.0f;
		waveDtH[idx] = 0.0f;
	}
}

__global__ void ConstraintDeleteFineParticles_D(uint* secondParticleGridIdx, REAL3* finePos, REAL3* coarsePos, uint* coarseType, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numFineParticles, uint numCoarseParticles, REAL* waveSeedAmplitude, REAL* waveH, REAL* waveDtH, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL cellSize = 1.0 / maintenanceParam._coarseRes;
	REAL3 fPos = finePos[idx];

	REAL levelSet = MetaballLevelSet(fPos, coarsePos, coarseType, gridIdx, cellStart, cellEnd, maintenanceParam);
	if (levelSet < -0.2f || levelSet > 1.2)
	{
		secondParticleGridIdx[idx] = numCoarseParticles * PER_PARTICLE;
		finePos[idx] = make_REAL3(-1, -1, -1);

		waveSeedAmplitude[idx] = 0.0f;
		waveH[idx] = 0.0f;
		waveDtH[idx] = 0.0f;
	}
}


//Compute Curvature
__global__ void ComputeCurvature_D(REAL3* finePos, REAL* tempCurvature, REAL3* surfaceNormal, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = maintenanceParam._coarseScaleLen;

	REAL3 pos = finePos[idx];
	REAL3 normal = surfaceNormal[idx];
	REAL curvature = 0.0f;

	int3 gridPos = calcGridPos(pos, 1.0 / maintenanceParam._fineRes);
	int width = (r * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				REAL3 pos2 = finePos[sortedIdx];
				curvature += Dot(normal, (pos - pos2)) * NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
			}
		}
	}END_FOR;

	tempCurvature[idx] = fabs(curvature);
}

__global__ void SmoothCurvature_D(REAL3* finePos, REAL* tempCurvature, REAL* curvature, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = maintenanceParam._coarseScaleLen;

	REAL3 pos = finePos[idx];
	REAL newCurvature = 0.0f;

	int3 gridPos = calcGridPos(pos, 1.0 / maintenanceParam._fineRes);
	int width = (r * maintenanceParam._fineRes) + 1;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
		uint startIdx = fineCellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = fineCellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = fineGridIdx[i];
				newCurvature += tempCurvature[sortedIdx] * NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
			}
		}
	}END_FOR;
	curvature[idx] = newCurvature;
}


//Seed Wave
__device__ REAL SmoothStep(REAL left, REAL right, REAL val)
{
	REAL x = fmax(0.0f, fmin((val - left) / (right - left), 1.0f));
	return x * x * (3.0 - 2.0 * x);
}

__global__ void SeedWave_D(REAL* curvature, REAL* waveSeedAmplitude, REAL* seed, REAL* waveH, uint numFineParticles, uint step, WaveParam waveParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;
	
	REAL source = 2.0 * SmoothStep(waveParam._waveSeedingCurvatureThresholdCenter - waveParam._waveSeedingCurvatureThresholdRadius, waveParam._waveSeedingCurvatureThresholdCenter + waveParam._waveSeedingCurvatureThresholdRadius, curvature[idx]) - 1.0; //edge값 추후 수정 가능
	REAL freq = waveParam._waveSeedFreq;
	REAL theta = waveParam._dt * (REAL)step * waveParam._waveSpeed * freq;
	REAL cosTheta = cos(theta);
	REAL maxSeedAmplitude = waveParam._waveMaxSeedingAmplitude * waveParam._waveMaxAmplitude;

	waveSeedAmplitude[idx] = fmax(0.0f, fmin(waveSeedAmplitude[idx] + source * waveParam._waveSeedStepSizeRatioOfMax * maxSeedAmplitude, maxSeedAmplitude));
	seed[idx] = waveSeedAmplitude[idx] * cosTheta;

	// source values for display (not used after this point anyway)
	curvature[idx] = (source >= 0) ? 1 : 0;

	//seed 더하기
	waveH[idx] += seed[idx];
}

__global__ void ComputeWaveNormal_D(REAL3* finePos, REAL* waveH, REAL3* waveNormal, REAL3* surfaceNormal, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles, uint numCoarseParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 3.0 * maintenanceParam._fineScaleLen;
	REAL3 pos = finePos[idx];
	REAL3 n = surfaceNormal[idx];
	REAL3 vx = make_REAL3(1, 0, 0);
	REAL3 vy = make_REAL3(0, 1, 0);
	REAL dotX = Dot(n, vx);
	REAL dotY = Dot(n, vy);
	REAL3 t1 = fabs(dotX) < fabs(dotY) ? Cross(n, vx) : Cross(n, vy);
	REAL3 t2 = Cross(n, t1);
	Normalize(t1);
	Normalize(t2);

	REAL3 pos1 = finePos[idx];
	int3 gridPos = calcGridPos(pos1, 1.0 / maintenanceParam._fineRes);
	int width = (r * maintenanceParam._fineRes) + 1;
	REAL sw = 0, swx = 0, swy = 0, swxy = 0, swx2 = 0, swy2 = 0, swxz = 0, swyz = 0, swz = 0;
	FOR_NEIGHBOR(width) {
		int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
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
				REAL z = waveH[sortedIdx];
				REAL w = NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
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

	if (det <= DBL_EPSILON && det >= -DBL_EPSILON)	waveNormal[idx] = make_REAL3(0, 0, 0);
	else {
		REAL3 abc = make_REAL3(
			swxz * (-swy * swy + sw * swy2) + swyz * (-sw * swxy + swx * swy) + swz * (swxy * swy - swx * swy2),
			swxz * (-sw * swxy + swx * swy) + swyz * (-swx * swx + sw * swx2) + swz * (swx * swxy - swx2 * swy),
			swxz * (swxy * swy - swx * swy2) + swyz * (swx * swxy - swx2 * swy) + swz * (-swxy * swxy + swx2 * swy2)
		) * (1.0 / det);

		REAL3 _waveNormal = (vx * abc.x + vy * abc.y - make_REAL3(0, 0, 1));
		Normalize(_waveNormal);
		_waveNormal *= -1;

		waveNormal[idx] = _waveNormal;
	}
}

__global__ void ComputeLaplacian_D(REAL3* finePos, REAL* laplacian, REAL3* waveNormal, REAL* waveH, REAL3* surfaceNormal, REAL* fineKernelDens, REAL* fineWeightSum, uint* fineGridIdx, uint* fineCellStart, uint* fineCellEnd, uint numFineParticles, MaintenanceParam maintenanceParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;

	REAL r = 3.0 * maintenanceParam._fineScaleLen;
	REAL3 pos1 = finePos[idx];
	REAL3 normal = surfaceNormal[idx];

	REAL3 vx = make_REAL3(1, 0, 0);
	REAL3 vy = make_REAL3(0, 1, 0);
	REAL dotX = Dot(normal, vx);
	REAL dotY = Dot(normal, vy);
	REAL3 t1 = fabs(dotX) < fabs(dotY) ? Cross(normal, vx) : Cross(normal, vy);
	REAL3 t2 = Cross(normal, t1);
	Normalize(t1);
	Normalize(t2);

	REAL l = 0.0f;
	REAL ph = waveH[idx];
	REAL3 waveN = waveNormal[idx];

	if ((waveN.y <= DBL_EPSILON && waveN.y >= -DBL_EPSILON)) laplacian[idx] = 0.0f;
	else
	{
		int3 gridPos = calcGridPos(pos1, 1.0 / maintenanceParam._fineRes);
		int width = (r * maintenanceParam._fineRes) + 1;
		FOR_NEIGHBOR(width) {
			int3 neighborPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
			uint neighHash = calcGridHash(neighborPos, maintenanceParam._fineRes);
			uint startIdx = fineCellStart[neighHash];

			if (startIdx != 0xffffffff)
			{
				uint endIdx = fineCellEnd[neighHash];
				for (uint i = startIdx; i < endIdx; i++)
				{
					uint sortedIdx = fineGridIdx[i];
					REAL3 pos2 = finePos[sortedIdx];
					REAL nh = waveH[sortedIdx];

					REAL3 dir = pos2 - pos1;
					REAL lengthDir = Length(dir);
					if (lengthDir < 1e-5) continue;

					REAL3 tangentDir = dir - normal * (Dot(dir, normal));
					Normalize(tangentDir);
					tangentDir = tangentDir * lengthDir;

					REAL dirX = Dot(tangentDir, t1);
					REAL dirY = Dot(tangentDir, t2);
					REAL dz = nh - ph - (-waveN.x / waveN.z) * dirX - (-waveN.y / waveN.z) * dirY;
					REAL w = NeighborFineWeight(r, finePos, fineKernelDens, fineWeightSum, idx, sortedIdx);
					l += fmax(-100.0f, fmin(w * 4.0f * dz / (lengthDir * lengthDir), 100.0f));
				}
			}
		}END_FOR;
		laplacian[idx] = l;
	}

}

__global__ void EvolveWave_D(REAL* waveDtH, REAL* waveH, REAL* laplacian, REAL* seed, uint numFineParticles, WaveParam waveParam)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;
	
	waveDtH[idx] += waveParam._waveSpeed * waveParam._waveSpeed * waveParam._dt * laplacian[idx];
	waveDtH[idx] /= (1.0 + waveParam._dt * waveParam._waveDamping);

	waveH[idx] += waveParam._dt * waveDtH[idx];
	waveH[idx] /= (1.0 + waveParam._dt * waveParam._waveDamping);
	waveH[idx] -= seed[idx];

	//clamp
	waveDtH[idx] = fmax(-waveParam._waveMaxFreq * waveParam._waveMaxAmplitude, fmin(waveDtH[idx], waveParam._waveMaxFreq * waveParam._waveMaxAmplitude));
	waveH[idx] = fmax(-waveParam._waveMaxAmplitude, fmin(waveH[idx], waveParam._waveMaxAmplitude));
}

__global__ void SetDisplayParticles_D(REAL3* displayPos, REAL3* finePos, REAL3* surfaceNormal, REAL* waveH, uint numFineParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numFineParticles)
		return;
	displayPos[idx] = finePos[idx] + surfaceNormal[idx] * waveH[idx];
}
#endif