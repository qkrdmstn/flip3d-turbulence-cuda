#ifndef __SURFACETURBULENCE_CUH__
#define __SURFACETURBULENCE_CUH__

#include "SurfaceTurbulence.h"
#include "Hash.cuh"

#include <cmath>
#include<stdio.h>

__global__ void Initialize_D(REAL3* coarsePos, uint* coarseType, uint numCoarseParticles, uint* numFineParticles, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint baseRes, REAL fineScaleLen, REAL outerRadius, REAL innerRadius)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numCoarseParticles)
		return;
	if (coarseType[idx] != FLUID)
		return;
	numFineParticles[0] = 0;

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
				
				if (GetNumFluidParticleAt(indexX, indexY, indexZ, coarsePos, coarseType, gridHash, gridIdx, cellStart, cellEnd, baseRes) == 0){
					nearSurface = true;
					break;
				}
			}
		}
	}

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
							if (idx != sortedIdx && LengthSquared(position - neighborPos) < outerRadius2) {
								valid = false;
								break;
							}
						}
					}
				}END_FOR;
				
				if (valid) {
					atomicAdd(&numFineParticles[0], 1);

					//파티클마다 영역을 두기. 첫 번째 파티클은 0-10번지, 두번째는 11-20.. 이케 해서 REAL4 로 w인자를 flag로 사용, w로 sorting을 하면 사용하는 위치값만 앞으로 나옴
				}
			}
		}
	}
	printf("numFineParticles: %d\n", numFineParticles[0]);
}

#endif