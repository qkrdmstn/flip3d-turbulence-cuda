#ifndef __WEIGHTKERNELS__CUH__
#define __WEIGHTKERNELS__CUH__

#include <cuda_runtime.h>
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"

__device__ static REAL SmoothKernel(REAL r2, REAL h)
{
	return max(1.0 - r2 / (h * h), 0.0);
}

__device__ static REAL SharpKernel(REAL r2, REAL h)
{
	return max(h * h / max(r2, 0.00001f) - 1.0, 0.0);
}

__device__ static REAL trilinearUnitKernel(REAL r) {
    r = abs(r);
    if (r > 1) return 0;
    return 1 - r;
}

__device__ static REAL trilinearHatKernel(REAL3 r, REAL support) {
    return trilinearUnitKernel(r.x / support) * trilinearUnitKernel(r.y / support) * trilinearUnitKernel(r.z / support);
}

__device__ static REAL DistKernel(REAL3 diff, REAL r) //거리에 반비례하는 커널
{
	REAL dist = Length(diff);
	if (dist > r)
		return 0.0;
	else
		return 1.0 - (dist / r);
}
#endif