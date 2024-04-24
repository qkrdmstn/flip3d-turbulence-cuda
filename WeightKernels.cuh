#ifndef __WEIGHTKERNELS__CUH__
#define __WEIGHTKERNELS__CUH__

__device__ REAL SmoothKernel(REAL r2, REAL h)
{
	return max(1.0 - r2 / (h * h), 0.0);
}

__device__ REAL SharpKernel(REAL r2, REAL h)
{
	return max(h * h / max(r2, 0.00001f) - 1.0, 0.0);
}

__device__ REAL trilinearUnitKernel(float r) {
    r = abs(r);
    if (r > 1) return 0;
    return 1 - r;
}

__device__ REAL trilinearHatKernel(float3 r, float support) {
    return trilinearUnitKernel(r.x / support) * trilinearUnitKernel(r.y / support) * trilinearUnitKernel(r.z / support);
}

#endif