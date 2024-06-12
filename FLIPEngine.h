#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include "FLIP3D_Cuda.h"
#include "SurfaceTurbulence.h"
#include "MarchingCubesCuda.h"

class FLIPEngine
{
public:
	FLIP3D_Cuda* _fluid;
	SurfaceTurbulence* _turbulence;
	MarchingCubes_CUDA* _MC;

public:
	AABB			_boundary;
public:
	REAL3			_gravity;
	REAL			_dt;
	uint			_frame;
public:
	FLIPEngine() {}
	FLIPEngine(REAL3& gravity, REAL dt)
	{
		init(gravity, dt);
	}
	~FLIPEngine() {}
public:
	void	init(REAL3& gravity, REAL dt);
public:
	void	simulation(bool advection, bool flag);
	void	reset(void);
public:
	void	draw(bool flag1, bool flag2, bool flag3, bool flag4);
	void	drawBoundary();
};

#endif