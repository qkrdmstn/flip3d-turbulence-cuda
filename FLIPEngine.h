#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <fstream>
#include <string>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_gl.h>
#include <helper_cuda.h> 
#include "Shader.h"
#include "FLIP3D_Cuda.h"
#include "SurfaceTurbulence.h"
#include "MarchingCubesCuda.h"
#include "FluidRenderer.h"

class FLIPEngine
{
public: //Simulation
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
	GLuint			posVbo;
	GLuint			colorVbo;
	struct cudaGraphicsResource* vboPosResource;
	struct cudaGraphicsResource* vboColorResource;

public: //Shader
	Shader*			_particleShader;

public: FluidRenderer* _renderer;

public:
	FLIPEngine()
	{
		init();
	}
	~FLIPEngine() 
	{
		cudaGraphicsUnregisterResource(vboPosResource);
		glDeleteBuffers(1, &posVbo);

		cudaGraphicsUnregisterResource(vboColorResource);
		glDeleteBuffers(1, &colorVbo);
	}
public:
	void	init(void);
	void	InitRenderer(void);
	void	InitSimulation(REAL3& gravity, REAL dt);
public:
	void	simulation(bool advection, bool flag);
	void	reset(void);
public:
	void	draw(bool flag1, bool flag2, bool flag3, bool flag4);
	void	drawBoundary();
	void	ExportObj(const char* filePath);

	float lerp(float a, float b, float t)
	{
		return a + t * (b - a);
	}

	// create a color ramp
	void colorRamp(float t, float* r)
	{
		const int ncolors = 7;
		float c[ncolors][3] =
		{
			{ 1.0, 0.0, 0.0, },
			{ 1.0, 0.5, 0.0, },
			{ 1.0, 1.0, 0.0, },
			{ 0.0, 1.0, 0.0, },
			{ 0.0, 1.0, 1.0, },
			{ 0.0, 0.0, 1.0, },
			{ 1.0, 0.0, 1.0, },
		};
		t = t * (ncolors - 1);
		int i = (int)t;
		float u = t - floorf(t);
		//r[0] = lerp(c[i][0], c[i + 1][0], u);
		//r[1] = lerp(c[i][1], c[i + 1][1], u);
		//r[2] = lerp(c[i][2], c[i + 1][2], u);

		r[0] = 1;
		r[1] = 1;
		r[2] = 1;
	}
};

#endif