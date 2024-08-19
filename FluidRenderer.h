#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_gl.h>
#include <helper_cuda.h> 
#include "Shader.h"
#include "FBO.h"

class FluidRenderer
{
public:
	int _numParticles;
	float _sphereRadius;

public: //VBO
	GLuint _posVbo;
	GLuint _colorVbo;
	struct cudaGraphicsResource* _vboPosResource;
	struct cudaGraphicsResource* _vboColorResource;

public: //FBO
	FBO _depthFBO;
	FBO _normalFBO;

public: // Texture
	GLuint _depthTex;
	GLuint _normalTex;

public: //Shader
	Shader* _depthShader;
	Shader* _normalShader;

public:
	FluidRenderer(void);
	~FluidRenderer(void);

	void InitializeFluidRenderer(int _numParticles);
	void InitShader(void);
	void InitVBO(void);
	void InitFBO(void);
public:
	float3* CudaGraphicResourceMapping(void);
	void CudaGraphicResourceUnMapping(void);
	
public:
	void Rendering(void);
	void GenerateDepth(void);
	void CalcNormal(void);

public:
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