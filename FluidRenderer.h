#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <helper_gl.h>
#include <helper_cuda.h> 
#include "Shader.h"

class FluidRenderer
{
public: //VBO
	GLuint			posVbo;
	GLuint			colorVbo;
	struct cudaGraphicsResource* vboPosResource;
	struct cudaGraphicsResource* vboColorResource;

public: //Shader
	Shader* _particleShader;

public:
	FluidRenderer(void);
	~FluidRenderer(void);

	void InitShader(void);
	void InitVBO(void);

public:
	void Rendering(void);


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