#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_gl.h>
#include <helper_cuda.h> 
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Shader.h"
#include "Camera.h"

class FluidRenderer
{
public:
	int _numParticles;
	int _width = 800;
	int _height = 800;
	float aspectRatio = _width / _height;
	glm::vec2 _screenSize = glm::vec2(_width, _height);
	glm::vec2 _blurDirX = glm::vec2(1.0f / _screenSize.x, 0.0f);
	glm::vec2 _blurDirY = glm::vec2(0.0f, 1.0f / _screenSize.y);
	glm::vec4 color = glm::vec4(0.275f, 0.75f, 0.85f, 0.8f);
	//glm::vec4 color = glm::vec4(0.5,0.5,1,1);
	float _sphereRadius = 0.125f * 0.075f;
	float _filterRadius = 15.0f;
	float _MaxFilterRadius = 20.0f;

public: //VBO
	GLuint _posVbo;
	GLuint _colorVbo;
	struct cudaGraphicsResource* _vboPosResource;
	struct cudaGraphicsResource* _vboColorResource;

public: //Shader
	Shader* _plane;
	Shader* _depthShader;
	BlurShader* _bilateralBlurShader;
	BlurShader* _narrowBlurShader;
	Shader* _thicknessShader;
	Shader* _fluidFinalShader;
	Shader* _finalShader;

public:
	Camera* _camera;

public:
	FluidRenderer(void);
	~FluidRenderer(void);

	void InitializeFluidRenderer(Camera *camera, int _numParticles);
	void InitShader(void);
	void InitVBO(void);
	void InitFBO(void);
public:
	float3* CudaGraphicResourceMapping(void);
	void CudaGraphicResourceUnMapping(void);

public:
	void Rendering(void);
	void InfintePlane(void);
	void GenerateDepth(void);
	void BilateralFilteringDepth(void);
	void NarrowFilteringDepth(void);
	void GenerateThickness(void);
	void FinalRendering(void);

public:
	void setInt(Shader* shader, const int& x, const GLchar* name);
	void setFloat(Shader* shader, const float& x, const GLchar* name);
	void setVec2(Shader* shader, const glm::vec2& v, const GLchar* name);
	void setVec3(Shader* shader, const glm::vec3& v, const GLchar* name);
	void setVec4(Shader* shader, const glm::vec4& v, const GLchar* name);
	void setMatrix(Shader* shader, const glm::mat4& m, const GLchar* name);

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