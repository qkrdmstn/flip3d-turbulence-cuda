#include "FluidRenderer.h"

FluidRenderer::FluidRenderer(void)
{

}

FluidRenderer::~FluidRenderer(void)
{
	cudaGraphicsUnregisterResource(_vboPosResource);
	glDeleteBuffers(1, &_posVbo);

	//cudaGraphicsUnregisterResource(_vboColorResource);
	//glDeleteBuffers(1, &_colorVbo);
}

void FluidRenderer::InitializeFluidRenderer(int numParticles)
{
	_numParticles = numParticles;
	InitShader();
	InitVBO();
	InitFBO();
}

void FluidRenderer::InitShader(void)
{
	_depthShader = new Shader("Shader\\particleSphere", "Shader\\particleSphere");
	_normalShader = new Shader("Shader\\particleSphere", "Shader\\normal");

}

void FluidRenderer::InitVBO(void)
{
	// allocate GPU data
	unsigned int memSize = 4 * sizeof(float) * _numParticles;

	// Pos VBO 持失
	glGenBuffers(1, &_posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, _posVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&_vboPosResource, _posVbo, cudaGraphicsMapFlagsNone);

//	// Color VBO 持失
//	glGenBuffers(1, &_colorVbo);
//	glBindBuffer(GL_ARRAY_BUFFER, _colorVbo);
//	glBufferData(GL_ARRAY_BUFFER, memSize, 0, GL_DYNAMIC_DRAW);
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//	cudaGraphicsGLRegisterBuffer(&_vboColorResource, _colorVbo, cudaGraphicsMapFlagsNone);
//
//	// fill color buffer
//	glBindBuffer(GL_ARRAY_BUFFER, _colorVbo);
//	float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
//	float* ptr = data;
//
//	for (unsigned int i = 0; i < _numParticles; i++)
//	{
//		float t = i / (float)_numParticles;
//#if 0
//		* ptr++ = rand() / (float)RAND_MAX;
//		*ptr++ = rand() / (float)RAND_MAX;
//		*ptr++ = rand() / (float)RAND_MAX;
//#else
//		colorRamp(t, ptr);
//		ptr += 3;
//#endif
//		* ptr++ = 1.0f;
//	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
}

void FluidRenderer::InitFBO(void)
{
	_depthFBO.creatFBO();
	_depthFBO.addBufferToFBO(_depthTex, 0);
	_depthFBO.bindFBO();
	GLenum DepthFBODrawBuffer[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, DepthFBODrawBuffer);
	_depthFBO.banFBO();
}

float3* FluidRenderer::CudaGraphicResourceMapping( )
{
	float3* dptr;
	size_t numBytes;
	cudaGraphicsMapResources(1, &_vboPosResource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &numBytes, _vboPosResource);

	return dptr;
}

void FluidRenderer::CudaGraphicResourceUnMapping(void)
{
	cudaGraphicsUnmapResources(1, &_vboPosResource, 0);
}

void FluidRenderer::Rendering(void)
{
	GenerateDepth();
	CalcNormal();
}

void FluidRenderer::GenerateDepth(void)
{
	_depthFBO.bindFBO();
	_depthShader->enable();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	GLuint PointRadiusLocation = glGetUniformLocation(_depthShader->_program, "pointRadius");
	GLuint PointScaleLocation = glGetUniformLocation(_depthShader->_program, "pointScale");
	glUniform1f(PointScaleLocation, 800 / tanf(60.0 * 0.5f * 3.141592 / 180.0f));
	glUniform1f(PointRadiusLocation, 0.125f * 0.15f);

	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	//VBO Rendering
	glBindBuffer(GL_ARRAY_BUFFER, _posVbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_POINTS, 0, _numParticles);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_DEPTH_TEST);

	_depthShader->ban();
	_depthFBO.banFBO();
}

void FluidRenderer::CalcNormal(void)
{
	_normalShader->enable();
	glEnable(GL_DEPTH_TEST);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _depthTex);

	GLuint PointRadiusLocation = glGetUniformLocation(_normalShader->_program, "pointRadius");
	GLuint PointScaleLocation = glGetUniformLocation(_normalShader->_program, "pointScale");
	GLuint depthSamplerLocation = glGetUniformLocation(_normalShader->_program, "depthTex");
	GLuint maxDepthLocation = glGetUniformLocation(_normalShader->_program, "maxDepth");
	GLuint screenSizeLocation = glGetUniformLocation(_normalShader->_program, "screenSize");
	glUniform1f(PointScaleLocation, 800 / tanf(60.0 * 0.5f * 3.141592 / 180.0f));
	glUniform1f(PointRadiusLocation, 0.125f * 0.15f);
	glUniform1i(depthSamplerLocation, 0);
	glUniform1f(maxDepthLocation, 1.0);
	glUniform2f(screenSizeLocation, 800, 800);

	//VBO Rendering
	glBindBuffer(GL_ARRAY_BUFFER, _posVbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_POINTS, 0, _numParticles);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindTexture(GL_TEXTURE_2D, 0);


	glDisable(GL_DEPTH_TEST);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	_normalShader->ban();
}