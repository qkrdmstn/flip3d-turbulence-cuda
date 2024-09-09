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

void FluidRenderer::InitializeFluidRenderer(Camera *camera, int numParticles)
{
	_numParticles = numParticles;
	_camera = camera;

	InitShader();
	InitVBO();
	InitFBO();
}

void FluidRenderer::InitShader(void)
{
	_plane = new Shader("Shader\\plane.vs", "Shader\\plane.fs");
	_depthShader = new Shader("Shader\\depth.vs", "Shader\\depth.fs");
	_blurShader = new BlurShader("Shader\\bilateralBlur.vs", "Shader\\bilateralBlur.fs");
	_thicknessShader = new Shader("Shader\\depth.vs", "Shader\\thickness.fs");
	_fluidFinalShader = new Shader("Shader\\fluidFinal.vs", "Shader\\fluidFinal.fs");
	//_finalShader = new Shader("Shader\\final.vs", "Shader\\final.fs");
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
	//Infinite Plane buffer
	_plane->initFBO(_plane->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, _plane->fbo);
	_plane->initTexture(_width, _height, GL_RGBA, GL_RGBA32F, _plane->tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _plane->tex, 0);

	//Depth buffer
	_depthShader->initFBO(_depthShader->fbo);
	//glBindFramebuffer(GL_FRAMEBUFFER, _depthShader->fbo);
	_depthShader->initTexture(_width, _height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, _depthShader->tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depthShader->tex, 0);

	//Blur buffer
	_blurShader->initFBO(_blurShader->fboV);
	glBindFramebuffer(GL_FRAMEBUFFER, _blurShader->fboV);
	_blurShader->initTexture(_width, _height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, _blurShader->texV);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _blurShader->texV, 0);

	_blurShader->initFBO(_blurShader->fboH);
	glBindFramebuffer(GL_FRAMEBUFFER, _blurShader->fboH);
	_blurShader->initTexture(_width, _height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, _blurShader->texH);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _blurShader->texH, 0);

	//Thickness buffer
	_thicknessShader->initFBO(_thicknessShader->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, _thicknessShader->fbo);
	_thicknessShader->initTexture(_width, _height, GL_RGBA, GL_RGBA32F, _thicknessShader->tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _thicknessShader->tex, 0);

	//Fluid Final buffer
	_fluidFinalShader->initFBO(_fluidFinalShader->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, _fluidFinalShader->fbo);
	_fluidFinalShader->initTexture(_width, _height, GL_RGBA, GL_RGBA32F, _fluidFinalShader->tex);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _fluidFinalShader->tex, 0);

	////Final buffer
	//_finalShader->initFBO(_finalShader->fbo);
	//glBindFramebuffer(GL_FRAMEBUFFER, _finalShader->fbo);
	//_finalShader->initTexture(_width, _height, GL_RGBA, GL_RGBA32F, _finalShader->tex);
	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _finalShader->tex, 0);
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
	//Clear buffer
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	InfintePlane();
	GenerateDepth();
	SmoothDepth();
	GenerateThickness();
	FinalRendering();
}

void FluidRenderer::InfintePlane(void)
{
	//----------------------Infinite Plane---------------------
	glUseProgram(_plane->program);
	glBindFramebuffer(GL_FRAMEBUFFER, _plane->fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_plane->shaderVAOInfinitePlane();

	setMatrix(_plane, _camera->GetModelViewMatrix(), "mView");
	setMatrix(_plane, _camera->GetProjectionMatrix(), "projection");

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void FluidRenderer::GenerateDepth(void)
{
	glUseProgram(_depthShader->program);
	glBindFramebuffer(GL_FRAMEBUFFER, _depthShader->fbo);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	_depthShader->bindPositionVAO(_posVbo, 0);

	setMatrix(_depthShader, _camera->GetModelViewMatrix(), "mView");
	setMatrix(_depthShader, _camera->GetProjectionMatrix(), "projection");
	setFloat(_depthShader, _sphereRadius, "pointRadius");
	setFloat(_depthShader, _width / tanf(60.0 * 0.5f * 3.141592 / 180.0f), "pointScale");
	
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, _numParticles);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

void FluidRenderer::SmoothDepth(void)
{
	////--------------------Particle Blur-------------------------
	glUseProgram(_blurShader->program);

	//Vertical blur
	glBindFramebuffer(GL_FRAMEBUFFER, _blurShader->fboV);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	_blurShader->shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _depthShader->tex);
	GLint depthMap = glGetUniformLocation(_blurShader->program, "depthMap");
	glUniform1i(depthMap, 0);

	setMatrix(_blurShader, _camera->GetModelViewMatrix(), "projection");
	setVec2(_blurShader, _screenSize, "screenSize");
	setVec2(_blurShader, _blurDirY, "blurDir");
	setFloat(_blurShader, _filterRadius, "filterRadius");
	//setFloat(blur, width / aspectRatio * (1.0f / (tanf(cam.zoom*0.5f))), "blurScale");
	setFloat(_blurShader, 0.1f, "blurScale");

	glEnable(GL_DEPTH_TEST);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	//Horizontal blur
	glBindFramebuffer(GL_FRAMEBUFFER, _blurShader->fboH);
	glDrawBuffer(GL_NONE);
	glReadBuffer(GL_NONE);

	glClear(GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _blurShader->texV);
	depthMap = glGetUniformLocation(_blurShader->program, "depthMap");
	glUniform1i(depthMap, 0);

	setMatrix(_blurShader, _camera->GetModelViewMatrix(), "projection");
	setVec2(_blurShader, _screenSize, "screenSize");
	setVec2(_blurShader, _blurDirX, "blurDir");
	setFloat(_blurShader, _filterRadius, "filterRadius");
	//setFloat(blur, width / aspectRatio * (1.0f / (tanf(cam.zoom*0.5f))), "blurScale");
	setFloat(_blurShader, 0.1f, "blurScale");

	glEnable(GL_DEPTH_TEST);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glDisable(GL_DEPTH_TEST);
}

void FluidRenderer::GenerateThickness(void)
{
	//--------------------Particle Thickness-------------------------
	glUseProgram(_thicknessShader->program);
	glBindFramebuffer(GL_FRAMEBUFFER, _thicknessShader->fbo);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_thicknessShader->bindPositionVAO(_posVbo, 0);

	setMatrix(_thicknessShader, _camera->GetModelViewMatrix(), "mView");
	setMatrix(_thicknessShader, _camera->GetProjectionMatrix(), "projection");
	setFloat(_depthShader, _sphereRadius * 2.5f, "pointRadius");
	setFloat(_depthShader, _width / tanf(60.0 * 0.5f * 3.141592 / 180.0f), "pointScale");

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glBlendEquation(GL_FUNC_ADD);
	glDepthMask(GL_FALSE);
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glDrawArrays(GL_POINTS, 0, _numParticles);

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void FluidRenderer::FinalRendering(void)
{
	////--------------------Particle fluidFinal-------------------------
	glUseProgram(_fluidFinalShader->program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_fluidFinalShader->shaderVAOQuad();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _blurShader->texH);
	GLuint depthMap = glGetUniformLocation(_fluidFinalShader->program, "depthMap");
	glUniform1i(depthMap, 0);

	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, _depthShader->tex);
	//GLuint depthMap = glGetUniformLocation(_fluidFinalShader->program, "depthMap");
	//glUniform1i(depthMap, 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, _thicknessShader->tex);
	GLint thicknessMap = glGetUniformLocation(_fluidFinalShader->program, "thicknessMap");
	glUniform1i(thicknessMap, 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, _plane->tex);
	GLint sceneMap = glGetUniformLocation(_fluidFinalShader->program, "sceneMap");
	glUniform1i(sceneMap, 2);

	setMatrix(_fluidFinalShader, _camera->GetProjectionMatrix(), "projection");
	setMatrix(_fluidFinalShader, _camera->GetModelViewMatrix(), "mView");
	setVec4(_fluidFinalShader, color, "color");
	setVec2(_fluidFinalShader, glm::vec2(1.0f / _width, 1.0f / _height), "invTexScale");

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void FluidRenderer::setInt(Shader* shader, const int& x, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader->program, name);
	glUniform1i(loc, x);
}

void FluidRenderer::setFloat(Shader* shader, const float& x, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader->program, name);
	glUniform1f(loc, x);
}

void FluidRenderer::setVec2(Shader* shader, const glm::vec2& v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader->program, name);
	glUniform2f(loc, v.x, v.y);
}

void FluidRenderer::setVec3(Shader* shader, const glm::vec3& v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader->program, name);
	glUniform3f(loc, v.x, v.y, v.z);
}

void FluidRenderer::setVec4(Shader* shader, const glm::vec4& v, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader->program, name);
	glUniform4f(loc, v.x, v.y, v.z, v.w);
}

void FluidRenderer::setMatrix(Shader* shader, const glm::mat4& m, const GLchar* name) {
	GLint loc = glGetUniformLocation(shader->program, name);
	glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(m));
}