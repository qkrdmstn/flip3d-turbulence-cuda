#pragma once
#include<iostream>
#include <GL/glew.h>

const int   TEXTURE_WIDTH = 800;  // NOTE: texture size cannot be larger than
const int   TEXTURE_HEIGHT = 800;  // the rendering window size in non-FBO mode

class FBO
{
public:
	FBO() {}
	~FBO() {}
	void creatFBO();
	void bindFBO();
	void banFBO();
	void addBufferToFBO(GLuint& bufferId, unsigned int n);
private:
	bool checkFramebufferStatus();
	GLuint m_fboId;
};