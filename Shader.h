#pragma once

#include <iostream>
#include <string>
#include <GL/glew.h>

using namespace std;

class Shader
{

public:
	static const unsigned int NUM_SHADER = 2;
	GLuint _program;
	GLuint _shaders[NUM_SHADER]; // vertex and fragment shaders
public:
	Shader(const string& vsfileName, const string& fsfileName);
	virtual ~Shader();
public:
	void		enable(void);
	void		ban(void);
};