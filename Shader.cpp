

#include <iostream>
#include <fstream>
#include "Shader.h"

using namespace std;

static GLuint CreateShader(const string& text, GLenum shaderType);
static string LoadShader(const string& fileName);
static void CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const string& errorMessage);

Shader::Shader(const string& fileName)
{
	_program = glCreateProgram();

	_shaders[0] = CreateShader(LoadShader(fileName + ".vs"), GL_VERTEX_SHADER);
	_shaders[1] = CreateShader(LoadShader(fileName + ".fs"), GL_FRAGMENT_SHADER);

	for (unsigned int i = 0; i < NUM_SHADER; i++)
		glAttachShader(_program, _shaders[i]);

	glLinkProgram(_program);
	CheckShaderError(_program, GL_LINK_STATUS, true, "Error: Program linking failed: ");

	glValidateProgram(_program);
	CheckShaderError(_program, GL_VALIDATE_STATUS, true, "Error: Program is invalid: ");
}

Shader::~Shader()
{
	for (unsigned int i = 0; i < NUM_SHADER; i++) {
		glDetachShader(_program, _shaders[i]);
		glDeleteShader(_shaders[i]);
	}
	glDeleteProgram(_program);
}

void Shader::enable(void)
{
	glUseProgram(_program);
}

void Shader::ban(void)
{
	glUseProgram(0);
}

static GLuint CreateShader(const string& text, GLenum shaderType)
{
	GLuint shader = glCreateShader(shaderType);

	if (shader == 0)
		cout << "Error: Shader Creation failed!" << endl;

	const GLchar* shaderSourceString[1];
	GLint shaderSourceStringLength[1];

	shaderSourceString[0] = text.c_str();
	shaderSourceStringLength[0] = (GLint)text.length();

	glShaderSource(shader, 1, shaderSourceString, shaderSourceStringLength);
	glCompileShader(shader);

	CheckShaderError(shader, GL_COMPILE_STATUS, false, "Error: Shader Compilation failed: ");

	return shader;
}

static string LoadShader(const string& fileName)
{
	ifstream file;
	file.open((fileName).c_str());

	string output;
	string line;

	if (file.is_open())
	{
		while (file.good())
		{
			getline(file, line);
			output.append(line + "\n");
		}
	}
	else
	{
		cout << "Unable to load shader: " << fileName << endl;
	}

	return output;
}

static void CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const string& errorMessage)
{
	GLint success = 0;
	GLchar error[1024] = { 0 };

	if (isProgram)
		glGetProgramiv(shader, flag, &success);
	else
		glGetShaderiv(shader, flag, &success);

	if (success == GL_FALSE)
	{
		if (isProgram)
			glGetProgramInfoLog(shader, sizeof(error), NULL, error);
		else
			glGetShaderInfoLog(shader, sizeof(error), NULL, error);

		cout << errorMessage << error << endl;
	}
}