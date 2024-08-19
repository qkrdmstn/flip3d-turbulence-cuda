

#pragma once

#include <iostream>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL 
#include <glm/gtx/transform.hpp>
#include "GL\glut.h"
using namespace std;


class Camera
{
public:
	glm::mat4 m_perspective;

	float m_Zoom;
	float m_Rotate[2];
	float m_Translate[2];
public:
	Camera()
	{
		m_Zoom = 0.0f;
		m_Rotate[0] = m_Rotate[1] = 0.0f;
		m_Translate[0] = m_Translate[1] = 0.0f;
	}

	inline glm::mat4 GetModelViewMatrix() const
	{
		glm::mat4 transMat, rotXMat, rotYMat;

		transMat = glm::translate(glm::vec3(m_Translate[0], m_Translate[1], -m_Zoom));
		rotXMat = glm::rotate(m_Rotate[0] * 3.14f / 180.0f, glm::vec3(1, 0, 0));
		rotYMat = glm::rotate(m_Rotate[1] * 3.14f / 180.0f, glm::vec3(0, 1, 0));

		return transMat * rotXMat * rotYMat;
	}

	inline glm::mat4 GetProjectionMatrix() const
	{
		return m_perspective;
	}

	inline glm::mat4 GetModelViewProjection() const
	{
		return GetProjectionMatrix() * GetModelViewMatrix();
	}

	inline void SetPerspective(float fov, float aspect, float zNear, float zFar)
	{
		m_perspective = glm::perspective(fov, aspect, zNear, zFar);
	}

	inline void CameraZoom(float factor)
	{
		m_Zoom += factor;
	}

	inline void CameraTranslate(float factorX, float factorY)
	{
		m_Translate[0] += factorX;
		m_Translate[1] += factorY;
	}

	inline void CameraRotate(float factorX, float factorY)
	{
		m_Rotate[0] += factorX;
		m_Rotate[1] += factorY;
	}

	inline void SetCameraForOpenGL()
	{
		glTranslatef(0.0, 0.0, -m_Zoom);
		glTranslatef(m_Translate[0], m_Translate[1], 0.0);
		glRotatef(m_Rotate[0], 1.0, 0.0, 0.0);
		glRotatef(m_Rotate[1], 0.0, 1.0, 0.0);
	}

	inline void PrintCameraInfo()
	{
		cout << "Zoom : " << m_Zoom << endl;
		cout << "Rotate : " << m_Rotate[0] << " " << m_Rotate[1] << endl;
		cout << "Translate : " << m_Translate[0] << " " << m_Translate[1] << endl;
		cout << endl;
	}
};