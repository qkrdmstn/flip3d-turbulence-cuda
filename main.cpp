#include <stdio.h>
#include <string>
#include <time.h>
#include <Windows.h> // exit error
#include "FLIPEngine.h"
#include "GL\glut.h"

using namespace std;


FLIPEngine* _engine;

void Init(void)
{
	_engine = new FLIPEngine();

}

void Update(void)
{
	_engine->idle();
}


void Draw(void)
{
	glEnable(GL_LIGHTING); // 조명 활성화
	glEnable(GL_LIGHT0); // 첫번째 조명

	_engine->draw();
	glDisable(GL_LIGHTING);
}

void Display(void)
{
	_engine->draw();
}

void Reshape(int w, int h)
{
	_engine->reshape(w, h);
}

void Keyboard(unsigned char key, int x, int y)
{
	_engine->keyboard(key, x, y);
}

void Mouse(int button, int state, int x, int y)
{
	_engine->mouse(button, state, x, y);
}

void Motion(int x, int y)
{
	_engine->motion(x, y);
}

void main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(800, 800);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("FLIP on GPU");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Update);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutKeyboardFunc(Keyboard);

	Init();
	_engine->_width = 800;
	_engine->_height = 800;
	glutMainLoop();
}