#include <stdio.h>
#include <string>
#include <time.h>
#include <Windows.h> // exit error
#include "GL\glut.h"
#include "FLIPEngine.h"

using namespace std;

#define SCREEN_CAPTURE 1
int _width = 800;
int _height = 800;
float _zoom = 1.959998f; // 화면 확대,축소
float _rot_x = 53.0f; // x축 회전
float _rot_y = -48.20f; // y축 회전
float _trans_x = 0.02f; // x축 이동
float _trans_y = 0.14f; // y축 이동
int _last_x = 0; // 이전 마우스 클릭 x위치
int _last_y = -0; // 이전 마우스 클릭 y위치
unsigned char _buttons[3] = { 0 }; // 마우스 상태(왼쪽,오른쪽,휠 버튼)
bool _simulation = false;

bool _fluidFlag = false;
bool _turbulenceDisplayFlag = false;
bool _turbulenceBaseFlag = false;
bool _surfaceReconstructionFlag = true;

int frame = 0, curTime, timebase = 0;
int _frame = 0;
double fps = 0;

FLIPEngine* _engine;

void Init(void)
{
	// 깊이값 사용 여부
	glEnable(GL_DEPTH_TEST);
	// 0.6e-2
	_engine = new FLIPEngine(make_REAL3(0, -9.81, 0.0), 0.6e-2);
}

void Capture(char* filename, int width, int height)
{
	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
	FILE* file;
	fopen_s(&file, filename, "wb");
	if (image != NULL) {
		if (file != NULL) {
			glReadPixels(0, 0, width, height, 0x80E0, GL_UNSIGNED_BYTE, image);
			memset(&bf, 0, sizeof(bf));
			memset(&bi, 0, sizeof(bi));
			bf.bfType = 'MB';
			bf.bfSize = sizeof(bf) + sizeof(bi) + width * height * 3;
			bf.bfOffBits = sizeof(bf) + sizeof(bi);
			bi.biSize = sizeof(bi);
			bi.biWidth = width;
			bi.biHeight = height;
			bi.biPlanes = 1;
			bi.biBitCount = 24;
			bi.biSizeImage = width * height * 3;
			fwrite(&bf, sizeof(bf), 1, file);
			fwrite(&bi, sizeof(bi), 1, file);
			fwrite(image, sizeof(unsigned char), height * width * 3, file);
			fclose(file);
		}
		free(image);
	}
}

void Update(void)
{
	if (_simulation) {
#if SCREEN_CAPTURE
		if (_frame <= 1000) {
			string path = "image\\FLIPGPU" + to_string(_frame) + ".jpg";
			char* strPath = const_cast<char*>((path).c_str());
			Capture(strPath, _width, _height);
		}
#endif
		frame++;
		curTime = glutGet(GLUT_ELAPSED_TIME); 
		if (curTime - timebase > 1000)
		{
			fps = frame * 1000.0 / (curTime - timebase);
			timebase = curTime;
			frame = 0;
		}

		_engine->simulation();

		//if (_frame == 600) {
		//	exit(0);
		//}
		_frame++;
	}
	glutPostRedisplay();
}

void DrawText(float x, float y, const char* text, void* font = NULL)
{
	glColor3f(1, 1, 1);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, (double)_width, 0.0, (double)_height, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	if (font == NULL) {
		font = GLUT_BITMAP_9_BY_15;
	}

	size_t len = strlen(text);

	glRasterPos2f(x, y);
	for (const char* letter = text; letter < text + len; letter++) {
		if (*letter == '\n') {
			y -= 12.0f;
			glRasterPos2f(x, y);
		}
		glutBitmapCharacter(font, *letter);
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glEnable(GL_DEPTH_TEST);
}

void Draw(void)
{
	glEnable(GL_LIGHTING); // 조명 활성화
	glEnable(GL_LIGHT0); // 첫번째 조명

	_engine->draw(_fluidFlag, _turbulenceBaseFlag, _turbulenceDisplayFlag, _surfaceReconstructionFlag);
	glDisable(GL_LIGHTING);

	char text[100];
	sprintf(text, "Frame: %d", _frame);
	DrawText(10.0f, 760.0f, text);

	sprintf(text, "FPS: %f", fps);
	DrawText(10.0f, 780.0f, text);
}

void Display(void)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glTranslatef(0, 0, -_zoom);
	glTranslatef(_trans_x, _trans_y, 0);
	glRotatef(_rot_x, 1, 0, 0);
	glRotatef(_rot_y, 0, 1, 0);

	glTranslatef(-0.5, -0.5, -0.5);
	Draw();
	glutSwapBuffers();
}

void Reshape(int w, int h)
{
	if (w == 0) {
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (float)w / h, 0.1, 100);
	glMatrixMode(GL_MODELVIEW); // 이걸 빼먹음!!!
	glLoadIdentity();
}

void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'Q':
	case 'q':
		exit(0);
	case '1':
		_fluidFlag = !_fluidFlag;
		if (_fluidFlag)
			printf("FLIP render On\n");
		else
			printf("FLIP render Off\n");
		break;
	case '2':
		_turbulenceBaseFlag = !_turbulenceBaseFlag;
		if (_turbulenceBaseFlag)
			printf("Turbulence Base render On\n");
		else
			printf("Turbulence Base render Off\n");
		break;
	case '3':
		_turbulenceDisplayFlag = !_turbulenceDisplayFlag;
		if (_turbulenceDisplayFlag)
			printf("Turbulence Display render On\n");
		else
			printf("Turbulence Display render Off\n");
		break;
	case '4':
		_surfaceReconstructionFlag = !_surfaceReconstructionFlag;
		if (_surfaceReconstructionFlag)
			printf("MarchingCube render On\n");
		else
			printf("MarchingCube render Off\n");
		break;
	case ' ':
		_simulation = !_simulation;
		if (!_simulation)
			printf("Simulation Pause\n");
		else
			printf("Simulation Start\n");
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
	_last_x = x;
	_last_y = y;

	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		_buttons[0] = state == GLUT_DOWN ? 1 : 0;
		break;
	case GLUT_MIDDLE_BUTTON:
		_buttons[1] = state == GLUT_DOWN ? 1 : 0;
		break;
	case GLUT_RIGHT_BUTTON:
		_buttons[2] = state == GLUT_DOWN ? 1 : 0;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void Motion(int x, int y)
{
	int diff_x = x - _last_x;
	int diff_y = y - _last_y;
	_last_x = x;
	_last_y = y;

	if (_buttons[2]) {
		_zoom -= (float)0.02f * diff_x;
	}
	else if (_buttons[1]) {
		_trans_x += (float)0.005f * diff_x;
		_trans_y -= (float)0.005f * diff_y;
	}
	else if (_buttons[0]) {
		_rot_x += (float)0.2f * diff_y;
		_rot_y += (float)0.2f * diff_x;
	}

	//printf("_zoom: %f, _transX: %f, _transY: %f, _rotX: %f, _rotY: %f \n", _zoom, _trans_x, _trans_y, _rot_x, _rot_y);
	glutPostRedisplay();
}

void main(int argc, char** argv)
{
	glutInit(&argc, argv);
	Init();
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(_width, _height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("FLIP on GPU");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Update);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutKeyboardFunc(Keyboard);
	glutMainLoop();
}