#include "FLIPEngine.h"
#include "Shader.h"

#define SCREEN_CAPTURE 0
#define OBJ_CAPTURE 0


#define RES 64
#define RENDERRES 256
#define TURBULENCE 0
#define SURFACERECONSTRUCTION 0

void FLIPEngine::init()
{
	InitOpenGL();
	InitCamera();
	InitSimulation(make_REAL3(0, -9.81, 0.0), 0.005);
	InitRenderer();
}

void FLIPEngine::InitOpenGL()
{
	// 깊이값 사용 여부
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.3f, 1.0f);

	glewInit();
	cudaGLSetGLDevice(0);
}

void FLIPEngine::InitCamera()
{
	_camera = new Camera();
	_camera->CameraZoom(1.959998f);
	_camera->CameraTranslate(0.02f, 0.14f);
	_camera->CameraRotate(53.0f, -48.20f);
}

void FLIPEngine::InitRenderer()
{
	_renderer = new FluidRenderer();
	_renderer->InitializeFluidRenderer(_camera, _fluid->_numFluidParticles);
}

void FLIPEngine::InitSimulation(REAL3& gravity, REAL dt)
{
	_gravity = gravity;
	_dt = dt;
	_step = 0u;

	_fluid = new FLIP3D_Cuda(RES);
#if TURBULENCE
	_turbulence = new SurfaceTurbulence(_fluid, RES);
#endif
	_fluid->CopyToHost();

#if SURFACERECONSTRUCTION
	_MC = new MarchingCubes_CUDA();
	_MC->init(_fluid, _turbulence, RENDERRES, RENDERRES, RENDERRES);
#endif
}

void FLIPEngine::simulation(void)
{
	printf("-------------- Step %d --------------\n", _step);
	if (advection || _step == 0)
	{
//		if (flag)
//		{
//			_fluid->PourWater();
//#if TURBULENCE
//			_turbulence->InsertNewCoarseNeighbor_kernel();
//#endif
//		}

		_fluid->SetHashTable_kernel();
		_fluid->ComputeParticleDensity_kernel();
		_fluid->ComputeExternalForce_kernel(_gravity, _dt);

		//if (_step <= 60)
		{
			_fluid->MoveObject();
			_fluid->CollisionObject_kernel(_dt);
		}

		_fluid->SolvePICFLIP();
 		_fluid->AdvectParticle_kernel(_dt);
		_fluid->Correct_kernel(_dt);

		//VBO Update
		float3* dptr = _renderer->CudaGraphicResourceMapping();
		_fluid->CopyPosToVBO(dptr);
		_renderer->CudaGraphicResourceUnMapping();

#if TURBULENCE
		_turbulence->Advection_kernel();

		if (_step == 0)
		{
			int iter1 = 24;
			for (int i = 0; i < iter1; i++)
			{
				_turbulence->SurfaceMaintenance();
				if (i % 6 == 0)
					printf("%.4f%\n", (float)(i + 1) / (float)iter1);
			}

		}
		else
		{
			int iter2 = 6;
			for (int i = 0; i < iter2; i++)
			{
				_turbulence->SurfaceMaintenance();
				if (i % 2 == 0)
					printf("%.4f%\n", (float)(i + 1) / (float)iter2);
			}
		}
	}

	printf("SurfaceParticles %d\n", _turbulence->_numFineParticles);
	if(_step != 0)
		_turbulence->WaveSimulation_kernel(_step);

#else
	}
#endif

#if SURFACERECONSTRUCTION
	//MC
	_MC->MarchingCubes();
#endif

	//_fluid->CopyToHost();

#if TURBULENCE
	_turbulence->CopyToHost();
#endif

}

void FLIPEngine::reset(void)
{

}


void FLIPEngine::idle()
{
	if (_simulation) {
		if (/*_step <= 1000 &&*/ _step % 3 == 1)
		{
#if SCREEN_CAPTURE

			string path = "capture\\image\\Fluid" + to_string(cnt) + ".jpg";
			char* strPath = const_cast<char*>((path).c_str());
			Capture(strPath, _width, _height);
#endif

#if OBJ_CAPTURE
			string objPath = "capture\\obj\\Fluid" + to_string(cnt) + ".obj";
			char* objStrPath = const_cast<char*>((objPath).c_str());
			_engine->ExportObj(objStrPath);
#endif	
			cnt++;
		}
		_frame++;
		_curTime = glutGet(GLUT_ELAPSED_TIME);
		if (_curTime - _timebase > 1000)
		{
			fps = _frame * 1000.0 / (_curTime - _timebase);
			_timebase = _curTime;
			_frame = 0;
		}

		simulation();
		_step++;

		//if (_step == 600) {
		//	exit(0);
		//}
	}
	glutPostRedisplay();
}

void FLIPEngine::draw()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glUseProgram(0);
	_camera->SetCameraForOpenGL();

	_renderer->Rendering();
	//if (_fluidFlag)
	//	_fluid->draw();

#if TURBULENCE
	if (_turbulenceBaseFlag)
		_turbulence->drawFineParticles();
	if (_turbulenceDisplayFlag)
		_turbulence->drawDisplayParticles();
#endif

#if SURFACERECONSTRUCTION
	if (_surfaceReconstructionFlag)
		_MC->renderSurface();
#endif
	_fluid->drawBoundingObject();
	//drawBoundary();

	DrawSimulationInfo();
	glutSwapBuffers();
}

void FLIPEngine::DrawSimulationInfo(void)
{
	glEnable(GL_LIGHTING); // 조명 활성화
	glEnable(GL_LIGHT0); // 첫번째 조명
	char text[100];

	sprintf(text, "FPS: %f", fps);
	DrawText(10.0f, 780.0f, text);

	sprintf(text, "Frame: %d", _step);
	DrawText(10.0f, 760.0f, text);

	sprintf(text, "FLIP Particles: %d", _fluid->_numParticles);
	DrawText(10.0f, 740.0f, text);

#if TURBULENCE
	sprintf(text, "Turbulence Particles: %d", _turbulence->_numFineParticles);
	DrawText(10.0f, 720.0f, text);
#endif
	glDisable(GL_LIGHTING);
}

void FLIPEngine::DrawText(float x, float y, const char* text, void* font)
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

void FLIPEngine::reshape(int w, int h)
{
	if (w == 0) {
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	_camera->SetPerspective(45, (float)w / h, 0.1f, 100000.0f);
	gluPerspective(45, (float)w / h, 0.1, 100000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void FLIPEngine::mouse(int mouse_event, int state, int x, int y)
{
	_mousePos[0] = x;
	_mousePos[1] = y;
	switch (mouse_event)
	{
	case GLUT_LEFT_BUTTON:
		_mouseEvent[0] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_MIDDLE_BUTTON:
		_mouseEvent[1] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_RIGHT_BUTTON:
		_mouseEvent[2] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void FLIPEngine::keyboard(unsigned char key, int x, int y)
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
	case 'c':
		advection = !advection;
		if (!advection)
			printf("advection Pause\n");
		else
			printf("advection Start\n");
		break;
	case 'v':
		flag = !flag;
		if (!flag)
			printf("flag false\n");
		else
			printf("flag true\n");
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void FLIPEngine::motion(int x, int y)
{
	int diffx = x - _mousePos[0];
	int diffy = y - _mousePos[1];

	_mousePos[0] = x;
	_mousePos[1] = y;

	if (_mouseEvent[0])
	{
		float factorX = (float)0.2 * diffy;
		float factorY = (float)0.2 * diffx;

		_camera->CameraRotate(factorX, factorY);
	}
	else if (_mouseEvent[1])
	{
		float factorX = (float)0.005f * diffx;
		float factorY = -(float)0.005f * diffy;

		_camera->CameraTranslate(factorX, factorY);
	}
	else if (_mouseEvent[2])
	{
		float factor = -(float)0.02f * diffx;

		_camera->CameraZoom(factor);
	}

	glutPostRedisplay();
}

void FLIPEngine::drawBoundary()
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(1.0);
	glLineWidth(0.5f);
	glColor3f(1.0f, 1.0f, 1.0f);

	glBegin(GL_LINES);
	glVertex3d(0.0f, 0.0f, 0.0f);
	glVertex3d(0.0f, 0.0f, 1.0f);
	glVertex3d(0.0f, 0.0f, 0.0f);
	glVertex3d(0.0f, 1.0f, 0.0f);
	glVertex3d(0.0f, 0.0f, 0.0f);
	glVertex3d(1.0f, 0.0f, 0.0f);
	glVertex3d(1.0f, 1.0f, 1.0f);
	glVertex3d(1.0f, 1.0f, 0.0f);
	glVertex3d(1.0f, 1.0f, 1.0f);
	glVertex3d(1.0f, 0.0f, 1.0f);
	glVertex3d(1.0f, 1.0f, 1.0f);
	glVertex3d(0.0f, 1.0f, 1.0f);
	glVertex3d(1.0f, 0.0f, 0.0f);
	glVertex3d(1.0f, 1.0f,0.0f);
	glVertex3d(1.0f, 0.0f, 0.0f);
	glVertex3d(1.0f, 0.0f, 1.0f);
	glVertex3d(0.0f, 1.0f, 0.0f);
	glVertex3d(0.0f, 1.0f, 1.0f);
	glVertex3d(0.0f, 1.0f, 0.0f);
	glVertex3d(1.0f, 1.0f, 0.0f);
	glVertex3d(0.0f, 0.0f, 1.0f);
	glVertex3d(0.0f, 1.0f, 1.0f);
	glVertex3d(1.0f, 0.0f, 1.0f);
	glVertex3d(0.0f, 0.0f, 1.0f);
	glEnd();

	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void FLIPEngine::ExportObj(const char* filePath)
{
	ofstream fout;
	fout.open(filePath);


	//for (int i = 0; i < numberFace; i++)
	//{
	//	glBegin(GL_POLYGON);
	//	for (int j = 0; j < 3; j++)
	//	{
	//		vec3 vertexNormal = h_VertexNormals[h_Faces[i * 3 + j]];
	//		vec3 vertex = h_Vertices[h_Faces[i * 3 + j]];
	//		glNormal3d(vertexNormal.x(), vertexNormal.y(), vertexNormal.z());
	//		glVertex3d(vertex.x(), vertex.y(), vertex.z());
	//	}
	//	glEnd();
	//}

	int numberFace = (int)_MC->h_Faces.size() / 3;
	for (int i = 0; i < _MC->h_Vertices.size(); i++)
	{
		string vStr = "v " + to_string(_MC->h_Vertices[i].x()) + " " + to_string(_MC->h_Vertices[i].y()) + " " + to_string(_MC->h_Vertices[i].z()) + "\n";
		fout.write(vStr.c_str(), vStr.size());
	}

	for (int i = 0; i < numberFace; i++)
	{
		string fStr = "f " + to_string(_MC->h_Faces[i * 3 + 0] + 1u) + " " + to_string(_MC->h_Faces[i * 3 + 1] + 1u) + " " + to_string(_MC->h_Faces[i * 3 + 2] + 1u) + "\n";
		fout.write(fStr.c_str(), fStr.size());
	}

	fout.close();
}

void FLIPEngine::Capture(char* filename, int width, int height)
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