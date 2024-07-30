#include "FLIPEngine.h"

#define RES 64
#define RENDERRES 256
#define TURBULENCE 0
#define SURFACERECONSTRUCTION 0
void FLIPEngine::init(REAL3& gravity, REAL dt)
{
	_gravity = gravity;
	_dt = dt;
	_frame = 0u;

	_fluid = new FLIP3D_Cuda(RES);
	_turbulence = new SurfaceTurbulence(_fluid, RES);
	_fluid->CopyToHost();

#if SURFACERECONSTRUCTION
	_MC = new MarchingCubes_CUDA();
	_MC->init(_fluid, _turbulence, RENDERRES, RENDERRES, RENDERRES);
#endif

	// allocate GPU data
	unsigned int memSize = sizeof(REAL3) * _fluid->_numParticles;

	// VBO 생성
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, &_fluid->h_CurPos, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&vboResource, vbo, cudaGraphicsMapFlagsNone);
}

void	FLIPEngine::simulation(bool advection, bool flag)
{
	printf("-------------- Step %d --------------\n", _frame);
	if (advection || _frame == 0)
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

		//if (_frame <= 60)
		{
			_fluid->MoveObject();
			_fluid->CollisionObject_kernel(_dt);
		}

		_fluid->SolvePICFLIP();

		_fluid->AdvectParticle_kernel(_dt);

		_fluid->Correct_kernel(_dt);


		REAL3* dptr;
		size_t numBytes;
		cudaGraphicsMapResources(1, &vboResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&dptr, &numBytes, vboResource);
		_fluid->CopyPosToVBO(dptr);

		cudaGraphicsUnmapResources(1, &vboResource, 0);

#if TURBULENCE
		_turbulence->Advection_kernel();

		if (_frame == 0)
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
	if(_frame != 0)
		_turbulence->WaveSimulation_kernel(_frame);

#else
	}
#endif

#if SURFACERECONSTRUCTION
	//MC
	_MC->MarchingCubes();
#endif

	_fluid->CopyToHost();

#if TURBULENCE
	_turbulence->CopyToHost();
#endif
	//if (_frame > 500)
	//	exit(0);
	_frame++;
}

void	FLIPEngine::reset(void)
{

}

void FLIPEngine::draw(bool flag1, bool flag2, bool flag3, bool flag4)
{
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//// OpenGL에서 VBO를 사용하여 렌더링
	//glEnableClientState(GL_VERTEX_ARRAY);
	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//glVertexPointer(3, GL_FLOAT, 0, 0);
	//glDrawArrays(GL_POINTS, 0, _fluid->_numParticles);
	//glDisableClientState(GL_VERTEX_ARRAY);

	//glutSwapBuffers();

	if (flag1)
		_fluid->draw();

#if TURBULENCE
	if (flag2)
		_turbulence->drawFineParticles();
	if (flag3)
		_turbulence->drawDisplayParticles();
#endif

#if SURFACERECONSTRUCTION
	if (flag4)
		_MC->renderSurface();
#endif
	//_fluid->drawBoundingObject();
	//drawBoundary();
}

void	FLIPEngine::drawBoundary()
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