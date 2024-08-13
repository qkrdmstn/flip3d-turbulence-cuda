#include "FLIPEngine.h"
#include "Shader.h"

#define RES 64
#define RENDERRES 256
#define TURBULENCE 0
#define SURFACERECONSTRUCTION 0

void FLIPEngine::init()
{
	InitSimulation(make_REAL3(0, -9.81, 0.0), 0.005);
	InitRenderer();
}

void	FLIPEngine::InitRenderer()
{
	_renderer = new FluidRenderer();
	_particleShader = new Shader("Shader\\particleSphere");
}

void	FLIPEngine::InitSimulation(REAL3& gravity, REAL dt)
{
	_gravity = gravity;
	_dt = dt;
	_frame = 0u;

	_fluid = new FLIP3D_Cuda(RES);
#if TURBULENCE
	_turbulence = new SurfaceTurbulence(_fluid, RES);
#endif
	_fluid->CopyToHost();

#if SURFACERECONSTRUCTION
	_MC = new MarchingCubes_CUDA();
	_MC->init(_fluid, _turbulence, RENDERRES, RENDERRES, RENDERRES);
#endif

	// allocate GPU data
	unsigned int memSize = 4 * sizeof(float) * _fluid->_numFluidParticles;

	// Pos VBO 생성
	glGenBuffers(1, &posVbo);
	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&vboPosResource, posVbo, cudaGraphicsMapFlagsNone);

	// Color VBO 생성
	glGenBuffers(1, &colorVbo);
	glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
	glBufferData(GL_ARRAY_BUFFER, memSize, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&vboColorResource, colorVbo, cudaGraphicsMapFlagsNone);

	// fill color buffer
	glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
	float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float* ptr = data;

	for (uint i = 0; i < _fluid->_numFluidParticles; i++)
	{
		float t = i / (float)_fluid->_numFluidParticles;
#if 0
		* ptr++ = rand() / (float)RAND_MAX;
		*ptr++ = rand() / (float)RAND_MAX;
		*ptr++ = rand() / (float)RAND_MAX;
#else
		colorRamp(t, ptr);
		ptr += 3;
#endif
		* ptr++ = 1.0f;
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
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
		cudaGraphicsMapResources(1, &vboPosResource, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&dptr, &numBytes, vboPosResource);
		
		_fluid->CopyPosToVBO(dptr);

		cudaGraphicsUnmapResources(1, &vboPosResource, 0);

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

	//_fluid->CopyToHost();

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
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	_particleShader->enable();
	glUniform1f(glGetUniformLocation(_particleShader->_program, "pointScale"),
		800 / tanf(60.0 * 0.5f * (float)M_PI / 180.0f));
	glUniform1f(glGetUniformLocation(_particleShader->_program, "pointRadius"),
		0.125f * 0.15f);

	// OpenGL에서 VBO를 사용하여 렌더링
	glColor3f(1, 1, 1);
	//Pos Setting
	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	//Color Setting
	glBindBuffer(GL_ARRAY_BUFFER, colorVbo);
	glColorPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, 0, _fluid->_numFluidParticles);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	_particleShader->ban();
	glDisable(GL_POINT_SPRITE_ARB);

	//if (flag1)
	//	_fluid->draw();

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
	_fluid->drawBoundingObject();
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