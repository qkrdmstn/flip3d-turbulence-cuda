#include "FLIPEngine.h"

#define RES 32
#define RENDERRES 256
#define TURBULENCE 1
#define SURFACERECONSTRUCTION 1
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
}

void	FLIPEngine::simulation(bool advection)
{
	printf("-------------- Step %d --------------\n", _frame);
	if (advection || _frame == 0)
	{
		_fluid->SetHashTable_kernel();
		_fluid->ComputeParticleDensity_kernel();
		_fluid->ComputeExternalForce_kernel(_gravity, _dt);
		//_fluid->CollisionMovingBox_kernel(_dt);

		_fluid->SolvePICFLIP();

		_fluid->AdvectParticle_kernel(_dt);

		_fluid->Correct_kernel(_dt);

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
			int iter2 = 4;
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
	_fluid->drawOBB();
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