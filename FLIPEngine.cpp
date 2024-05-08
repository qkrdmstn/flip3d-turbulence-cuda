#include "FLIPEngine.h"

#define RES 64
#define TURBULENCE 1
void FLIPEngine::init(REAL3& gravity, REAL dt)
{
	_gravity = gravity;
	_dt = dt;
	_frame = 0u;

	_fluid = new FLIP3D_Cuda(RES);
	_turbulence = new SurfaceTurbulence(_fluid, RES);
	_fluid->CopyToHost();
}

void	FLIPEngine::simulation(void)
{
	printf("-------------- Step %d --------------\n", _frame);
	_fluid->SetHashTable_kernel();
	_fluid->ComputeParticleDensity_kernel();
	_fluid->ComputeExternalForce_kernel(_gravity, _dt);

	_fluid->SolvePICFLIP();

	_fluid->AdvectParticle_kernel(_dt);

	_fluid->Correct_kernel(_dt);

#if TURBULENCE
	_turbulence->Advection_kernel();

	if (_frame == 0) {
		for (int i = 0; i < 24; i++) {
			_turbulence->SurfaceMaintenance();
			//if (i % 6 == 0)
			//	printf("%.4f%\n", (float)(i + 1) / 24.0);
		}

	}
	else {
		for (int i = 0; i < 4; i++) {
			_turbulence->SurfaceMaintenance();
			//if (i % 2 == 0)
			//	printf("%.4f%\n", (float)(i + 1) / 4);
		}
	}
	printf("-------------- fineParticles %d --------------\n", _turbulence->_numFineParticles);
	_turbulence->WaveSimulation_kernel(_frame);
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

void FLIPEngine::draw(int option)
{
	if (option == 1) {
		_fluid->draw();
#if TURBULENCE
		_turbulence->draw();
#endif
	}
	else if (option == 2) {
		_fluid->draw();
#if TURBULENCE
		_turbulence->draw();
#endif
	}
	else if (option == 3)
		_fluid->draw();
	else if (option == 4) {
		//turbulence->drawDisplayParticles();

	}
	else if (option == 5) {
#if TURBULENCE
		_turbulence->draw();
#endif
	}

}