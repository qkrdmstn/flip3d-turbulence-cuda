#include "FLIPEngine.h"

void FLIPEngine::init(REAL3& gravity, REAL dt)
{
	_gravity = gravity;
	_dt = dt;
	_frame = 0u;

	_fluid = new FLIP3D_Cuda(16u);
}

void	FLIPEngine::simulation(void)
{
	printf("-------------- Step %d --------------\n", _frame);
	_fluid->SetHashTable_kernel();
	//_fluid->ComputeParticleDensity_kernel();
	//_fluid->ComputeExternalForce_kernel(_gravity, _dt);
	//
	//_fluid->SolvePICFLIP();
	//
	//_fluid->AdvectParticle_kernel(_dt);




	_fluid->CopyToHost();

	//if (_frame > 500)
	//	exit(0);
	_frame++;
}

void	FLIPEngine::reset(void)
{

}

void FLIPEngine::draw(void)
{
	_fluid->draw();
}