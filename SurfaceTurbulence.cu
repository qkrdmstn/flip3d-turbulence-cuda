#include "SurfaceTurbulence.cuh"

SurfaceTurbulence::SurfaceTurbulence()
{

}

SurfaceTurbulence:: ~SurfaceTurbulence()
{
	FreeDeviceMem();
}

void SurfaceTurbulence::Initialize_kernel()
{
	_fluid->SetHashTable_kernel();

	Initialize_D << <divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(_fluid->d_CurPos(), _fluid->d_Type(), _fluid->_numParticles, d_NumFineParticles(), _fluid->d_GridHash(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), _baseRes,
			_fineScaleLen, outerRadius, innerRadius);
}

void SurfaceTurbulence::InitHostMem(void)
{
	
	//Surface Maintenance
	h_Pos.resize(_fluid->_numParticles * PER_PARTICLE);
	h_Vel.resize(_fluid->_numParticles * PER_PARTICLE);
	h_SurfaceNormal.resize(_fluid->_numParticles * PER_PARTICLE);
	h_TempNormal.resize(_fluid->_numParticles * PER_PARTICLE);
	h_TempPos.resize(_fluid->_numParticles * PER_PARTICLE);
	h_Tangent.resize(_fluid->_numParticles * PER_PARTICLE);
	h_KernelDens.resize(_fluid->_numParticles * PER_PARTICLE);

	//Wave Simulation
	h_Curvature.resize(_fluid->_numParticles * PER_PARTICLE);
	h_TempCurvature.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveH.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveDtH.resize(_fluid->_numParticles * PER_PARTICLE);
	h_Seed.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveSeedAmp.resize(_fluid->_numParticles * PER_PARTICLE);
	h_Laplacian.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveNormal.resize(_fluid->_numParticles * PER_PARTICLE);

	h_NumFineParticles.resize(1);

}

void SurfaceTurbulence::InitDeviceMem()
{
	//Surface Maintenance
	d_Pos.resize(_fluid->_numParticles * PER_PARTICLE);				d_Pos.memset(0);
	d_Vel.resize(_fluid->_numParticles * PER_PARTICLE);				d_Vel.memset(0);
	d_SurfaceNormal.resize(_fluid->_numParticles * PER_PARTICLE);		d_SurfaceNormal.memset(0);
	d_TempNormal.resize(_fluid->_numParticles * PER_PARTICLE);			d_TempNormal.memset(0);
	d_TempPos.resize(_fluid->_numParticles * PER_PARTICLE);			d_TempPos.memset(0);
	d_Tangent.resize(_fluid->_numParticles * PER_PARTICLE);			d_Tangent.memset(0);
	d_KernelDens.resize(_fluid->_numParticles * PER_PARTICLE);			d_KernelDens.memset(0);

	//Wave Simulation
	d_Curvature.resize(_fluid->_numParticles * PER_PARTICLE);			d_Curvature.memset(0);
	d_TempCurvature.resize(_fluid->_numParticles * PER_PARTICLE);		d_TempCurvature.memset(0);
	d_WaveH.resize(_fluid->_numParticles * PER_PARTICLE);				d_WaveH.memset(0);
	d_WaveDtH.resize(_fluid->_numParticles * PER_PARTICLE);			d_WaveDtH.memset(0);
	d_Seed.resize(_fluid->_numParticles * PER_PARTICLE);				d_Seed.memset(0);
	d_WaveSeedAmp.resize(_fluid->_numParticles * PER_PARTICLE);		d_WaveSeedAmp.memset(0);
	d_Laplacian.resize(_fluid->_numParticles * PER_PARTICLE);			d_Laplacian.memset(0);
	d_WaveNormal.resize(_fluid->_numParticles * PER_PARTICLE);			d_WaveNormal.memset(0);

	////Hash
	//d_GridHash.resize(_numParticles);			d_GridHash.memset(0);
	//d_GridIdx.resize(_numParticles);			d_GridIdx.memset(0);
	//d_CellStart.resize(_gridRes * _gridRes * _gridRes);			d_CellStart.memset(0);
	//d_CellEnd.resize(_gridRes * _gridRes * _gridRes);			d_CellEnd.memset(0);

	d_NumFineParticles.resize(1);			d_NumFineParticles.memset(0);
}

void SurfaceTurbulence::FreeDeviceMem()
{
	//Surface Maintenance
	d_Pos.free();
	d_Vel.free();
	d_SurfaceNormal.free();
	d_TempNormal.free();
	d_TempPos.free();
	d_Tangent.free();
	d_KernelDens.free();

	//Wave Simulation
	d_Curvature.free();
	d_TempCurvature.free();
	d_WaveH.free();
	d_WaveDtH.free();
	d_Seed.free();
	d_WaveSeedAmp.free();
	d_Laplacian.free();
	d_WaveNormal.free();

	//Hash
	d_GridHash.free();
	d_GridIdx.free();
	d_CellStart.free();
	d_CellEnd.free();

	d_NumFineParticles.free();
}

void SurfaceTurbulence::CopyToDevice()
{
	//Surface Maintenance
	d_Pos.copyFromHost(h_Pos);
	d_Vel.copyFromHost(h_Vel);
	d_SurfaceNormal.copyFromHost(h_SurfaceNormal);
	d_TempNormal.copyFromHost(h_TempNormal);
	d_TempPos.copyFromHost(h_TempPos);
	d_Tangent.copyFromHost(h_Tangent);
	d_KernelDens.copyFromHost(h_KernelDens);

	//Wave Simulation
	d_Curvature.copyFromHost(h_Curvature);
	d_TempCurvature.copyFromHost(h_TempCurvature);
	d_WaveH.copyFromHost(h_WaveH);
	d_WaveDtH.copyFromHost(h_WaveDtH);
	d_Seed.copyFromHost(h_Seed);
	d_WaveSeedAmp.copyFromHost(h_WaveSeedAmp);
	d_Laplacian.copyFromHost(h_Laplacian);
	d_WaveNormal.copyFromHost(h_WaveNormal);

	d_NumFineParticles.copyFromHost(h_NumFineParticles);
}

void SurfaceTurbulence::CopyToHost()
{
	//Surface Maintenance
	d_Pos.copyToHost(h_Pos);
	d_Vel.copyToHost(h_Vel);
	d_SurfaceNormal.copyToHost(h_SurfaceNormal);
	d_TempNormal.copyToHost(h_TempNormal);
	d_TempPos.copyToHost(h_TempPos);
	d_Tangent.copyToHost(h_Tangent);
	d_KernelDens.copyToHost(h_KernelDens);

	//Wave Simulation
	d_Curvature.copyToHost(h_Curvature);
	d_TempCurvature.copyToHost(h_TempCurvature);
	d_WaveH.copyToHost(h_WaveH);
	d_WaveDtH.copyToHost(h_WaveDtH);
	d_Seed.copyToHost(h_Seed);
	d_WaveSeedAmp.copyToHost(h_WaveSeedAmp);
	d_Laplacian.copyToHost(h_Laplacian);
	d_WaveNormal.copyToHost(h_WaveNormal);

	d_NumFineParticles.copyToHost(h_NumFineParticles);
}