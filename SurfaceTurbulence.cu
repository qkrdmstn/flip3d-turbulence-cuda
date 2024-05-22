#include "SurfaceTurbulence.cuh"

SurfaceTurbulence::SurfaceTurbulence()
{

}

SurfaceTurbulence::SurfaceTurbulence(FLIP3D_Cuda* fluid, uint gridRes) {
	_fluid = fluid;

	_coarseScaleLen = 1.0 / gridRes;
	_baseRes = gridRes;
	_fineScaleLen = PI * (_coarseScaleLen + (_coarseScaleLen / 2.0)) / SURFACE_DENSITY;
	_hashGridRes = _baseRes * 4;

	_outerRadius = _coarseScaleLen; //_coarseScaleLen;
	_innerRadius = _outerRadius / 2.0;  //_outerRadius / 2;

	InitHostMem();
	InitDeviceMem();
	CopyToDevice();

	InitWaveParam();
	Initialize_kernel();
	printf("Coarse Scale Length: %f\n", _coarseScaleLen);
	printf("Fine Scale Length: %f\n", _fineScaleLen);

	CopyToHost();
	printf("Initialize coarse-particles number is %d\n", _fluid->_numParticles);
	printf("Initialize fine-particles number is %d\n", _numFineParticles);
	cudaDeviceSynchronize();
}

SurfaceTurbulence:: ~SurfaceTurbulence()
{
	FreeDeviceMem();
}

void SurfaceTurbulence::InitWaveParam(void)
{
	//waveParam._dt = 0.00125;
	//waveParam._waveSpeed = 16.0;
	//waveParam._waveDamping = 0.0f;
	//waveParam._waveSeedFreq = 4.0;
	//waveParam._waveMaxAmplitude = _coarseScaleLen;
	//waveParam._waveMaxFreq = 800.0;
	//waveParam._waveMaxSeedingAmplitude = 2 * _coarseScaleLen; // as multiple of max amplitude
	//waveParam._waveSeedingCurvatureThresholdMinimum = _coarseScaleLen * 0.077; // any curvature higher than this value will seed waves
	//waveParam._waveSeedingCurvatureThresholdMaximum = _coarseScaleLen * 0.15;
	//waveParam._waveSeedStepSizeRatioOfMax = 0.9; // higher values will result in faster and more violent wave seeding

	waveParam._dt = _coarseScaleLen * 0.005f;
	waveParam._waveSpeed = _coarseScaleLen * 16.0;
	waveParam._waveDamping = 0.0f;
	waveParam._waveSeedFreq = _coarseScaleLen * 4.0;
	waveParam._waveMaxAmplitude = _coarseScaleLen * 0.25;
	waveParam._waveMaxFreq = _coarseScaleLen * 800.0;
	waveParam._waveMaxSeedingAmplitude = _coarseScaleLen * 0.5; // as multiple of max amplitude
	waveParam._waveSeedingCurvatureThresholdCenter = _coarseScaleLen * 0.025; // any curvature higher than this value will seed waves
	waveParam._waveSeedingCurvatureThresholdRadius = _coarseScaleLen * 0.01;
	waveParam._waveSeedStepSizeRatioOfMax = _coarseScaleLen * 0.05f; // higher values will result in faster and more violent wave seeding
}

void SurfaceTurbulence::ThrustScanWrapper_kernel(uint* output, uint* input, uint numElements)
{
	thrust::exclusive_scan(thrust::device_ptr<uint>(input),
		thrust::device_ptr<uint>(input + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<uint>(output));
}

void SurfaceTurbulence::Initialize_kernel()
{
	_fluid->SetHashTable_kernel();

	Initialize_D << <divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(_fluid->d_CurPos(), _fluid->d_Type(), this->d_Pos(), this->d_ParticleGridIndex(), _fluid->_numParticles, _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), _baseRes,
			_fineScaleLen, _outerRadius, _innerRadius);

	thrust::sort_by_key(thrust::device_ptr<uint>(d_ParticleGridIndex()),
		thrust::device_ptr<uint>(d_ParticleGridIndex() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL3>(d_Pos()));

	StateCheck_D << <divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(this->d_Pos(), this->d_ParticleGridIndex(), this->d_StateData(), _fluid->_numParticles);

	ThrustScanWrapper_kernel(this->d_StateData(), this->d_StateData(), (_fluid->_numParticles * PER_PARTICLE));

	CUDA_CHECK(cudaMemcpy((void*)&_numFineParticles, (void*)(this->d_StateData() + (_fluid->_numParticles * PER_PARTICLE) - 1), sizeof(uint), cudaMemcpyDeviceToHost));
}

void SurfaceTurbulence::Advection_kernel(void)
{
	_fluid->SetHashTable_kernel();
	REAL r = 2.0 * _coarseScaleLen;

	ComputeCoarseDens_D << <divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
	(r, _fluid->d_CurPos(), _fluid->d_Type(), _fluid->d_KernelDens(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), 
		_baseRes, _fluid->_numParticles);
 
	Advection_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(this->d_Pos(), _fluid->d_CurPos(), _fluid->d_BeforePos(), _fluid->d_Type(), _fluid->d_KernelDens(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), 
			_numFineParticles, _coarseScaleLen, _baseRes, d_Flag());
}

void SurfaceTurbulence::SurfaceConstraint_kernel(void)
{
	SurfaceConstraint_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(this->d_Pos(), _fluid->d_CurPos(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), _outerRadius, _innerRadius, _baseRes, _numFineParticles, d_SurfaceNormal());
	
}

void SurfaceTurbulence::ComputeSurfaceNormal_kernel(void)
{
	REAL r = _coarseScaleLen;
	ComputeCoarseDens_D << <divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, _fluid->d_CurPos(), _fluid->d_Type(), _fluid->d_KernelDens(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(),
			_baseRes, _fluid->_numParticles);

	ComputeFineDens_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeFineNeighborWeightSum_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeSurfaceNormal_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(_fluid->d_CurPos(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), _baseRes,
			d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_TempNormal(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles,
			_outerRadius, _innerRadius);

	SmoothNormal_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_TempNormal(), d_SurfaceNormal(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles, _baseRes);
}

void SurfaceTurbulence::NormalRegularization_kernel(void)
{
	REAL r = _coarseScaleLen;

	ComputeFineDens_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeFineNeighborWeightSum_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	NormalRegularization_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_TempPos(), d_SurfaceNormal(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles, _baseRes);
}

void SurfaceTurbulence::TangentRegularization_kernel(void)
{
	REAL r = 3.0 * _fineScaleLen;

	ComputeFineDens_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeFineNeighborWeightSum_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	TangentRegularization_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_TempPos(), d_SurfaceNormal(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

}

void SurfaceTurbulence::Regularization_kernel(void)
{
	ComputeSurfaceNormal_kernel();

	CopyToTempPos_D << < divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_TempPos(), _numFineParticles);

	NormalRegularization_kernel();
	TangentRegularization_kernel();

	CopyToPos_D << < divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_TempPos(), _numFineParticles);
}

void SurfaceTurbulence::InsertFineParticles(void)
{
	REAL tangentRadius = 3.0 * _fineScaleLen;

	//Normal 계산
	ComputeSurfaceNormal_kernel();

	//Tangent Regularization을 위한 가중치 밀도 설정
	ComputeFineDens_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(tangentRadius, d_Pos(), d_KernelDens(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeFineNeighborWeightSum_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(tangentRadius, d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	InsertFineParticles_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_ParticleGridIndex(), d_Pos(), d_SurfaceNormal(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles, _fluid->_numParticles,
			d_WaveSeedAmp(), d_WaveH(), d_WaveDtH());

	//Copy Key
	Dvector<uint> d_key1, d_key2, d_key3;
	d_key1.resize(_fluid->_numParticles * PER_PARTICLE);
	d_key2.resize(_fluid->_numParticles * PER_PARTICLE);
	d_key3.resize(_fluid->_numParticles * PER_PARTICLE);

	d_ParticleGridIndex.copyToDevice(d_key1);
	d_ParticleGridIndex.copyToDevice(d_key2);
	d_ParticleGridIndex.copyToDevice(d_key3);

	thrust::sort_by_key(thrust::device_ptr<uint>(d_ParticleGridIndex()),
		thrust::device_ptr<uint>(d_ParticleGridIndex() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL3>(d_Pos()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key1()),
		thrust::device_ptr<uint>(d_key1() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL>(d_WaveSeedAmp()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key2()),
		thrust::device_ptr<uint>(d_key2() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL>(d_WaveH()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key3()),
		thrust::device_ptr<uint>(d_key3() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL>(d_WaveDtH()));


	StateCheck_D << <divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_ParticleGridIndex(), d_StateData(), _fluid->_numParticles);

	ThrustScanWrapper_kernel(d_StateData(), d_StateData(), (_fluid->_numParticles * PER_PARTICLE));

	CUDA_CHECK(cudaMemcpy((void*)&_numFineParticles, (void*)(this->d_StateData() + (_fluid->_numParticles * PER_PARTICLE) - 1), sizeof(uint), cudaMemcpyDeviceToHost));

	d_key1.free();
	d_key2.free();
	d_key3.free();
}

void SurfaceTurbulence::DeleteFineParticles(void)
{
	DeleteFineParticles_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_ParticleGridIndex(), d_Pos(), d_SurfaceNormal(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles, _fluid->_numParticles, d_WaveSeedAmp(), d_WaveH(), d_WaveDtH());

	AdvectionDeleteFineParticles_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_ParticleGridIndex(), d_Pos(), _fluid->d_CurPos(), _fluid->d_Type(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), _numFineParticles, _fluid->_numParticles, _baseRes, d_WaveSeedAmp(), d_WaveH(), d_WaveDtH());

	ConstraintDeleteFineParticles_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_ParticleGridIndex(), d_Pos(), _fluid->d_CurPos(), _fluid->d_GridIdx(), _fluid->d_CellStart(), _fluid->d_CellEnd(), _numFineParticles, _fluid->_numParticles, _baseRes, _outerRadius, _innerRadius, d_WaveSeedAmp(), d_WaveH(), d_WaveDtH());

	//Copy Key
	Dvector<uint> d_key1, d_key2, d_key3;
	d_key1.resize(_fluid->_numParticles * PER_PARTICLE);	
	d_key2.resize(_fluid->_numParticles* PER_PARTICLE);
	d_key3.resize(_fluid->_numParticles* PER_PARTICLE);

	d_ParticleGridIndex.copyToDevice(d_key1);
	d_ParticleGridIndex.copyToDevice(d_key2);
	d_ParticleGridIndex.copyToDevice(d_key3);

	thrust::sort_by_key(thrust::device_ptr<uint>(d_ParticleGridIndex()),
		thrust::device_ptr<uint>(d_ParticleGridIndex() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL3>(d_Pos()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key1()),
		thrust::device_ptr<uint>(d_key1() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL>(d_WaveSeedAmp()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key2()),
		thrust::device_ptr<uint>(d_key2() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL>(d_WaveH()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key3()),
		thrust::device_ptr<uint>(d_key3() + (_fluid->_numParticles * PER_PARTICLE)),
		thrust::device_ptr<REAL>(d_WaveDtH()));

	StateCheck_D << <divup(_fluid->_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_ParticleGridIndex(), d_StateData(), _fluid->_numParticles);

	ThrustScanWrapper_kernel(d_StateData(), d_StateData(), (_fluid->_numParticles * PER_PARTICLE));

	CUDA_CHECK(cudaMemcpy((void*)&_numFineParticles, (void*)(this->d_StateData() + (_fluid->_numParticles * PER_PARTICLE) - 1), sizeof(uint), cudaMemcpyDeviceToHost));

	d_key1.free();
	d_key2.free();
	d_key3.free();
}

void SurfaceTurbulence::SurfaceMaintenance(void)
{
	SurfaceConstraint_kernel();

	SetHashTable_kernel();
	Regularization_kernel();

	SetHashTable_kernel();
	InsertFineParticles();
	DeleteFineParticles();
}

void  SurfaceTurbulence::ComputeCurvature_kernel(void)
{
	SetHashTable_kernel();
	ComputeSurfaceNormal_kernel();

	ComputeCurvature_D << < divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_TempCurvature(), d_SurfaceNormal(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles, _baseRes);

	SmoothCurvature_D << < divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_TempCurvature(), d_Curvature(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles, _baseRes);
}

void SurfaceTurbulence::SeedWave_kernel(int step)
{
	SetHashTable_kernel();

	REAL r = 3.0 * _fineScaleLen;
	ComputeFineDens_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeFineNeighborWeightSum_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	SeedWave_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Curvature(), d_WaveSeedAmp(), d_Seed(), d_WaveH(), _numFineParticles, step, waveParam);
}

void SurfaceTurbulence::ComputeWaveNormal_kernel(void)
{
	REAL r = 3.0 * _fineScaleLen;
	ComputeFineDens_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeFineNeighborWeightSum_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeWaveNormal_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_WaveH(), d_WaveNormal(), d_SurfaceNormal(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles, _fluid->_numParticles);
}

void SurfaceTurbulence::ComputeLaplacian_kernel(void)
{
	SetHashTable_kernel();

	REAL r = 3.0 * _fineScaleLen;
	ComputeFineDens_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeFineNeighborWeightSum_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(r, d_Pos(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);

	ComputeLaplacian_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_Laplacian(), d_WaveNormal(), d_WaveH(), d_SurfaceNormal(), d_KernelDens(), d_NeighborWeightSum(), d_GridIdx(), d_CellStart(), d_CellEnd(), _hashGridRes, _numFineParticles);
}

void SurfaceTurbulence::EvolveWave_kernel(void)
{
	EvolveWave_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_WaveDtH(), d_WaveH(), d_Laplacian(), d_Seed(), _numFineParticles, waveParam);
}

void SurfaceTurbulence::WaveSimulation_kernel(int step)
{
	ComputeCurvature_kernel();
	SeedWave_kernel(step);
	ComputeWaveNormal_kernel();
	ComputeLaplacian_kernel();

	EvolveWave_kernel();

	SetDisplayParticles_D <<<divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_DisplayPos(), d_Pos(), d_SurfaceNormal(), d_WaveH(), _numFineParticles);
}

void SurfaceTurbulence::SetHashTable_kernel(void)
{
	CalculateHash_kernel();
	SortParticle_kernel();
	FindCellStart_kernel();
}

void SurfaceTurbulence::CalculateHash_kernel(void)
{
	CalculateHash_D << <divup(_numFineParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_GridHash(), d_GridIdx(), d_Pos(), _hashGridRes, _numFineParticles);
}

void SurfaceTurbulence::SortParticle_kernel(void)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(d_GridHash()),
		thrust::device_ptr<uint>(d_GridHash() + _numFineParticles),
		thrust::device_ptr<uint>(d_GridIdx()));
}

void SurfaceTurbulence::FindCellStart_kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numFineParticles, 128, numBlocks, numThreads);

	uint smemSize = sizeof(uint) * (numThreads + 1);
	FindCellStart_D << <numBlocks, numThreads, smemSize >> >
		(d_GridHash(), d_CellStart(), d_CellEnd(), _numFineParticles);
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
	h_NeighborWeightSum.resize(_fluid->_numParticles * PER_PARTICLE);
	h_Flag.resize(_fluid->_numParticles * PER_PARTICLE);

	////Wave Simulation
	h_Curvature.resize(_fluid->_numParticles * PER_PARTICLE);
	h_TempCurvature.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveH.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveDtH.resize(_fluid->_numParticles * PER_PARTICLE);
	h_Seed.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveSeedAmp.resize(_fluid->_numParticles * PER_PARTICLE);
	h_Laplacian.resize(_fluid->_numParticles * PER_PARTICLE);
	h_WaveNormal.resize(_fluid->_numParticles * PER_PARTICLE);

	//Display Particle
	h_DisplayPos.resize(_fluid->_numParticles * PER_PARTICLE);
}

void SurfaceTurbulence::InitDeviceMem()
{
	//Initialize
	d_ParticleGridIndex.resize(_fluid->_numParticles * PER_PARTICLE);	d_ParticleGridIndex.memset(0);
	d_StateData.resize(_fluid->_numParticles * PER_PARTICLE);			d_StateData.memset(0);

	//Surface Maintenance
	d_Pos.resize(_fluid->_numParticles * PER_PARTICLE);					d_Pos.memset(0);
	d_Vel.resize(_fluid->_numParticles * PER_PARTICLE);					d_Vel.memset(0);
	d_SurfaceNormal.resize(_fluid->_numParticles * PER_PARTICLE);		d_SurfaceNormal.memset(0);
	d_TempNormal.resize(_fluid->_numParticles * PER_PARTICLE);			d_TempNormal.memset(0);
	d_TempPos.resize(_fluid->_numParticles * PER_PARTICLE);				d_TempPos.memset(0);
	d_Tangent.resize(_fluid->_numParticles * PER_PARTICLE);				d_Tangent.memset(0);
	d_KernelDens.resize(_fluid->_numParticles * PER_PARTICLE);			d_KernelDens.memset(0);
	d_NeighborWeightSum.resize(_fluid->_numParticles * PER_PARTICLE);	d_NeighborWeightSum.memset(0);
	d_Flag.resize(_fluid->_numParticles * PER_PARTICLE);				d_Flag.memset(0);

	////Wave Simulation
	d_Curvature.resize(_fluid->_numParticles * PER_PARTICLE);			d_Curvature.memset(0);
	d_TempCurvature.resize(_fluid->_numParticles * PER_PARTICLE);		d_TempCurvature.memset(0);
	d_WaveH.resize(_fluid->_numParticles * PER_PARTICLE);				d_WaveH.memset(0);
	d_WaveDtH.resize(_fluid->_numParticles * PER_PARTICLE);			d_WaveDtH.memset(0);
	d_Seed.resize(_fluid->_numParticles * PER_PARTICLE);				d_Seed.memset(0);
	d_WaveSeedAmp.resize(_fluid->_numParticles * PER_PARTICLE);		d_WaveSeedAmp.memset(0);
	d_Laplacian.resize(_fluid->_numParticles * PER_PARTICLE);			d_Laplacian.memset(0);
	d_WaveNormal.resize(_fluid->_numParticles * PER_PARTICLE);			d_WaveNormal.memset(0);

	//Display Particle
	d_DisplayPos.resize(_fluid->_numParticles * PER_PARTICLE);			d_DisplayPos.memset(0);

	//Hash
	d_GridHash.resize(_fluid->_numParticles * PER_PARTICLE);			d_GridHash.memset(0);
	d_GridIdx.resize(_fluid->_numParticles * PER_PARTICLE);			d_GridIdx.memset(0);
	d_CellStart.resize(_hashGridRes * _hashGridRes * _hashGridRes);			d_CellStart.memset(0);
	d_CellEnd.resize(_hashGridRes * _hashGridRes * _hashGridRes);			d_CellEnd.memset(0);
}

void SurfaceTurbulence::FreeDeviceMem()
{
	//Initialize
	d_ParticleGridIndex.free();
	d_StateData.free();

	//Surface Maintenance
	d_Pos.free();
	d_Vel.free();
	d_SurfaceNormal.free();
	d_TempNormal.free();
	d_TempPos.free();
	d_Tangent.free();
	d_KernelDens.free();
	d_NeighborWeightSum.free();
	d_Flag.free();

	////Wave Simulation
	d_Curvature.free();
	d_TempCurvature.free();
	d_WaveH.free();
	d_WaveDtH.free();
	d_Seed.free();
	d_WaveSeedAmp.free();
	d_Laplacian.free();
	d_WaveNormal.free();

	//Display Particle
	d_DisplayPos.free();

	//Hash
	d_GridHash.free();
	d_GridIdx.free();
	d_CellStart.free();
	d_CellEnd.free();
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
	d_NeighborWeightSum.copyFromHost(h_NeighborWeightSum);
	d_Flag.copyFromHost(h_Flag);

	////Wave Simulation
	d_Curvature.copyFromHost(h_Curvature);
	d_TempCurvature.copyFromHost(h_TempCurvature);
	d_WaveH.copyFromHost(h_WaveH);
	d_WaveDtH.copyFromHost(h_WaveDtH);
	d_Seed.copyFromHost(h_Seed);
	d_WaveSeedAmp.copyFromHost(h_WaveSeedAmp);
	d_Laplacian.copyFromHost(h_Laplacian);
	d_WaveNormal.copyFromHost(h_WaveNormal);

	//Display Particle
	d_DisplayPos.copyFromHost(h_DisplayPos);
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
	d_NeighborWeightSum.copyToHost(h_NeighborWeightSum);
	d_Flag.copyToHost(h_Flag);

	////Wave Simulation
	d_Curvature.copyToHost(h_Curvature);
	d_TempCurvature.copyToHost(h_TempCurvature);
	d_WaveH.copyToHost(h_WaveH);
	d_WaveDtH.copyToHost(h_WaveDtH);
	d_Seed.copyToHost(h_Seed);
	d_WaveSeedAmp.copyToHost(h_WaveSeedAmp);
	d_Laplacian.copyToHost(h_Laplacian);
	d_WaveNormal.copyToHost(h_WaveNormal);

	//Display Particle
	d_DisplayPos.copyToHost(h_DisplayPos);
}

void SurfaceTurbulence::drawFineParticles(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(1.0);
	glLineWidth(1.0);
	for (uint i = 0u; i < _numFineParticles; i++)
	{
		REAL3 position = h_Pos[i];
		REAL3 surfaceNormal = h_SurfaceNormal[i];
		//REAL3 surfaceNormal = h_TempNormal[i];
		REAL3 waveNormal = h_WaveNormal[i];
		REAL curvature = h_Curvature[i];
		REAL laplacian = h_Laplacian[i];
		REAL waveH = h_WaveH[i];
		BOOL flag = h_Flag[i];

		//////Draw normal
		//glColor3f(1.0f, 1.0f, 1.0f);
		//double scale = 0.02;
		//glBegin(GL_LINES);
		//glVertex3d(position.x, position.y, position.z);
		//glVertex3d(position.x + surfaceNormal.x * scale, position.y + surfaceNormal.y * scale, position.z + surfaceNormal.z * scale);
		//glEnd();

		//////Draw waveNormal
		//glColor3f(1.0f, 1.0f, 1.0f);
		//double scale = 0.03;
		//glBegin(GL_LINES);
		//glVertex3d(position.x, position.y, position.z);
		//glVertex3d(position.x + waveNormal.x * scale, position.y + waveNormal.y * scale, position.z + waveNormal.z * scale);
		//glEnd();

		////general visualize
		glColor3f(0.0f, 1.0f, 1.0f);

		////////Curvature visualize
		//REAL3 color = ScalarToColor(curvature * 1000);
		//glColor3f(color.x, color.y, color.z);
		
		//////WaveH visualize
		//REAL3 color = ScalarToColor(waveH * 1000);
		//glColor3f(color.x, color.y, color.z);
		
		////////Laplacian visualize
		//REAL3 color = ScalarToColor(laplacian * 10000);
		//glColor3f(color.x, color.y, color.z);

		//if (flag) {
		//	glColor3f(1.0f, 0.0f, 0.0f);
		//}
		//else
		//	glColor3f(1.0f, 1.0f, 1.0f);
		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();
	}
	//printf("NUM:::: %d\n", cnt);
	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void SurfaceTurbulence::drawDisplayParticles(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(1.0);
	glLineWidth(1.0);
	for (uint i = 0u; i < _numFineParticles; i++)
	{
		REAL3 position = h_DisplayPos[i];
		REAL waveDt = h_WaveDtH[i];

		////general visualize
		glColor3f(0.0f, 1.0f, 1.0f);

		////WaveH visualize
		REAL3 color = VelocityToColor(waveDt * 1000);
		glColor3f(color.x, color.y, color.z);

		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();
	}
	//printf("NUM:::: %d\n", cnt);
	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

REAL3 SurfaceTurbulence::ScalarToColor(double val)
{
	double fColorMap[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };   //Red->Blue
	double v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	double t = v - low;
	REAL x = (fColorMap[low][0]) * (1 - t) + (fColorMap[high][0]) * t;
	REAL y = (fColorMap[low][1]) * (1 - t) + (fColorMap[high][1]) * t;
	REAL z = (fColorMap[low][2]) * (1 - t) + (fColorMap[high][2]) * t;
	REAL3 color = make_REAL3(x, y, z);
	return color;
}

REAL3 SurfaceTurbulence::VelocityToColor(double val)
{
	double fColorMap[3][3] = { { 0.164705882,0.337254902,0.937254902 },{ 0.862745098,1,1 },{ 1,1,1 } };
	double v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 2.0;
	int low = (int)floor(v), high = (int)ceil(v);
	double t = v - low;
	REAL x = ((fColorMap[low][0]) * (1 - t) + (fColorMap[high][0]) * t);
	REAL y = ((fColorMap[low][1]) * (1 - t) + (fColorMap[high][1]) * t);
	REAL z = ((fColorMap[low][2]) * (1 - t) + (fColorMap[high][2]) * t);
	REAL3 color = make_REAL3(x, y, z);
	return color;
}
