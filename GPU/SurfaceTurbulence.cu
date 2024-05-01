#include "SurfaceTurbulence.cuh"

SurfaceTurbulence::SurfaceTurbulence()
{

}

SurfaceTurbulence::SurfaceTurbulence(FLIP3D_Cuda* fluid, uint gridRes) {
	_fluid = fluid;

	_coarseScaleLen = 1.0 / gridRes;
	_baseRes = gridRes;
	_fineScaleLen = PI * (_coarseScaleLen + (_coarseScaleLen / 2.0)) / SURFACE_DENSITY;
	_hashGridRes = 1.0 / _fineScaleLen;

	_outerRadius = _coarseScaleLen;
	_innerRadius = _outerRadius / 2.0;

	_waveSeedingCurvatureThresholdMinimum = _coarseScaleLen * 0.005; //°î·ü ÀÓ°è°ª (Á¶Á¤ ÇÊ¿ä)
	_waveSeedingCurvatureThresholdMaximum = _coarseScaleLen * 0.077;

	InitHostMem();
	InitDeviceMem();
	CopyToDevice();

	Initialize_kernel();
	printf("Coarse Scale Length: %f\n", _coarseScaleLen);
	printf("Fine Scale Length: %f\n", _fineScaleLen);

	CopyToHost();
	printf("Initialize coarse-particles number is %d\n", _fluid->_numParticles);
	printf("Initialize fine-particles number is %d\n", _numFineParticles);
}

SurfaceTurbulence:: ~SurfaceTurbulence()
{
	FreeDeviceMem();
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
			_numFineParticles, _coarseScaleLen, _baseRes, _fluid->d_Flag());
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

	////Wave Simulation
	//h_Curvature.resize(_fluid->_numParticles * PER_PARTICLE);
	//h_TempCurvature.resize(_fluid->_numParticles * PER_PARTICLE);
	//h_WaveH.resize(_fluid->_numParticles * PER_PARTICLE);
	//h_WaveDtH.resize(_fluid->_numParticles * PER_PARTICLE);
	//h_Seed.resize(_fluid->_numParticles * PER_PARTICLE);
	//h_WaveSeedAmp.resize(_fluid->_numParticles * PER_PARTICLE);
	//h_Laplacian.resize(_fluid->_numParticles * PER_PARTICLE);
	//h_WaveNormal.resize(_fluid->_numParticles * PER_PARTICLE);

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

	////Wave Simulation
	//d_Curvature.resize(_fluid->_numParticles * PER_PARTICLE);			d_Curvature.memset(0);
	//d_TempCurvature.resize(_fluid->_numParticles * PER_PARTICLE);		d_TempCurvature.memset(0);
	//d_WaveH.resize(_fluid->_numParticles * PER_PARTICLE);				d_WaveH.memset(0);
	//d_WaveDtH.resize(_fluid->_numParticles * PER_PARTICLE);			d_WaveDtH.memset(0);
	//d_Seed.resize(_fluid->_numParticles * PER_PARTICLE);				d_Seed.memset(0);
	//d_WaveSeedAmp.resize(_fluid->_numParticles * PER_PARTICLE);		d_WaveSeedAmp.memset(0);
	//d_Laplacian.resize(_fluid->_numParticles * PER_PARTICLE);			d_Laplacian.memset(0);
	//d_WaveNormal.resize(_fluid->_numParticles * PER_PARTICLE);			d_WaveNormal.memset(0);

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

	////Wave Simulation
	//d_Curvature.free();
	//d_TempCurvature.free();
	//d_WaveH.free();
	//d_WaveDtH.free();
	//d_Seed.free();
	//d_WaveSeedAmp.free();
	//d_Laplacian.free();
	//d_WaveNormal.free();

	////Hash
	//d_GridHash.free();
	//d_GridIdx.free();
	//d_CellStart.free();
	//d_CellEnd.free();

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

	////Wave Simulation
	//d_Curvature.copyFromHost(h_Curvature);
	//d_TempCurvature.copyFromHost(h_TempCurvature);
	//d_WaveH.copyFromHost(h_WaveH);
	//d_WaveDtH.copyFromHost(h_WaveDtH);
	//d_Seed.copyFromHost(h_Seed);
	//d_WaveSeedAmp.copyFromHost(h_WaveSeedAmp);
	//d_Laplacian.copyFromHost(h_Laplacian);
	//d_WaveNormal.copyFromHost(h_WaveNormal);

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

	////Wave Simulation
	//d_Curvature.copyToHost(h_Curvature);
	//d_TempCurvature.copyToHost(h_TempCurvature);
	//d_WaveH.copyToHost(h_WaveH);
	//d_WaveDtH.copyToHost(h_WaveDtH);
	//d_Seed.copyToHost(h_Seed);
	//d_WaveSeedAmp.copyToHost(h_WaveSeedAmp);
	//d_Laplacian.copyToHost(h_Laplacian);
	//d_WaveNormal.copyToHost(h_WaveNormal);

}

void SurfaceTurbulence::draw(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(2.0);
	glLineWidth(1.0);
	int cnt = 0;
	for (uint i = 0u; i < _numFineParticles; i++)
	{
		REAL3 position = h_Pos[i];
		
		//printf("pos %f %f %f %f\n", position.x , position.y, position.z, position.w);
		cnt++;
		////general visualize
		glColor3f(1.0f, 0.0f, 0.0f);

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

