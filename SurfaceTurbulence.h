#ifndef __SURFACETURBULENCE_H__
#define __SURFACETURBULENCE_H__

#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "thrust/sort.h"
#include <GL/freeglut.h>
#include "FLIP3D_Cuda.h"

using namespace std;

__host__ __device__
struct WaveParam
{
	//Wave Simulation Coefficient
	REAL _dt = 0.005;
	REAL _waveSpeed = 8.0;
	REAL _waveDamping = 0.0f;
	REAL _waveSeedFreq = 4.0;
	REAL _waveMaxAmplitude;
	REAL _waveMaxFreq = 400.0;
	REAL _waveMaxSeedingAmplitude = 0.025;
	REAL _waveSeedingCurvatureThresholdCenter;
	REAL _waveSeedingCurvatureThresholdRadius;
	REAL _waveSeedStepSizeRatioOfMax = 0.025;
};

__host__ __device__
struct MaintenanceParam
{
	//Surface Maintenance Coefficient
	REAL _coarseScaleLen;
	REAL _fineScaleLen;
	REAL _outerRadius;
	REAL _innerRadius;
	REAL _normalRadius;

	uint _fineRes;
	uint _coarseRes;

	uint _arrayMaxSize;
};


#define SURFACE_DENSITY 10.0
#define PER_PARTICLE 100

class SurfaceTurbulence
{
public: //Device
	//Particle
	//Initialize
	Dvector<uint> d_ParticleGridIndex;
	Dvector<uint> d_StateData;

	//Surface Maintenance
	Dvector<REAL3> d_Pos;
	Dvector<REAL3> d_Vel;
	Dvector<REAL3> d_SurfaceNormal;
	Dvector<REAL3> d_TempNormal;
	Dvector<REAL3> d_TempPos;
	Dvector<REAL3> d_Tangent;
	Dvector<REAL> d_KernelDens;
	Dvector<REAL> d_NeighborWeightSum;
	Dvector<BOOL> d_Flag;
	
	////Wave Simulation
	Dvector<REAL> d_Curvature;
	Dvector<REAL> d_TempCurvature;
	Dvector<REAL> d_WaveH;
	Dvector<REAL> d_WaveDtH;
	Dvector<REAL> d_Seed;
	Dvector<REAL> d_WaveSeedAmp;
	Dvector<REAL> d_Laplacian;
	Dvector<REAL3> d_WaveNormal;

	//Display Particle
	Dvector<REAL3> d_DisplayPos;

public: //Host
	//Particle
	//Surface Maintenance
	vector<REAL3> h_Pos;
	vector<REAL3> h_Vel;
	vector<REAL3> h_SurfaceNormal;
	vector<REAL3> h_TempNormal;
	vector<REAL3> h_TempPos;
	vector<REAL3> h_Tangent;
	vector<BOOL> h_Flag;

	//Wave Simulation
	vector<REAL> h_Curvature;
	vector<REAL> h_TempCurvature;
	vector<REAL> h_WaveH;
	vector<REAL> h_WaveDtH;
	vector<REAL> h_Seed;
	vector<REAL> h_WaveSeedAmp;
	vector<REAL> h_Laplacian;
	vector<REAL3> h_WaveNormal;

	//Display Particle
	vector<REAL3> h_DisplayPos;

public: //Hash
	Dvector<uint> d_GridHash;
	Dvector<uint> d_GridIdx;
	Dvector<uint> d_CellStart;
	Dvector<uint> d_CellEnd;

public:
	MaintenanceParam maintenanceParam;
	WaveParam waveParam;

public:
	uint _numFineParticles;

public:
	FLIP3D_Cuda* _fluid;
	
public:
	SurfaceTurbulence();
	SurfaceTurbulence(FLIP3D_Cuda* fluid, uint gridRes);
	~SurfaceTurbulence();

public: //Initialize
	void InitMaintenanceParam(uint gridRes);
	void InitWaveParam(void);
	void Initialize_kernel(void);
	void ThrustScanWrapper_kernel(uint* output, uint*input, uint numElements);

	void InsertNewCoarseNeighbor_kernel(void);

public: //Surface Maintenance
	void Advection_kernel(void);
	void SurfaceConstraint_kernel(void);
	void Regularization_kernel(void);
	void InsertFineParticles(void);
	void DeleteFineParticles(void);
	void SurfaceMaintenance(void);

public: //Wave Simulation func
	void AddSeed_kernel(void);
	void ComputeCurvature_kernel(void);
	void SeedWave_kernel(int step);
	void ComputeWaveNormal_kernel(void);
	void ComputeLaplacian_kernel(void);
	void EvolveWave_kernel(void);
	void WaveSimulation_kernel(int step);

public: //Regularization func
	void ComputeSurfaceNormal_kernel(void);
	void NormalRegularization_kernel(void);
	void TangentRegularization_kernel(void);

public:	//Hash
	void SetHashTable_kernel(void);
	void CalculateHash_kernel(void);
	void SortParticle_kernel(void);
	void FindCellStart_kernel(void);

public:		//Cuda
	void InitHostMem(void);
	void InitDeviceMem(void);
	void FreeDeviceMem(void);
	void CopyToDevice(void);
	void CopyToHost(void);
	void ComputeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = divup(n, numThreads);
	}

public:
	void drawFineParticles(void);
	void drawDisplayParticles(void);
	REAL3 ScalarToColor(double val);
	REAL3 VelocityToColor(double val);
};
#endif