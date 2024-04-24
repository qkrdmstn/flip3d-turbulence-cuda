#ifndef __SURFACETURBULENCE_H__
#define __SURFACETURBULENCE_H__

#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "thrust/sort.h"
#include <GL/freeglut.h>
#include "FLIP3D_Cuda.h"

using namespace std;

#define SURFACE_DENSITY 20.0
#define PER_PARTICLE 10

class SurfaceTurbulence
{
public: //Device
	//Particle
	//Surface Maintenance
	Dvector<REAL3> d_Pos;
	Dvector<REAL3> d_Vel;
	Dvector<REAL3> d_SurfaceNormal;
	Dvector<REAL3> d_TempNormal;
	Dvector<REAL3> d_TempPos;
	Dvector<REAL3> d_Tangent;
	Dvector<REAL> d_KernelDens;
	
	//Wave Simulation
	Dvector<REAL> d_Curvature;
	Dvector<REAL> d_TempCurvature;
	Dvector<REAL> d_WaveH;
	Dvector<REAL> d_WaveDtH;
	Dvector<REAL> d_Seed;
	Dvector<REAL> d_WaveSeedAmp;
	Dvector<REAL> d_Laplacian;
	Dvector<REAL3> d_WaveNormal;

public: //Host
	//Particle
	//Surface Maintenance
	vector<REAL3> h_Pos;
	vector<REAL3> h_Vel;
	vector<REAL3> h_SurfaceNormal;
	vector<REAL3> h_TempNormal;
	vector<REAL3> h_TempPos;
	vector<REAL3> h_Tangent;
	vector<REAL> h_KernelDens;

	//Wave Simulation
	vector<REAL> h_Curvature;
	vector<REAL> h_TempCurvature;
	vector<REAL> h_WaveH;
	vector<REAL> h_WaveDtH;
	vector<REAL> h_Seed;
	vector<REAL> h_WaveSeedAmp;
	vector<REAL> h_Laplacian;
	vector<REAL3> h_WaveNormal;

public: //Hash
	Dvector<uint> d_GridHash;
	Dvector<uint> d_GridIdx;
	Dvector<uint> d_CellStart;
	Dvector<uint> d_CellEnd;


public: // Surface Maintenance Coefficient
	REAL _coarseScaleLen;
	REAL _fineScaleLen;
	REAL outerRadius;
	REAL innerRadius;

public:
	//Wave Simulation Coefficient
	REAL dt = 0.00125;
	REAL waveSpeed = 8.0;
	REAL waveSeedFreq = 2.0;
	REAL waveMaxAmplitude = 0.025;
	REAL waveMaxFreq = 400.0;
	REAL waveMaxSeedingAmplitude = 0.05;
	REAL waveSeedingCurvatureThresholdMinimum;
	REAL waveSeedingCurvatureThresholdMaximum;
	REAL waveSeedStepSizeRatioOfMax = 0.025;

public:
	Dvector<uint> d_NumFineParticles;
	vector<uint> h_NumFineParticles;

public:
	FLIP3D_Cuda* _fluid;
	uint _baseRes;

public:
	SurfaceTurbulence();
	SurfaceTurbulence(FLIP3D_Cuda* fluid, uint gridRes) {
		_fluid = fluid;

		_baseRes = gridRes;
		_coarseScaleLen = 1.0 / gridRes;
		_fineScaleLen = PI * (_coarseScaleLen + (_coarseScaleLen / 2.0)) / SURFACE_DENSITY;
		outerRadius = _coarseScaleLen;
		innerRadius = outerRadius / 2.0;

		waveSeedingCurvatureThresholdMinimum = _coarseScaleLen * 0.005; //°î·ü ÀÓ°è°ª (Á¶Á¤ ÇÊ¿ä)
		waveSeedingCurvatureThresholdMaximum = _coarseScaleLen * 0.077;

		InitHostMem();
		InitDeviceMem();
		CopyToDevice();

		Initialize_kernel();
		printf("Coarse Scale Length: %f\n", _coarseScaleLen);
		printf("Fine Scale Length: %f\n", _fineScaleLen);

		CopyToHost();
		printf("Initialize coarse-particles number is %d\n", _fluid->_numParticles);
		printf("Initialize fine-particles number is %d\n", h_NumFineParticles[0]);
	}
	~SurfaceTurbulence();

public:
	void Initialize_kernel(void);

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

};
#endif