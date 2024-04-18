#ifndef __FLIP_CUDA_H__
#define __FLIP_CUDA_H__

#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "thrust/sort.h"
#include <GL/freeglut.h>
#include "FLIPGrid.h"

#define BLOCK_SIZE 1024

#define AIR		0
#define FLUID	1
#define WALL	2

#define BOX		0
#define SPHERE	1

#define GLASS	1
#define GRAY	2
#define RED		3

#define PI          3.14159265

using namespace std;

struct Object {
	uint type;
	uint shape;
	uint material;
	BOOL visible;
	REAL r; //Sphere's radius (SPHEREÀÏ °æ¿ì)
	REAL3 c; //Sphere's  center
	REAL3 p[2]; //Box's min, max position
};

class FLIP3D_Cuda
{//particle
public:		//Device
	//Particle
	Dvector<REAL3> d_BeforePos;
	Dvector<REAL3> d_CurPos;
	Dvector<REAL3> d_Vel;
	Dvector<REAL3> d_Normal;
	Dvector<uint> d_Type;
	Dvector<uint> d_Visible;
	Dvector<uint> d_Remove;
	Dvector<REAL> d_Mass;
	Dvector<REAL> d_Dens;

	Dvector<BOOL> d_Flag;

	//grid visualize
	Dvector<REAL3> d_gridPos;
	Dvector<REAL3> d_gridVel;
	Dvector<REAL> d_gridPress;
	Dvector<REAL> d_gridDens;
	Dvector<REAL> d_gridLevelSet;
	Dvector<uint> d_gridContent;

public:		//Hash
	Dvector<uint> d_GridHash;
	Dvector<uint> d_GridIdx;
	Dvector<uint> d_CellStart;
	Dvector<uint> d_CellEnd;

public:		//Host
	//Particle
	vector<REAL3> h_BeforePos;
	vector<REAL3> h_CurPos;
	vector<REAL3> h_Vel;
	vector<REAL3> h_Normal;
	vector<uint> h_Type;
	vector<uint> h_Visible;
	vector<uint> h_Remove;
	vector<REAL> h_Mass;
	vector<REAL> h_Dens;
	
	vector<BOOL> h_Flag;

	//grid visualize
	vector<REAL3> h_gridPos;
	vector<REAL3> h_gridVel;
	vector<REAL> h_gridPress;
	vector<REAL> h_gridDens;
	vector<REAL> h_gridLevelSet;
	vector<uint> h_gridContent;
public:
	FLIPGRID* _grid;

public:
	uint _iterations = 100u;
	uint _numParticles;
	uint _gridRes;

	REAL _wallThick;
	REAL _dens;
	REAL _maxDens = 92.9375;
	REAL3 _externalForce = make_REAL3(0, 0, 0);

	REAL _cellPhysicalSize; //hash table


public:
	vector<Object> objects;

public:
	FLIP3D_Cuda();
	FLIP3D_Cuda(uint res)
		: _gridRes(res)
	{
		Init();
	}
	~FLIP3D_Cuda();

public:		//Initialize
	void Init(void);
	void ParticleInit(void);
	void PlaceObjects(void);
	void PlaceWalls(void);
	void WaterDropTest(void);
	void DamBreakTest(void);
	void PushParticle(REAL x, REAL y, REAL z, uint type);
	void ComputeWallNormal(void);

public:		//Simulation
	void ResetCell_kernel(void);
	void ComputeParticleDensity_kernel(void);
	void ComputeExternalForce_kernel(REAL3& extForce, REAL dt);
	void SolvePICFLIP(void);
	void TrasnferToGrid_kernel(void);
	void MarkWater_kernel(void);
	void EnforceBoundary_kernel(void);
	void ComputeDivergence_kernel(void);
	void ComputeLevelSet_kernel(void);
	void ComputeGridDensity_kernel(void);
	void SolvePressureJacobi_kernel(void);
	void ComputeVelocityWithPress_kernel(void);
	void ExtrapolateVelocity_kernel(void);
	void SubtarctGrid_kernel(void);
	void TrasnferToParticle_kernel(void);

	void AdvectParticle_kernel(REAL dt);

public:	//Hash
	void SetHashTable_kernel(void);
	void CalculateHash_kernel(void);
	void SortParticle_kernel(void);
	void FindCellStart_kernel(void);

public:		//Cuda
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
	void gridValueVisualize(void);
	void draw(void);
	REAL3 ScalarToColor(double val);
};
#endif
