#ifndef __FLIP_CUDA_H__
#define __FLIP_CUDA_H__

#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "thrust/sort.h"
#include <GL/freeglut.h>
#include "FLIPGrid.h"
#include "BoundingBox.h"
#include <stdio.h>

#define BLOCK_SIZE 1024

#define AIR		0
#define FLUID	1
#define WALL	2

#define BOX		0
#define SPHERE	1

#define GLASS	1
#define GRAY	2
#define RED		3

#define PI          3.141592

#define MAXPARTICLENUM 50000

#define GRIDRENDER 0
using namespace std;

struct Object 
{
	uint type;
	uint shape;
	uint material;
	REAL r; //Sphere's radius (SPHEREÀÏ °æ¿ì)
	REAL3 c; //Sphere's  center
	REAL3 p[2]; //Box's min, max position
};

class FLIP3D_Cuda
{//particle
public:		//Device
	//Particle
	Dvector<REAL3> d_CurPos;
	Dvector<REAL3> d_BeforePos;
	Dvector<REAL3> d_Vel;
	Dvector<REAL3> d_Normal;
	Dvector<uint> d_Type;
	Dvector<REAL> d_Mass;
	Dvector<REAL> d_Dens;
	Dvector<REAL> d_KernelDens;

	Dvector<BOOL> d_Flag;


#if GRIDRENDER
	//grid visualize
	Dvector<REAL3> d_gridPos;
	Dvector<REAL3> d_gridVel;
	Dvector<REAL> d_gridPress;
	Dvector<REAL> d_gridDens;
	Dvector<REAL> d_gridLevelSet;
	Dvector<REAL> d_gridDiv;
	Dvector<uint> d_gridContent;
#endif
	//OBB
	Dvector<OBB> d_Boxes;

public:		//Host
	//Particle
	vector<REAL3> h_CurPos;
	vector<REAL3> h_BeforePos;
	vector<REAL3> h_Vel;
	vector<REAL3> h_Normal;
	vector<uint> h_Type;
	vector<REAL> h_Mass;
	vector<REAL> h_Dens;
	vector<REAL> h_KernelDens;

	vector<BOOL> h_Flag;

#if GRIDRENDER
	//grid visualize
	vector<REAL3> h_gridPos;
	vector<REAL3> h_gridVel;
	vector<REAL> h_gridPress;
	vector<REAL> h_gridDens;
	vector<REAL> h_gridLevelSet;
	vector<REAL> h_gridDiv;
	vector<uint> h_gridContent;
#endif
	//OBB
	vector<OBB> h_Boxes;

public:		//Hash
	Dvector<uint> d_GridHash;
	Dvector<uint> d_GridIdx;
	Dvector<uint> d_CellStart;
	Dvector<uint> d_CellEnd;

public:
	FLIPGRID* _grid;

public:
	uint _numParticles;
	uint _gridRes;

	REAL _wallThick;
	REAL _dens = 0.5;
	REAL _maxDens = 92.9375;
	REAL3 _externalForce = make_REAL3(0, 0, 0);

	REAL _cellPhysicalSize; //hash table

public:
	vector<Object> objects;
	uint _numBoxes;

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
	void RotateBoxesTest(void);
	void MoveBoxTest(void);
	void PushParticle(REAL x, REAL y, REAL z, uint type);
	void ComputeWallParticleNormal_kernel(void);

public:		//Simulation
	void ResetCell_kernel(void);
	void ComputeParticleDensity_kernel(void);
	void ComputeExternalForce_kernel(REAL3& extForce, REAL dt);
	void CollisionMovingBox_kernel(REAL dt);
	void SolvePICFLIP(void);
	void TrasnferToGrid_kernel(void);
	void MarkWater_kernel(void);
	void EnforceBoundary_kernel(void);
	void InsertFLIPParticles_kernel(/*REAL3* d_newPos, REAL3* d_newVel, REAL* d_newMass, uint numInsertParticles*/);
	void DeleteFLIPParticles_kernel(/*uint* deleteIdxes, uint deletNum*/);
	void ThrustScanWrapper_kernel(uint* output, uint* input, uint numElements);

public:
	//Solver
	void SolvePressure(void);
	void ComputeDivergence_kernel(void);
	void ComputeLevelSet_kernel(void);
	void BuildPreconditioner_kernel(REAL* P, REAL* L, uint* A, uint gridSize, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads);
	void Solver_kernel(uint* A, REAL* P, REAL* L, REAL* x, REAL* b, REAL* r, REAL* z, REAL* s, uint size, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads);
	void Op_Kernel(uint* A, REAL* x, REAL* y, REAL* ans, REAL a, uint size, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads);
	void DotHost(uint* A, REAL* x, REAL* y, uint size, REAL* result, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads);
	void Apply_Preconditioner(REAL* z, REAL* r, REAL* P, REAL* L, uint* A, uint size, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads);
	void ComputeVelocityWithPress_kernel(void);

	void ExtrapolateVelocity_kernel(void);
	void SubtarctGrid_kernel(void);
	void TrasnferToParticle_kernel(void);
	void AdvectParticle_kernel(REAL dt);

	void Correct_kernel(REAL dt);


public:	//Hash
	void SetHashTable_kernel(void);
	void CalculateHash_kernel(void);
	void SortParticle_kernel(void);
	void FindCellStart_kernel(void);

public:	//Cuda
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
	void GridValueVisualize(void);
	void draw(void);
	void drawOBB(void);
	REAL3 ScalarToColor(double val);

};
#endif
