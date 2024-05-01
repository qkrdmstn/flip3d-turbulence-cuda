#ifndef __SURFACETURBULENCE_H__
#define __SURFACETURBULENCE_H__

#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "Particle.h"
#include "GL\glut.h"
#include "sorter.h"
#include "common.h"
#include "Vec3.h"
#include "Hash.h"

using namespace std;

#define SURFACE_DENSITY 20.0

class SurfaceTurbulencen
{
public:
	double _coarseScaleLen;
	double _fineScaleLen;
	double outerRadius;
	double innerRadius;

	int deleteInfo = 0;
	int insertInfo = 0;


public:
	double dt = 0.00125;
	double waveSpeed = 8.0;
	double waveSeedFreq = 2.0;
	double waveMaxAmplitude = 0.025;
	double waveMaxFreq = 400.0;
	double waveMaxSeedingAmplitude = 0.05;
	double waveSeedingCurvatureThresholdMinimum;
	double waveSeedingCurvatureThresholdMaximum;
	double waveSeedStepSizeRatioOfMax = 0.025;

public:
	vector<Particle *>	_fineParticles; // 입자들
	vector<particle *>	_coarseParticles;
	vector<Particle*>	_displayParticles;
	Sorter* _sorter; //coarse particle hash
	Hash* _hash; //fine particle hash


public:
	SurfaceTurbulencen(double _cellNum, vector<particle*>& _coarse, Sorter *_s)
	{
		_coarseParticles = _coarse;
		_coarseScaleLen = 1.0 / _cellNum;
		_fineScaleLen = PI * (_coarseScaleLen + (_coarseScaleLen / 2.0)) / SURFACE_DENSITY;
		outerRadius = _coarseScaleLen;
		innerRadius = outerRadius / 2.0;

		waveSeedingCurvatureThresholdMinimum = _coarseScaleLen * 0.005; //곡률 임계값 (조정 필요)
		waveSeedingCurvatureThresholdMaximum = _coarseScaleLen * 0.077;

		_sorter = _s;
		Initialize();

		_hash = new Hash(_fineScaleLen, _fineParticles.size());
		printf("Coarse Scale Length: %f\n", _coarseScaleLen);
		printf("Fine Scale Length: %f\n", _fineScaleLen);
	}

public: //Surface Maintenance func
	void Initialize(void);
	void Advection(void);
	void SurfaceConstarint(void);
	void Regularization(void);
	void InsertDeleteFineParticles(void);
	void SurfaceMaintenance(void);
	
public: //Wave Simulation func
	void ComputeCurvature(void);
	void SmoothCurvature(void);
	void SeedWave(int step);
	void EvolveWave(void);
	void WaveSimulation(int step);

public:
	void SetDisplayParticles(void);

public:
	//Weight func
	void ComputeFineDensKernel(double r);
	void ComputeCoarseDensKernel(double r);
	double DistKernel(Vec3<double> diff, double r);
	double NeighborWeight(Particle* p1, Particle* p2, double r, vector<Particle*> neighbors); //Fine-Fine 가중치
	double NeighborWeight(Particle* p1, particle* p2, double r, vector<particle*> neighbors); //Fine-Coarse 가중치
	vector<particle*> GetNeighborCoarseParticles(vec3 pos, int w, int h, int d);
	vector<Particle*> GetNeighborFineParticles(vec3 pos, double maxDist);

	//Metaball func
	vec3 MetaballConstraintGradient(Particle* p1, vector<particle*> neighbors);
	double MetaballDens(double dist); //metaball 밀도
	double MetaballLevelSet(Particle* p1, vector<particle*> neighbors);

	//Regularization func
	void ComputeSurfaceNormal(void);
	void SmoothNormal(void);
	void NormalRegularization(void);
	void TangentRegularization(void);

	//Wave Seed func
	void ComputeWaveNormal(void);
	void ComputeLaplacian(void);
	double SmoothStep(double left, double right, double val);

	//Simulation Bound
	bool IsInDomain(vec3 pos);
public:
	void drawFineParticles(void);
	void drawDisplayParticles(void);
	vec3 ScalarToColor(double val);
};
#endif