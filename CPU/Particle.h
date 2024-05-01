#ifndef __PARTICLE_H__
#define __PARTICLE_H__


#pragma once
#include "Vec3.h"

class Particle
{
public:
	//Surface Maintenance
	vec3 _pos;
	vec3 _surfaceNormal;
	vec3 _tempSurfaceNormal;
	vec3 _tangent;
	vec3 _tempPos;
	double _kernelDens;

public:
	vec3 _vel;

public: 
	//Wave Simulation
	double _curvature;
	double _tempCurvature;
	double _waveH;
	double _waveDtH;
	double _seed;
	double _waveSeedAmplitude;
	double _laplacian;

	vec3 _waveNormal;
public:
	//Visualize
	vec3 _prevPos; //시각화를 위한 이전 frame Position
	bool _flag = false;

public:
	Particle(vec3 pos, bool flag)
	{
		_pos = pos;
		_vel = vec3(0, 0, 0);
		_surfaceNormal = vec3(0, 0, 0);
		_kernelDens = 0;

		_curvature = 0;
		_waveH = 0;
		_waveDtH = 0;
		_seed = 0;
		_waveSeedAmplitude = 0;
		_laplacian = 0;

		_flag = flag;
	}
};
#endif