#pragma once
#include "Vec3.h"
#include <vector>
#include <math.h>
#include "Particle.h"

using namespace std;

class Hash
{
public:
	double spacing;
	int tableSize;
	int numParticles;

	vector<int> cellStart; //cell Index 스타트 배열
	vector<Particle *> cellEntries; //파티클 배열

public:
	Hash();
	Hash(double _spacing, int _numParticles);
	~Hash();

public:
	int intCoord(double coord);
	int hashCoords(int xi, int yi, int zi);
	int hashPos(vec3 pos);
	void create(vector<Particle*> _fineParticles);
	vector<Particle*> query(vec3 pos, double maxDist);
};