#ifndef __FLIPGRID_H__
#define __FLIPGRID_H__

#include <iostream>
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "thrust/sort.h"
#include "VolumeData.cuh"
#include "surface_indirect_functions.h"

using namespace std;

#define CONTENT_AIR		0
#define CONTENT_FLUID	1
#define CONTENT_WALL	2

#define get3D(arr, x, y, z) arr[(x)*(sizeY + 1)*(sizeZ+1)+(y)*(sizeZ+1)+(z)]

__host__ __device__
struct VolumeCollection {
	VolumeData content;
	VolumeData pressure;
	VolumeData fluidIndex;
	VolumeData divergence;
	VolumeData particleCount;

	VolumeData velocityAccumWeight;
	VolumeData hasVelocity;

	VolumeData velocity;
	VolumeData newVelocity;

	VolumeData volumeFractions;
	VolumeData newVolumeFractions;

	VolumeData density;
};

class FLIPGRID
{//particle
public:		
	VolumeCollection d_Volumes;

public:
	uint _gridRes;
	uint _gridCellCount;
	REAL _cellPhysicalSize;
	REAL _physicalSize;

public:
	dim3 _cudaGridSize;
	dim3 _cudaBlockSize = dim3(8, 8, 8);
	
public:
	FLIPGRID();
	FLIPGRID(uint res, REAL cellPhysicalSize);
	~FLIPGRID();

public:
	void Init(void);


public:
	void InitDeviceMem(void);
	void FreeDeviceMem(void);
	void CopyToDevice(void);
	void CopyToHost(void);
};

#endif