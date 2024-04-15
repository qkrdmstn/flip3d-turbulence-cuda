#include "FLIPGrid.cuh"

FLIPGRID::FLIPGRID()
{

}

FLIPGRID::FLIPGRID(uint res, REAL cellPhysicalSize):
	_gridRes(res), _gridCellCount((res+1)* (res + 1)* (res + 1)), _cellPhysicalSize(cellPhysicalSize), _physicalSize(res * cellPhysicalSize)
{
	_cudaGridSize = dim3(divup(_gridRes + 1, _cudaBlockSize.x), divup(_gridRes + 1, _cudaBlockSize.y), divup(_gridRes + 1, _cudaBlockSize.z));
	printf("cudaGridSize: %d %d %d\n", _cudaGridSize.x, _cudaGridSize.y, _cudaGridSize.z);

	d_Volumes.vel = createField3D<REAL4>(_gridRes + 1, _gridRes + 1, _gridRes + 1, _cudaGridSize, _cudaBlockSize, make_REAL4(0, 0, 0, 0), true);
	d_Volumes.velSave = createField3D<REAL4>(_gridRes + 1, _gridRes + 1, _gridRes + 1, _cudaGridSize, _cudaBlockSize, make_REAL4(0, 0, 0, 0), true);
	d_Volumes.hasVel = createField3D<uint4>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, make_uint4(0, 0, 0, 0), false);

	d_Volumes.content = createField3D<uint>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, CONTENT_AIR, false);
	d_Volumes.levelSet = createField3D<REAL>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0.0, false);
	d_Volumes.press = createField3D<REAL>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0.0, false);
	d_Volumes.divergence = createField3D<REAL>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0.0, false);

	d_Volumes.wallNormal = createField3D<REAL4>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, make_REAL4(0, 0, 0, 0), false);

	Init();
}

FLIPGRID::~FLIPGRID()
{
	releaseField3D(d_Volumes.vel);
	releaseField3D(d_Volumes.velSave);
	releaseField3D(d_Volumes.content);
	releaseField3D(d_Volumes.levelSet);
	releaseField3D(d_Volumes.press);
	releaseField3D(d_Volumes.divergence);
	releaseField3D(d_Volumes.hasVel);
	releaseField3D(d_Volumes.wallNormal);
}


void FLIPGRID::Init(void)
{
}

void FLIPGRID::InitDeviceMem(void)
{

}

void FLIPGRID::FreeDeviceMem(void)
{

}

void FLIPGRID::CopyToDevice(void)
{

}

void FLIPGRID::CopyToHost(void)
{

}