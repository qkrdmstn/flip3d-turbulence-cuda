#include "FLIPGrid.cuh"

FLIPGRID::FLIPGRID()
{

}

FLIPGRID::FLIPGRID(uint res, REAL cellPhysicalSize):
	_gridRes(res), _gridCellCount((res+1)* (res + 1)* (res + 1)), _cellPhysicalSize(cellPhysicalSize), _physicalSize(res * cellPhysicalSize)
{
	_cudaGridSize = dim3(divup(_gridRes + 1, _cudaBlockSize.x), divup(_gridRes + 1, _cudaBlockSize.y), divup(_gridRes + 1, _cudaBlockSize.z));
	printf("cudaGridSize: %d %d %d\n", _cudaGridSize.x, _cudaGridSize.y, _cudaGridSize.z);

	d_Volumes.content = createField3D<int>(_gridRes + 1, _gridRes + 1, _gridRes + 1, _cudaGridSize, _cudaBlockSize, CONTENT_AIR, false);
	d_Volumes.pressure = createField3D<float>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0.f, false);
	d_Volumes.fluidIndex = createField3D<int>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0, false);
	d_Volumes.divergence = createField3D<float>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0.f, false);
	d_Volumes.particleCount = createField3D<int>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0, false);

	d_Volumes.velocityAccumWeight = createField3D<float4>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, make_float4(0, 0, 0, 0), false);
	d_Volumes.hasVelocity = createField3D<int4>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, make_int4(0, 0, 0, 0), false);


	d_Volumes.velocity = createField3D<float4>(_gridRes + 1, _gridRes + 1, _gridRes + 1, _cudaGridSize, _cudaBlockSize, make_float4(0, 0, 0, 0), true);

	d_Volumes.newVelocity = createField3D<float4>(_gridRes + 1, _gridRes + 1, _gridRes + 1, _cudaGridSize, _cudaBlockSize, make_float4(0, 0, 0, 0), true);


	d_Volumes.volumeFractions = createField3D<float4>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, make_float4(0, 0, 0, 0), false);
	d_Volumes.newVolumeFractions = createField3D<float4>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, make_float4(0, 0, 0, 0), false);

	d_Volumes.density = createField3D<float>(_gridRes, _gridRes, _gridRes, _cudaGridSize, _cudaBlockSize, 0.f, false);

	Init();
}

FLIPGRID::~FLIPGRID()
{
	releaseField3D(d_Volumes.content);
	releaseField3D(d_Volumes.pressure);
	releaseField3D(d_Volumes.fluidIndex);
	releaseField3D(d_Volumes.divergence);
	releaseField3D(d_Volumes.particleCount);

	releaseField3D(d_Volumes.velocityAccumWeight);
	releaseField3D(d_Volumes.hasVelocity);

	releaseField3D(d_Volumes.velocity);
	releaseField3D(d_Volumes.newVelocity);

	releaseField3D(d_Volumes.volumeFractions);
	releaseField3D(d_Volumes.newVolumeFractions);
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