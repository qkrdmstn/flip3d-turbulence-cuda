#include "MarchingCubesCuda.h"
#include "Hash.cuh"

void MarchingCubes_CUDA::SetHashTable_kernel(void)
{
	CalculateHash_kernel();
	SortParticle_kernel();
	FindCellStart_kernel();
}

void MarchingCubes_CUDA::CalculateHash_kernel(void)
{
	CalculateHash_D << <divup(_numTotalParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_GridHash, d_GridIdx, d_TotalParticles, _gridSize.x, _numTotalParticles);
}

void MarchingCubes_CUDA::SortParticle_kernel(void)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(d_GridHash),
		thrust::device_ptr<uint>(d_GridHash + _numTotalParticles),
		thrust::device_ptr<uint>(d_GridIdx));
}

void ComputeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = divup(n, numThreads);
}

void MarchingCubes_CUDA::FindCellStart_kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numTotalParticles, 128, numBlocks, numThreads);

	uint smemSize = sizeof(uint) * (numThreads + 1);
	FindCellStart_D << <numBlocks, numThreads, smemSize >> >
		(d_GridHash, d_CellStart, d_CellEnd, _numTotalParticles);
}

