#include "Hash.cuh"

__global__ void CalculateHash_D(uint* gridHash, uint* gridIdx, REAL3* pos, uint gridRes, uint numParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numParticles)
		return;

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);
	uint hash = calcGridHash(gridPos, gridRes);

	gridHash[idx] = hash;
	gridIdx[idx] = idx;
}

__global__ void FindCellStart_D(uint* gridHash, uint* cellStart, uint* cellEnd, uint numParticles)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];  // blockSize + 1 elements
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	//if (idx >= numParticles)
	//	return;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (idx < numParticles) {
		hash = gridHash[idx];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (idx > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridHash[idx - 1];
		}
	}

	cg::sync(cta);

	if (idx < numParticles) {

		if (idx == 0 || hash != sharedHash[threadIdx.x]) {
			cellStart[hash] = idx;

			if (idx > 0) cellEnd[sharedHash[threadIdx.x]] = idx;
		}

		if (idx == numParticles - 1) {
			cellEnd[hash] = idx + 1;
		}
	}
}

