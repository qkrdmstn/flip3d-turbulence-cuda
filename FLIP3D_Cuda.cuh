#ifndef __FLIP_CUDA_CUH__
#define __FLIP_CUDA_CUH__

#include "FLIP3D_Cuda.h"
#include <cmath>
#include "Hash.cuh"
#include "WeightKernels.cuh"

__device__ REAL LevelSet(int3 gridPos, REAL3* pos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	uint neighHash = calcGridHash(gridPos, gridRes);
	uint startIdx = cellStart[neighHash];

	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(gridPos.x + 0.5, gridPos.y + 0.5, gridPos.z + 0.5) * cellSize;

	REAL accm = 0.0;
	if (startIdx != 0xffffffff)
	{
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];

			REAL3 dist = pos[sortedIdx] - centerPos;
			REAL d2 = LengthSquared(dist);
			if (d2 > cellSize * cellSize)
				continue;


			if (type[sortedIdx] == FLUID)
				accm += dens[sortedIdx];
			else
				return 1.0;
		}
	}
	REAL n0 = 1.0 / (densVal * densVal * densVal);
	return 0.2 * n0 - accm;
}

__global__ void ComputeWallParticleNormal_D(REAL3* pos, uint* type, REAL3* normal, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numParticles, uint gridRes)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	REAL cellSize = 1.0 / gridRes;
	REAL wallThick = 1.0 / gridRes;
	REAL3 position = pos[idx];

	REAL3 normalVector = make_REAL3(0.0f, 0.0f, 0.0f);
	if (type[idx] == WALL) {
		if (position.x <= 1.1 * wallThick) {
			normalVector.x = 1.0;
		}
		if (position.x >= 1.0 - 1.1 * wallThick) {
			normalVector.x = -1.0;
		}
		if (position.y <= 1.1 * wallThick) {
			normalVector.y = 1.0;
		}
		if (position.y >= 1.0 - 1.1 * wallThick) {
			normalVector.y = -1.0;
		}
		if (position.z <= 1.1 * wallThick) {
			normalVector.z = 1.0;
		}
		if (position.z >= 1.0 - 1.1 * wallThick) {
			normalVector.z = -1.0;
		}

		if (normalVector.x == 0.0f && normalVector.y == 0.0f && normalVector.z == 0.0f) {
			int3 gridPos = calcGridPos(position, cellSize);
			FOR_NEIGHBOR(3) {

				int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
				uint neighHash = calcGridHash(neighbourPos, gridRes);
				uint startIdx = cellStart[neighHash];

				if (startIdx != 0xffffffff)
				{
					uint endIdx = cellEnd[neighHash];
					for (uint i = startIdx; i < endIdx; i++)
					{
						uint sortedIdx = gridIdx[i];
						if (sortedIdx != idx && type[sortedIdx] == WALL) {
							REAL d = Length(position - pos[sortedIdx]);
							REAL w = 1.0 / d;
							normalVector += w * (position - pos[sortedIdx]) / d;
						}

					}
				}
			}END_FOR;
		}
	}

	Normalize(normalVector);
	normal[idx] = normalVector;
}

__global__ void ResetCell_D(VolumeCollection volumes, uint gridRes) {

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	volumes.content.writeSurface<uint>(CONTENT_AIR, x, y, z);

	volumes.hasVel.writeSurface<uint4>(make_uint4(0, 0, 0, 0), x, y, z);
	volumes.vel.writeSurface<REAL4>(make_REAL4(0, 0, 0, 0), x, y, z);
	volumes.velSave.writeSurface<REAL4>(make_REAL4(0, 0, 0, 0), x, y, z);

}

__global__ void ComputeParticleDensity_D(REAL3* pos, uint* type, REAL* dens, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, uint numParticles, REAL densVal, REAL maxDens)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	if (type[idx] == WALL) {
		dens[idx] = 1.0;
		return;
	}

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);

	REAL wsum = 0.0;
	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (sortedIdx != idx)
				{
					if (type[sortedIdx] == WALL)
						continue;
					
					REAL3 dist = pos[sortedIdx] - pos[idx];
					REAL d2 = LengthSquared(dist);
					//if (d2 > cellSize * cellSize)
					//	continue;

					REAL w = mass[sortedIdx] * SmoothKernel(d2, 4.0 * densVal / gridRes);
					wsum += w;
				}
			}
		}
	} END_FOR;
	dens[idx] = wsum / maxDens;
}

__global__ void CompExternlaForce_D(REAL3* pos, REAL3* vel, REAL3 gravity, REAL3 ext, uint numParticles, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	REAL3 v = vel[idx];
	v += gravity * dt;
	v += ext * dt;
	vel[idx] = v;
}

__global__ void CollisionMovingBox_D(OBB* boxes, REAL3* _pos, REAL3* _vel, uint* type, uint numParticles, uint numBoxes, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID) return;

	REAL delta_tau = 0.01;
	REAL new_phi = 0.0, phi = 0.0;
	REAL friction_coeff = 0.3;
	REAL collision_objects_normal_variation = 0.0;
	REAL particle_normal_variation = 0.0;
	REAL new_particle_noraml_variaion = 0.0;

	REAL3 velocity_collision_objects = make_REAL3(0, 0, 0);
	REAL3 collision_objects_tangential_variation = make_REAL3(0, 0, 0);
	REAL3 particle_tangential_variation = make_REAL3(0, 0, 0);
	REAL3 relative_tangential_vel = make_REAL3(0, 0, 0);
	REAL3 new_relative_tangential_vel = make_REAL3(0, 0, 0);
	REAL3 new_particle_tangential_vel = make_REAL3(0, 0, 0);
	REAL3 collision_normal = make_REAL3(0, 0, 0);
	REAL offset = 0.01;

	for (int i = 0; i < numBoxes; i++) {

		for (int j = 0; j < 20; j++) {
			REAL3 box_vel = boxes[i]._center - boxes[i]._center0;

			REAL3 pos = _pos[idx];
			phi = getDist(boxes[i], pos);
			REAL3 vel = _vel[idx];

			collision_normal.x = getDist(boxes[i], make_REAL3(pos.x + offset, pos.y, pos.z)) - getDist(boxes[i], make_REAL3(pos.x - offset, pos.y, pos.z));
			collision_normal.y = getDist(boxes[i], make_REAL3(pos.x, pos.y + offset, pos.z)) - getDist(boxes[i], make_REAL3(pos.x, pos.y - offset, pos.z));
			collision_normal.z = getDist(boxes[i], make_REAL3(pos.x, pos.y, pos.z + offset)) - getDist(boxes[i], make_REAL3(pos.x, pos.y, pos.z - offset));
			Normalize(collision_normal);

			new_phi = phi + Dot(((vel - box_vel) * dt), collision_normal);
			if (new_phi < 0.0f)
			{
				collision_objects_normal_variation = Dot(velocity_collision_objects, collision_normal);
				particle_normal_variation = Dot(vel, collision_normal);
				collision_objects_tangential_variation = velocity_collision_objects - (collision_normal * collision_objects_normal_variation);
				particle_tangential_variation = vel - (collision_normal * particle_normal_variation);
				new_particle_noraml_variaion = particle_normal_variation - new_phi / delta_tau;
				relative_tangential_vel = particle_tangential_variation - collision_objects_tangential_variation;
				new_relative_tangential_vel = relative_tangential_vel
					* fmax(0.0, 1.0 - (friction_coeff * ((new_particle_noraml_variaion - particle_normal_variation) / Length(relative_tangential_vel))));

				new_particle_tangential_vel = collision_objects_tangential_variation + new_relative_tangential_vel;
				REAL3 new_vel = (collision_normal * new_particle_noraml_variaion) + new_particle_tangential_vel;
				_vel[idx] = new_vel;

				//_pos[idx] = make_REAL3(-1, -1, -1);
			}
		}
	}
}

__global__ void CollisionMovingSphere_D(BoundingSphere* spheres, REAL3* _pos, REAL3* _vel, uint* type, uint numParticles, uint numSphere, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID) return;

	REAL delta_tau = 0.01;
	REAL new_phi = 0.0, phi = 0.0;
	REAL friction_coeff = 0.3;
	REAL collision_objects_normal_variation = 0.0;
	REAL particle_normal_variation = 0.0;
	REAL new_particle_noraml_variaion = 0.0;

	REAL3 velocity_collision_objects = make_REAL3(0, 0, 0);
	REAL3 collision_objects_tangential_variation = make_REAL3(0, 0, 0);
	REAL3 particle_tangential_variation = make_REAL3(0, 0, 0);
	REAL3 relative_tangential_vel = make_REAL3(0, 0, 0);
	REAL3 new_relative_tangential_vel = make_REAL3(0, 0, 0);
	REAL3 new_particle_tangential_vel = make_REAL3(0, 0, 0);
	REAL3 collision_normal = make_REAL3(0, 0, 0);
	REAL offset = 0.01;

	for (int i = 0; i < numSphere; i++) {

		for (int j = 0; j < 20; j++) {
			REAL3 box_vel = spheres[i]._center - spheres[i]._center0;

			REAL3 pos = _pos[idx];
			phi = getDist(spheres[i], pos);
			REAL3 vel = _vel[idx];

			collision_normal.x = getDist(spheres[i], make_REAL3(pos.x + offset, pos.y, pos.z)) - getDist(spheres[i], make_REAL3(pos.x - offset, pos.y, pos.z));
			collision_normal.y = getDist(spheres[i], make_REAL3(pos.x, pos.y + offset, pos.z)) - getDist(spheres[i], make_REAL3(pos.x, pos.y - offset, pos.z));
			collision_normal.z = getDist(spheres[i], make_REAL3(pos.x, pos.y, pos.z + offset)) - getDist(spheres[i], make_REAL3(pos.x, pos.y, pos.z - offset));
			Normalize(collision_normal);

			new_phi = phi + Dot(((vel - box_vel) * dt), collision_normal);
			if (new_phi < 0.0f)
			{
				collision_objects_normal_variation = Dot(velocity_collision_objects, collision_normal);
				particle_normal_variation = Dot(vel, collision_normal);
				collision_objects_tangential_variation = velocity_collision_objects - (collision_normal * collision_objects_normal_variation);
				particle_tangential_variation = vel - (collision_normal * particle_normal_variation);
				new_particle_noraml_variaion = particle_normal_variation - new_phi / delta_tau;
				relative_tangential_vel = particle_tangential_variation - collision_objects_tangential_variation;
				new_relative_tangential_vel = relative_tangential_vel
					* fmax(0.0, 1.0 - (friction_coeff * ((new_particle_noraml_variaion - particle_normal_variation) / Length(relative_tangential_vel))));

				new_particle_tangential_vel = collision_objects_tangential_variation + new_relative_tangential_vel;
				REAL3 new_vel = (collision_normal * new_particle_noraml_variaion) + new_particle_tangential_vel;
				_vel[idx] = new_vel;
			}
		}
	}
}

__global__ void InsertFLIPParticles_D(REAL3* curPos, REAL3* beforePos, REAL3* vel, REAL3* normal, uint* type, REAL* mass, REAL* dens, REAL* kernelDens, BOOL* flag, uint numParticles, REAL maxDens, REAL3* newPos, REAL3* newVel, REAL* newMass, uint numInsert)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numInsert)
		return;

	REAL3 _newPos = newPos[idx];
	REAL3 _newVel = newVel[idx];
	REAL _newMass = newMass[idx];

	uint insertIdx = numParticles + idx;
	beforePos[insertIdx] = _newPos;
	curPos[insertIdx] = _newPos;
	vel[insertIdx] = _newVel;
	normal[insertIdx] = make_REAL3(0.0, 0.0, 0.0);
	dens[insertIdx] = maxDens;
	type[insertIdx] = FLUID;
	mass[insertIdx] = _newMass;
	kernelDens[insertIdx] = 0.0;
	flag[insertIdx] = false;
}

__global__ void InitParticleIdx_D(uint* particleIdx, uint numParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < numParticles)
		particleIdx[idx] = idx;
	else if (idx < MAXPARTICLENUM)
		particleIdx[idx] = MAXPARTICLENUM;
	else
		return;
}

__global__ void StateCheck_D(REAL3* curPos, uint* stateData)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= MAXPARTICLENUM)
		return;


	if (curPos[idx].x < -0.5 && curPos[idx].y < -0.5 && curPos[idx].z < -0.5)
	{
		stateData[idx] = 0;
	}
	else
	{
		stateData[idx] = 1;
	}
}

__global__ void DeleteFLIPParticles_D(uint* particleIdx, REAL3* curPos, REAL3* beforePos, REAL3* vel, REAL3* normal, uint* type, REAL* mass, REAL* dens, REAL* kernelDens, BOOL* flag, uint numParticles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;

	if (curPos[idx].x < -0.5 && curPos[idx].y < -0.5 && curPos[idx].z < -0.5)
	{
		beforePos[idx] = make_REAL3(-1.0, -1.0, -1.0);
		curPos[idx] = make_REAL3(-1.0, -1.0, -1.0);
		vel[idx] = make_REAL3(0.0, 0.0, 0.0);
		normal[idx] = make_REAL3(0.0, 0.0, 0.0);
		dens[idx] = 0.0f;
		type[idx] = FLUID;
		mass[idx] = 0.0f;
		kernelDens[idx] = 0.0;
		flag[idx] = false;

		particleIdx[idx] = MAXPARTICLENUM;
	}
}



__global__ void TransferToGrid_D(VolumeCollection volumes, REAL3* pos, REAL3* vel, uint* type, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, uint numParticles)
{
	int3 gridPos;
	gridPos.x = blockIdx.x * blockDim.x + threadIdx.x;
	gridPos.y = blockIdx.y * blockDim.y + threadIdx.y;
	gridPos.z = blockIdx.z * blockDim.z + threadIdx.z;

	if (gridPos.x >= gridRes || gridPos.y >= gridRes || gridPos.z >= gridRes) return;

	int cellCount = (gridRes) * (gridRes) * (gridRes);
	REAL cellPhysicalSize = 1.0 / gridRes;

	REAL3 xVelocityPos = make_REAL3(gridPos.x, (gridPos.y + 0.5), (gridPos.z + 0.5)) * cellPhysicalSize;
	REAL3 yVelocityPos = make_REAL3((gridPos.x + 0.5), gridPos.y, (gridPos.z + 0.5)) * cellPhysicalSize;
	REAL3 zVelocityPos = make_REAL3((gridPos.x + 0.5), (gridPos.y + 0.5), gridPos.z) * cellPhysicalSize;

	uint4 hasVelocity = volumes.hasVel.readSurface<uint4>(gridPos.x, gridPos.y, gridPos.z);
	REAL4 velocity = make_REAL4(0, 0, 0, 0);
	REAL4 weight = make_REAL4(0, 0, 0, 0);

	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];

				if (type[sortedIdx] == WALL)
					continue;

				REAL3 pPosition = pos[sortedIdx];
				REAL3 pVelocity = vel[sortedIdx];
				REAL thisWeightX = mass[sortedIdx] * trilinearHatKernel(pPosition - xVelocityPos, cellPhysicalSize);
				REAL thisWeightY = mass[sortedIdx] * trilinearHatKernel(pPosition - yVelocityPos, cellPhysicalSize);
				REAL thisWeightZ = mass[sortedIdx] * trilinearHatKernel(pPosition - zVelocityPos, cellPhysicalSize);

				velocity.x += thisWeightX * pVelocity.x;
				velocity.y += thisWeightY * pVelocity.y;
				velocity.z += thisWeightZ * pVelocity.z;

				weight.x += thisWeightX;
				weight.y += thisWeightY;
				weight.z += thisWeightZ;
			}
		}
	}END_FOR;

	if (weight.x > 0) {
		velocity.x /= weight.x;
		hasVelocity.x = true;
	}

	if (weight.y > 0) {
		velocity.y /= weight.y;
		hasVelocity.y = true;
	}

	if (weight.z > 0) {
		velocity.z /= weight.z;
		hasVelocity.z = true;
	}

	volumes.vel.writeSurface<REAL4>(velocity, gridPos.x, gridPos.y, gridPos.z);
	volumes.velSave.writeSurface<REAL4>(velocity, gridPos.x, gridPos.y, gridPos.z);
	volumes.hasVel.writeSurface<uint4>(hasVelocity, gridPos.x, gridPos.y, gridPos.z);
}

__global__ void MarkWater_D(VolumeCollection volumes, REAL3* pos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	int3 gridPos;
	gridPos.x = blockIdx.x * blockDim.x + threadIdx.x;
	gridPos.y = blockIdx.y * blockDim.y + threadIdx.y;
	gridPos.z = blockIdx.z * blockDim.z + threadIdx.z;

	if (gridPos.x >= gridRes || gridPos.y >= gridRes || gridPos.z >= gridRes) return;

	volumes.content.writeSurface<uint>(CONTENT_AIR, gridPos.x, gridPos.y, gridPos.z);

	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(gridPos.x + 0.5, gridPos.y + 0.5, gridPos.z + 0.5) * cellSize;

	uint neighHash = calcGridHash(gridPos, gridRes);
	uint startIdx = cellStart[neighHash];
	if (startIdx != 0xffffffff)
	{
		uint endIdx = cellEnd[neighHash];
		for (uint i = startIdx; i < endIdx; i++)
		{
			uint sortedIdx = gridIdx[i];
			
			REAL3 dist = pos[sortedIdx] - centerPos; //정확히 그 칸에 있는 애만 집어야 함 옆 cell의 벽 파티클까지 해서 그런듯
			REAL d2 = LengthSquared(dist);
			if (d2 > cellSize * cellSize * 0.25)
				continue;
			if (type[sortedIdx] == WALL) {
				volumes.content.writeSurface<uint>(CONTENT_WALL, gridPos.x, gridPos.y, gridPos.z);
			}
		}
		if (volumes.content.readSurface<uint>(gridPos.x, gridPos.y, gridPos.z) != CONTENT_WALL)
		{
			REAL levelSet = LevelSet(gridPos, pos, type, dens, gridHash, gridIdx, cellStart, cellEnd, densVal, gridRes);

			if (levelSet < 0.0)
				volumes.content.writeSurface<uint>(CONTENT_FLUID, gridPos.x, gridPos.y, gridPos.z);
			else
				volumes.content.writeSurface<uint>(CONTENT_AIR, gridPos.x, gridPos.y, gridPos.z);
		}
	}
}

__device__ REAL WallCheck(uint type)
{
	if (type == CONTENT_WALL)
		return 1.0;
	return -1.0;
}

__global__ void EnforceBoundary_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;
	
	REAL4 velocity = volumes.vel.readSurface<REAL4>(x, y, z);
	
	if (x == 0 || x == gridRes)
		velocity.x = 0.0;
	if (x < gridRes && x>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x - 1, y, z)) < 0)
	{
		velocity.x = 0.0;
	}

	if (y == 0 || y == gridRes)
		velocity.y = 0.0;
	if (y < gridRes && y>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y - 1, z)) < 0) //여기가 문제
	{
		velocity.y = 0.0;
	}

	if (z == 0 || z == gridRes)
		velocity.z = 0.0;
	if (z < gridRes && z>0 && WallCheck(volumes.content.readSurface<uint>(x, y, z)) * WallCheck(volumes.content.readSurface<uint>(x, y, z - 1)) < 0)
	{
		velocity.z = 0.0;
	}

	volumes.vel.writeSurface<REAL4>(velocity, x, y, z);
}

__global__ void ComputeDivergence_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	//Compute Divergence
	REAL cellSize = 1.0 / gridRes;
	if (volumes.content.readSurface<uint>(x, y, z) == CONTENT_FLUID)
	{
		REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
		REAL4 rightVel = volumes.vel.readSurface<REAL4>(x + 1, y, z);
		REAL4 upVel = volumes.vel.readSurface<REAL4>(x, y + 1, z);
		REAL4 frontVel = volumes.vel.readSurface<REAL4>(x, y, z + 1);

		REAL div = ((rightVel.x - curVel.x) + (upVel.y - curVel.y) + (frontVel.z - curVel.z)) / cellSize;

		volumes.divergence.writeSurface<REAL>(div, x, y, z);
	}
	else
		volumes.divergence.writeSurface<REAL>(0.0f, x, y, z);

}

__global__ void ComputeLevelSet_D(VolumeCollection volumes, REAL3* pos, uint* type, REAL* dens, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL densVal, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	int3 gridPos = make_int3(x, y, z);
	REAL levelSet = LevelSet(gridPos, pos, type, dens, gridHash, gridIdx, cellStart, cellEnd, densVal, gridRes);

	volumes.levelSet.writeSurface<REAL>(levelSet, x, y, z);
}

__global__ void CopyToSolver_D(VolumeCollection volumes, uint* _airD, REAL* _levelsetD, REAL* _pressureD, REAL* _divergenceD, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;
	uint index = (gridRes * gridRes) * z + gridRes * y + x;

	_airD[index] = volumes.content.readSurface<uint>(x, y, z);
	_levelsetD[index] = volumes.levelSet.readSurface<REAL>(x, y, z);
	_pressureD[index] = volumes.press.readSurface<REAL>(x, y, z);
	_divergenceD[index] = -volumes.divergence.readSurface<REAL>(x, y, z); //must be flip divergence!!
}

__device__ double A_Ref(uint* A, int i, int j, int k, int qi, int qj, int qk, int n)
{
	if (i<0 || i>n - 1 || j<0 || j>n - 1 || k<0 || k>n - 1 || A[(n * n) * k + n * j + i] != CONTENT_FLUID) {
		return 0.0;
	}
	if (qi<0 || qi>n - 1 || qj<0 || qj>n - 1 || qk<0 || qk>n - 1 || A[(n * n) * qk + n * qj + qi] != CONTENT_FLUID) {
		return 0.0;
	}
	return -1.0;
}

__device__ double A_Diag(uint* A, float* L, int i, int j, int k, int n)
{
	float diag = 6.0f;
	if (A[(n * n) * k + n * j + i] != CONTENT_FLUID) {
		return diag;
	}
	int q[][3] = { { i - 1, j, k }, { i + 1, j, k }, { i, j - 1, k }, { i, j + 1, k }, { i, j, k - 1 }, { i, j, k + 1 } };
	for (int m = 0; m < 6; m++) {
		int qi = q[m][0];
		int qj = q[m][1];
		int qk = q[m][2];
		if (qi<0 || qi>n - 1 || qj<0 || qj>n - 1 || qk<0 || qk>n - 1 || A[(n * n) * qk + n * qj + qi] == CONTENT_WALL) {
			diag -= 1.0;
		}
		else if (A[(n * n) * qk + n * qj + qi] == CONTENT_AIR) {
			diag -= L[(n * n) * qk + n * qj + qi] / (float)(min(1.0e-6f, L[(n * n) * k + n * j + i]));
		}
	}
	return diag;
}

template <class T>
__device__ double P_Ref(T* P, int i, int j, int k, int n)
{
	if (i<0 || i>n - 1 || j<0 || j>n - 1 || k<0 || k>n - 1 || P[(n * n) * k + n * j + i] != CONTENT_FLUID) {
		return 0.0;
	}
	return P[(n * n) * k + n * j + i];
}

__device__ double x_Ref(uint* A, float* L, float* x, int fi, int fj, int fk, int i, int j, int k, int n)
{
	i = min(max(0, i), n - 1);
	j = min(max(0, j), n - 1);
	k = min(max(0, k), n - 1);
	if (A[(n * n) * k + n * j + i] == CONTENT_FLUID) return x[(n * n) * k + n * j + i];
	else if (A[(n * n) * k + n * j + i] == CONTENT_WALL) return x[(n * n) * fk + n * fj + fi];
	return L[(n * n) * k + n * j + i] / min(1.0e-6f, L[(n * n) * fk + n * fj + fi]) * x[(n * n) * fk + n * fj + fi];
}

__global__ void   BuildPreconditioner_D(REAL* P,
	REAL* L,
    uint* A,
    uint         size,
	REAL         one_over_n2,
	REAL         one_over_n3,
    uint   sizeOfData,
	dim3 blockDim)
{
    double tmp1;
    unsigned int t1, i, j, k, index_in, gridIndex;

    // step 1: compute gridIndex in 1-D   and 1-D data index "index_in"
    gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
    index_in = (gridIndex * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

    // step 2: extract 3-D data index via 
    // index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
    // where xIndex = i-1, yIndex = j-1, zIndex = k-1   
    if (index_in < sizeOfData) {
        // build indices
        tmp1 = __uint2double_rn(index_in);
        tmp1 = tmp1 * one_over_n3;
        t1 = __double2uint_rz(tmp1);
        k = index_in - size * t1;
        tmp1 = __uint2double_rn(t1);
        tmp1 = tmp1 * one_over_n2;
        i = __double2uint_rz(tmp1);
        j = t1 - size * i;
        // compute
        double a = 0.25f;
        int index = (size * size) * k + size * j + i;
        if (A[index] == CONTENT_FLUID) {
            double left = A_Ref(A, i - 1, j, k, i, j, k, size) * P_Ref(P, i - 1, j, k, size);
            double bottom = A_Ref(A, i, j - 1, k, i, j, k, size) * P_Ref(P, i, j - 1, k, size);
            double back = A_Ref(A, i, j, k - 1, i, j, k, size) * P_Ref(P, i, j, k - 1, size);
            double diag = A_Diag(A, L, i, j, k, size);
            //         double e      = diag - pow(left,2.0) - pow(bottom,2.0) - pow(back,2.0);
            double e = diag - (left * left) - (bottom * bottom) - (back * back);
            if (e < a * diag) {
                e = diag;
            }
            P[index] = 1.0 / sqrt(e);
        }
    }
}


__global__ void Compute_Ax_D(uint* A,
	REAL* L,
	REAL* x,
	REAL* ans,
	uint         size,
	REAL      one_over_n2,
	REAL      one_over_n3,
	uint   sizeOfData,
	dim3         grid,
	dim3         threads,
	dim3 blockDim)
{
	float h2 = (float)(1.0 / (size * size));
	double tmp1;
	unsigned int t1, i, j, k, index_in, gridIndex;

	// step 1: compute gridIndex in 1-D   and 1-D data index "index_in"
	gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
	index_in = (gridIndex * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

	// step 2: extract 3-D data index via 
	// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
	// where xIndex = i-1, yIndex = j-1, zIndex = k-1   
	if (index_in < sizeOfData) {
		// build indices
		tmp1 = __uint2double_rn(index_in);
		tmp1 = tmp1 * one_over_n3;
		t1 = __double2uint_rz(tmp1);
		k = index_in - size * t1;
		tmp1 = __uint2double_rn(t1);
		tmp1 = tmp1 * one_over_n2;
		i = __double2uint_rz(tmp1);
		j = t1 - size * i;
		// compute
		int index = (size * size) * k + size * j + i;
		if (A[index] == FLUID) {
			ans[index] = (6.0f * x[index]
				- (float)x_Ref(A, L, x, i, j, k, i + 1, j, k, size) - (float)x_Ref(A, L, x, i, j, k, i - 1, j, k, size)
				- (float)x_Ref(A, L, x, i, j, k, i, j + 1, k, size) - (float)x_Ref(A, L, x, i, j, k, i, j - 1, k, size)
				- (float)x_Ref(A, L, x, i, j, k, i, j, k + 1, size) - (float)x_Ref(A, L, x, i, j, k, i, j, k - 1, size)) / h2;
		}
		else {
			ans[index] = 0.0f;
		}
	}
}

__global__ void Operator_Kernel(uint* A,
	REAL* x,
	REAL* y,
	REAL* ans,   // copy
	REAL         a,
	uint            size,
	REAL         one_over_n2,
	REAL         one_over_n3,
	uint   sizeOfData,
	dim3         grid,
	dim3         threads,
	dim3 blockDim)
{
	double tmp1;
	unsigned int t1, i, j, k, index_in, gridIndex;

	// step 1: compute gridIndex in 1-D   and 1-D data index "index_in"
	gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
	index_in = (gridIndex * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

	// step 2: extract 3-D data index via 
	// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
	// where xIndex = i-1, yIndex = j-1, zIndex = k-1   
	if (index_in < sizeOfData) {
		// build indices
		tmp1 = __uint2double_rn(index_in);
		tmp1 = tmp1 * one_over_n3;
		t1 = __double2uint_rz(tmp1);
		k = index_in - size * t1;
		tmp1 = __uint2double_rn(t1);
		tmp1 = tmp1 * one_over_n2;
		i = __double2uint_rz(tmp1);
		j = t1 - size * i;
		// compute
		int index = (size * size) * k + size * j + i;
		if (A[index] == CONTENT_FLUID) {
			ans[index] = x[index] + a * y[index];
		}
		else {
			ans[index] = 0.0;
		}
	}
}

__global__ void Copy_Kernel(REAL* x,
	REAL* y,
	uint            size,
	REAL         one_over_n2,
	REAL         one_over_n3,
	uint   sizeOfData,
	dim3         grid,
	dim3         threads,
	dim3 blockDim)
{
	double tmp1;
	unsigned int t1, i, j, k, index_in, gridIndex;

	// step 1: compute gridIndex in 1-D   and 1-D data index "index_in"
	gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
	index_in = (gridIndex * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

	// step 2: extract 3-D data index via 
	// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
	// where xIndex = i-1, yIndex = j-1, zIndex = k-1   
	if (index_in < sizeOfData) {
		// build indices
		tmp1 = __uint2double_rn(index_in);
		tmp1 = tmp1 * one_over_n3;
		t1 = __double2uint_rz(tmp1);
		k = index_in - size * t1;
		tmp1 = __uint2double_rn(t1);
		tmp1 = tmp1 * one_over_n2;
		i = __double2uint_rz(tmp1);
		j = t1 - size * i;
		// compute
		int index = (size * size) * k + size * j + i;
		x[index] = y[index];
	}
}

__global__ void Apply_Preconditioner_Kernel(REAL* z,
	REAL* r,
	REAL* P,
	REAL* L,
	uint* A,
	REAL* q,   // tmp : muse be
	uint            size,
	REAL         one_over_n2,
	REAL         one_over_n3,
	uint   sizeOfData,
	dim3         grid,
	dim3         threads,
	dim3 blockDim)
{
	double tmp1;
	unsigned int t1, i, j, k, index_in, gridIndex;

	// step 1: compute gridIndex in 1-D   and 1-D data index "index_in"
	gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
	index_in = (gridIndex * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

	// step 2: extract 3-D data index via 
	// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
	// where xIndex = i-1, yIndex = j-1, zIndex = k-1   
	if (index_in < sizeOfData) {
		// build indices
		tmp1 = __uint2double_rn(index_in);
		tmp1 = tmp1 * one_over_n3;
		t1 = __double2uint_rz(tmp1);
		k = index_in - size * t1;
		tmp1 = __uint2double_rn(t1);
		tmp1 = tmp1 * one_over_n2;
		i = __double2uint_rz(tmp1);
		j = t1 - size * i;
		// compute
		int index = (size * size) * k + size * j + i;
		// Lq = r
		if (A[index] == CONTENT_FLUID) {
			double left = A_Ref(A, i - 1, j, k, i, j, k, size) * P_Ref(P, i - 1, j, k, size) * P_Ref(q, i - 1, j, k, size);
			double bottom = A_Ref(A, i, j - 1, k, i, j, k, size) * P_Ref(P, i, j - 1, k, size) * P_Ref(q, i, j - 1, k, size);
			double back = A_Ref(A, i, j, k - 1, i, j, k, size) * P_Ref(P, i, j, k - 1, size) * P_Ref(q, i, j, k - 1, size);
			double t = r[index] - left - bottom - back;
			q[index] = t * P[index];
		}
	}
}

__global__ void Apply_Trans_Preconditioner_Kernel(REAL* z,
	REAL* r,
	REAL* P,
	REAL* L,
	uint* A,
	REAL* q,   // tmp : muse be
	uint         size,
	REAL      one_over_n2,
	REAL      one_over_n3,
	uint   sizeOfData,
	dim3         grid,
	dim3         threads,
	dim3 blockDim)
{
	double tmp1;
	unsigned int t1, i, j, k, index_in, gridIndex;

	// step 1: compute gridIndex in 1-D   and 1-D data index "index_in"
	gridIndex = blockIdx.y * gridDim.x + blockIdx.x;
	index_in = (gridIndex * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;

	// step 2: extract 3-D data index via 
	// index_in = row-map(i,j,k) = (i-1)*n2*n3 + (j-1)*n3 + (k-1)
	// where xIndex = i-1, yIndex = j-1, zIndex = k-1   
	if (index_in < sizeOfData) {
		// build indices
		tmp1 = __uint2double_rn(index_in);
		tmp1 = tmp1 * one_over_n3;
		t1 = __double2uint_rz(tmp1);
		k = index_in - size * t1;
		tmp1 = __uint2double_rn(t1);
		tmp1 = tmp1 * one_over_n2;
		i = __double2uint_rz(tmp1);
		j = t1 - size * i;
		// compute
		unsigned int ti, tj, tk;
		ti = size - 1 - i;
		tj = size - 1 - j;
		tk = size - 1 - k;
		int tIndex = (size * size) * tk + size * tj + ti;
		// L^T z = q
		if (A[tIndex] == CONTENT_FLUID) {
			double right = A_Ref(A, ti, tj, tk, ti + 1, tj, tk, size) * P_Ref(P, ti, tj, tk, size) * P_Ref(z, ti + 1, tj, tk, size);
			double top = A_Ref(A, ti, tj, tk, ti, tj + 1, tk, size) * P_Ref(P, ti, tj, tk, size) * P_Ref(z, ti, tj + 1, tk, size);
			double front = A_Ref(A, ti, tj, tk, ti, tj, tk + 1, size) * P_Ref(P, ti, tj, tk, size) * P_Ref(z, ti, tj, tk + 1, size);
			double t = q[tIndex] - right - top - front;
			z[tIndex] = (float)(t * P[tIndex]);
		}
	}
}

__global__ void CopyToGrid_D(VolumeCollection volumes, uint* _airD, REAL* _levelsetD, REAL* _pressureD, REAL* _divergenceD, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;
	uint index = (gridRes * gridRes) * z + gridRes * y + x;

	volumes.content.writeSurface<uint>(_airD[index], x, y, z);
	volumes.levelSet.writeSurface<REAL>(_levelsetD[index], x, y, z);
	volumes.press.writeSurface<REAL>(_pressureD[index], x, y, z);
	volumes.divergence.writeSurface<REAL>(_divergenceD[index], x, y, z);
}



__global__ void ComputeVelocityWithPress_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	REAL cellSize = 1.0 / gridRes;
	REAL levelSet = volumes.levelSet.readSurface<REAL>(x, y, z);

	REAL4 curVel = volumes.vel.readSurface<REAL4>(x, y, z);
	if (x < gridRes && x>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressX = volumes.press.readSurface<REAL>(x - 1, y, z);
		REAL levelSetX = volumes.levelSet.readSurface<REAL>(x - 1, y, z);

		if (levelSet * levelSetX < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(1.0e-3f, levelSetX) * volumes.press.readSurface<REAL>(x - 1, y, z);
			pressX = levelSetX < 0.0 ? volumes.press.readSurface<REAL>(x - 1, y, z) : levelSetX / min(1.0e-6f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		curVel.x = curVel.x - ((press - pressX) / cellSize);
	}

	if (y < gridRes && y>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressY = volumes.press.readSurface<REAL>(x, y - 1, z);
		REAL levelSetY = volumes.levelSet.readSurface<REAL>(x, y - 1, z);

		if (levelSet * levelSetY < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(1.0e-3f, levelSetY) * volumes.press.readSurface<REAL>(x, y - 1, z);
			pressY = levelSetY < 0.0 ? volumes.press.readSurface<REAL>(x, y - 1, z) : levelSetY / min(1.0e-6f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		curVel.y = curVel.y - ((press - pressY)/ cellSize);
	}

	if (z < gridRes && z>0) {
		REAL press = volumes.press.readSurface<REAL>(x, y, z);
		REAL pressZ = volumes.press.readSurface<REAL>(x, y, z - 1);
		REAL levelSetZ = volumes.levelSet.readSurface<REAL>(x, y, z - 1);

		if (levelSet * levelSetZ < 0.0) {
			press = levelSet < 0.0 ? volumes.press.readSurface<REAL>(x, y, z) : levelSet / min(1.0e-3f, levelSetZ) * volumes.press.readSurface<REAL>(x, y, z - 1);
			pressZ = levelSetZ < 0.0 ? volumes.press.readSurface<REAL>(x, y, z - 1) : levelSetZ / min(1.0e-6f, levelSet) * volumes.press.readSurface<REAL>(x, y, z);
		}
		curVel.z = curVel.z - ((press - pressZ) /cellSize);
	}

	volumes.vel.writeSurface<REAL4>(curVel, x, y, z);
}

__device__ uint3 Mark(VolumeCollection volumes, uint x, uint y, uint z, uint gridRes)
{
	uint3 mark = make_uint3(0, 0, 0);

	uint thisContent = volumes.content.readSurface<uint>(x, y, z);
	//uint leftContent = volumes.content.readSurface<uint>(x - 1, y, z);
	if ((x > 0 && volumes.content.readSurface<uint>(x - 1, y, z) == CONTENT_FLUID) || (x < gridRes && thisContent == CONTENT_FLUID))
		mark.x = true;

	//uint downContent = volumes.content.readSurface<uint>(x, y - 1, z);
	if ((y > 0 && volumes.content.readSurface<uint>(x, y - 1, z) == CONTENT_FLUID) || (y < gridRes && thisContent == CONTENT_FLUID))
		mark.y = true;

	//uint backContent = volumes.content.readSurface<uint>(x, y, z - 1);
	if ((z > 0 && volumes.content.readSurface<uint>(x, y, z - 1) == CONTENT_FLUID) || (z < gridRes && thisContent == CONTENT_FLUID))
		mark.z = true;

	return mark;
}

__device__ uint3 WallMark(VolumeCollection volumes, uint x, uint y, uint z, uint gridRes)
{
	uint3 wall_mark = make_uint3(0, 0, 0);

	uint thisContent = volumes.content.readSurface<uint>(x, y, z);
	//uint leftContent = volumes.content.readSurface<uint>(x - 1, y, z);
	if ((x <= 0 || volumes.content.readSurface<uint>(x - 1, y, z) == CONTENT_WALL) && (x >= gridRes || thisContent == CONTENT_WALL))
		wall_mark.x = true;

	//uint downContent = volumes.content.readSurface<uint>(x, y - 1, z);
	if ((y <= 0 || volumes.content.readSurface<uint>(x, y - 1, z) == CONTENT_WALL) && (y >= gridRes || thisContent == CONTENT_WALL))
		wall_mark.y = true;

	//uint backContent = volumes.content.readSurface<uint>(x, y, z - 1);
	if ((z <= 0 || volumes.content.readSurface<uint>(x, y, z - 1) == CONTENT_WALL) && (z >= gridRes || thisContent == CONTENT_WALL))
		wall_mark.z = true;

	return wall_mark;
}

__global__ void ExtrapolateVelocity_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	REAL4 newVel = volumes.vel.readSurface<REAL4>(x, y, z);
	uint3 mark = Mark(volumes, x, y, z, gridRes);
	uint3 wall_mark = WallMark(volumes, x, y, z, gridRes);

	if (!mark.x && wall_mark.x) {
		uint wsum = 0;
		REAL sum = 0.0;
		uint q[][3] = { {x - 1,y,z}, {x + 1,y,z}, {x,y - 1,z}, {x,y + 1,z}, {x,y,z - 1}, {x,y,z + 1} };
		for (int qk = 0; qk < 6; qk++) {
			if (q[qk][0] >= 0 && q[qk][0] < gridRes && q[qk][1] >= 0 && q[qk][1] < gridRes && q[qk][2] >= 0 && q[qk][2] < gridRes) {
				uint3 newMark = Mark(volumes, q[qk][0], q[qk][1], q[qk][2], gridRes);
				if (newMark.x) {
					wsum++;
					sum += volumes.vel.readSurface<REAL4>(q[qk][0], q[qk][1], q[qk][2]).x;
				}
			}
		}

		if (wsum) newVel.x = sum / wsum;
	}

	if (!mark.y && wall_mark.y) {
		uint wsum = 0;
		REAL sum = 0.0;
		uint q[][3] = { {x - 1,y,z}, {x + 1,y,z}, {x,y - 1,z}, {x,y + 1,z}, {x,y,z - 1}, {x,y,z + 1} };
		for (int qk = 0; qk < 6; qk++) {
			if (q[qk][0] >= 0 && q[qk][0] < gridRes && q[qk][1] >= 0 && q[qk][1] < gridRes && q[qk][2] >= 0 && q[qk][2] < gridRes) {
				uint3 newMark = Mark(volumes, q[qk][0], q[qk][1], q[qk][2], gridRes);
				if (newMark.y) {
					wsum++;
					sum += volumes.vel.readSurface<REAL4>(q[qk][0], q[qk][1], q[qk][2]).y;
				}
			}
		}
		if (wsum) newVel.y = sum / wsum;
	}

	if (!mark.z && wall_mark.z) {
		uint wsum = 0;
		REAL sum = 0.0;
		uint q[][3] = { {x - 1,y,z}, {x + 1,y,z}, {x,y - 1,z}, {x,y + 1,z}, {x,y,z - 1}, {x,y,z + 1} };
		for (int qk = 0; qk < 6; qk++) {
			if (q[qk][0] >= 0 && q[qk][0] < gridRes && q[qk][1] >= 0 && q[qk][1] < gridRes && q[qk][2] >= 0 && q[qk][2] < gridRes) {
				uint3 newMark = Mark(volumes, q[qk][0], q[qk][1], q[qk][2], gridRes);
				if (newMark.z) {
					wsum++;
					sum += volumes.vel.readSurface<REAL4>(q[qk][0], q[qk][1], q[qk][2]).z;
				}
			}
		}
		if (wsum) newVel.z = sum / wsum;
	}
	volumes.vel.writeSurface<REAL4>(newVel, x, y, z);
}

__global__ void SubtarctGrid_D(VolumeCollection volumes, uint gridRes)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= gridRes || y >= gridRes || z >= gridRes) return;

	REAL4 beforeVel = volumes.velSave.readSurface<REAL4>(x, y, z);
	REAL4 afterVel = volumes.vel.readSurface<REAL4>(x, y, z);

	REAL subVelX = afterVel.x - beforeVel.x;
	REAL subVelY = afterVel.y - beforeVel.y;
	REAL subVelZ = afterVel.z - beforeVel.z;
	REAL4 newVel = make_float4(subVelX, subVelY, subVelZ, 0.0);

	volumes.velSave.writeSurface<REAL4>(newVel, x, y, z);
}

//__device__ inline REAL3 lerp(REAL3 v0, REAL3 v1, REAL t) {
//	return (1 - t) * v0 + t * v1;
//}

__device__ REAL3 GetLerpValueAtPoint(VolumeData data, REAL x, REAL y, REAL z, uint gridRes)
{
	x = max(min(x, gridRes - 1.f), 0.f);
	y = max(min(y, gridRes - 1.f), 0.f);
	z = max(min(z, gridRes - 1.f), 0.f);

	uint i = floor(x);
	uint j = floor(y);
	uint k = floor(z);

	REAL tx = x - (REAL)i;
	REAL ty = y - (REAL)j;
	REAL tz = z - (REAL)k;
	
	REAL3 a = make_REAL3(data.readSurface<REAL4>(i, j, k).x, data.readSurface<REAL4>(i, j, k).y, data.readSurface<REAL4>(i, j, k).z);
	REAL3 b = make_REAL3(data.readSurface<REAL4>(i, j, k + 1).x, data.readSurface<REAL4>(i, j, k + 1).y, data.readSurface<REAL4>(i, j, k + 1).z);
	REAL3 x0y0 =
		lerp(a, b, tz);

	REAL3 c = make_REAL3(data.readSurface<REAL4>(i, j + 1, k).x, data.readSurface<REAL4>(i, j + 1, k).y, data.readSurface<REAL4>(i, j + 1, k).z);
	REAL3 d = make_REAL3(data.readSurface<REAL4>(i, j + 1, k + 1).x, data.readSurface<REAL4>(i, j + 1, k + 1).y, data.readSurface<REAL4>(i, j + 1, k + 1).z);
	REAL3 x0y1 =
		lerp(c, d, tz);

	REAL3 e = make_REAL3(data.readSurface<REAL4>(i + 1, j, k).x, data.readSurface<REAL4>(i + 1, j, k).y, data.readSurface<REAL4>(i + 1, j, k).z);
	REAL3 f = make_REAL3(data.readSurface<REAL4>(i + 1, j, k + 1).x, data.readSurface<REAL4>(i + 1, j, k + 1).y, data.readSurface<REAL4>(i + 1, j, k + 1).z);
	REAL3 x1y0 =
		lerp(e, f, tz);

	REAL3 g = make_REAL3(data.readSurface<REAL4>(i + 1, j + 1, k).x, data.readSurface<REAL4>(i + 1, j + 1, k).y, data.readSurface<REAL4>(i + 1, j + 1, k).z);
	REAL3 h = make_REAL3(data.readSurface<REAL4>(i + 1, j + 1, k + 1).x, data.readSurface<REAL4>(i + 1, j + 1, k + 1).y, data.readSurface<REAL4>(i + 1, j + 1, k + 1).z);
	REAL3 x1y1 =
		lerp(g, h, tz);

	REAL3 x0 = lerp(x0y0, x0y1, ty);
	REAL3 x1 = lerp(x1y0, x1y1, ty);

	REAL3 result = lerp(x0, x1, tx);
	return result;
}

__device__ REAL3 GetPointSaveVelocity(REAL3 physicalPos,  uint gridRes, VolumeCollection volume) {
	REAL cellPhysicalSize = 1.0 / gridRes;

	REAL x = physicalPos.x / cellPhysicalSize;
	REAL y = physicalPos.y / cellPhysicalSize;
	REAL z = physicalPos.z / cellPhysicalSize;

	REAL3 result;

	result.x = GetLerpValueAtPoint(volume.velSave, x, y - 0.5, z - 0.5, gridRes).x;
	result.y = GetLerpValueAtPoint(volume.velSave, x - 0.5, y, z - 0.5, gridRes).y;
	result.z = GetLerpValueAtPoint(volume.velSave, x - 0.5, y - 0.5, z, gridRes).z;

	return result;
}

__device__ REAL3 GetPointAfterVelocity(REAL3 physicalPos, uint gridRes, VolumeCollection volume) {
	REAL cellPhysicalSize = 1.0 / gridRes;

	REAL x = physicalPos.x / cellPhysicalSize;
	REAL y = physicalPos.y / cellPhysicalSize;
	REAL z = physicalPos.z / cellPhysicalSize;

	REAL3 result;

	result.x = GetLerpValueAtPoint(volume.vel, x, y - 0.5, z - 0.5, gridRes).x;
	result.y = GetLerpValueAtPoint(volume.vel, x - 0.5, y, z - 0.5, gridRes).y;
	result.z = GetLerpValueAtPoint(volume.vel, x - 0.5, y - 0.5, z, gridRes).z;

	return result;
}

__global__ void TrasnferToParticle_D(VolumeCollection volumes, uint gridRes, REAL3* pos, REAL3* vel, uint numParticles)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles) return;

	REAL3 position = pos[index];
	REAL3 oldGridVel = GetPointSaveVelocity(position, gridRes, volumes);
	REAL3 newGridVel = GetPointAfterVelocity(position, gridRes, volumes);

	REAL3 FLIP = vel[index] + oldGridVel;
	REAL3 PIC = newGridVel;

	REAL FLIPCoeff = 0.95;
	vel[index] = FLIPCoeff * FLIP + (1.0 - FLIPCoeff) * PIC;
}

__global__ void AdvecParticle_D(VolumeCollection volumes, REAL3* beforePos, REAL3* curPos, REAL3* vel, uint* type, uint gridRes, uint numParticles, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID)
		return;

	beforePos[idx] = curPos[idx];

	REAL3 lerpVel = GetPointAfterVelocity(curPos[idx], gridRes, volumes);
	curPos[idx] += dt * lerpVel;
}

__global__ void ConstraintOuterWall_D(REAL3* pos, REAL3* vel, REAL3* normal, uint* type, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint numParticles, REAL gridRes, REAL densVal)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID)
		return;

	REAL wallThick = 1.0 / gridRes;
	pos[idx].x = max(wallThick, min((REAL)(1.0 - wallThick), pos[idx].x));
	pos[idx].y = max(wallThick, min((REAL)(1.0 - wallThick), pos[idx].y));
	pos[idx].z = max(wallThick, min((REAL)(1.0 - wallThick), pos[idx].z));

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);

	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];
		REAL re = 1.5 * densVal / gridRes;

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (type[sortedIdx] == WALL) {
					REAL dist = Length(pos[idx] - pos[sortedIdx]);
					if (dist < re) {
						REAL3 normalVector = normal[sortedIdx];
						if (normalVector.x == 0.0f && normalVector.y == 0.0f && normalVector.z == 0.0f)
							normalVector = (pos[idx] - pos[sortedIdx]) / dist;

						pos[idx] += (re - dist) * normalVector;
						REAL dot = Dot(vel[idx], normalVector);
						vel[idx] -= dot * normalVector;
					}
				}
			}

		}

	}END_FOR;
}

__device__ REAL3 Resample(REAL3 curPos, REAL3 curVel, REAL3* pos, REAL3* vel, REAL* mass, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, REAL re)
{
	REAL wsum = 0.0f;
	REAL3 save = make_REAL3(curVel.x, curVel.y, curVel.z);
	curVel = make_REAL3(0, 0, 0);


	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(curPos.x + dx, curPos.y + dy, curPos.z + dz);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];
		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				REAL3 dist = curPos - pos[sortedIdx];
				REAL d2 = LengthSquared(dist);
				REAL w = mass[sortedIdx] * SharpKernel(d2, re);
				curVel += w * vel[sortedIdx];
				wsum += w;
			}
		}
	}END_FOR;

	if (wsum)
		curVel /= wsum;
	else
		curVel = save;

	return curVel;
}

__global__ void Correct_D(REAL3* pos, REAL3* vel, REAL3* normal, REAL* mass, uint* type, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, uint gridRes, uint numParticles, REAL dt, REAL re, uint r1, uint r2, uint r3)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= numParticles)
		return;
	if (type[idx] != FLUID)
		return;

	REAL springCoeff = 50.0f;
	REAL3 spring = make_REAL3(0, 0, 0);

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);

	FOR_NEIGHBOR(1) {

		int3 neighbourPos = make_int3(gridPos.x + dx, gridPos.y + dy, gridPos.z + dz);
		uint neighHash = calcGridHash(neighbourPos, gridRes);
		uint startIdx = cellStart[neighHash];

		if (startIdx != 0xffffffff)
		{
			uint endIdx = cellEnd[neighHash];
			for (uint i = startIdx; i < endIdx; i++)
			{
				uint sortedIdx = gridIdx[i];
				if (sortedIdx != idx)
				{
					REAL3 dist = pos[sortedIdx] - pos[idx];
					REAL d = Length(dist);
					REAL w = springCoeff * mass[sortedIdx] * SmoothKernel(d * d, re);
					if (d > 0.1 * re) {
						spring += w * (pos[idx] - pos[sortedIdx]) / d * re;
					}
					else {
						if (type[sortedIdx] == FLUID) {
							spring.x += 0.01 * re / dt * (r1 % 101) / 100.0;
							spring.y += 0.01 * re / dt * (r2 % 101) / 100.0;
							spring.z += 0.01 * re / dt * (r3 % 101) / 100.0;
						}
						else
						{
							spring.x += 0.05 * re / dt * normal[sortedIdx].x;
							spring.y += 0.05 * re / dt * normal[sortedIdx].y;
							spring.z += 0.05 * re / dt * normal[sortedIdx].z;
						}
					}
					
				}
			}
		}
	} END_FOR;
	REAL3 temp = pos[idx] + dt * spring;

	REAL3 temp2 = vel[idx];
	temp2 = Resample(temp, temp2, pos, vel, mass, gridHash, gridIdx, cellStart, cellEnd, gridRes, re);

	pos[idx] = temp;
	vel[idx] = temp2;
}

__global__ void GridVisualize_D(VolumeCollection volumes, uint gridRes, REAL3* gridPos, REAL3* gridVel, REAL* gridPress, REAL* gridDens, REAL* gridLevelSet, REAL* gridDiv, uint* gridContent)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= gridRes || y >= gridRes || z >= gridRes) return;


	REAL cellSize = 1.0 / gridRes;
	REAL3 centerPos = make_REAL3(x + 0.5, y + 0.5, z + 0.5) * cellSize;
	int index = x * (gridRes * gridRes) + y * (gridRes)+z;

	gridPos[index] = make_REAL3(centerPos.x, centerPos.y, centerPos.z);

	REAL4 vel = volumes.vel.readSurface<REAL4>(x, y, z);
	gridVel[index] = make_REAL3(vel.x, vel.y, vel.z);

	REAL press = volumes.press.readSurface<REAL>(x, y, z);
	gridPress[index] = press;

	REAL dens = volumes.density.readSurface<REAL>(x, y, z);
	gridDens[index] = dens;

	REAL levelSet = volumes.levelSet.readSurface<REAL>(x, y, z);
	gridLevelSet[index] = levelSet;

	REAL divergence = volumes.divergence.readSurface<REAL>(x, y, z);
	gridDiv[index] = divergence;
	
	uint content = volumes.content.readSurface<uint>(x, y, z);
	gridContent[index] = content;

	
}


#endif