#include "FLIP3D_Cuda.cuh"
#define VEL 0
#define PRESS 0
#define LEVEL 0
#define DENSITY 0
#define DIV 0
#define CONTENT 0

FLIP3D_Cuda::FLIP3D_Cuda()
{
}

FLIP3D_Cuda:: ~FLIP3D_Cuda()
{
	FreeDeviceMem();
}

void FLIP3D_Cuda::Init(void)
{
	_wallThick = 1.0 / _gridRes;
	_cellPhysicalSize = 1.0 / _gridRes;

	_grid = new FLIPGRID(_gridRes, _cellPhysicalSize);

	_numParticles = 0;

	InitHostMem();
	ParticleInit();
	printf("Num FLIP particles: %d\n", _numParticles);

	////For grid visualize
	//h_gridPos.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 
	//h_gridVel.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 
	//h_gridPress.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 
	//h_gridDens.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));
	//h_gridLevelSet.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));
	//h_gridDiv.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));
	//h_gridContent.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 

	InitDeviceMem();
	CopyToDevice();

	ComputeWallParticleNormal_kernel();
	cudaDeviceSynchronize();
}

void FLIP3D_Cuda::ParticleInit()
{
	PlaceObjects();

	// Place Fluid Particles
	double w = _dens * _wallThick;
	for (int i = 0; i < _gridRes / _dens; i++) {
		for (int j = 0; j < _gridRes / _dens; j++) {
			for (int k = 0; k < _gridRes / _dens; k++) {
				double x = i * w + w / 2.0;
				double y = j * w + w / 2.0;
				double z = k * w + w / 2.0;
				
				if (x > _wallThick && x < 1.0 - _wallThick &&
					y > _wallThick && y < 1.0 - _wallThick &&
					z > _wallThick && z < 1.0 - _wallThick) {
					PushParticle(x, y, z, FLUID);
				}
			}
		}
	}

	// Place Wall Particles
	w = 1.0 / _gridRes;
	for (int i = 0; i < _gridRes; i++) {
		for (int j = 0; j < _gridRes; j++) {
			for (int k = 0; k < _gridRes; k++) {
				double x = i * w + w / 2.0;
				double y = j * w + w / 2.0;
				double z = k * w + w / 2.0;
				PushParticle(x, y, z, WALL);
			}
		}
	}

	//for (int iter = 0; iter < h_CurPos.size(); iter++)
	//{
	//	if (h_Type[iter] == WALL) {
	//		iter++;
	//		continue;
	//	}
	//	int i = fmin(_gridRes - 1, fmax(0, h_CurPos[iter].x * _gridRes));
	//	int j = fmin(_gridRes - 1, fmax(0, h_CurPos[iter].y * _gridRes));
	//	int k = fmin(_gridRes - 1, fmax(0, h_CurPos[iter].z * _gridRes));
	//}
}

void FLIP3D_Cuda::PlaceObjects()
{
	PlaceWalls();

	//WaterDropTest();
	//DamBreakTest();
	RotateBoxesTest();
	//MoveBoxTest();
	//MoveSphereTest();
}

void FLIP3D_Cuda::PlaceWalls()
{
	Object obj;

	// Left Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.p[0].x = 0.0;			obj.p[1].x = _wallThick; //Box min, max 값
	obj.p[0].y = 0.0;			obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;			obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Right Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.p[0].x = 1.0 - _wallThick;	obj.p[1].x = 1.0;
	obj.p[0].y = 0.0;				obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;				obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Floor Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.p[0].x = 0.0;	obj.p[1].x = 1.0;
	obj.p[0].y = 0.0;	obj.p[1].y = _wallThick;
	obj.p[0].z = 0.0;	obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Ceiling Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.p[0].x = 0.0;				obj.p[1].x = 1.0;
	obj.p[0].y = 1.0 - _wallThick;	obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;				obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Front Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.p[0].x = 0.0;	obj.p[1].x = 1.0;
	obj.p[0].y = 0.0;	obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;	obj.p[1].z = _wallThick;
	objects.push_back(obj);

	// Back Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.p[0].x = 0.0;				obj.p[1].x = 1.0;
	obj.p[0].y = 0.0;				obj.p[1].y = 1.0;
	obj.p[0].z = 1.0 - _wallThick;	obj.p[1].z = 1.0;
	objects.push_back(obj);
}

void FLIP3D_Cuda::WaterDropTest()
{
	Object obj;

	obj.type = FLUID;
	obj.shape = BOX;
	obj.p[0].x = _wallThick;	obj.p[1].x = 1.0 - _wallThick;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.1;
	obj.p[0].z = _wallThick;	obj.p[1].z = 1.0 - _wallThick;
	objects.push_back(obj);

	//obj.type = FLUID;
	//obj.shape = SPHERE;
	//obj.c.x = 0.4;
	//obj.c.y = 0.25;
	//obj.c.z = 0.5;
	//obj.r = 0.05;
	//objects.push_back(obj);

	obj.type = FLUID;
	obj.shape = SPHERE;
	obj.c.x = 0.5;
	obj.c.y = 0.3;
	obj.c.z = 0.5;
	obj.r = 0.02;
	objects.push_back(obj);

	//obj.type = FLUID;
	//obj.shape = SPHERE;
	//obj.c.x = 0.5;
	//obj.c.y = 0.4;
	//obj.c.z = 0.47;
	//obj.r = 0.02;
	//objects.push_back(obj);

	//obj.type = FLUID;
	//obj.shape = SPHERE;
	//obj.c.x = 0.3;
	//obj.c.y = 0.7;
	//obj.c.z = 0.3;
	//obj.r = 0.02;
	//objects.push_back(obj);

	//obj.type = FLUID;
	//obj.shape = SPHERE;
	//obj.c.x = 0.5;
	//obj.c.y = 0.25;
	//obj.c.z = 0.75;
	//obj.r = 0.02;
	//objects.push_back(obj);

	//obj.type = FLUID;
	//obj.shape = SPHERE;
	//obj.c.x = 0.25;
	//obj.c.y = 0.3;
	//obj.c.z = 0.5;
	//obj.r = 0.02;
	//objects.push_back(obj);

	//obj.type = FLUID;
	//obj.shape = SPHERE;
	//obj.c.x = 0.5;
	//obj.c.y = 0.35;
	//obj.c.z = 0.25;
	//obj.r = 0.02;
	//objects.push_back(obj);

}

void FLIP3D_Cuda::DamBreakTest()
{
	Object obj;

	obj.type = FLUID;
	obj.shape = BOX;
	obj.p[0].x = 0.2;	obj.p[1].x = 0.4;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.4;
	obj.p[0].z = 0.2;	obj.p[1].z = 0.8;
	objects.push_back(obj);

	//obj.type = FLUID;
	//obj.shape = BOX;
	//obj.p[0].x = 0.2;	obj.p[1].x = 0.4;
	//obj.p[0].y = _wallThick;	obj.p[1].y = 0.25;
	//obj.p[0].z = 0.3;	obj.p[1].z = 0.8;
	//objects.push_back(obj);

	obj.type = FLUID;
	obj.shape = BOX;
	obj.p[0].x = _wallThick;	obj.p[1].x = 1.0 - _wallThick;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.06;
	obj.p[0].z = _wallThick;	obj.p[1].z = 1.0 - _wallThick;
	objects.push_back(obj);
}

void FLIP3D_Cuda::RotateBoxesTest(void)
{
	OBB box;
	box._center = make_REAL3(0.3, 0.15, 0.5);
	box._center0 = box._center;
	box._radius = make_REAL3(0.1, 0.1, 0.06);
	computeCorners(box);
	h_Boxes.push_back(box);

	//box._center = make_REAL3(0.7, 0.12, 0.5);
	//box._center0 = box._center;
	//box._radius = make_REAL3(0.06, 0.12, 0.03);
	//computeCorners(box);
	//h_Boxes.push_back(box);

	Object obj;
	obj.type = FLUID;
	obj.shape = BOX;
	obj.p[0].x = _wallThick;	obj.p[1].x = 1.0 - _wallThick;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.1;
	obj.p[0].z = _wallThick;	obj.p[1].z = 1.0 - _wallThick;
	objects.push_back(obj);

	_numBoxes = h_Boxes.size();
}

void FLIP3D_Cuda::MoveBoxTest(void)
{
	OBB box;
	box._center = make_REAL3(0.0, 0.06, 0.5);
	box._center0 = box._center;
	box._radius = make_REAL3(0.12, 0.5, 0.5);
	box.flag = true;
	computeCorners(box);
	h_Boxes.push_back(box);

	//box._center = make_REAL3(0.7, 0.12, 0.5);
	//box._center0 = box._center;
	//box._radius = make_REAL3(0.06, 0.12, 0.03);
	//computeCorners(box);
	//h_Boxes.push_back(box);

	Object obj;
	obj.type = FLUID;
	obj.shape = BOX;
	obj.p[0].x = _wallThick;	obj.p[1].x = 1.0 - _wallThick;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.05;
	obj.p[0].z = _wallThick;	obj.p[1].z = 1.0 - _wallThick;
	objects.push_back(obj);

	_numBoxes = h_Boxes.size();
}

void FLIP3D_Cuda::MoveSphereTest(void)
{
	BoundingSphere sphere;
	sphere._center = make_REAL3(0.5, 0.06, 0.5);
	sphere._center0 = sphere._center;
	sphere._radius = 0.25;
	sphere.flag = true;
	h_Spheres.push_back(sphere);

	Object obj;
	obj.type = FLUID;
	obj.shape = BOX;
	obj.p[0].x = _wallThick;	obj.p[1].x = 1.0 - _wallThick;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.1;
	obj.p[0].z = _wallThick;	obj.p[1].z = 1.0 - _wallThick;
	objects.push_back(obj);

	_numSpheres = h_Spheres.size();
}

void FLIP3D_Cuda::PourWater(void)
{
	vector<REAL3> h_newPos;
	vector<REAL3> h_newVel;
	vector<REAL> h_newMass;

	h_newPos.clear();
	h_newVel.clear();
	h_newMass.clear();

	REAL pourPosY = 0.8;
	REAL pourPosZ = 0.5;
	REAL pourRad = 0.1;

	double w = _dens / _gridRes;
	for (REAL y = w + w / 2.0; y < 1.0 - w / 2.0; y += w)
	{
		for (REAL z = w + w / 2.0; z < 1.0 - w / 2.0; z += w)
		{
			if (hypot(y - pourPosY, z - pourPosZ) < pourRad)
			{
				h_newPos.push_back(make_REAL3(z, y, 1.0 - _wallThick - 0.2));
				h_newVel.push_back(make_REAL3(0.0f, 0.0f, -_dens / _gridRes / 0.005f));
				h_newMass.push_back(1.0);
			}
		}
	}
	uint numInsertParticles = h_newPos.size();

	Dvector<REAL3> d_newPos;
	Dvector<REAL3> d_newVel;
	Dvector<REAL> d_newMass;

	d_newPos.resize(numInsertParticles);
	d_newVel.resize(numInsertParticles);
	d_newMass.resize(numInsertParticles);

	d_newPos.memset(0);
	d_newVel.memset(0);
	d_newMass.memset(0);

	d_newPos.copyFromHost(h_newPos);
	d_newVel.copyFromHost(h_newVel);
	d_newMass.copyFromHost(h_newMass);

	//삽입
	InsertFLIPParticles_D << < divup(numInsertParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_BeforePos(), d_Vel(), d_Normal(), d_Type(), d_Mass(), d_Dens(), d_KernelDens(), d_Flag(), _numParticles, _maxDens,
			d_newPos(), d_newVel(), d_newMass(), numInsertParticles);

	_numParticles += numInsertParticles;

	d_newPos.free();
	d_newVel.free();
	d_newMass.free();

	cudaDeviceSynchronize();
}

void FLIP3D_Cuda::PushParticle(REAL x, REAL y, REAL z, uint type)
{
	Object* inside_obj = NULL;
	for (int n = 0; n < objects.size(); n++) {
		Object& obj = objects[n];

		bool found = false;
		REAL thickness = 3.0 / _gridRes;
		if (obj.shape == BOX) {
			if (x > obj.p[0].x && x < obj.p[1].x &&
				y > obj.p[0].y && y < obj.p[1].y &&
				z > obj.p[0].z && z < obj.p[1].z) {

				if (obj.type == WALL &&
					x > obj.p[0].x + thickness && x < obj.p[1].x - thickness &&
					y > obj.p[0].y + thickness && y < obj.p[1].y - thickness &&
					z > obj.p[0].z + thickness && z < obj.p[1].z - thickness) {
					// 벽 obj일 경우 일정 깊이 안에는 particle 생성 X 
					inside_obj = NULL;
					break;
				}
				else {
					found = true;
				}
			}
		}
		else if (obj.shape == SPHERE) {
			REAL3 p = make_REAL3(x, y, z);
			REAL3 c = make_REAL3(obj.c.x, obj.c.y, obj.c.z);
		
			REAL len = Length(p - c);
			if (len < obj.r) {
				if (obj.type == WALL) {
					found = true;
					if (len < obj.r - thickness) {
						// 벽 obj일 경우 일정 깊이 안에는 particle 생성 X 
						inside_obj = NULL;
						break;
					}
				}
				else if (obj.type == FLUID) {
					found = true;
				}
			}
		}

		if (found) {
			if (objects[n].type == type) {
				inside_obj = &objects[n]; // Found
				break;
			}
		}
	}

	if (type == FLUID)
	{
		for (int i = 0; i < h_Boxes.size(); i++)
		{
			for (auto box : h_Boxes)
			{
				if (getDist(box, make_REAL3(x, y, z)) < 0.0f)
				{
					inside_obj = NULL;
				}
			}
		}

		for (int i = 0; i < h_Spheres.size(); i++)
		{
			for (auto sphere : h_Spheres)
			{
				if (getDist(sphere, make_REAL3(x, y, z)) < 0.0f)
				{
					inside_obj = NULL;
				}
			}
		}
	}

	if (inside_obj) {
		REAL _x = x + 0.01 * (inside_obj->type == FLUID) * 0.2 * ((rand() % 101) / 50.0 - 1.0) / _gridRes;
		REAL _y = y + 0.01 * (inside_obj->type == FLUID) * 0.2 * ((rand() % 101) / 50.0 - 1.0) / _gridRes;
		REAL _z = z + 0.01 * (inside_obj->type == FLUID) * 0.2 * ((rand() % 101) / 50.0 - 1.0) / _gridRes;

		REAL3 beforePos = make_REAL3(0.0, 0.0, 0.0);
		REAL3 curPos = make_REAL3(_x, _y, _z);
		REAL3 vel = make_REAL3(0.0, 0.0, 0.0);
		REAL3 normal = make_REAL3(0.0, 0.0, 0.0);
		REAL dens = 10.0;
		uint type = inside_obj->type;
		REAL mass = 1.0;
		REAL kernelDens = 0.0;
		BOOL flag = false;

		h_BeforePos[_numParticles] = beforePos;
		h_CurPos[_numParticles] = curPos;
		h_Vel[_numParticles] = vel;
		h_Normal[_numParticles] = normal;
		h_Dens[_numParticles] = dens;
		h_Type[_numParticles] = type;
		h_Mass[_numParticles] = mass;
		h_KernelDens[_numParticles] = kernelDens;
		h_Flag[_numParticles] = false;
		_numParticles++;
	}
}

void FLIP3D_Cuda::ComputeWallParticleNormal_kernel()
{
	SetHashTable_kernel();

	ComputeWallParticleNormal_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_Type(), d_Normal(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _numParticles, _gridRes);
	printf("Normal compute\n");
}

void FLIP3D_Cuda::ComputeParticleDensity_kernel()
{
	ComputeParticleDensity_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_Type(), d_Dens(), d_Mass(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _gridRes, _numParticles, _dens, _maxDens);
}

void FLIP3D_Cuda::ComputeExternalForce_kernel(REAL3& gravity, REAL dt)
{
	CompExternlaForce_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_Vel(), gravity, _externalForce, _numParticles, dt);
}

void FLIP3D_Cuda::MoveObject()
{
	if (h_Boxes.size() > 0)
	{
		d_Boxes.copyToHost(h_Boxes);
		//LinearMovingBox_kernel(h_Boxes[0]);
		RotateMovingBox_kernel(h_Boxes[0], true);
		//RotateMovingBox_kernel(h_Boxes[1], false);
		d_Boxes.copyFromHost(h_Boxes);
	}

	if (h_Spheres.size() > 0)
	{
		d_Spheres.copyToHost(h_Spheres);
		LinearMovingSphere_kernel(h_Spheres[0]);
		d_Spheres.copyFromHost(h_Spheres);
	}
}

void FLIP3D_Cuda::CollisionObject_kernel(REAL dt)
{
	if (h_Boxes.size() > 0)
	{
		CollisionMovingBox_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
			(d_Boxes(), d_CurPos(), d_Vel(), d_Type(), _numParticles, _numBoxes, dt);
	}
	if (h_Spheres.size() > 0)
	{
		CollisionMovingSphere_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
			(d_Spheres(), d_CurPos(), d_Vel(), d_Type(), _numParticles, _numSpheres, dt);
	}
}

void FLIP3D_Cuda::InsertFLIPParticles_kernel(/*REAL3* d_newPos, REAL3* d_newVel, REAL* d_newMass, uint numInsertParticles*/)
{
	//임시 파티클 삽입 로직
	vector<REAL3> h_newPos;
	vector<REAL3> h_newVel;
	vector<REAL> h_newMass;

	h_newPos.clear();
	h_newVel.clear();
	h_newMass.clear();

	REAL pourPosX = 0.5;
	REAL pourPosZ = 0.5;
	REAL pourRad = 0.1;

	double w = _dens / _gridRes;
	for (REAL y = w + w / 2.0; y < 1.0 - w / 2.0; y += w) {
		for (REAL z = w + w / 2.0; z < 1.0 - w / 2.0; z += w) {
			if (hypot(y - pourPosX, z - pourPosZ) < pourRad) {
				h_newPos.push_back(make_REAL3(1.0 - _wallThick - 0.2, y, z));
				h_newVel.push_back(make_REAL3(- _dens / _gridRes / 0.005f, 0.0f , 0.0f));
				h_newMass.push_back(1.0);
			}
		}
	}
	uint numInsertParticles = h_newPos.size();

	Dvector<REAL3> d_newPos;
	Dvector<REAL3> d_newVel;
	Dvector<REAL> d_newMass;

	d_newPos.resize(numInsertParticles);
	d_newVel.resize(numInsertParticles);
	d_newMass.resize(numInsertParticles);

	d_newPos.memset(0);
	d_newVel.memset(0);
	d_newMass.memset(0);

	d_newPos.copyFromHost(h_newPos);
	d_newVel.copyFromHost(h_newVel);
	d_newMass.copyFromHost(h_newMass);

	//삽입
	InsertFLIPParticles_D << < divup(numInsertParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_BeforePos(), d_Vel(), d_Normal(), d_Type(), d_Mass(), d_Dens(), d_KernelDens(), d_Flag(), _numParticles, _maxDens,
			d_newPos(), d_newVel(), d_newMass(), numInsertParticles);

	_numParticles += numInsertParticles;

	d_newPos.free();
	d_newVel.free();
	d_newMass.free();

	cudaDeviceSynchronize();
}

void FLIP3D_Cuda::DeleteFLIPParticles_kernel(/*uint* d_deleteIdxes, uint deletNum*/)
{
	Dvector<uint> d_ParticleGridIdx;
	d_ParticleGridIdx.resize(MAXPARTICLENUM);
	d_ParticleGridIdx.memset(0);
	InitParticleIdx_D << < divup(MAXPARTICLENUM, BLOCK_SIZE), BLOCK_SIZE >> > (d_ParticleGridIdx(), _numParticles);

	//삭제
	DeleteFLIPParticles_D << < divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_ParticleGridIdx(), d_CurPos(), d_BeforePos(), d_Vel(), d_Normal(), d_Type(), d_Mass(), d_Dens(), d_KernelDens(), d_Flag(), _numParticles);

	//vel,pos,beforePos, normal, mass, dens, 유지
	//Copy Key
	Dvector<uint> d_key1, d_key2, d_key3, d_key4, d_key5, d_key6, d_key7;
	d_key1.resize(MAXPARTICLENUM);
	d_key2.resize(MAXPARTICLENUM);
	d_key3.resize(MAXPARTICLENUM);
	d_key4.resize(MAXPARTICLENUM);
	d_key5.resize(MAXPARTICLENUM);
	d_key6.resize(MAXPARTICLENUM);
	d_key7.resize(MAXPARTICLENUM);

	d_ParticleGridIdx.copyToDevice(d_key1);
	d_ParticleGridIdx.copyToDevice(d_key2);
	d_ParticleGridIdx.copyToDevice(d_key3);
	d_ParticleGridIdx.copyToDevice(d_key4);
	d_ParticleGridIdx.copyToDevice(d_key5);
	d_ParticleGridIdx.copyToDevice(d_key6);
	d_ParticleGridIdx.copyToDevice(d_key7);

	thrust::sort_by_key(thrust::device_ptr<uint>(d_ParticleGridIdx()),
		thrust::device_ptr<uint>(d_ParticleGridIdx() + MAXPARTICLENUM),
		thrust::device_ptr<REAL3>(d_CurPos()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key1()),
		thrust::device_ptr<uint>(d_key1() + MAXPARTICLENUM),
		thrust::device_ptr<REAL3>(d_BeforePos()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key2()),
		thrust::device_ptr<uint>(d_key2() + MAXPARTICLENUM),
		thrust::device_ptr<REAL3>(d_Vel()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key3()),
		thrust::device_ptr<uint>(d_key3() + MAXPARTICLENUM),
		thrust::device_ptr<REAL3>(d_Normal()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key4()),
		thrust::device_ptr<uint>(d_key4() + MAXPARTICLENUM),
		thrust::device_ptr<uint>(d_Type()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key5()),
		thrust::device_ptr<uint>(d_key5() + MAXPARTICLENUM),
		thrust::device_ptr<REAL>(d_Mass()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key6()),
		thrust::device_ptr<uint>(d_key6() + MAXPARTICLENUM),
		thrust::device_ptr<REAL>(d_Dens()));

	thrust::sort_by_key(thrust::device_ptr<uint>(d_key7()),
		thrust::device_ptr<uint>(d_key7() + MAXPARTICLENUM),
		thrust::device_ptr<BOOL>(d_Flag()));

	Dvector<uint> d_StateData;
	d_StateData.resize(MAXPARTICLENUM);
	d_StateData.memset(0);

	StateCheck_D << <divup(MAXPARTICLENUM, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_StateData());

	ThrustScanWrapper_kernel(d_StateData(), d_StateData(), MAXPARTICLENUM);

	CUDA_CHECK(cudaMemcpy((void*)&_numParticles, (void*)(d_StateData() + MAXPARTICLENUM - 1), sizeof(uint), cudaMemcpyDeviceToHost));

	d_ParticleGridIdx.free();
	d_StateData.free();
}

void FLIP3D_Cuda::ThrustScanWrapper_kernel(uint* output, uint* input, uint numElements)
{
	thrust::exclusive_scan(thrust::device_ptr<uint>(input),
		thrust::device_ptr<uint>(input + (numElements)),
		thrust::device_ptr<uint>(output));
}

void FLIP3D_Cuda::SolvePICFLIP()
{
	ResetCell_kernel();

	TrasnferToGrid_kernel();
	MarkWater_kernel();

	EnforceBoundary_kernel();
	SolvePressure();
	ComputeVelocityWithPress_kernel();
	EnforceBoundary_kernel();
	ExtrapolateVelocity_kernel();

	SubtarctGrid_kernel();
	TrasnferToParticle_kernel();

	GridValueVisualize();
}

void FLIP3D_Cuda::ResetCell_kernel()
{
	ResetCell_D << <_grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes);
}

void FLIP3D_Cuda::TrasnferToGrid_kernel()
{
	TransferToGrid_D << <_grid->_cudaGridSize, _grid->_cudaBlockSize >> >
		(_grid->d_Volumes, d_CurPos(), d_Vel(), d_Type(), d_Mass(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _gridRes, _numParticles);
}

void FLIP3D_Cuda::MarkWater_kernel()
{
	MarkWater_D << <_grid->_cudaGridSize, _grid->_cudaBlockSize >> >
		(_grid->d_Volumes, d_CurPos(), d_Type(), d_Dens(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _dens, _gridRes);
}

void FLIP3D_Cuda::EnforceBoundary_kernel() 
{
	EnforceBoundary_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes);
}

void FLIP3D_Cuda::ComputeDivergence_kernel()
{
	ComputeDivergence_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes);
}

void FLIP3D_Cuda::ComputeLevelSet_kernel()
{
	ComputeLevelSet_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, d_CurPos(), d_Type(), d_Dens(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _dens, _gridRes);
}

void FLIP3D_Cuda::SolvePressure() 
{
	ComputeDivergence_kernel();
	ComputeLevelSet_kernel();

	// step 1: compute # of threads per block
	uint _threads = 16 * 16;
	// step 2: compute # of blocks needed
	uint _blocks = (_gridRes * _gridRes * _gridRes + _threads - 1) / _threads;
	// step 3: find grid's configuration	
	REAL _doubleThreads = (REAL)_blocks;
	uint _k0 = (uint)floor(sqrt(_doubleThreads));
	uint _k1 = (uint)ceil(_doubleThreads / ((REAL)_k0));
	dim3 _numThreads;
	_numThreads.x = 16;
	_numThreads.y = 16;
	_numThreads.z = 1;
	dim3 _numGrid;
	_numGrid.x = _k1;
	_numGrid.y = _k0;
	_numGrid.z = 1;
	uint _size = _gridRes * _gridRes * _gridRes;
	REAL _eps = 1.E-12;
	REAL _oneOverRes2 = (1.0 + _eps) / ((REAL)_gridRes);
	REAL _oneOverRes3 = (1.0 + _eps) / ((REAL)_gridRes);

	//Copy To Solver
	uint* _airD;
	REAL* _levelsetD;
	REAL* _pressureD;
	REAL* _divergenceD;
	cudaMalloc((void**)&_airD, sizeof(uint) * _size);
	cudaMalloc((void**)&_levelsetD, sizeof(REAL) * _size);
	cudaMalloc((void**)&_pressureD, sizeof(REAL) * _size);
	cudaMalloc((void**)&_divergenceD, sizeof(REAL) * _size);

	cudaMemset(_airD, 0, sizeof(uint) * _size);
	cudaMemset(_levelsetD, 0, sizeof(REAL) * _size);
	cudaMemset(_pressureD, 0, sizeof(REAL) * _size);
	cudaMemset(_divergenceD, 0, sizeof(REAL) * _size);

	//////Grid -> Solver Copy
	CopyToSolver_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> >
		(_grid->d_Volumes, _airD, _levelsetD, _pressureD, _divergenceD, _gridRes);

	REAL* _preconditionerD;
	cudaMalloc((void**)&_preconditionerD, sizeof(REAL) * _size);
	cudaMemset(_preconditionerD, 0, sizeof(REAL) * _size);

	BuildPreconditioner_kernel(_preconditionerD, _levelsetD, _airD, _gridRes, _oneOverRes2, _oneOverRes3, _size, _numGrid, _numThreads);

	REAL* _raid1;
	REAL* _raid2;
	REAL* _raid3;
	cudaMalloc((void**)&_raid1, sizeof(REAL) * _size);
	cudaMalloc((void**)&_raid2, sizeof(REAL) * _size);
	cudaMalloc((void**)&_raid3, sizeof(REAL) * _size);

	Solver_kernel(_airD,
		_preconditionerD,
		_levelsetD,
		_pressureD,
		_divergenceD,
		_raid1,
		_raid2,
		_raid3,
		_gridRes,
		_oneOverRes2,
		_oneOverRes3,
		_size,
		_numGrid,
		_numThreads);

	//Solver -> Grid Copy
	CopyToGrid_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> >
		(_grid->d_Volumes, _airD, _levelsetD, _pressureD, _divergenceD, _gridRes);


	//Free
	cudaFree(_airD);
	cudaFree(_levelsetD);
	cudaFree(_pressureD);
	cudaFree(_divergenceD);
	cudaFree(_preconditionerD);
	cudaFree(_raid1);
	cudaFree(_raid2);
	cudaFree(_raid3);
}

void FLIP3D_Cuda::BuildPreconditioner_kernel(REAL* P, REAL* L, uint* A, uint gridSize, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads)
{
	BuildPreconditioner_D << < grid, threads >> > (P, L, A, gridSize, one_over_n2, one_over_n3, sizeOfData, dim3(16, 16, 16));
}

void FLIP3D_Cuda::Solver_kernel(uint* A, REAL* P, REAL* L, REAL* x, REAL* b, REAL* r, REAL* z, REAL* s, uint size, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads)
{
	// host variables
	REAL a_host = 0.0f;
	REAL a2_host = 0.0f;
	REAL tmp1_host = 0.0f;
	REAL error2_0_host = 0.0f;
	REAL error2_host = 0.0f;

	// device variables
	REAL* a_device = NULL;
	REAL* a2_device = NULL;
	REAL* tmp1_device = NULL;
	REAL* error2_device = NULL;
	REAL* error2_0_device = NULL;

	// init buffers
	cudaMalloc(&a_device, sizeof(REAL));
	cudaMalloc(&a2_device, sizeof(REAL));
	cudaMalloc(&tmp1_device, sizeof(REAL));
	cudaMalloc(&error2_device, sizeof(REAL));
	cudaMalloc(&error2_0_device, sizeof(REAL));

	// z=applyA(x)
	Compute_Ax_D << < grid, threads >> > (A, L, x, z, size, one_over_n2, one_over_n3, sizeOfData, grid, threads, dim3(16, 16, 16));

	// r=b-Ax
	Op_Kernel(A, b, z, r, -1.0, size, one_over_n2, one_over_n3, sizeOfData, grid, threads);
	error2_0_host = 0.0f;
	cudaMemcpy(error2_0_device, &error2_0_host, sizeof(REAL), cudaMemcpyHostToDevice);

	// error2_0 = r.r
	//   DotKernel<<< grid, threads >>>(A, r, r, size, error2_0_device, one_over_n2, one_over_n3, sizeOfData, grid, threads); 
	DotHost(A, r, r, size, error2_0_device, one_over_n2, one_over_n3, sizeOfData, grid, threads);
	cudaMemcpy(&error2_0_host, error2_0_device, sizeof(float), cudaMemcpyDeviceToHost);

	// Apply conditioner z = f(r)
	Apply_Preconditioner(z, r, P, L, A, size, one_over_n2, one_over_n3, sizeOfData, grid, threads);
	cudaMemcpy(a_device, &a_host, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(error2_device, &error2_host, sizeof(float), cudaMemcpyHostToDevice);

	// a = z . r
	//   DotKernel<<< grid, threads >>>(A, z, r, size, a_device, one_over_n2, one_over_n3, sizeOfData, grid, threads);
	DotHost(A, z, r, size, a_device, one_over_n2, one_over_n3, sizeOfData, grid, threads);
	cudaMemcpy(&a_host, a_device, sizeof(float), cudaMemcpyDeviceToHost);

	// s = z
	cudaMemcpy(s, z, sizeof(float) * sizeOfData, cudaMemcpyDeviceToDevice);
	cudaMemcpy(a2_device, &a2_host, sizeof(float), cudaMemcpyHostToDevice);

	// tolerance setting
	double eps = 1.0e-2 * (sizeOfData);
	cudaMemcpy(tmp1_device, &tmp1_host, sizeof(float), cudaMemcpyHostToDevice);

	for (int k = 0; k < (int)sizeOfData; k++) {
		// z = applyA(s)
		Compute_Ax_D << < grid, threads >> > (A, L, s, z, size, one_over_n2, one_over_n3, sizeOfData, grid, threads, dim3(16, 16, 16));

		// alpha = a/(z . s)
		//      cudaMemset(tmp1_device, 0, sizeof(float));
		//      DotKernel<<< grid, threads >>>(A, z, s, size, tmp1_device, one_over_n2, one_over_n3, sizeOfData, grid, threads); 
		DotHost(A, z, s, size, tmp1_device, one_over_n2, one_over_n3, sizeOfData, grid, threads);
		cudaMemcpy(&tmp1_host, tmp1_device, sizeof(float), cudaMemcpyDeviceToHost);
		float alpha = a_host / tmp1_host;

		// x = x + alpha*s
		Op_Kernel(A, x, s, x, (float)alpha, size, one_over_n2, one_over_n3, sizeOfData, grid, threads);

		// r = r - alpha*z;
		Op_Kernel(A, r, z, r, (float)-alpha, size, one_over_n2, one_over_n3, sizeOfData, grid, threads);

		// error2 = r.r
		//      cudaMemset(error2_device, 0, sizeof(float));
		//      DotKernel<<< grid, threads >>>(A, r, r, size, error2_device, one_over_n2, one_over_n3, sizeOfData, grid, threads); 
		DotHost(A, r, r, size, error2_device, one_over_n2, one_over_n3, sizeOfData, grid, threads);
		cudaMemcpy(&error2_host, error2_device, sizeof(float), cudaMemcpyDeviceToHost);
		error2_0_host = max(error2_0_host, error2_host);

		if (error2_host <= eps) {
			break;
		}

		// Apply Conditioner z = f(r)
		Apply_Preconditioner(z, r, P, L, A, size, one_over_n2, one_over_n3, sizeOfData, grid, threads);

		// a2 = z . r
		//      cudaMemset(a2_device, 0, sizeof(float));
		//      DotKernel<<< grid, threads >>>(A, z, r, size, a2_device, one_over_n2, one_over_n3, sizeOfData, grid, threads); 
		DotHost(A, z, r, size, a2_device, one_over_n2, one_over_n3, sizeOfData, grid, threads);
		cudaMemcpy(&a2_host, a2_device, sizeof(float), cudaMemcpyDeviceToHost);

		// beta = a2 / a   
		float beta = a2_host / a_host;

		// s = z + beta*s
		Op_Kernel(A, z, s, s, (float)beta, size, one_over_n2, one_over_n3, sizeOfData, grid, threads);
		a_host = a2_host;
	}

	// Free
	cudaFree(a_device);
	cudaFree(a2_device);
	cudaFree(tmp1_device);
	cudaFree(error2_device);
	cudaFree(error2_0_device);

	a_device = NULL;
	a2_device = NULL;
	tmp1_device = NULL;
	error2_device = NULL;
	error2_0_device = NULL;
}

void FLIP3D_Cuda::Op_Kernel(uint* A,
	REAL* x,
	REAL* y,
	REAL* ans,   // copy
	REAL      a,
	uint         size,
	REAL      one_over_n2,
	REAL      one_over_n3,
	uint   sizeOfData,
	dim3         grid,
	dim3         threads)
{
	REAL* tmp;
	cudaMalloc(&tmp, sizeof(REAL) * sizeOfData);
	// operation
	Operator_Kernel << < grid, threads >> > (A,
		x,
		y,
		tmp,
		a,
		size,
		one_over_n2,
		one_over_n3,
		sizeOfData,
		grid,
		threads,
		dim3(16, 16, 16));
	// copy
	Copy_Kernel << < grid, threads >> > (ans,
		tmp,
		size,
		one_over_n2,
		one_over_n3,
		sizeOfData,
		grid,
		threads,
		dim3(16, 16, 16));
	cudaFree(tmp);
}

void FLIP3D_Cuda::DotHost(uint* A, REAL* x, REAL* y, uint size, REAL* result, REAL one_over_n2, REAL one_over_n3, uint sizeOfData, dim3 grid, dim3 threads)
{
	uint* A_host = new uint[sizeOfData];
	REAL* x_host = new REAL[sizeOfData];
	REAL* y_host = new REAL[sizeOfData];
	REAL   result_host = 0.0;
	cudaMemcpy(A_host, A, sizeof(uint) * sizeOfData, cudaMemcpyDeviceToHost);
	cudaMemcpy(x_host, x, sizeof(REAL) * sizeOfData, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_host, y, sizeof(REAL) * sizeOfData, cudaMemcpyDeviceToHost);

#pragma omp for reduction(+:result_host)
	for (int i = 0; i < (int)sizeOfData; i++) {
		if (A_host[i] == FLUID)   result_host += x_host[i] * y_host[i];
	}

	cudaMemcpy(result, &result_host, sizeof(REAL), cudaMemcpyHostToDevice);
	delete[] A_host;
	delete[] x_host;
	delete[] y_host;
}

void FLIP3D_Cuda::Apply_Preconditioner(REAL* z,
	REAL* r,
	REAL* P,
	REAL* L,
	uint* A,
	uint         size,
	REAL      one_over_n2,
	REAL      one_over_n3,
	uint   sizeOfData,
	dim3         grid,
	dim3         threads)
{
	REAL* q;
	cudaMalloc((void**)&q, sizeof(REAL) * sizeOfData);
	cudaMemset(q, 0, sizeof(REAL) * sizeOfData);

	// Lq = r
	Apply_Preconditioner_Kernel << < grid, threads >> > (z,
		r,
		P,
		L,
		A,
		q,
		size,
		one_over_n2,
		one_over_n3,
		sizeOfData,
		grid,
		threads,
		dim3(16, 16, 16));
	// L^T z = q
	Apply_Trans_Preconditioner_Kernel << < grid, threads >> > (z,
		r,
		P,
		L,
		A,
		q,
		size,
		one_over_n2,
		one_over_n3,
		sizeOfData,
		grid,
		threads,
		dim3(16, 16, 16));
	cudaFree(q);

	/*
	Non_Preconditioner(z,
	r,
	P,
	L,
	A,
	size,
	one_over_n2,
	one_over_n3,
	sizeOfData,
	grid,
	threads);
	*/
}


void FLIP3D_Cuda::ComputeVelocityWithPress_kernel()
{
	ComputeVelocityWithPress_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes);
}

void FLIP3D_Cuda::ExtrapolateVelocity_kernel()
{
	ExtrapolateVelocity_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes);
}

void FLIP3D_Cuda::SubtarctGrid_kernel()
{
	SubtarctGrid_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes);
}

void FLIP3D_Cuda::TrasnferToParticle_kernel()
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numParticles, 128, numBlocks, numThreads);
	TrasnferToParticle_D << <numBlocks, numThreads >> > (_grid->d_Volumes, _gridRes, d_CurPos(), d_Vel(), _numParticles);
}

void FLIP3D_Cuda::AdvectParticle_kernel(REAL dt)
{
	AdvecParticle_D << < divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> > 
		(_grid->d_Volumes, d_BeforePos(), d_CurPos(), d_Vel(), d_Type(), _gridRes, _numParticles, dt);

	SetHashTable_kernel();

	ConstraintOuterWall_D << < divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_Vel(), d_Normal(), d_Type(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _numParticles, _gridRes, _dens);
}

void FLIP3D_Cuda::Correct_kernel(REAL dt)
{
	SetHashTable_kernel();

	uint r1 = rand();
	uint r2 = rand();
	uint r3 = rand();
	Correct_D << < divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_Vel(), d_Normal(), d_Mass(), d_Type(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _gridRes, _numParticles, dt, _dens / _gridRes, r1, r2, r3);
}

void FLIP3D_Cuda::SetHashTable_kernel(void)
{
	CalculateHash_kernel();
	SortParticle_kernel();
	FindCellStart_kernel();
}

void FLIP3D_Cuda::CalculateHash_kernel(void)
{
	CalculateHash_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_GridHash(), d_GridIdx(), d_CurPos(), _gridRes, _numParticles);
}

void FLIP3D_Cuda::SortParticle_kernel(void)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(d_GridHash()),
		thrust::device_ptr<uint>(d_GridHash() + _numParticles),
		thrust::device_ptr<uint>(d_GridIdx()));
}

void FLIP3D_Cuda::FindCellStart_kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numParticles, 128, numBlocks, numThreads);

	uint smemSize = sizeof(uint) * (numThreads + 1);
	FindCellStart_D << <numBlocks, numThreads, smemSize >> >
		(d_GridHash(), d_CellStart(), d_CellEnd(), _numParticles);
}

void FLIP3D_Cuda::InitHostMem(void)
{
	h_CurPos.resize(MAXPARTICLENUM, make_REAL3(-1.0, -1.0, -1.0));
	h_BeforePos.resize(MAXPARTICLENUM, make_REAL3(-1.0, -1.0, -1.0));
	h_Vel.resize(MAXPARTICLENUM);
	h_Normal.resize(MAXPARTICLENUM);
	h_Type.resize(MAXPARTICLENUM);
	h_Mass.resize(MAXPARTICLENUM);
	h_Dens.resize(MAXPARTICLENUM);
	h_KernelDens.resize(MAXPARTICLENUM);
	h_Flag.resize(MAXPARTICLENUM);
}

void FLIP3D_Cuda::InitDeviceMem(void)
{
	//Particles
	d_BeforePos.resize(MAXPARTICLENUM);							d_BeforePos.memset(0);
	d_CurPos.resize(MAXPARTICLENUM);								d_CurPos.memset(0);
	d_Vel.resize(MAXPARTICLENUM);								d_Vel.memset(0);
	d_Normal.resize(MAXPARTICLENUM);								d_Normal.memset(0);
	d_Type.resize(MAXPARTICLENUM);								d_Type.memset(0);
	d_Mass.resize(MAXPARTICLENUM);								d_Mass.memset(0);
	d_Dens.resize(MAXPARTICLENUM);								d_Dens.memset(0);
	d_KernelDens.resize(MAXPARTICLENUM);							d_KernelDens.memset(0);
	d_Flag.resize(MAXPARTICLENUM);								d_Flag.memset(0);

	//Hash
	d_GridHash.resize(MAXPARTICLENUM);							d_GridHash.memset(0);
	d_GridIdx.resize(MAXPARTICLENUM);							d_GridIdx.memset(0);
	d_CellStart.resize(_gridRes * _gridRes * _gridRes);			d_CellStart.memset(0);
	d_CellEnd.resize(_gridRes * _gridRes * _gridRes);			d_CellEnd.memset(0);

	////Visualize
	//d_gridPos.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridPos.memset(0);
	//d_gridVel.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridVel.memset(0);
	//d_gridPress.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridPress.memset(0);
	//d_gridDens.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridDens.memset(0);
	//d_gridLevelSet.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridLevelSet.memset(0);
	//d_gridDiv.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridDiv.memset(0);
	//d_gridContent.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridContent.memset(0);
	//printf("Size: %d\n", (_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));

	//Collision Object
	d_Boxes.resize(h_Boxes.size());	d_Boxes.memset(0);
	d_Spheres.resize(h_Spheres.size());	d_Spheres.memset(0);
}

void FLIP3D_Cuda::FreeDeviceMem(void)
{
	//Particles
	d_BeforePos.free();
	d_CurPos.free();
	d_Vel.free();
	d_Normal.free();
	d_Type.free();
	d_Mass.free();
	d_Dens.free();
	d_KernelDens.free();
	d_Flag.free();

	//Hash
	d_GridHash.free();
	d_GridIdx.free();
	d_CellStart.free();
	d_CellEnd.free();

	////Visualize
	//d_gridPos.free();
	//d_gridVel.free();
	//d_gridPress.free();
	//d_gridDens.free();
	//d_gridLevelSet.free();
	//d_gridDiv.free();
	//d_gridContent.free();

	//Collision Object
	d_Boxes.free();
	d_Spheres.free();
}

void FLIP3D_Cuda::CopyToDevice(void)
{
	//Particles
	d_CurPos.copyFromHost(h_CurPos);
	d_BeforePos.copyFromHost(h_BeforePos);
	d_Vel.copyFromHost(h_Vel);
	d_Normal.copyFromHost(h_Normal);
	d_Type.copyFromHost(h_Type);
	d_Mass.copyFromHost(h_Mass);
	d_Dens.copyFromHost(h_Dens);
	d_KernelDens.copyFromHost(h_KernelDens);
	d_Flag.copyFromHost(h_Flag);

	////Visualize
	//d_gridPos.copyFromHost(h_gridPos);
	//d_gridVel.copyFromHost(h_gridVel);
	//d_gridPress.copyFromHost(h_gridPress);
	//d_gridDens.copyFromHost(h_gridDens);
	//d_gridLevelSet.copyFromHost(h_gridLevelSet);
	//d_gridDiv.copyFromHost(h_gridDiv);
	//d_gridContent.copyFromHost(h_gridContent);

	//OBB
	d_Boxes.copyFromHost(h_Boxes);
	d_Spheres.copyFromHost(h_Spheres);
}

void FLIP3D_Cuda::CopyToHost(void)
{
	//Particles
	d_CurPos.copyToHost(h_CurPos);
	d_BeforePos.copyToHost(h_BeforePos);
	d_Vel.copyToHost(h_Vel);
	d_Normal.copyToHost(h_Normal);
	d_Type.copyToHost(h_Type);
	d_Mass.copyToHost(h_Mass);
	d_Dens.copyToHost(h_Dens);
	d_KernelDens.copyToHost(h_KernelDens);
	d_Flag.copyToHost(h_Flag);

	////Visualize
	//d_gridPos.copyToHost(h_gridPos);
	//d_gridVel.copyToHost(h_gridVel);
	//d_gridPress.copyToHost(h_gridPress);
	//d_gridDens.copyToHost(h_gridDens);
	//d_gridLevelSet.copyToHost(h_gridLevelSet);
	//d_gridDiv.copyToHost(h_gridDiv);
	//d_gridContent.copyToHost(h_gridContent);

	//OBB
	d_Boxes.copyToHost(h_Boxes);
	d_Spheres.copyToHost(h_Spheres);
}

void FLIP3D_Cuda::GridValueVisualize(void)
{
	//GridVisualize_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes, d_gridPos(), d_gridVel(), d_gridPress(), d_gridDens(), d_gridLevelSet(), d_gridDiv(), d_gridContent());
}

void FLIP3D_Cuda::draw(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(1.0);
	for (uint i = 0u; i < _numParticles; i++)
	{
		REAL3 position = h_CurPos[i];
		REAL3 velocity = h_Vel[i];
		REAL3 normal = h_Normal[i];
		REAL density = h_Dens[i];
		uint type = h_Type[i];
		BOOL flag = h_Flag[i];

		//if (h_Flag[i]) {
		//	//glPointSize(3.0);
		//	glColor3f(1.0f, 0.0f, 0.0f);
		//}
		//else
		//{
		//	//glPointSize(1.0);
		//	//continue;
		//	glColor3f(0.0f, 0.0f, 1.0f);
		//}

		if (type == WALL ) {
			continue;
			glColor3f(1.0f, 1.0f, 1.0f);
		}
		else
			glColor3f(0.0f, 1.0f, 1.0f);
		////glColor3f(1.0, 1.0, 1.0);
		
		////////////cout << h_Dens[i] << endl;
		//REAL3 color = ScalarToColor(density);
		//glColor3f(color.x, color.y, color.z);

		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();

		//glColor3f(1.0f, 1.0f, 1.0f);
		//glLineWidth(1.0f);
		//glBegin(GL_LINES);
		//float c = 0.2f;
		//glVertex3d(position.x, position.y, position.z);
		//glVertex3d(position.x + velocity.x * c, position.y + velocity.y * c, position.z + velocity.z * c);
		//glEnd();
	}

	//printf("cnt: %d\n", cnt);

#if GRIDRENDER
	for (uint i = 0u; i < _gridRes * _gridRes * _gridRes; i++)
	{
		REAL3 position = h_gridPos[i];
		REAL3 velocity = h_gridVel[i];
		REAL pressure = h_gridPress[i];
		REAL density = h_gridDens[i];
		REAL levelSet = h_gridLevelSet[i];
		REAL divergence = h_gridDiv[i];
		uint content = h_gridContent[i];

		//int x = 15; 
		//int y = 2;
		//int z = 15;
		//int idx = x * _gridRes * _gridRes + y * _gridRes + z;
		//if (i == idx) {
		//	printf("velocity: %f %f %f\n", velocity.x ,velocity.y, velocity.z);
		//	printf("div: %f\n", divergence);
		//	printf("final: %f\n", pressure);
		//}
#if VEL

		glColor3f(1.0f, 1.0f, 1.0f);
		glLineWidth(1.0f);
		glBegin(GL_LINES);
		float c = 0.02f;
		glVertex3d(position.x, position.y, position.z);
		glVertex3d(position.x + velocity.x * c, position.y + velocity.y * c, position.z + velocity.z * c);
		glEnd();
#endif

#if PRESS
		//Visualize Pressure
		if (pressure == 0  )
			continue;

		//if (content == CONTENT_WALL)
		//	printf("%f\n", pressure);

		REAL3 color = ScalarToColor(pressure );
		glColor3f(color.x, color.y, color.z);

		glPointSize(15.0);
		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();
#endif

#if DENSITY
		////Visualize Dens
		if (density == 0 || content == CONTENT_WALL || content == CONTENT_AIR)
			continue;
		REAL3 color = ScalarToColor(density);
		glColor3f(color.x, color.y, color.z);

		glPointSize(15.0);
		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();
#endif

#if LEVEL
		//Visualize Level
		if (content == CONTENT_WALL )
			continue;
		REAL3 color = ScalarToColor(abs(levelSet) * 0.1);
		glColor3f(color.x, color.y, color.z);

		if (content == CONTENT_FLUID)
			glPointSize(15.0);
		else if (content == CONTENT_AIR) {
			continue;
			glPointSize(2.0);
		}
		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();
#endif

#if DIV
		//Visualize Level
		if (content == CONTENT_WALL)
			continue;
		REAL3 color = ScalarToColor(abs(divergence) * 0.1);
		glColor3f(color.x, color.y, color.z);

		if (content == CONTENT_FLUID)
			glPointSize(15.0);
		else if (content == CONTENT_AIR)
			glPointSize(2.0);
		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();
#endif

#if CONTENT
		//////Visualize Content
		if (content == CONTENT_FLUID) {
			//continue;
			glColor3f(0, 0, 1);
			glPointSize(15.0);

		}
		else if (content == CONTENT_AIR) {
			continue;
			glColor3f(0, 1, 0);
			glPointSize(2.0);

		}
		else if (content == CONTENT_WALL) {
			continue;
			glColor3f(1, 1, 1);
			glPointSize(1.0);
		}

		glBegin(GL_POINTS);
		glVertex3d(position.x, position.y, position.z);
		glEnd();

#endif
	}
#endif
	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void  FLIP3D_Cuda::drawBoundingObject(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	for (int i = 0; i < h_Boxes.size(); i++)
	{
		REAL3 center = h_Boxes[i]._center;
		REAL3* corner = h_Boxes[i]._corners;

		//glPointSize(5.0);
		//glColor3f(1.0f, 0.0f, 0.0f);
		//glBegin(GL_POINTS);
		//glVertex3f(center.x, center.y, center.z);
		//glVertex3f((GLfloat)corner[0].x, (GLfloat)corner[0].y, (GLfloat)corner[0].z);
		//glVertex3f((GLfloat)corner[1].x, (GLfloat)corner[1].y, (GLfloat)corner[1].z);
		//glVertex3f((GLfloat)corner[2].x, (GLfloat)corner[2].y, (GLfloat)corner[2].z);
		//glVertex3f((GLfloat)corner[3].x, (GLfloat)corner[3].y, (GLfloat)corner[3].z);
		//glVertex3f((GLfloat)corner[4].x, (GLfloat)corner[4].y, (GLfloat)corner[4].z);
		//glVertex3f((GLfloat)corner[5].x, (GLfloat)corner[5].y, (GLfloat)corner[5].z);
		//glVertex3f((GLfloat)corner[6].x, (GLfloat)corner[6].y, (GLfloat)corner[6].z);
		//glVertex3f((GLfloat)corner[7].x, (GLfloat)corner[7].y, (GLfloat)corner[7].z);
		//glEnd();

		glColor3f(1.0f, 0.0f, 0.0f);
		glBegin(GL_LINES);
		glVertex3f((GLfloat)corner[0].x, (GLfloat)corner[0].y, (GLfloat)corner[0].z);
		glVertex3f((GLfloat)corner[1].x, (GLfloat)corner[1].y, (GLfloat)corner[1].z);
		glVertex3f((GLfloat)corner[2].x, (GLfloat)corner[2].y, (GLfloat)corner[2].z);
		glVertex3f((GLfloat)corner[3].x, (GLfloat)corner[3].y, (GLfloat)corner[3].z);
		glVertex3f((GLfloat)corner[4].x, (GLfloat)corner[4].y, (GLfloat)corner[4].z);
		glVertex3f((GLfloat)corner[5].x, (GLfloat)corner[5].y, (GLfloat)corner[5].z);
		glVertex3f((GLfloat)corner[6].x, (GLfloat)corner[6].y, (GLfloat)corner[6].z);
		glVertex3f((GLfloat)corner[7].x, (GLfloat)corner[7].y, (GLfloat)corner[7].z);
		glVertex3f((GLfloat)corner[0].x, (GLfloat)corner[0].y, (GLfloat)corner[0].z);
		glVertex3f((GLfloat)corner[4].x, (GLfloat)corner[4].y, (GLfloat)corner[4].z);
		glVertex3f((GLfloat)corner[1].x, (GLfloat)corner[1].y, (GLfloat)corner[1].z);
		glVertex3f((GLfloat)corner[5].x, (GLfloat)corner[5].y, (GLfloat)corner[5].z);
		glVertex3f((GLfloat)corner[2].x, (GLfloat)corner[2].y, (GLfloat)corner[2].z);
		glVertex3f((GLfloat)corner[6].x, (GLfloat)corner[6].y, (GLfloat)corner[6].z);
		glVertex3f((GLfloat)corner[3].x, (GLfloat)corner[3].y, (GLfloat)corner[3].z);
		glVertex3f((GLfloat)corner[7].x, (GLfloat)corner[7].y, (GLfloat)corner[7].z);
		glVertex3f((GLfloat)corner[0].x, (GLfloat)corner[0].y, (GLfloat)corner[0].z);
		glVertex3f((GLfloat)corner[2].x, (GLfloat)corner[2].y, (GLfloat)corner[2].z);
		glVertex3f((GLfloat)corner[1].x, (GLfloat)corner[1].y, (GLfloat)corner[1].z);
		glVertex3f((GLfloat)corner[3].x, (GLfloat)corner[3].y, (GLfloat)corner[3].z);
		glVertex3f((GLfloat)corner[4].x, (GLfloat)corner[4].y, (GLfloat)corner[4].z);
		glVertex3f((GLfloat)corner[6].x, (GLfloat)corner[6].y, (GLfloat)corner[6].z);
		glVertex3f((GLfloat)corner[5].x, (GLfloat)corner[5].y, (GLfloat)corner[5].z);
		glVertex3f((GLfloat)corner[7].x, (GLfloat)corner[7].y, (GLfloat)corner[7].z);
		glEnd();
	}
	glPopMatrix();
	glEnable(GL_LIGHTING);

	for (int i = 0; i < h_Spheres.size(); i++)
	{
		REAL3 center = h_Spheres[i]._center;
		REAL radius = h_Spheres[i]._radius;
		glPushMatrix();
		glColor3f(1, 1, 1);
		glTranslatef(center.x, center.y, center.z);
		glutSolidSphere(radius, 20, 20);
		glPushMatrix();
	}
}

REAL3 FLIP3D_Cuda::ScalarToColor(double val)
{
	double fColorMap[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };   //Red->Blue
	double v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	double t = v - low;
	REAL x = (fColorMap[low][0]) * (1 - t) + (fColorMap[high][0]) * t;
	REAL y = (fColorMap[low][1]) * (1 - t) + (fColorMap[high][1]) * t;
	REAL z = (fColorMap[low][2]) * (1 - t) + (fColorMap[high][2]) * t;
	REAL3 color = make_REAL3(x, y, z);
	return color;
}
