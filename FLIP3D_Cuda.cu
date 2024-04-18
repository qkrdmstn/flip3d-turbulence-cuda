#include "FLIP3D_Cuda.cuh"

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
	_dens = 0.5;
	_maxDens = 92.9375;

	_grid = new FLIPGRID(_gridRes, _cellPhysicalSize);

	ParticleInit();
	_numParticles = h_CurPos.size();
	cout << _numParticles << endl;
	ComputeWallNormal();


	//For grid visualize
	h_gridPos.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 
	h_gridVel.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 
	h_gridPress.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 
	h_gridContent.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1)); 

	InitDeviceMem();
	CopyToDevice();
	//cudaDeviceSynchronize();
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

	for (int iter = 0; iter < h_CurPos.size(); iter++)
	{
		if (h_Type[iter] == WALL) {
			iter++;
			continue;
		}
		int i = fmin(_gridRes - 1, fmax(0, h_CurPos[iter].x * _gridRes));
		int j = fmin(_gridRes - 1, fmax(0, h_CurPos[iter].y * _gridRes));
		int k = fmin(_gridRes - 1, fmax(0, h_CurPos[iter].z * _gridRes));
	}
}

void FLIP3D_Cuda::PlaceObjects()
{
	PlaceWalls();

	WaterDropTest();
	//DamBreakTest();
}

void FLIP3D_Cuda::PlaceWalls()
{
	Object obj;

	// Left Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.visible = 0;
	obj.p[0].x = 0.0;			obj.p[1].x = _wallThick; //Box min, max 값
	obj.p[0].y = 0.0;			obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;			obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Right Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.visible = 0;
	obj.p[0].x = 1.0 - _wallThick;	obj.p[1].x = 1.0;
	obj.p[0].y = 0.0;				obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;				obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Floor Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.visible = 0;
	obj.p[0].x = 0.0;	obj.p[1].x = 1.0;
	obj.p[0].y = 0.0;	obj.p[1].y = _wallThick;
	obj.p[0].z = 0.0;	obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Ceiling Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.visible = 0;
	obj.p[0].x = 0.0;				obj.p[1].x = 1.0;
	obj.p[0].y = 1.0 - _wallThick;	obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;				obj.p[1].z = 1.0;
	objects.push_back(obj);

	// Front Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.visible = 0;
	obj.p[0].x = 0.0;	obj.p[1].x = 1.0;
	obj.p[0].y = 0.0;	obj.p[1].y = 1.0;
	obj.p[0].z = 0.0;	obj.p[1].z = _wallThick;
	objects.push_back(obj);

	// Back Wall
	obj.type = WALL;
	obj.shape = BOX;
	obj.material = GLASS;
	obj.visible = 0;
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
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.2;
	obj.p[0].z = _wallThick;	obj.p[1].z = 1.0 - _wallThick;
	objects.push_back(obj);

	obj.type = FLUID;
	obj.shape = SPHERE;
	obj.c.x = 0.5;
	obj.c.y = 0.6;
	obj.c.z = 0.5;
	obj.r = 0.12;
	objects.push_back(obj);
}

void FLIP3D_Cuda::DamBreakTest()
{
	Object obj;

	obj.type = FLUID;
	obj.shape = BOX;
	obj.visible = true;
	obj.p[0].x = 0.2;	obj.p[1].x = 0.4;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.4;
	obj.p[0].z = 0.2;	obj.p[1].z = 0.8;

	objects.push_back(obj);

	obj.type = FLUID;
	obj.shape = BOX;
	obj.visible = true;
	obj.p[0].x = _wallThick;	obj.p[1].x = 1.0 - _wallThick;
	obj.p[0].y = _wallThick;	obj.p[1].y = 0.06;
	obj.p[0].z = _wallThick;	obj.p[1].z = 1.0 - _wallThick;

	objects.push_back(obj);
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
		uint visible = inside_obj->visible;
		REAL mass = 1.0;
		BOOL flag = false;

		h_BeforePos.push_back(beforePos);
		h_CurPos.push_back(curPos);
		h_Vel.push_back(vel);
		h_Normal.push_back(normal);
		h_Dens.push_back(dens);
		h_Type.push_back(type);
		h_Visible.push_back(visible);
		h_Mass.push_back(mass);
		h_Flag.push_back(false);
	}
}

void FLIP3D_Cuda::ComputeWallNormal()
{
	for (int i = 0; i < _numParticles; i++)
	{
		if (h_Type[i] == WALL)
		{
			if (h_CurPos[i].x <= 1.1 * _wallThick) {
				h_Normal[i].x = 1.0;
			}
			if (h_CurPos[i].x >= 1.0 - 1.1 * _wallThick) {
				h_Normal[i].x = -1.0;
			}
			if (h_CurPos[i].y <= 1.1 * _wallThick) {
				h_Normal[i].y = 1.0;
			}
			if (h_CurPos[i].y >= 1.0 - 1.1 * _wallThick) {
				h_Normal[i].y = -1.0;
			}
			if (h_CurPos[i].z <= 1.1 * _wallThick) {
				h_Normal[i].z = 1.0;
			}
			if (h_CurPos[i].z >= 1.0 - 1.1 * _wallThick) {
				h_Normal[i].z = -1.0;
			}
		}
	}
}

void FLIP3D_Cuda::ComputeParticleDensity_kernel()
{
	ComputeParticleDensity_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_Type(), d_Dens(), d_Mass(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _gridRes, _numParticles, _dens, _maxDens, d_Flag());
}

void FLIP3D_Cuda::ComputeExternalForce_kernel(REAL3& gravity, REAL dt)
{
	CompExternlaForce_D << <divup(_numParticles, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_CurPos(), d_Vel(), gravity, _externalForce, _numParticles, dt);
}

void FLIP3D_Cuda::SolvePICFLIP()
{
	//ResetCell_kernel();

	TrasnferToGrid_kernel();
	MarkWater_kernel();

	ComputeGridDensity_kernel();
	EnforceBoundary_kernel();
	ComputeDivergence_kernel();
	ComputeLevelSet_kernel();
	SolvePressureJacobi_kernel();
	ComputeVelocityWithPress_kernel();
	EnforceBoundary_kernel();
	ExtrapolateVelocity_kernel();
	gridValueVisualize();

	SubtarctGrid_kernel();
	TrasnferToParticle_kernel();
}

void FLIP3D_Cuda::ResetCell_kernel()
{
	ResetCell_D << <_grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes);
}

void FLIP3D_Cuda::TrasnferToGrid_kernel()
{
	TrasnferToGrid_D << <_grid->_cudaGridSize, _grid->_cudaBlockSize >> >
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

	//uint total = _gridRes * _gridRes;
	//uint numThreads = min(1024u, total);
	//uint numBlocks = divup(total, numThreads);
	//FixBoundaryX << < numBlocks, numThreads >> > (_grid->d_Volumes, _gridRes);
	//cudaDeviceSynchronize();

	//FixBoundaryY << < numBlocks, numThreads >> > (_grid->d_Volumes, _gridRes);
	//cudaDeviceSynchronize();

	//FixBoundaryZ << < numBlocks, numThreads >> > (_grid->d_Volumes, _gridRes);
	//cudaDeviceSynchronize();
}

void FLIP3D_Cuda::ComputeDivergence_kernel()
{
	ComputeDivergence_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, d_Dens(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(),_gridRes);
}

void FLIP3D_Cuda::ComputeLevelSet_kernel()
{
	ComputeLevelSet_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, d_CurPos(), d_Type(), d_Dens(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _dens, _gridRes);
}

void FLIP3D_Cuda::ComputeGridDensity_kernel()
{
	ComputeGridDensity_D << <_grid->_cudaGridSize, _grid->_cudaBlockSize >> >
		(_grid->d_Volumes, d_CurPos(), d_Type(), d_Mass(), d_GridHash(), d_GridIdx(), d_CellStart(), d_CellEnd(), _dens, _maxDens, _gridRes, d_Flag());
}

void FLIP3D_Cuda::SolvePressureJacobi_kernel()
{
	for (int i = 0; i < _iterations; i++)
	{
		SolvePressureJacobi_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > 
			(_grid->d_Volumes, _gridRes, d_gridPress());
	}
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
		(d_BeforePos(), d_CurPos(), d_Vel(), d_Type(), _gridRes, _numParticles, dt);
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
	FindCellStart_D << <numBlocks, numThreads >> >
		(d_GridHash(), d_CellStart(), d_CellEnd(), _numParticles);
}


void FLIP3D_Cuda::InitDeviceMem(void)
{
	d_GridHash.resize(_numParticles);			d_GridHash.memset(0);
	d_GridIdx.resize(_numParticles);			d_GridIdx.memset(0);
	d_CellStart.resize(_gridRes * _gridRes * _gridRes);			d_CellStart.memset(0);
	d_CellEnd.resize(_gridRes * _gridRes * _gridRes);			d_CellEnd.memset(0);

	d_BeforePos.resize(_numParticles);			d_BeforePos.memset(0);
	d_CurPos.resize(_numParticles);			d_CurPos.memset(0);
	d_Vel.resize(_numParticles);			d_Vel.memset(0);
	d_Normal.resize(_numParticles);			d_Normal.memset(0);
	d_Type.resize(_numParticles);			d_Type.memset(0);
	d_Visible.resize(_numParticles);		d_Visible.memset(0);
	d_Remove.resize(_numParticles);			d_Remove.memset(0);
	d_Mass.resize(_numParticles);			d_Mass.memset(0);
	d_Dens.resize(_numParticles);			d_Dens.memset(0);

	d_Flag.resize(_numParticles);			d_Flag.memset(0);

	d_gridPos.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridPos.memset(0);
	d_gridVel.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridVel.memset(0);
	d_gridPress.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridPress.memset(0);
	d_gridContent.resize((_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));		d_gridContent.memset(0);
	printf("Size: %d\n", (_gridRes + 1) * (_gridRes + 1) * (_gridRes + 1));
}

void FLIP3D_Cuda::FreeDeviceMem(void)
{
	d_GridHash.free();
	d_GridIdx.free();
	d_CellStart.free();
	d_CellEnd.free();

	d_BeforePos.free();
	d_CurPos.free();
	d_Vel.free();
	d_Normal.free();
	d_Type.free();
	d_Visible.free();
	d_Remove.free();
	d_Mass.free();
	d_Dens.free();

	d_Flag.free();
	d_gridPos.free();
	d_gridVel.free();
	d_gridPress.free();
	d_gridContent.free();
}

void FLIP3D_Cuda::CopyToDevice(void)
{
	d_BeforePos.copyFromHost(h_BeforePos);
	d_CurPos.copyFromHost(h_CurPos);
	d_Vel.copyFromHost(h_Vel);
	d_Normal.copyFromHost(h_Normal);
	d_Type.copyFromHost(h_Type);
	d_Visible.copyFromDevice(h_Visible);
	d_Remove.copyFromHost(h_Remove);
	d_Mass.copyFromHost(h_Mass);
	d_Dens.copyFromHost(h_Dens);

	d_Flag.copyFromHost(h_Flag);
	d_gridPos.copyFromHost(h_gridPos);
	d_gridVel.copyFromHost(h_gridVel);
	d_gridPress.copyFromHost(h_gridPress);
	d_gridContent.copyFromHost(h_gridContent);
}

void FLIP3D_Cuda::CopyToHost(void)
{
	d_BeforePos.copyToHost(h_BeforePos);
	d_CurPos.copyToHost(h_CurPos);
	d_Vel.copyToHost(h_Vel);
	d_Normal.copyToHost(h_Normal);
	d_Type.copyToHost(h_Type);
	d_Visible.copyToHost(h_Visible);
	d_Remove.copyToHost(h_Remove);
	d_Mass.copyToHost(h_Mass);
	d_Dens.copyToHost(h_Dens);

	d_Flag.copyToHost(h_Flag);
	d_gridPos.copyToHost(h_gridPos);
	d_gridVel.copyToHost(h_gridVel);
	d_gridPress.copyToHost(h_gridPress);
	d_gridContent.copyToHost(h_gridContent);
}

void FLIP3D_Cuda::gridValueVisualize(void)
{
	GridVisualize_D << < _grid->_cudaGridSize, _grid->_cudaBlockSize >> > (_grid->d_Volumes, _gridRes, d_gridPos(), d_gridVel(), d_gridPress(), d_gridContent());
}

void FLIP3D_Cuda::draw(void)
{
	int cnt = 0;
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(2.0);
	for (uint i = 0u; i < _numParticles; i++)
	{
		//if (h_Flag[i])
		//	glColor3f(1.0f, 0.0f, 0.0f);
		//else
		//{
		//	//continue;
		//	glColor4f(0.0f, 0.0f, 1.0f, 0.3);
		//}
		
		////////cout << h_Dens[i] << endl;
		//REAL3 color = ScalarToColor(h_Dens[i]);
		//glColor3f(color.x, color.y, color.z);

		if (h_Type[i] == WALL) {
			continue;
			glColor3f(1.0f, 1.0f, 1.0f);
		}
		else
			glColor3f(0.0f, 1.0f, 1.0f);
		//glColor3f(1.0, 1.0, 1.0);


		glBegin(GL_POINTS);
		glVertex3d(h_CurPos[i].x, h_CurPos[i].y, h_CurPos[i].z);
		glEnd();
	}


	for (uint i = 0u; i < _gridRes * _gridRes * _gridRes; i++)
	{
		//glColor3f(1.0f, 1.0f, 1.0f);
		//glLineWidth(1.0f);
		//glBegin(GL_LINES);
		//float c = 0.2f;
		//glVertex3d(h_gridPos[i].x, h_gridPos[i].y, h_gridPos[i].z);
		//glVertex3d(h_gridPos[i].x + h_gridVel[i].x * c, h_gridPos[i].y + h_gridVel[i].y * c, h_gridPos[i].z + h_gridVel[i].z * c);
		//glEnd();


		//Visualize Pressure
		if (h_gridPress[i] == 0)
			continue;
		REAL3 color = ScalarToColor(h_gridPress[i] * 10);
		glColor3f(color.x, color.y, color.z);

		//// Draw Solid Sphere
		//glColor3f(0.0f, 0.0f, 1.0f);
		//glPushMatrix();
		//glTranslatef(h_gridPos[i].x, h_gridPos[i].y, h_gridPos[i].z);

		////glutSolidSphere(0.1/_gridRes, 20, 20);
		//glutWireSphere(0.3 / _gridRes, 20, 20);

		//glColor3f(1.0f, 1.0f, 1.0f);
		//glPopMatrix();

		glPointSize(15.0);
		glBegin(GL_POINTS);
		glVertex3d(h_gridPos[i].x, h_gridPos[i].y, h_gridPos[i].z);
		glEnd();


		////Visualize Content
		//uint content = h_gridContent[i];
		////if (A[i][j][k] == WALL)
		//	//	continue;

		//if (content == CONTENT_FLUID) {
		//	//continue;
		//	glColor3f(0, 0, 1);
		//	glPointSize(15.0);

		//}
		//else if (content == CONTENT_AIR) {
		//	//continue;
		//	glColor3f(0, 1, 0);
		//	glPointSize(2.0);

		//}
		//else if (content == CONTENT_WALL) {
		//	//continue;
		//	glColor3f(1, 1, 1);
		//	glPointSize(1.0);
		//}

		//glBegin(GL_POINTS);
		//glVertex3d(h_gridPos[i].x, h_gridPos[i].y, h_gridPos[i].z);
		//glEnd();

	}

	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
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