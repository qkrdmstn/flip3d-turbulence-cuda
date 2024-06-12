#include "MarchingCubesCuda.h"

MarchingCubes_CUDA::MarchingCubes_CUDA()
{
	_numVoxel = 0;				
	_maxVertices = 0;				
	_numActiveVoxel = 0;			
	_numVertex = 0;	
	_volume = 0;					
	_noise = 0;					
	_voxelVertices = 0;			
	_voxelVerticesScan = 0;		
	_compactedVoxelArray = 0;
	_voxelOccupied = 0;		
	_voxelOccupiedScan = 0;	
	_voxelCubeIndex = 0;
	_edgeOccupied = 0;			
	_edgeOccupiedScan = 0;		
	_edgeVertices = 0;			
	_compactedEdgeVertices = 0;	
	_indexArray = 0;			
	_vertexNormals = 0;
	_numVoxelTriangle = 0;			
	_numVoxelTriangleScan = 0;		
	_4VertexPos = 0;	
	_3TriangleIndex = 0;		
	_scan = 0;
	_maxVertices = 0;
	_vertexStore = 4;
	_set = false;
	_validSurface = false;
}

MarchingCubes_CUDA::~MarchingCubes_CUDA()
{
	free();
}

void MarchingCubes_CUDA::init(FLIP3D_Cuda* fluid, SurfaceTurbulence* turbulence, int rx, int ry, int rz)
{
	_gridSize = make_uint3(rx, ry, rz);
	_numVoxel = _gridSize.x * _gridSize.y * _gridSize.z;
	_voxelMin = make_float3(0.0f, 0.0f, 0.0f);
	_voxelMax = make_float3(1.0f, 1.0f, 1.0f);
	float3 range = make_float3(_voxelMax.x - _voxelMin.x, _voxelMax.y - _voxelMin.y, _voxelMax.z - _voxelMin.z);
	h = make_float3(range.x / _gridSize.x, range.y / _gridSize.y, range.z / _gridSize.z);
	
	_edgeSize[0] = make_uint3(_gridSize.x, _gridSize.y + 1, _gridSize.z + 1);
	_edgeSize[1] = make_uint3(_gridSize.x + 1, _gridSize.y, _gridSize.z + 1);
	_edgeSize[2] = make_uint3(_gridSize.x + 1, _gridSize.y + 1, _gridSize.z);
	_numEdge[0] = _gridSize.x*(_gridSize.y + 1)*(_gridSize.z + 1);
	_numEdge[1] = (_gridSize.x + 1)*_gridSize.y*(_gridSize.z + 1);
	_numEdge[2] = (_gridSize.x + 1)*(_gridSize.y + 1)*_gridSize.z;
	_numEdge[3] = _numEdge[0] + _numEdge[1] + _numEdge[2];

	int vm = 10;
	_maxVertices = _gridSize.x*_gridSize.y*vm;
	_vertexStore = vm;
	_numTriangle = 0;
	_numVertex = 0;

	int size = _gridSize.x*_gridSize.y*_gridSize.z*sizeof(float);
	cudaMalloc((void**)&_volume, size);
	cudaMemset((void*)_volume, 0, size);
	cudaMalloc((void**)&_noise, size);
	cudaMemset((void*)_noise, 0, size);

	InitTableMC();

	unsigned int memSize = sizeof(uint)*_numVoxel;
	cudaMalloc((void**)&_voxelVertices, memSize);
	cudaMalloc((void**)&_voxelVerticesScan, memSize);

	cudaMalloc((void**)&_voxelOccupied, memSize);
	cudaMalloc((void**)&_voxelOccupiedScan, memSize);
	cudaMalloc((void**)&_compactedVoxelArray, memSize);

	cudaMalloc((void**)&_voxelCubeIndex, memSize);

	cudaMalloc((void**)&_numVoxelTriangle, memSize);
	cudaMalloc((void**)&_numVoxelTriangleScan, memSize);

	cudaMemset((void*)_voxelCubeIndex, 0, memSize);
	cudaMemset((void*)_numVoxelTriangle, 0, memSize);
	cudaMemset((void*)_numVoxelTriangleScan, 0, memSize);

	memSize = sizeof(uint)*_numEdge[3];
	cudaMalloc((void**)&_edgeOccupied, memSize);
	cudaMalloc((void**)&_edgeOccupiedScan, memSize);

	cudaMemset((void*)_edgeOccupied, 0, memSize);
	cudaMemset((void*)_edgeOccupiedScan, 0, memSize);

	memSize = sizeof(float)* 4 * _numEdge[3];
	cudaMalloc((void**)&_edgeVertices, memSize);
	cudaMalloc((void**)&_compactedEdgeVertices, memSize);

	cudaMemset((void*)_edgeVertices, 0, memSize);
	cudaMemset((void*)_compactedEdgeVertices, 0, memSize);

	memSize = sizeof(float)* 3 * _numEdge[3];
	cudaMalloc((void**)&_vertexNormals, memSize);
	cudaMemset((void*)_vertexNormals, 0, memSize);

	memSize = sizeof(uint)* 3 * _maxVertices * 3;
	cudaMalloc((void**)&_indexArray, memSize);
	cudaMemset((void*)_indexArray, 0, memSize);

	_4VertexPos = (float4*)malloc(sizeof(float4)*_maxVertices);
	memset(_4VertexPos, 0, sizeof(float4)*_maxVertices);

	_3TriangleIndex = (uint3*)malloc(sizeof(uint3)*_maxVertices * 3);
	memset(_3TriangleIndex, 0, sizeof(uint3)*_maxVertices * 3);

	_scan = (uint*)malloc(sizeof(uint)*_numEdge[3]);
	memset(_scan, 0, sizeof(uint)*_numEdge[3]);

	_set = true;

	_fluid = fluid;
	_turbulence = turbulence;
	hashRes = _fluid->_gridRes * 2;

	memSize = sizeof(REAL3) * _numVoxel;
	cudaMalloc((void**)&d_gridPosition, memSize);
	cudaMemset((void*)d_gridPosition, 0, memSize);

	memSize = sizeof(uint) * _numVoxel;
	cudaMalloc(&d_CellStart, memSize);
	cudaMalloc(&d_CellEnd, memSize);

	//Visualize Init
	h_gridPosition = (REAL3*)malloc(sizeof(REAL3) * _numVoxel * 1);
	memset(h_gridPosition, 0, sizeof(REAL3)* _numVoxel * 1);
	h_level = (REAL*)malloc(sizeof(REAL)* _numVoxel * 1);
	memset(h_level, 0, sizeof(REAL)* _numVoxel * 1);

	h_Vertices.resize(_maxVertices * 2);
	h_VertexNormals.resize(_maxVertices * 2);
	h_Faces.resize(_maxVertices * 4);
}

void MarchingCubes_CUDA::free(void)
{
	if (_set) {
		cudaFree(_volume);
		cudaFree(_noise);
		cudaFree(_voxelVertices);
		cudaFree(_voxelVerticesScan);
		cudaFree(_voxelOccupied);
		cudaFree(_voxelOccupiedScan);
		cudaFree(_compactedVoxelArray);		
		ClearTableMC();
		cudaFree(_voxelCubeIndex);
		cudaFree(_numVoxelTriangle);
		cudaFree(_numVoxelTriangleScan);
		cudaFree(_indexArray);
		cudaFree(_edgeOccupied);
		cudaFree(_edgeOccupiedScan);
		cudaFree(_edgeVertices);
		cudaFree(_compactedEdgeVertices);
		cudaFree(_vertexNormals);
		if (_3TriangleIndex != 0) std::free(_3TriangleIndex);
		if (_scan != 0) std::free(_scan);
		if (_4VertexPos != 0) std::free(_4VertexPos);
		_set = false;

		cudaFree(d_gridPosition);
		cudaFree(d_CellStart);
		cudaFree(d_CellEnd);
	}
}

void MarchingCubes_CUDA::surfaceRecon(float threshold)
{
	_numVertex = 0;

	CuMCCalTriNum(_volume, _voxelCubeIndex, _voxelVertices, _voxelVerticesScan,
		_numVoxelTriangle, _numVoxelTriangleScan, _voxelOccupied, _voxelOccupiedScan, _compactedVoxelArray,
		_gridSize, _numVoxel, h, threshold,
		_numActiveVoxel, _numVertex, _numTriangle);

	CuMCCalEdgeVrts(_volume, _edgeVertices, _compactedEdgeVertices,
		_edgeOccupied, _edgeOccupiedScan, _edgeSize, _numEdge,
		_gridSize, _numVoxel, h, _voxelMin, threshold,
		_numVertex);

	if (_numVertex){
		CuMCCalTri(_indexArray, _voxelCubeIndex, _numVoxelTriangleScan, _compactedVoxelArray,
			_edgeOccupiedScan, _edgeSize, _numEdge,
			_gridSize, _numVoxel, h, threshold,
			_numActiveVoxel, _numVertex, _numTriangle, _compactedEdgeVertices);
	} else {
		_numTriangle = 0;
	}
}

void MarchingCubes_CUDA::copyCPU(vector<vec3> &vertices, vector<vec3> &normals, vector<int> &faces)
{
	faces.clear();
	normals.clear();
	vertices.clear();

	float *vertexPtr = new float[_numVertex * 4];
	uint *trianglePtr = new uint[_numTriangle * 3];

	cudaMemcpy(vertexPtr, _compactedEdgeVertices, _numVertex * 4 * sizeof(float), cudaMemcpyDeviceToHost);
	vertices.resize(_numVertex);

	for (uint i = 0; i < _numVertex; ++i){
		vertices[i][0] = vertexPtr[4 * i];
		vertices[i][1] = vertexPtr[4 * i + 1];
		vertices[i][2] = vertexPtr[4 * i + 2];
	}
	delete[] vertexPtr;
	vertexPtr = nullptr;

	normals.resize(_numVertex);
	faces.resize(_numTriangle * 3);
	cudaMemcpy(trianglePtr, _indexArray, _numTriangle * 3 * sizeof(uint), cudaMemcpyDeviceToHost);

	for (uint i = 0; i < _numTriangle; ++i){
		faces[3 * i] = trianglePtr[3 * i];
		faces[3 * i + 1] = trianglePtr[3 * i + 1];
		faces[3 * i + 2] = trianglePtr[3 * i + 2];	
		vec3 v0 = vertices[faces[3 * i]];
		vec3 v1 = vertices[faces[3 * i + 1]];
		vec3 v2 = vertices[faces[3 * i + 2]];
		vec3 vn = ((v1 - v0).cross(v2 - v0));
		vn.normalize();
		normals[faces[3 * i]] += vn;
		normals[faces[3 * i + 1]] += vn;
		normals[faces[3 * i + 2]] += vn;
	}

	for (uint i = 0; i < _numVertex; ++i){
		normals[i].normalize();
	}

	delete[] trianglePtr;
	trianglePtr = nullptr;

	cudaMemcpy(h_level, _volume, sizeof(REAL) * _numVoxel, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_gridPosition, d_gridPosition, sizeof(REAL3) * _numVoxel, cudaMemcpyDeviceToHost);
}

void MarchingCubes_CUDA::SetHashTable_kernel(void)
{
	CalculateHash_kernel(d_GridHash, d_GridIdx, d_TotalParticles, hashRes, _numTotalParticles);
	SortParticle_kernel(d_GridHash, d_GridIdx, _numTotalParticles);
	FindCellStart_kernel(d_GridHash, d_CellStart, d_CellEnd, _numTotalParticles);
}

void MarchingCubes_CUDA::Smoothing(void)
{
	// Build connection of vertices
	vector<vector<int>> connections;
	connections.resize(h_Vertices.size());
	int numberFace = (int)h_Faces.size() / 3;

	for (int i = 0; i < numberFace; i++) {
		connections[h_Faces[i * 3]].push_back(h_Faces[i * 3 + 1]);
		connections[h_Faces[i * 3]].push_back(h_Faces[i * 3 + 2]);
		connections[h_Faces[i * 3 + 1]].push_back(h_Faces[i * 3]);
		connections[h_Faces[i * 3 + 1]].push_back(h_Faces[i * 3 + 2]);
		connections[h_Faces[i * 3 + 2]].push_back(h_Faces[i * 3]);
		connections[h_Faces[i * 3 + 2]].push_back(h_Faces[i * 3 + 1]);
	}

	// Smoothing
	vector<vec3> copy_vetices;
	vector<vec3> copy_normals;

	copy_vetices.resize(h_Vertices.size());
	copy_normals.resize(h_VertexNormals.size());

	int numit = 2;
	int do_vertices = 1;
	int do_normal = 1;

	for (int k = 0; k < numit; k++) {
#pragma omp parallel for
		for (int vi = 0; vi < connections.size(); vi++) {
			FLOAT wsum = 0.0;
			double v[3] = { 0.0, 0.0, 0.0 };
			double vn[3] = { 0.0, 0.0, 0.0 };

			// Add itself
			{
				bool isOK = true;
				for (int n = 0; n < 3; n++) {
					if (!_finite(h_Vertices[vi][n])) isOK = false;
					if (!_finite(h_VertexNormals[vi][n])) isOK = false;
				}

				if (isOK) {
					FLOAT w = 1.0;
					for (int n = 0; n < 3; n++) {
						v[n] += w * h_Vertices[vi][n];
						vn[n] += w * h_VertexNormals[vi][n];
					}
					wsum += w;
				}
			}

			// Add neighbors
			for (int i = 0; i < connections[vi].size(); i++) {
				int ni = connections[vi][i];
				FLOAT w = 1.0;

				bool isOK = true;
				for (int n = 0; n < 3; n++) {
					if (!_finite(h_Vertices[ni][n])) isOK = false;
					if (!_finite(h_VertexNormals[ni][n])) isOK = false;
				}

				if (isOK) {
					for (int n = 0; n < 3; n++) {
						v[n] += w * h_Vertices[ni][n];
						vn[n] += w * h_VertexNormals[ni][n];
					}
					wsum += w;
				}
			}

			double len = _hypotf((float)vn[0], _hypotf((float)vn[1], (float)vn[2]));
			for (int n = 0; n < 3; n++) {
				if (wsum) copy_vetices[vi][n] = v[n] / (double)wsum;
				if (len) copy_normals[vi][n] = vn[n] / len;
			}
		}

#pragma omp parallel for
		for (int vi = 0; vi < connections.size(); vi++) {
			FLOAT a = 1.0;
			if (_isnan(a)) {
				a = 1.0;
				for (int n = 0; n < 3; n++) {
					h_Vertices[vi][n] = h_VertexNormals[vi][n] = 0.0;
				}
			}

			for (int n = 0; n < 3; n++) {
				if (do_vertices) h_Vertices[vi][n] = copy_vetices[vi][n];
				if (do_normal) h_VertexNormals[vi][n] = copy_normals[vi][n];
			}
		}
	}
}

void MarchingCubes_CUDA::MarchingCubes()
{
	_numTotalParticles = _fluid->_numParticles + _turbulence->_numFineParticles;
	printf("Num of MC Particles %d\n", _numTotalParticles);
	
	unsigned int memSize = sizeof(REAL3) * _numTotalParticles;
	cudaMalloc(&d_TotalParticles, memSize);
	cudaMemset(d_TotalParticles, 0, memSize);

	memSize = sizeof(uint) * _numTotalParticles;
	cudaMalloc(&d_Type, memSize);
	cudaMemset(d_Type, 0, memSize);

	cudaMalloc(&d_GridHash, memSize);
	cudaMemset(d_GridHash, 0, memSize);

	cudaMalloc(&d_GridIdx, memSize);
	cudaMemset(d_GridIdx, 0, memSize);

	memSize = sizeof(uint) * _numVoxel;
	cudaMemset(d_CellStart, 0, memSize);
	cudaMemset(d_CellEnd, 0, memSize);

	CopyToTotalParticles_kernel(_fluid, _turbulence, d_TotalParticles, d_Type, _numTotalParticles);
	SetHashTable_kernel();

	ComputeLevelset_kernel(d_gridPosition, d_TotalParticles, d_Type, _volume, d_GridIdx,d_CellStart, d_CellEnd, _numTotalParticles, _gridSize, hashRes, _fluid->_numParticles, _turbulence->d_SurfaceNormal());
	surfaceRecon(0.0f);
	Smoothing();

	copyCPU(h_Vertices, h_VertexNormals, h_Faces);

	cudaFree(d_TotalParticles);
	cudaFree(d_Type);
	cudaFree(d_GridHash);
	cudaFree(d_GridIdx);

}

REAL3 ScalarToColor(double val)
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

void MarchingCubes_CUDA::renderSurface(void)
{
	float diffuse[] = { 0.34117647058823529411764705882353f, 0.52156862745098039215686274509804f, 0.84705882352941176470588235294118f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);

	glPushMatrix();
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHTING);
	int numberFace = (int)h_Faces.size() / 3;
	for (int i = 0; i < numberFace; i++) {
		glBegin(GL_POLYGON);
		for (int j = 0; j < 3; j++) {
			vec3 vertexNormal = h_VertexNormals[h_Faces[i * 3 + j]];
			vec3 vertex = h_Vertices[h_Faces[i * 3 + j]];
			glNormal3d(vertexNormal.x(), vertexNormal.y(), vertexNormal.z());
			glVertex3d(vertex.x(), vertex.y(), vertex.z());
		}
		glEnd();
	}
	glShadeModel(GL_FLAT);
	glPopMatrix();
	diffuse[0] = 1.0f;
	diffuse[1] = 1.0f;
	diffuse[2] = 1.0f;
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glFlush();

	//glPushMatrix();
	//glDisable(GL_LIGHTING);
	//glPointSize(1.0);
	//for (uint i = 0u; i < _numVoxel; i++)
	//{
	//	REAL3 position = h_gridPosition[i];
	//	REAL level = h_level[i];

	//	//if (isnan(level))
	//	//	continue;
	//	if (fabs(level) > 0.8501f)
	//		continue;
	//	REAL3 color = ScalarToColor(fabs(level));
	//	glColor3f(color.x, color.y, color.z);

	//	glBegin(GL_POINTS);
	//	glVertex3d(position.x, position.y, position.z);
	//	glEnd();
	//}
	//glPointSize(1.0);
	//glEnable(GL_LIGHTING);
	//glPopMatrix();
}




