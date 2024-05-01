#include "SurfaceTurbulence.h"
#include <time.h>
#include <cmath>

void SurfaceTurbulencen::Initialize(void)
{
	_sorter->sort(_coarseParticles);

	unsigned int discretization = (unsigned int)PI * (outerRadius + innerRadius) / _fineScaleLen;
	double dtheta = 2.0 * _fineScaleLen / (outerRadius + innerRadius);
	double outerRadius2 = outerRadius * outerRadius;

	_fineParticles.clear();
	for (int i = 0; i < _coarseParticles.size(); i++) {
		particle* cp = _coarseParticles[i];
		if (i % 500 == 0) printf("Initializing surface points : %.4f %\n", (float)i / (float)_coarseParticles.size());
		if (cp->type != FLUID) continue;

		//check flag if we are near surface
		bool nearSurface = false;
		vec3 pos(cp->p[0], cp->p[1], cp->p[2]);
		for (int i = -1; i <= 1; i++) { 
			for (int j = -1; j <= 1; j++) {
				for (int k = -1; k <= 1; k++) {

					double cell_size = 1.0 / _coarseScaleLen;
					double x = max(0.0, min(cell_size, cell_size * pos.x()));
					double y = max(0.0, min(cell_size, cell_size * pos.y()));
					double z = max(0.0, min(cell_size, cell_size * pos.z()));

					int indexX = x + i;
					int indexY = y + j;
					int indexZ = z + k;
					if (indexX < 0 || indexY < 0 || indexZ < 0) continue;

					if (_sorter->getNumFluidParticleAt(x + i, y + j, z + k) == 0) {
						nearSurface = true;
						break;
					}
				}
			}
		}

		if (nearSurface) {
			for (unsigned int i = 0; i <= discretization / 2; ++i) {
				double discretization2 = double(floor(2.0 * PI * sin(i * dtheta) / dtheta) + 1);
				for (double phi = 0; phi < 2.0 * PI; phi += double(2.0 * PI / discretization2)) {
					double theta = i * dtheta;
					vec3 normal(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
					vec3 position = pos + normal * outerRadius;

					bool valid = true;
					vector<particle*> neighbors = GetNeighborCoarseParticles(position, 2, 2, 2);
					for (auto ncp : neighbors) {
						vec3 neighborPos(ncp->p[0], ncp->p[1], ncp->p[2]);
						if (cp != ncp && (position - neighborPos).lengthSquared() < outerRadius2) {
							valid = false;
							break;
						}
					}

					if (valid) {
						_fineParticles.push_back(new Particle(position, false));
					}
				}
			}
		}
	}
	printf("Initialize fine-particles number is %d\n", _fineParticles.size());
}

void SurfaceTurbulencen::Advection(void)
{
	double r = 2.0 * _coarseScaleLen;

	_sorter->sort(_coarseParticles);
	ComputeCoarseDensKernel(r);

	for (auto fp : _fineParticles) {
		vec3 displacement;
		vector<particle*> neighbors = GetNeighborCoarseParticles(fp->_pos, 2, 2, 2);

		for (auto cp : neighbors) {
			if (cp->type != FLUID) continue;

			vec3 curPos(cp->p[0], cp->p[1], cp->p[2]);
			vec3 beforePos(cp->p2[0], cp->p2[1], cp->p2[2]);
			displacement += (curPos - beforePos) * NeighborWeight(fp, cp, r, neighbors);
		}
		fp->_pos += displacement;
		fp->_flag = false;
	}
}

double SurfaceTurbulencen::MetaballDens(double dist)
{
	return exp(-2 * pow((dist / _coarseScaleLen), 2));
}

double SurfaceTurbulencen::MetaballLevelSet(Particle* p1, vector<particle*> neighbors)
{
	double R = outerRadius; //outer Sphere
	double r = innerRadius; //inner Sphere
	double u = (3.0 / 2.0) * R;
	double a = log(2.0 / (1.0 + MetaballDens(u))) / (pow(u / 2.0, 2.0) - pow(r, 2.0));

	double f = 0.0;
	for (auto np : neighbors) {
		vec3 pos1(np->p[0], np->p[1], np->p[2]);
		f += exp(-a * (p1->_pos - pos1).lengthSquared());
	}
	if (f > 1.0) f = 1.0;

	f = (sqrt(-log(f) / a) - r) / (R - r);
	return f;
}

vec3 SurfaceTurbulencen::MetaballConstraintGradient(Particle* p1, vector<particle*> neighbors)
{
	double R = outerRadius; //outer Sphere
	double r = innerRadius; //inner Sphere
	double u = (3.0 / 2.0) * R;
	double a = log(2.0 / (1.0 + MetaballDens(u))) / (pow(u / 2.0, 2) - pow(r, 2));

	vec3 gradient;
	for (auto np : neighbors) {
		vec3 pos1(np->p[0], np->p[1], np->p[2]);

		gradient += (p1->_pos - pos1) * 2.0 * a * exp(-a * (p1->_pos - pos1).lengthSquared());
	}
	gradient.normalize();
	return gradient;
}

void SurfaceTurbulencen::SurfaceConstarint(void)
{
	double R = outerRadius; //outer Sphere
	double r = innerRadius; //inner Sphere

	for (auto fp : _fineParticles) {
		vector<particle*> neighbor = GetNeighborCoarseParticles(fp->_pos, 2, 2, 2); //LevelSet Compute

		double levelSet = MetaballLevelSet(fp, neighbor);
		if (levelSet <= 1.0 && levelSet >= 0.0) continue;

		vec3 gradient = MetaballConstraintGradient(fp, neighbor); // Constraints Projection
		if (levelSet < 0.0) {
			fp->_pos -= gradient * (R - r) * levelSet;
		}
		else if (levelSet > 1.0) {
			fp->_pos -= gradient * (R - r) * (levelSet - 1);
		}
	}
}

void SurfaceTurbulencen::ComputeSurfaceNormal(void)
{
	//normal 계산을 위한 가중치 밀도 설정
	double r = _coarseScaleLen;
	ComputeCoarseDensKernel(r);
	ComputeFineDensKernel(r);

	for (auto fp : _fineParticles) {
		//gradient로 normal 근사
		vector<particle*> coarseNeighbor = GetNeighborCoarseParticles(fp->_pos, 1, 1, 1);
		vec3 gradient = MetaballConstraintGradient(fp, coarseNeighbor);

		//tangent frame
		vec3 n = gradient;
		vec3 vx(1, 0, 0);
		vec3 vy(0, 1, 0);
		double dotX = n.dot(vx);
		double dotY = n.dot(vy);
		vec3 t1 = fabs(dotX) < fabs(dotY) ? n.cross(vx) : n.cross(vy);
		vec3 t2 = n.cross(t1);
		t1.normalize();
		t2.normalize();

		// least-square plane fitting with tangent frame
		vector<Particle*> fineNeighbor = GetNeighborFineParticles(fp->_pos, r);
		double sw = 0, swx = 0, swy = 0, swxy = 0, swx2 = 0, swy2 = 0, swxz = 0, swyz = 0, swz = 0;
		for (auto nfp : fineNeighbor) { //neighbor로 순회
			double x = (nfp->_pos - fp->_pos).dot(t1);
			double y = (nfp->_pos - fp->_pos).dot(t2);
			double z = (nfp->_pos - fp->_pos).dot(n);
			double w = NeighborWeight(fp, nfp, r, fineNeighbor);
			swx2 += w * x * x;
			swy2 += w * y * y;
			swxy += w * x * y;
			swxz += w * x * z;
			swyz += w * y * z;
			swx += w * x;
			swy += w * y;
			swz += w * z;
			sw += w;
		}
		double det = -sw * swxy * swxy + 2.0 * swx * swxy * swy - swx2 * swy * swy - swx * swx * swy2 + sw * swx2 * swy2;
		if (det == 0.0) fp->_surfaceNormal = vec3(0, 0, 0);
		else {
			vec3 abc = vec3(
				swxz * (-swy * swy + sw * swy2) + swyz * (-sw * swxy + swx * swy) + swz * (swxy * swy - swx * swy2),
				swxz * (-sw * swxy + swx * swy) + swyz * (-swx * swx + sw * swx2) + swz * (swx * swxy - swx2 * swy),
				swxz * (swxy * swy - swx * swy2) + swyz * (swx * swxy - swx2 * swy) + swz * (-swxy * swxy + swx2 * swy2)
			) * (1.0 / det);

			vec3 normal = (t1 * abc.x() + t2 * abc.y() - n);
			normal.normalize();
			normal *= -1;

			if (gradient.dot(normal) < 0.0) { normal = normal * -1; }
			fp->_tempSurfaceNormal = normal;
		}
	}
}

void SurfaceTurbulencen::SmoothNormal(void)
{
	double r = _coarseScaleLen;
	for (auto fp : _fineParticles) {
		vec3 newNormal;
		vector<Particle*> fineNeighbor = GetNeighborFineParticles(fp->_pos, r);
		for (auto nfp : fineNeighbor) { //neighbor로 순회
			newNormal += nfp->_tempSurfaceNormal* NeighborWeight(fp, nfp, r, fineNeighbor);
		}
		fp->_surfaceNormal = newNormal;
		fp->_surfaceNormal.normalize();
	}
}

void SurfaceTurbulencen::NormalRegularization(void)
{
	//Normal Regularization을 위한 가중치 밀도 설정
	ComputeFineDensKernel(_coarseScaleLen);

	for (auto fp1 : _fineParticles) {
		vec3 pos = fp1->_pos;
		vec3 normal = fp1->_surfaceNormal;
		vec3 displacementNormal(0, 0, 0);

		vector<Particle*> fineNeighbor = GetNeighborFineParticles(pos, _coarseScaleLen);
		for (auto nfp : fineNeighbor) { //여기도 neighbor
			vec3 dir = pos - nfp->_pos;
			vec3 dn = normal * dir.dot(normal);

			vec3 crossVec = normal.cross(-dir);
			crossVec.normalize();
			vec3 projectedNormal = nfp->_surfaceNormal - crossVec * crossVec.dot(nfp->_surfaceNormal);
			projectedNormal.normalize();

			if (projectedNormal.dot(normal) < 0 || abs(normal.dot(normal + projectedNormal)) < 1e-6) continue;
			dn = -normal * ((normal + projectedNormal).dot(dir) / (2 * normal.dot(normal + projectedNormal)));

			double w = NeighborWeight(fp1, nfp, _coarseScaleLen, fineNeighbor);
			displacementNormal += dn * w;
		}
		//printf("normal: ");
		//displacementNormal.print();
		fp1->_tempPos += displacementNormal;
	}
}

void SurfaceTurbulencen::TangentRegularization(void)
{
	//Tangent 계산 이웃 반경
	double tagnetRadius = 3.0 * _fineScaleLen;

	//Tangent Regularization을 위한 가중치 밀도 설정
	ComputeFineDensKernel(tagnetRadius);
	for (auto fp1 : _fineParticles) {
		vec3 pos = fp1->_pos;
		vec3 normal = fp1->_surfaceNormal;
		normal.normalize();

		vec3 displacementTangent(0, 0, 0);

		vector<Particle*> fineNeighbor = GetNeighborFineParticles(pos, tagnetRadius);
		for (auto nfp : fineNeighbor) { //여기도 neighbor
			vec3 dir = pos - nfp->_pos;
			vec3 dn = normal * dir.dot(normal); //dir의 normal 방향 성분
			vec3 dt = dir - dn; //dir의 tangent 방향 성분
			dt.normalize();

			double w = NeighborWeight(fp1, nfp, tagnetRadius, fineNeighbor);
			displacementTangent += dt * w;
		}
		displacementTangent *= 0.5 * _fineScaleLen;
		fp1->_tempPos += displacementTangent;
		
		//visualization of tangent regularization  
		fp1->_tangent = displacementTangent;
		fp1->_tangent.normalize();
		//printf("tan: %d\n", fineNeighbor.size());
		//displacementTangent.print();
	}
}

void SurfaceTurbulencen::Regularization(void)
{
	//_sorter->sort(_coarseParticles); //grid hash with coarse Particles
	_hash->create(_fineParticles); //grid hash with fine Particles

	//Normal 계산
	ComputeSurfaceNormal();
	SmoothNormal();

	//Regularization
	for (auto fp : _fineParticles)
		fp->_tempPos = fp->_pos;

	NormalRegularization();
	TangentRegularization();

	//Update
	for (auto fp : _fineParticles) {
		fp->_prevPos = fp->_pos;
		fp->_pos = fp->_tempPos;
	}
}

void SurfaceTurbulencen::InsertDeleteFineParticles(void)
{	
	_hash->create(_fineParticles);

	//Insert
	//Tangent 계산 이웃 반경
	double tangentRadius = 3.0 * _fineScaleLen;

	//삽입 반경
	double insertRadius = 2.0 * _fineScaleLen;
	int insertCnt = 0;

	//Normal 계산
	ComputeSurfaceNormal();
	SmoothNormal();
	
	//Tangent Regularization을 위한 가중치 밀도 설정
	ComputeFineDensKernel(tangentRadius);

	int fixedSize = _fineParticles.size();
	for (int i = 0; i < fixedSize; i++) {
		Particle* fp1 = _fineParticles[i];
		vec3 pos = fp1->_pos;
		vec3 normal = fp1->_surfaceNormal;
		normal.normalize();

		vec3 displacementTangent(0, 0, 0);

		vector<Particle*> fineNeighbor = GetNeighborFineParticles(pos, tangentRadius);
		for (auto nfp : fineNeighbor) { //여기도 neighbor
			if (nfp == fp1) continue;

			vec3 dir = pos - nfp->_pos;
			vec3 dn = normal * dir.dot(normal); //dir의 normal 방향 성분
			vec3 dt = dir - dn; //dir의 tangent 방향 성분
			dt.normalize();

			double w = NeighborWeight(fp1, nfp, tangentRadius, fineNeighbor);
			displacementTangent += dt * w;
		}
		displacementTangent.normalize();

		vec3 center = pos + (displacementTangent * insertRadius);
		if (!IsInDomain(center)) continue;

		////visualize
		//fp1->_tangent = displacementTangent * insertRadius;
		//fp1->_tangent.normalize();

		int cnt = 0;
		vector<Particle*> centerNeighbors = GetNeighborFineParticles(center, insertRadius - 1e-6);
		for (auto cnp : centerNeighbors) {
			if ((cnp->_pos - center).length() < insertRadius)
				cnt++;
		}

		if (cnt == 0) {
			_fineParticles.push_back(new Particle(center, true));
			insertCnt++;
		}
	}
	insertInfo += insertCnt;
	//printf("Dense Insert: %d\n", insertCnt);

	//delete fine particles if too low dense
	int densDeleteCnt = 0;
	fixedSize = _fineParticles.size();
	double deleteRadius = (3.0 / 4.0) * _fineScaleLen;
	for (int i = 0; i < fixedSize; i++) {
		Particle* fp = _fineParticles[i];
		vec3 pos = fp->_pos;

		vector<Particle*> fineNeighbors = GetNeighborFineParticles(fp->_pos, deleteRadius);
		for (auto nfp : fineNeighbors) {
			if (nfp == fp) continue;
			if ((nfp->_pos - fp->_pos).length() <= deleteRadius || !IsInDomain(fp->_pos)) {

				_fineParticles.erase(remove(_fineParticles.begin(), _fineParticles.end(), nfp), _fineParticles.end());
				fixedSize--;

				//_fineParticles.erase(_fineParticles.begin() + i);
				//i--;
	
				densDeleteCnt++;
				break;
			}
		}
		if (i >= _fineParticles.size())
			break;
	}
	deleteInfo += densDeleteCnt;
	//printf("Dense Delete: %d\n", densDeleteCnt);

	//// delete fine particles if no coarse neighbors in advection radius
	int advectionDeleteCnt = 0;
	double r = 2.0 * _coarseScaleLen;
	for (int i = 0; i < _fineParticles.size(); i++) {
		Particle* fp = _fineParticles[i];
		vector<particle*> neighbors = GetNeighborCoarseParticles(fp->_pos, 2, 2, 2);

		int cnt = 0;
		for (auto cp : neighbors) {
			if (cp->type != FLUID) continue;
			vec3 curPos(cp->p[0], cp->p[1], cp->p[2]);
			if ((curPos - fp->_pos).length() <= r) {
				cnt++;
				break;
			}
		}

		if (cnt == 0) {
			_fineParticles.erase(_fineParticles.begin() + i);
			i--;
			advectionDeleteCnt++;
			//fp->_flag = true;
		}
		if (i >= _fineParticles.size())
			break;
	}
	deleteInfo += advectionDeleteCnt;
	//printf("advection Delete: %d\n", advectionDeleteCnt);

	// delete fine particles if too far from constraint
	int constraintDeleteCnt = 0;
	for (int i = 0; i < _fineParticles.size(); i++) {
		Particle* fp = _fineParticles[i];
		vector<particle*> neighbor = GetNeighborCoarseParticles(fp->_pos, 2, 2, 2); //LevelSet Compute

		double levelSet = MetaballLevelSet(fp, neighbor);
		if (levelSet < -0.2 || levelSet > 1.2) {
			_fineParticles.erase(_fineParticles.begin() + i);
			i--;
			constraintDeleteCnt++;
			//fp->_flag = true;
		}
		if (i >= _fineParticles.size())
			break;
	}
	deleteInfo += constraintDeleteCnt;
	//printf("constraint Delete: %d\n", constraintDeleteCnt);
}

void SurfaceTurbulencen::SurfaceMaintenance(void) //첫 프레임에서 24회 정도.. 나머진 4회정도
{
	//clock_t s, f;
	//s = clock();

	SurfaceConstarint();
	Regularization();
	InsertDeleteFineParticles();

	for (auto fp : _fineParticles) {
		
		if (isnan(fp->_pos.x()) || isnan(fp->_pos.y()) || isnan(fp->_pos.z())) {
			fp->_pos.print();
			exit(0);
		}
	}

	//f = clock();
	//cout << f - s << endl;
}

void SurfaceTurbulencen::ComputeCurvature(void)
{
	_hash->create(_fineParticles); //grid hash with fine Particles

	//Normal 계산
	ComputeSurfaceNormal();
	SmoothNormal();

	//곡률 계산
	double r = _coarseScaleLen;
	for (auto fp : _fineParticles) {
		double curvature = 0.0;
		vector<Particle*> fineNeighbor = GetNeighborFineParticles(fp->_pos, r);
		for (auto nfp : fineNeighbor) { //neighbor로 순회
			curvature += fp->_surfaceNormal.dot(fp->_pos - nfp->_pos) * NeighborWeight(fp, nfp, r, fineNeighbor);
		}
		fp->_tempCurvature = fabs(curvature);
	}
}

void SurfaceTurbulencen::SmoothCurvature(void)
{
	double r = _coarseScaleLen;
	for (auto fp : _fineParticles) {
		double newCurvature = 0.0;
		vector<Particle*> fineNeighbor = GetNeighborFineParticles(fp->_pos, r);
		for (auto nfp : fineNeighbor) { //neighbor로 순회
			newCurvature += nfp->_tempCurvature * NeighborWeight(fp, nfp, r, fineNeighbor);
		}
		fp->_curvature = newCurvature;
	}
}

double SurfaceTurbulencen::SmoothStep(double left, double right, double val)
{
	//printf("val: %f, left: %f, right: %f, result: %f\n",val, left, right, (val - left) / (right - left));
	double x = max(0.0, min((val - left) / (right - left), 1.0));
	return x * x * (3.0 - 2.0 * x);
}

void SurfaceTurbulencen::SeedWave(int step)
{
	for (auto fp : _fineParticles) {
		//printf("%f\n", fp->_curvature);
		double source = 2.0 * SmoothStep(waveSeedingCurvatureThresholdMinimum, waveSeedingCurvatureThresholdMaximum, fp->_curvature) - 1.0; //edge값 추후 수정 가능
		double freq = waveSeedFreq;
		double theta = dt * (double)step * waveSpeed * freq;
		double cosTheta = cos(theta);
		double maxSeedAmplitude = waveMaxSeedingAmplitude * waveMaxAmplitude;

		fp->_waveSeedAmplitude = max(0.0, min(fp->_waveSeedAmplitude + source * waveSeedStepSizeRatioOfMax * maxSeedAmplitude, maxSeedAmplitude));
		fp->_seed = fp->_waveSeedAmplitude * cosTheta;

		// source values for display (not used after this point anyway)
		fp->_curvature = (source >= 0) ? 1 : 0;

		//if (fp->_curvature != 0)
		//	fp->_flag = true;

		//seed 더하기
		fp->_waveH += fp->_seed;

	}
}

void SurfaceTurbulencen::ComputeWaveNormal(void)
{
	_hash->create(_fineParticles); //grid hash with fine Particles

	double r = 3.0 * _fineScaleLen;
	ComputeFineDensKernel(r);

	for (auto fp : _fineParticles) {
		//tangent frame
		vec3 n = fp->_surfaceNormal;
		vec3 vx(1, 0, 0);
		vec3 vy(0, 1, 0);
		double dotX = n.dot(vx);
		double dotY = n.dot(vy);
		vec3 t1 = fabs(dotX) < fabs(dotY) ? n.cross(vx) : n.cross(vy);
		vec3 t2 = n.cross(t1);
		t1.normalize();
		t2.normalize();

		// least-square plane fitting with tangent frame
		vector<Particle*> fineNeighbor = GetNeighborFineParticles(fp->_pos, r);
		double sw = 0, swx = 0, swy = 0, swxy = 0, swx2 = 0, swy2 = 0, swxz = 0, swyz = 0, swz = 0;
		for (auto nfp : fineNeighbor) { //neighbor로 순회
			double x = (nfp->_pos - fp->_pos).dot(t1);
			double y = (nfp->_pos - fp->_pos).dot(t2);
			double z = nfp->_waveH;
			double w = NeighborWeight(fp, nfp, r, fineNeighbor);
			swx2 += w * x * x;
			swy2 += w * y * y;
			swxy += w * x * y;
			swxz += w * x * z;
			swyz += w * y * z;
			swx += w * x;
			swy += w * y;
			swz += w * z;
			sw += w;
		}
		double det = -sw * swxy * swxy + 2.0 * swx * swxy * swy - swx2 * swy * swy - swx * swx * swy2 + sw * swx2 * swy2;
		//if (swxz != 0 && swyz != 0 && swz != 0) {
		//	printf("x2: %f, y2: %f, xy: %f, xz: %f, yz: %f, x: %f, y: %f, z: %f, w: %f\n", swx2, swy2, swxy, swxz, swyz, swx, swy, swz, sw);
		//	printf("1: %f, 2: %f, 3: %f, 4: %f, 5: %f\n", -sw * swxy * swxy, 2.0 * swx * swxy * swy, -swx2 * swy * swy, -swx * swx * swy2, sw * swx2 * swy2);
		//	printf("det: %e\n", det);
		//}

		if (det == 0.0) fp->_waveNormal = vec3(0, 0, 0);
		else {
			vec3 abc = vec3(
				swxz * (-swy * swy + sw * swy2) + swyz * (-sw * swxy + swx * swy) + swz * (swxy * swy - swx * swy2),
				swxz * (-sw * swxy + swx * swy) + swyz * (-swx * swx + sw * swx2) + swz * (swx * swxy - swx2 * swy),
				swxz * (swxy * swy - swx * swy2) + swyz * (swx * swxy - swx2 * swy) + swz * (-swxy * swxy + swx2 * swy2)
			) * (1.0 / det);

			vec3 waveNormal = (vx * abc.x() + vy * abc.y() - vec3(0, 0, 1));
			waveNormal.normalize();
			waveNormal *= -1;

			fp->_waveNormal = waveNormal;
		}
	}
}

void SurfaceTurbulencen::ComputeLaplacian(void)
{
	//laplacian 계산을 위한 가중치 밀도 설정
	double r = 3.0 * _fineScaleLen;
	ComputeFineDensKernel(r);

	for (auto fp : _fineParticles) {
		double laplacian = 0;
		vec3 normal = fp->_surfaceNormal;

		vec3 vx(1, 0, 0);
		vec3 vy(0, 1, 0);
		double dotX = normal.dot(vx);
		double dotY = normal.dot(vy);
		vec3 t1 = fabs(dotX) < fabs(dotY) ? normal.cross(vx) : normal.cross(vy);
		vec3 t2 = normal.cross(t1);
		t1.normalize();
		t2.normalize();

		vec3 waveNormal = fp->_waveNormal;
		double ph = fp->_waveH;

		if (waveNormal.y() == 0) fp->_laplacian = 0;
		else {
			// least-square plane fitting with tangent frame
			vector<Particle*> fineNeighbor = GetNeighborFineParticles(fp->_pos, r);
			for (auto nfp : fineNeighbor) { //neighbor로 순회
				double nh = nfp->_waveH;

				vec3 dir = nfp->_pos - fp->_pos;
				double lengthDir = dir.getNorm();
				if (lengthDir < 1e-5) continue;

				vec3 tangentDir = dir - normal * (dir.dot(normal));
				tangentDir.normalize();
				tangentDir = tangentDir * lengthDir;

				double dirX = tangentDir.dot(t1);
				double dirY = tangentDir.dot(t2);
				double dz = nh - ph - (-waveNormal.x () / waveNormal.z()) * dirX - (-waveNormal.y() / waveNormal.z()) * dirY;
				laplacian += max(-100.0, min(NeighborWeight(fp, nfp, r, fineNeighbor) * 4.0 * dz / (lengthDir * lengthDir), 100.0));
			}
			fp->_laplacian = laplacian;
		}
	}
}

void SurfaceTurbulencen::EvolveWave(void)
{
	double damping = 0.0;
	for (auto fp : _fineParticles) {
		fp->_waveDtH += waveSpeed * waveSpeed * dt * fp->_laplacian; //wave의 H 방향 속도
		fp->_waveDtH /= (1.0 + dt * damping);

		fp->_waveH += dt * fp->_waveDtH;
		fp->_waveH /= (1.0 + dt * damping);
		fp->_waveH -= fp->_seed; //seed 값 빼기

		//clamp
		fp->_waveDtH = max(-waveMaxFreq * waveMaxAmplitude, min(fp->_waveDtH, waveMaxFreq * waveMaxAmplitude));
		fp->_waveH = max(- waveMaxAmplitude, min(fp->_waveH, waveMaxAmplitude));
	}
}

void SurfaceTurbulencen::WaveSimulation(int step)
{
	ComputeCurvature();
	SmoothCurvature();

	////printf("step: %d\n", step);
	SeedWave(step);
	ComputeWaveNormal();
	ComputeLaplacian();
	EvolveWave();
}

void SurfaceTurbulencen::SetDisplayParticles(void)
{
	_displayParticles.clear();
	for (auto fp : _fineParticles) {
		vec3 displayPos = fp->_pos + fp->_surfaceNormal * fp->_waveH;
		_displayParticles.push_back(new Particle(displayPos, false));
	}
	printf("size: %d\n", _displayParticles.size());
}

double SurfaceTurbulencen::DistKernel(Vec3<double> diff, double r) //거리에 반비례하는 커널
{
	double dist = diff.length();
	if (dist > r)
		return 0.0;
	else
		return 1.0 - (dist / r);
}

void SurfaceTurbulencen::ComputeFineDensKernel(double r) //DistKernel의 합을 이용해 커널 밀도 계산
{
	for (auto p1 : _fineParticles) { //fine-particles
		for (auto p2 : _fineParticles) {
			p1->_kernelDens += DistKernel(p1->_pos - p2->_pos, r);
		}
	}
}

void SurfaceTurbulencen::ComputeCoarseDensKernel(double r) //DistKernel의 합을 이용해 커널 밀도 계산
{
	for (auto p1 : _coarseParticles) { //coarse-particles
		vec3 pos1(p1->p[0], p1->p[1], p1->p[2]);
		vector<particle *> neighbors = GetNeighborCoarseParticles(pos1, 2, 2, 2);

		for (auto np : neighbors) { 
			if (p1->type != FLUID || np->type != FLUID) continue;
			vec3 pos2(np->p[0], np->p[1], np->p[2]);
			p1->_kernelDens += DistKernel(pos1 - pos2, r);
		}
	}
}

double SurfaceTurbulencen::NeighborWeight(Particle* fp1, Particle* fp2, double r, vector<Particle*> neighbors) //fine-fine 가중치
{
	double weight = DistKernel(fp1->_pos - fp2->_pos, r) / fp2->_kernelDens; //p2의 커널 밀도로 정규화
	if (weight == 0) return 0;

	double weightSum = 0.0;
	for (auto p : neighbors) {
		if ((fp1->_pos - p->_pos).length() > r) continue;
		weightSum += (DistKernel(fp1->_pos - p->_pos, r) / p->_kernelDens);
	}

	return weight / weightSum;
}

double SurfaceTurbulencen::NeighborWeight(Particle* fp1, particle* cp2, double r, vector<particle*> neighbors) //fine-coarse 가중치
{
	vec3 pos2(cp2->p[0], cp2->p[1], cp2->p[2]);
	double weight = DistKernel(fp1->_pos - pos2, r) / cp2->dens; //p2의 커널 밀도로 정규화
	if (weight == 0) return 0;

	double weightSum = 0.0;
	for (auto np : neighbors) { //sort의 그리드 탐색(hash) 이용, 최적화
		if (np->type != FLUID) continue;

		vec3 pos(np->p[0], np->p[1], np->p[2]);
		if ((fp1->_pos - pos).length() > r) continue;

		weightSum += (DistKernel(fp1->_pos - pos, r) / np->dens);
	}

	return weight / weightSum; //check
}

vector<particle*> SurfaceTurbulencen::GetNeighborCoarseParticles(vec3 pos, int w, int h, int d)
{
	double cell_size = 1.0 / _coarseScaleLen;
	double x = max(0.0, min(cell_size, cell_size * pos.x()));
	double y = max(0.0, min(cell_size, cell_size * pos.y()));
	double z = max(0.0, min(cell_size, cell_size * pos.z()));

	vector<particle*> neighbors = _sorter->getNeigboringParticles_cell(x, y, z, w, h, d);
	return neighbors;
}

vector<Particle*> SurfaceTurbulencen::GetNeighborFineParticles(vec3 pos, double maxDist)
{
	return _hash->query(pos, maxDist);
}

bool SurfaceTurbulencen::IsInDomain(vec3 pos)
{
	double x = pos.x();
	double y = pos.y();
	double z = pos.z();

	if (0.0 <= x && x <= 1.0 &&
		0.0 <= y && y <= 1.0 &&
		0.0 <= z && z <= 1.0)
		return true;
	else
		return false;
}

void SurfaceTurbulencen::drawFineParticles(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(2.0);
	glLineWidth(1.0);
	for (auto fp: _fineParticles) {

		//flag visualize
		//if(IsInDomain(fp->_pos))
		//if(!fp->_flag)
		//	glColor3f(1.0f, 0.0f, 0.0f); 
		//else 
		//	glColor3f(0.0f, 1.0f, 0.0f);
		
		//////Curvature visualize
		//vec3 color = ScalarToColor(fp->_curvature * 1000);
		//glColor3f(color.x(), color.y(), color.z());

		//////Laplacian visualize
		//vec3 color = ScalarToColor(fp->_laplacian * 0.05);
		//glColor3f(color.x(), color.y(), color.z());
		
		////general visualize
		glColor3f(1.0f, 0.0f, 0.0f); 

		//Draw Fine Particles
		glBegin(GL_POINTS);
		glVertex3d(fp->_pos.x(), fp->_pos.y(), fp->_pos.z());
		glEnd();

		////Draw normal
		//glColor3f(1.0f, 1.0f, 1.0f);
		//double scale = 0.02;
		//glBegin(GL_LINES);
		//glVertex3d(fp->_pos.x(), fp->_pos.y(), fp->_pos.z());
		//glVertex3d(fp->_pos.x() + fp->_surfaceNormal.x() * scale, fp->_pos.y()+fp->_surfaceNormal.y() * scale, fp->_pos.z() + fp->_surfaceNormal.z() * scale);
		//glEnd();
		
		////////Draw tangent
		//double scale = 0.02;
		//glColor3f(1.0f, 1.0f, 1.0f);
		//glBegin(GL_LINES);
		//glVertex3d( fp->_pos.x(),fp->_pos.y(), fp->_pos.z());
		//glVertex3d(fp->_pos.x() + fp->_tangent.x() * scale, fp->_pos.y() + fp->_tangent.y() * scale, fp->_pos.z() + fp->_tangent.z() * scale);
		//glEnd();
	
		//////Draw wave normal
		//glColor3f(1.0f, 1.0f, 1.0f);
		//double scale = 0.04;
		//glBegin(GL_LINES);
		//glVertex3d(fp->_pos.x(), fp->_pos.y(), fp->_pos.z());
		//glVertex3d(fp->_pos.x() + fp->_waveNormal.x() * scale, fp->_pos.y()+fp->_waveNormal.y() * scale, fp->_pos.z() + fp->_waveNormal.z() * scale);
		//glEnd();
	}
	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

void SurfaceTurbulencen::drawDisplayParticles(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glPointSize(2.0);
	glLineWidth(1.0);
	for (auto dp : _displayParticles) {

		////general visualize
		glColor3f(1.0f, 0.0f, 0.0f);

		//Draw Fine Particles
		glBegin(GL_POINTS);
		glVertex3d(dp->_pos.x(), dp->_pos.y(), dp->_pos.z());
		glEnd();

	}
	glPointSize(1.0);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

vec3 SurfaceTurbulencen::ScalarToColor(double val)
{
	double fColorMap[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };   //Red->Blue
	double v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	double t = v - low;
	vec3 color;
	color.x((fColorMap[low][0]) * (1 - t) + (fColorMap[high][0]) * t);
	color.y((fColorMap[low][1]) * (1 - t) + (fColorMap[high][1]) * t);
	color.z((fColorMap[low][2]) * (1 - t) + (fColorMap[high][2]) * t);
	return color;
}

//To do
// 1. insert/delete 함 확인해보기.
// 2. wave simulation
// 3. parameter 잘 조절하기.
// 4. 물이 내려올 때, 주변 파티클이 없어서 tangent 계산이 안되나 봄..
// 5. tangent 계산 반경 잘 생각해보기, insert랑, regularization 둘 다!