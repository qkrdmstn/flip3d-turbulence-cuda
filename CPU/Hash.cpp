#include "Hash.h";

Hash::Hash()
{

}

Hash::Hash(double _spacing, int _numParticles)
{
	spacing = _spacing;

	numParticles = _numParticles;
	tableSize = 5 * _numParticles; //table size를 파티클 수의 5배로

	cellStart.resize(tableSize + 1);
	cellEntries.resize(numParticles + 1);

}

Hash::~Hash()
{

}

int Hash::intCoord(double coord) //double 값을 int 값으로 내림 (cell Pos(int)로 변환)
{
	return floor(coord / spacing);
}

int Hash::hashCoords(int xi, int yi, int zi) //hash function으로 배열 index get
{
	int h = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481); //hash function
	return abs(h) % tableSize;
}

int Hash::hashPos(vec3 pos) //정점의 위치를 입력받아서 int형으로 내림한 뒤, hash function을 돌려서 index 반환
{
	return hashCoords(intCoord(pos.x()), intCoord(pos.y()), intCoord(pos.z()));
}

void Hash::create(vector<Particle*>	_fineParticles)
{
	//Initialize
	cellStart.assign(cellStart.size(), 0); 
	cellEntries.resize(_fineParticles.size());
	cellEntries.assign(cellEntries.size(), NULL);

	//Count
	for (int i = 0; i < _fineParticles.size(); i++) {
		int h = hashPos(_fineParticles[i]->_pos); //해당 정점에 해당하는 cell Index를 배열에 더하기
		cellStart[h]++;
	}

	//Partial Sums
	int start = 0;
	for (int i = 0; i < tableSize; i++) {
		start += cellStart[i];
		cellStart[i] = start;
	}
	cellStart[tableSize] = start; //마지막 칸 채우기

	//Fill in
	for (int i = 0; i < _fineParticles.size(); i++) {
		int h = hashPos(_fineParticles[i]->_pos);
		cellStart[h]--;

		cellEntries[cellStart[h]] = _fineParticles[i];
	}
}

vector<Particle*> Hash::query(vec3 pos, double maxDist)
{
	int x0 = intCoord(pos.x() - maxDist); //현재 정점에서 - maxDist 만큼의 위치
	int y0 = intCoord(pos.y() - maxDist);
	int z0 = intCoord(pos.z() - maxDist);

	int x1 = intCoord(pos.x() + maxDist); //현재 정점에서 + maxDist 만큼의 위치
	int y1 = intCoord(pos.y() + maxDist);
	int z1 = intCoord(pos.z() + maxDist);

	vector<Particle*> neighbor;
	for (int xi = x0; xi <= x1; xi++) //x, y, z 값을 1씩 더하면서 maxDist 내에 있는 셀을 순회
	{
		for (int yi = y0; yi <= y1; yi++)
		{
			for (int zi = z0; zi <= z1; zi++)
			{
				int h = hashCoords(xi, yi, zi); //해당 위치의 index 반환받고
				int start = cellStart[h];
				int end = cellStart[h + 1];

				for (int i = start; i < end; i++) //해당 index에 있는 정점들을 모두 queryIds에 대입
				{
					if ((cellEntries[i]->_pos - pos).length() <= maxDist * 1.5)
						neighbor.push_back(cellEntries[i]);
				}
			}
		}
	}

	return neighbor;
}