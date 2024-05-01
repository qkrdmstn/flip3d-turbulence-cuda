#include "Hash.h";

Hash::Hash()
{

}

Hash::Hash(double _spacing, int _numParticles)
{
	spacing = _spacing;

	numParticles = _numParticles;
	tableSize = 5 * _numParticles; //table size�� ��ƼŬ ���� 5���

	cellStart.resize(tableSize + 1);
	cellEntries.resize(numParticles + 1);

}

Hash::~Hash()
{

}

int Hash::intCoord(double coord) //double ���� int ������ ���� (cell Pos(int)�� ��ȯ)
{
	return floor(coord / spacing);
}

int Hash::hashCoords(int xi, int yi, int zi) //hash function���� �迭 index get
{
	int h = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481); //hash function
	return abs(h) % tableSize;
}

int Hash::hashPos(vec3 pos) //������ ��ġ�� �Է¹޾Ƽ� int������ ������ ��, hash function�� ������ index ��ȯ
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
		int h = hashPos(_fineParticles[i]->_pos); //�ش� ������ �ش��ϴ� cell Index�� �迭�� ���ϱ�
		cellStart[h]++;
	}

	//Partial Sums
	int start = 0;
	for (int i = 0; i < tableSize; i++) {
		start += cellStart[i];
		cellStart[i] = start;
	}
	cellStart[tableSize] = start; //������ ĭ ä���

	//Fill in
	for (int i = 0; i < _fineParticles.size(); i++) {
		int h = hashPos(_fineParticles[i]->_pos);
		cellStart[h]--;

		cellEntries[cellStart[h]] = _fineParticles[i];
	}
}

vector<Particle*> Hash::query(vec3 pos, double maxDist)
{
	int x0 = intCoord(pos.x() - maxDist); //���� �������� - maxDist ��ŭ�� ��ġ
	int y0 = intCoord(pos.y() - maxDist);
	int z0 = intCoord(pos.z() - maxDist);

	int x1 = intCoord(pos.x() + maxDist); //���� �������� + maxDist ��ŭ�� ��ġ
	int y1 = intCoord(pos.y() + maxDist);
	int z1 = intCoord(pos.z() + maxDist);

	vector<Particle*> neighbor;
	for (int xi = x0; xi <= x1; xi++) //x, y, z ���� 1�� ���ϸ鼭 maxDist ���� �ִ� ���� ��ȸ
	{
		for (int yi = y0; yi <= y1; yi++)
		{
			for (int zi = z0; zi <= z1; zi++)
			{
				int h = hashCoords(xi, yi, zi); //�ش� ��ġ�� index ��ȯ�ް�
				int start = cellStart[h];
				int end = cellStart[h + 1];

				for (int i = start; i < end; i++) //�ش� index�� �ִ� �������� ��� queryIds�� ����
				{
					if ((cellEntries[i]->_pos - pos).length() <= maxDist * 1.5)
						neighbor.push_back(cellEntries[i]);
				}
			}
		}
	}

	return neighbor;
}