/*
 *  corrector.h
 *  flip3D
 */

#include "common.h"
#include "sorter.h"
#include <vector>

namespace corrector {
	void resample( Sorter *sort, FLOAT p[3], FLOAT u[3], FLOAT re );
	void correct( Sorter *sort, std::vector<particle *> &particle, FLOAT dt, FLOAT re);
};