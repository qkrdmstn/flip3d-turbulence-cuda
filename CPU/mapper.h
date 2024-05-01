/*
 *  mapper.h
 *  flip3D
 */

#include "common.h"
#include "sorter.h"
#include <vector>

class Sorter;
namespace mapper {
	void mapP2G( Sorter *sort, std::vector<particle *> &particles, FLOAT ****grid, int gn );
	void mapG2P( std::vector<particle *> &particles, FLOAT ****grid, int gn );
	void fetchVelocity( FLOAT p[3], FLOAT u[3], FLOAT ****grid, int gn );
}