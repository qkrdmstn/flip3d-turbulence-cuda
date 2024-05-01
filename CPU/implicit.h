/*
 *  implicit.h
 *  flip3D
 */

#include "common.h"
#include "sorter.h"
#include <vector>

namespace implicit {
	double implicit_func( Sorter *sort, FLOAT p[3], FLOAT density );
};