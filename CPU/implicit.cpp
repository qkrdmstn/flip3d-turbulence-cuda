/*
 *  implicit.cpp
 *  flip3D
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "implicit.h"
#include "utility.h"
using namespace std;

static double implicit_func( vector<particle *> &neighbors, FLOAT p[3], FLOAT density, int gn) {
	double phi = 8.0*density/gn;
	for( int m=0; m<neighbors.size(); m++ ) {
		particle &np = *neighbors[m];
		if( np.type == WALL ) {
			if( length(np.p,p) < density/gn ) return 4.5*density/gn;
			continue;
		}
		double d = length(np.p,p);
		if( d < phi ) {
			phi = d;
		}
	}
	return phi - density/gn;
}

double implicit::implicit_func( Sorter *sort, FLOAT p[3], FLOAT density ) {	
	int gn = sort->getCellSize();	 
	vector<particle *> neighbors = sort->getNeigboringParticles_cell(
			fmax(0,fmin(gn-1,gn*p[0])),
			fmax(0,fmin(gn-1,gn*p[1])),
			fmax(0,fmin(gn-1,gn*p[2])),2,2,2
			);
	return implicit_func( neighbors, p, density, gn );
}


