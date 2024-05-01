/*
 *  flip3D.h
 *  flip3D
 *
 */

#include "common.h"
#include "GL\glut.h"

namespace flip3D {
	// Simulation Function
	void init(int load_step = 0);
	void simulateStep();
	void drawFLIP(void);
	void draw(int option); //FLIP + Turbulence
}