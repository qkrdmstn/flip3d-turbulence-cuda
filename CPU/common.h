/*
 *  common.h
 *  flip3D
 *
 */

#ifndef _COMMON_H
#define _COMMON_H

#define FLOAT	float
#define AIR		0
#define FLUID	1
#define WALL	2

#define BOX		0
#define SPHERE	1

#define GLASS	1
#define GRAY	2
#define RED		3

#define PI          3.14159265

typedef struct {
	char type;
	char shape;
	char material;
	bool visible;
	FLOAT r; //Sphere's radius (SPHERE일 경우)
	FLOAT c[3]; //Sphere's  center
	FLOAT p[2][3]; //Box's min, max position
} Object;

typedef struct _particle {  //이전 위치로 fineParticle Advection 하기
	FLOAT p2[3]; //이전 위치
	FLOAT p[3]; //current position
	FLOAT u[3]; //velocity
	FLOAT n[3]; //normal 아마..
	char type;
	char visible;
	char remove;
	char thinparticle;
	FLOAT tmp[2][3];
	FLOAT m;
	FLOAT dens;
	double _kernelDens;
} particle;

typedef struct _ipos {
	int i; int j; int k;
} ipos;

#endif
