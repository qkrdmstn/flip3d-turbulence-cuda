#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#include <fstream>
#include <string>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_gl.h>
#include <helper_cuda.h> 
#include "Shader.h"
#include "FLIP3D_Cuda.h"
#include "SurfaceTurbulence.h"
#include "MarchingCubesCuda.h"
#include "FluidRenderer.h"
#include "Camera.h"

class FLIPEngine
{
public: //Simulation
	FLIP3D_Cuda* _fluid;
	SurfaceTurbulence* _turbulence;
	MarchingCubes_CUDA* _MC;

public:
	AABB			_boundary;
public:
	REAL3			_gravity;
	REAL			_dt;

public:
	bool _simulation = false;
	int _frame = 0;
	int _curTime, _timebase = 0;
	int _step = 0;
	double fps = 0;
	int cnt = 0;

	bool advection = true;
	bool flag = false;

public: //Rendering
	bool _fluidFlag = true;
	bool _turbulenceBaseFlag = false;
	bool _turbulenceDisplayFlag = false;
	bool _surfaceReconstructionFlag = false;

	int				_width;
	int				_height;
	int				_mousePos[2];
	unsigned char	_mouseEvent[3];
	Camera*			_camera;
	FluidRenderer*	_renderer;

public:
	FLIPEngine()
	{
		init();
	}
	~FLIPEngine() 
	{

	}
public:
	void	init(void);
	void	InitOpenGL(void);
	void	InitCamera(void);
	void	InitRenderer(void);
	void	InitSimulation(REAL3& gravity, REAL dt);
public:
	void	simulation();
	void	reset(void);

public:
	void	idle();
	void	draw();
	void	DrawSimulationInfo(void);
	void	DrawText(float x, float y, const char* text, void* font = NULL);
	void	drawBoundary();
	void	motion(int x, int y);
	void	reshape(int w, int h);
	void	mouse(int mouse_event, int state, int x, int y);
	void	keyboard(unsigned char key, int x, int y);

public:
	void	ExportObj(const char* filePath);
	void	Capture(char* filename, int width, int height);
};

#endif