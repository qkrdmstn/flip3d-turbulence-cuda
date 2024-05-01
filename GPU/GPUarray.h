/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
   Class to represent an array in GPU and CPU memory
*/
#ifndef __GPUARRAY_H__
#define __GPUARRAY_H__

#include <stdlib.h>
#include <stdio.h>
#include <GL/glew.h>
#if defined (WIN32)
#include <GL/wglew.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

#include "init_PBF.h"
template <class T>
class GpuArray
{
    public:
        GpuArray();
        ~GpuArray();

        enum Direction
        {
            HOST_TO_DEVICE,
            DEVICE_TO_HOST,
			DEVICE_TO_DEVICE,
        };

		enum MemType
		{
			HOST_MEMORY,
			DEVICE_MEMORY
		};
		
        // allocate and free
        void alloc(size_t size, bool vbo=false, bool useElementArray=false);
        void free();

        // swap buffers for double buffering
        

        // when using vbo, must map before getting device ptr
        void map();
		void resize(size_t size);
        void unmap();

        void copy(Direction dir, uint start=0, uint count=0);
        void memset(T value, uint start=0, uint count=0);

        T *getDevicePtr(){
			
            return m_dptr;
        }

		
		//call getDevicePtr();
	
        GLuint getVbo(){
            return m_vbo;
        }

       
        T *getHostPtr(){
            return m_hptr;
        }

        size_t getSize() const
        {
            return m_size;
        }
		void dumpMemory(char* fileName, int size);
    private:
        GLuint createVbo(size_t size, bool useElementArray);

        void allocDevice();
		
        void allocVbo(bool useElementArray);
        void allocHost();

        void freeDevice();
		
        void freeVbo();
        void freeHost();
		
        size_t m_size;
		T *m_hptr;
        T *m_dptr;
		
        GLuint m_vbo;
        struct cudaGraphicsResource *m_cuda_vbo_resource; // handles OpenGL-CUDA exchange

        

        bool m_useVBO;
		bool m_useElementArray;

		
        
        
};

template <class T>
void GpuArray<T>::dumpMemory(char* fileName, int size){
	FILE* fp = fopen(fileName,"wb");
 	copy(DEVICE_TO_HOST);
	float4* tmp = m_hptr;
	printf("%f,%f,%f\n",tmp[0].x,tmp[0].y,tmp[0].z);
 	fwrite((void*)m_hptr, sizeof(T)*size, 1, fp);
	fflush(fp);
	fclose(fp);
}

template <class T>
void GpuArray<T>::resize(size_t size){

	uint minSize = min(size, m_size);
	
	T* tmp_hptr = (T *) new T [size];
	T* tmp_dptr;
	copy(GpuArray::DEVICE_TO_HOST);
	GLuint vbo = 0;
	struct cudaGraphicsResource *tmp_cuda_vbo_resource = NULL;
	
	if (m_hptr)
	{
		delete [] m_hptr;
		m_hptr = 0;
	}
	m_hptr = tmp_hptr;
	
	

	if (m_useVBO)
	{
		
		glGenBuffers(1, &vbo);

		if (m_useElementArray)
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, size*sizeof(T), 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}
		else
		{
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, size*sizeof(T), 0, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}


		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&tmp_cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));
	}else{
		checkCudaErrors(cudaMalloc((void **) &tmp_dptr, size*sizeof(T)));
		cudaMemset(tmp_dptr, 0x00000000, size*sizeof(T));
		
	}



	if (vbo){
		map();
		checkCudaErrors(cudaGraphicsMapResources(1, &tmp_cuda_vbo_resource, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&tmp_dptr, &num_bytes, tmp_cuda_vbo_resource));
		
	}

	checkCudaErrors(cudaMemcpy((void *)(tmp_dptr), (void *)(m_dptr), minSize*sizeof(T), cudaMemcpyDeviceToDevice));


	if (vbo){
		unmap();
		checkCudaErrors(cudaGraphicsUnmapResources(1, &tmp_cuda_vbo_resource, 0));
		tmp_dptr = 0;
		
	}


	if (m_vbo)
	{
		freeVbo();
		m_vbo = vbo;
		//m_cuda_vbo_resource = tmp_cuda_vbo_resource;
		cudaGraphicsUnregisterResource(tmp_cuda_vbo_resource);
		cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource, m_vbo, cudaGraphicsMapFlagsWriteDiscard);
	}

	if (m_dptr)
	{
		freeDevice();
		m_dptr = tmp_dptr;
	}

	m_size = size;
	

}


template <class T>
GpuArray<T>::GpuArray() :
    m_size(0),
    m_hptr(0)
{
    m_dptr = 0;
	m_vbo = 0;
    

    m_cuda_vbo_resource = NULL;
    
}

template <class T>
GpuArray<T>::~GpuArray()
{
    free();
}

template <class T>
void
GpuArray<T>::alloc(size_t size, bool vbo, bool useElementArray)
{
    m_size = size;
	
    m_useVBO = vbo;
	m_useElementArray = useElementArray;

    allocHost();

    if (vbo)
    {
        allocVbo(m_useElementArray);
    }
    else
    {
        allocDevice();
    }
	
}

template <class T>
void
GpuArray<T>::free()
{
    freeHost();

    if (m_vbo)
    {
        freeVbo();
    }
	
    if (m_dptr)
    {
        freeDevice();
    }
}

template <class T>
void
GpuArray<T>::allocHost()
{
    m_hptr = (T *) new T [m_size];
}

template <class T>
void
GpuArray<T>::freeHost()
{
    if (m_hptr)
    {
        delete [] m_hptr;
        m_hptr = 0;
    }
}

template <class T>
void
GpuArray<T>::allocDevice()
{
    checkCudaErrors(cudaMalloc((void **) &m_dptr, m_size*sizeof(T)));
	cudaMemset(m_dptr, 0x00000000, m_size*sizeof(T));
   
}




template <class T>
void
GpuArray<T>::freeDevice()
{
    if (m_dptr)
    {
        checkCudaErrors(cudaFree(m_dptr));
        m_dptr = 0;
    }

}

template <class T>
GLuint
GpuArray<T>::createVbo(size_t size, bool useElementArray)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);

    if (useElementArray)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    else
    {
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    return vbo;
}

template <class T>
void
GpuArray<T>::allocVbo(bool useElementArray)
{
    m_vbo = createVbo(m_size*sizeof(T), useElementArray);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_resource, m_vbo,
        cudaGraphicsMapFlagsNone));

   
}

template <class T>
void
GpuArray<T>::freeVbo()
{
    if (m_vbo)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(m_cuda_vbo_resource));
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

  
}


template <class T>
void
GpuArray<T>::map()
{
    if (m_vbo)
    {
        checkCudaErrors(cudaGraphicsMapResources(1, &m_cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&m_dptr, &num_bytes,
                                                             m_cuda_vbo_resource));
    }

   
}

template <class T>
void
GpuArray<T>::unmap()
{
    if (m_vbo)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &m_cuda_vbo_resource, 0));
        m_dptr = 0;
    }

   
}

template <class T>
void
GpuArray<T>::copy(Direction dir, uint start, uint count)
{
    if (count==0)
    {
        count = (uint) m_size;
    }

    map();

    switch (dir)
    {
        case HOST_TO_DEVICE:
            checkCudaErrors(cudaMemcpy((void *)(m_dptr + start), (void *)(m_hptr + start), count*sizeof(T), cudaMemcpyHostToDevice));
            break;

        case DEVICE_TO_HOST:
            checkCudaErrors(cudaMemcpy((void *)(m_hptr + start), (void *)(m_dptr + start), count*sizeof(T), cudaMemcpyDeviceToHost));
            break;
    }

    unmap();
}


template <class T>
void
GpuArray<T>::memset(T value, uint start, uint count)
{
	if (count==0)
	{
		count = (uint) m_size;
	}
	cudaMemset((void *)(m_dptr + start),value,count*sizeof(T));
}

#endif