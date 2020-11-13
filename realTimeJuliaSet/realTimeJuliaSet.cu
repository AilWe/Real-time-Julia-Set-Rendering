/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#include "cuda.h"
#include "cuda_gl_interop.h"

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

#define     DIM    512
#define MY_PI 3.1415926

static float rotatef = 296.0;
static int updateTime = 1000;
static float updateAngle = 0.5;

struct cuComplex{
  float r;
  float i;
  __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}
  __device__ float magnitude2(void) {
    return r * r + i * i;
  }
  __device__ cuComplex operator*(const cuComplex& a){
    return cuComplex(r*a.r-i*a.i, i*a.r+r*a.i);
  }
  __device__ cuComplex operator+(const cuComplex& a){
    return cuComplex(r+a.r, i+a.i);
  }
};

__device__ int julia(int x, int y, float angle) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM/2 - x)/(DIM/2);
  float jy = scale * (float)(DIM/2 - y)/(DIM/2);

  float factor = 0.578;
  cuComplex c(factor * cosf(angle), factor * sinf(angle));
//  cuComplex c(-0.8, angle / (2*MY_PI));
//  cuComplex c(-0.8, 0.156);
  cuComplex a(jx, jy);

  int i = 0;
  for (i = 0; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000)
      return 0;
  }
  return i;
}

GLuint  bufferObj;
cudaGraphicsResource *resource;

// based on ripple code, but uses uchar4 which is the type of data
// graphic inter op uses. see screenshot - basic2.png
__global__ void kernel( uchar4 *ptr , float angle) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y, angle/180.0*MY_PI);
    // accessing uchar4 vs unsigned char*
    ptr[offset].x = 255 * juliaValue;
    ptr[offset].y = 55 * juliaValue;
    ptr[offset].z = 25 * juliaValue;
    ptr[offset].w = 255;
}

static void key_func( unsigned char key, int x, int y ) {
    switch (key) {
        case 27:
            // clean up OpenGL and CUDA
            HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &bufferObj );
            exit(0);
    }
}

static void draw_func( void ) {
    // we pass zero as the last parameter, because out bufferObj is now
    // the source, and the field switches from being a pointer to a
    // bitmap to now mean an offset into a bitmap object
    HANDLE_ERROR( 
        cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone ) );
    HANDLE_ERROR( cudaGraphicsMapResources( 1, &resource, NULL ) );
    uchar4* devPtr;
    size_t  size;
    HANDLE_ERROR( 
        cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
                                              &size, 
                                              resource) );
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>( devPtr , rotatef);
    glDrawPixels( DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();
}

void update(int value){
  rotatef += updateAngle;
  printf ("rotatef: %f\n", rotatef);
  if (rotatef > 360.f)
    rotatef -= 360;
  glutPostRedisplay();
  glutTimerFunc(updateTime, update, 0);
}

int main( int argc, char **argv ) {
    cudaDeviceProp  prop;
    int dev;

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );

    // tell CUDA which dev we will be using for graphic interop
    // from the programming guide:  Interoperability with OpenGL
    //     requires that the CUDA device be specified by
    //     cudaGLSetGLDevice() before any other runtime calls.

    HANDLE_ERROR( cudaGLSetGLDevice( dev ) );

    // these GLUT calls need to be made before the other OpenGL
    // calls, else we get a seg fault
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( DIM, DIM );
    glutCreateWindow( "bitmap" );

    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    // the first three are standard OpenGL, the 4th is the CUDA reg 
    // of the bitmap these calls exist starting in OpenGL 1.5
    glGenBuffers( 1, &bufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB );

//    HANDLE_ERROR( 
//        cudaGraphicsGLRegisterBuffer( &resource, 
//                                      bufferObj, 
//                                      cudaGraphicsMapFlagsNone ) );

    // do work with the memory dst being on the GPU, gotten via mapping
//    HANDLE_ERROR( cudaGraphicsMapResources( 1, &resource, NULL ) );
//    uchar4* devPtr;
//    size_t  size;
//    HANDLE_ERROR( 
//        cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
//                                              &size, 
//                                              resource) );

//    dim3    grids(DIM/16,DIM/16);
//    dim3    threads(16,16);
//    kernel<<<grids,threads>>>( devPtr , rotatef);
//    HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &resource, NULL ) );

    // set up GLUT and kick off main loop
    glutKeyboardFunc( key_func );
    glutDisplayFunc( draw_func );
    glutTimerFunc(updateTime, update, 0);
    glutMainLoop();
}
