#pragma once

#if defined (__APPLE__) || defined(MACOSX)
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
#define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif
#include <CL/cl_gl.h>

#include <GL/glew.h>
#define FREEGLUT_LIB_PRAGMAS 0
#include <GL/glut.h>

// glx is a non-Windows header
#ifndef _WIN32
#include "GL/glx.h"
#endif


namespace sycl_cts 
{
namespace gl_util
{ 
static int gl_init = false;
struct cl_gl_context
{
private:
    static const int m_prop_size = 32;
    cl_context m_context;
    cl_context_properties m_properties[m_prop_size];
public:
    int get_prop_max_size() { return m_prop_size; }
    cl_context get_context() { return m_context; }
    void set_cl_context( cl_context c ) { m_context = c; }
    cl_context_properties * get_properties() { return m_properties; }
};

extern "C" 
{
bool init_gl_context( cl_gl_context * in );
bool gl_set_up();
void gl_clean_up();
}
} // end of gl_framework namespace
} // end of sycl_cts namespace
