#include "gl_util.h"
namespace sycl_cts 
{
namespace gl_util
{ 
extern "C" 
{ 
bool init_gl_context( cl_gl_context * in ) {

    cl_int error = CL_SUCCESS;
    //get platforms

    cl_uint platform_id_size = 0;
    clGetPlatformIDs( 0,0,&platform_id_size );

    if ( platform_id_size == 0 )
    {
        return false;
    }

    cl_platform_id * platform_ids_list = new cl_platform_id[platform_id_size];
    clGetPlatformIDs( platform_id_size, platform_ids_list, 0 );

    cl_platform_id platform_id = 0;
    cl_device_id device_id = 0;
    cl_context context = 0;

    for( size_t i = 0; i < platform_id_size &&  context == 0; ++i )
    {
        const cl_platform_id platform_to_try = platform_ids_list[i];

        cl_uint dev_id = 0;
        clGetDeviceIDs( platform_to_try, CL_DEVICE_TYPE_CPU, 0, 0, &dev_id );

        if( dev_id == 0 )
            continue;

        cl_device_id * device_id_list = new cl_device_id[ dev_id ];
        clGetDeviceIDs( platform_to_try, CL_DEVICE_TYPE_CPU, dev_id, device_id_list, 0 );

        cl_context_properties context_properties[] = {
            #if defined (WIN32)
            // We should first check for cl_khr_gl_sharing extension.
            CL_GL_CONTEXT_KHR , (cl_context_properties) wglGetCurrentContext() ,
            CL_WGL_HDC_KHR , (cl_context_properties) wglGetCurrentDC() ,
            #elif defined (__linux__)
            // We should first check for cl_khr_gl_sharing extension.
            CL_GL_CONTEXT_KHR , (cl_context_properties) glXGetCurrentContext() ,
            CL_GLX_DISPLAY_KHR , (cl_context_properties) glXGetCurrentDisplay() ,
            #elif defined (__APPLE__)
            // We should first check for cl_APPLE_gl_sharing extension.
            #if 0
            // This doesn't work.
            CL_GL_CONTEXT_KHR , (cl_context_properties) CGLGetCurrentContext() ,
            CL_CGL_SHAREGROUP_KHR , (cl_context_properties) CGLGetShareGroup( CGLGetCurrentContext() ) ,
            #else
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE , (cl_context_properties) CGLGetShareGroup( CGLGetCurrentContext() ) ,
            #endif
            #endif
            CL_CONTEXT_PLATFORM , (cl_context_properties) platform_to_try ,
            0 , 0 ,
        };

        //look for context
        for ( size_t j = 0; j < dev_id; ++j )
        {
            cl_device_id device_to_try = device_id_list[j];
            cl_context context_to_try = 0;
            context_to_try = clCreateContext(
                        context_properties,
                        1, &device_to_try,
                        0, 0,
                        &error
                        );

            if ( error == CL_SUCCESS )
            {
                platform_id = platform_to_try;
                device_id = device_to_try;
                context = context_to_try;
                break;
            }
        }

        if( device_id_list )
        {
            delete [] device_id_list;
        }
    }

    // no compatible OpenCL device found
    if ( device_id == 0 )
    {
        if( platform_ids_list )
            delete [] platform_ids_list;
        return false;
    }

    in->set_cl_context( context );

    if( platform_ids_list )
        delete [] platform_ids_list;

    return true;
}
bool gl_set_up() {
    if(gl_init)
        return true;
    /* Initialize the main window */
    int argc = 0;
    int glut_time = glutGet(GLUT_ELAPSED_TIME);
    if(glut_time > 0){
        glutInit(&argc, nullptr);
    }
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(300, 300);
    glutCreateWindow("Sphere");
    glEnable(GL_DEPTH_TEST);

    /* Launch GLEW processing */
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        return false;
    }
    gl_init = true;
    return true;
}
void gl_clean_up()
{
}
}
} // end of gl_framework namespace
} // end of sycl_cts namespace
