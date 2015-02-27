/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"
#include "../../gl_util/gl_util.h"


/// should we introduce dependency or shall we just put the gl header in common?

#define TEST_NAME opengl_context

namespace opengl_context__
{
using namespace sycl_cts;
using namespace cl::sycl;

/** Test for the SYCL buffer OpenGL interoperation interface
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl
{
public:
    /** return information about this test
     */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute this test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            // set up gl_framework
            if ( !gl_util::gl_set_up() )
            {
                FAIL( log, "gl_set_up failed." );
            }

            gl_util::cl_gl_context cl_gl_ctx;
            if ( gl_util::init_gl_context( &cl_gl_ctx ) )
            {
                // creat sycl context of cl_context that contains gl info.
                {
                    context sycl_ctx( cl_gl_ctx.get_context() );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create sycl context of device_selector & cl_context_properties
                {
                    cts_selector sel;
                    context sycl_ctx( sel, cl_gl_ctx.get_properties() );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create sycl context of device & cl_context_properties
                {
                    device dev;
                    context sycl_ctx( dev, cl_gl_ctx.get_properties() );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create sycl context of device & cl_context_properties
                {
                    device dev;
                    device dev_ref( dev );
                    context sycl_ctx( dev_ref, cl_gl_ctx.get_properties() );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create sycl context of platform & cl_context_properties
                {
                    platform plat;
                    context sycl_ctx( plat, cl_gl_ctx.get_properties() );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create of vector_class<device>
                {
                    VECTOR_CLASS<device> dev_vec;
                    device a;

                    device b;
                    dev_vec.push_back( a );
                    dev_vec.push_back( b );

                    context sycl_ctx( dev_vec, cl_gl_ctx.get_properties() );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                function_class<void( cl::sycl::exception_list )> fn = [&]( exception_list l )
                {
                    if ( l.size() > 1 )
                        FAIL( log, "Exception thrown during execution of kernel" );
                };

                // device_selector, cl_context_properties & async_handler
                {
                    cts_selector sel;
                    context sycl_ctx( sel, cl_gl_ctx.get_properties(), fn );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create sycl context of device, cl_context_properties & async_handler
                {
                    device dev;
                    context sycl_ctx( dev, cl_gl_ctx.get_properties(), fn );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create sycl context of platform, cl_context_properties, & async_handler
                {
                    platform plat;
                    context sycl_ctx( plat, cl_gl_ctx.get_properties(), fn );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }

                // create of vector_class<device> & async_handler
                {
                    VECTOR_CLASS<device> dev_vec;
                    device a;
                    device b;
                    dev_vec.push_back( a );
                    dev_vec.push_back( b );

                    context sycl_ctx( dev_vec, cl_gl_ctx.get_properties(), fn );
                    device dev = sycl_ctx.get_gl_current_device();
                    VECTOR_CLASS<device> devs = sycl_ctx.get_gl_context_devices();
                }
            }
            else
            {
                FAIL( log, "cl_context creation failed." );
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_buffer__ */
