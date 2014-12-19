/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME accessor_images

namespace accessor_images__
{
using namespace sycl_cts;
using namespace cl::sycl;

/**
 * test cl::sycl::image initialization
 **/
class TEST_NAME : public util::test_base
{
public:
    /**
     * return information about this test
     * @param info, test_base::info structure as output
     **/
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /**
     * execute the test
     * @param log, test transcript logging class
     **/
    virtual void run( util::logger &log )
    {
        static const int size = 64;
        static const int dims = 2;
        try
        {
            float4  data[size * size];
            float4 oData[size * size];

            // white image
            for ( int i = 0; i < size * size; i++ )
            {
                 data[i] = { 1.0f, 1.f, 1.f, 1.f };
                oData[i] = { 0.0f, 0.f, 0.f, 1.f };
            }

            queue q;
            {
                range<dims> r( size, size );

                image<dims>  img(  data, CL_RGBA, CL_FLOAT, r );
                image<dims> oImg( oData, CL_RGBA, CL_FLOAT, r );

                command_group( q, [&]()
                {
                    cl::sycl::nd_range<dims> nd_r = nd_range<dims>( r, r );

                    accessor<float4, dims, access::read,  access::image>  in( img );
                    accessor<float4, dims, access::write, access::image> out( oImg );
                    sampler smpl( true, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST );

                    parallel_for<class TEST_NAME>( nd_r, [=] ( item<dims> item )
                    {
                        cl::sycl::id<dims> i( item );
                        auto datum = in[i];
                        out[i] = datum;
                    } );
                } );
                accessor<float4, dims, access::read_write, access::host_image> hostPtr( oImg );
            }
            for(int i = 0; i < size; i++)
            {
                CHECK_VALUE(log, oData[i], data[i], i );
            }
        }
        catch ( cl::sycl::exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace accessor_images__ */
