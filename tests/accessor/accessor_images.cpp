/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
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
 */
class TEST_NAME : public util::test_base
{
public:
    /**
     * return information about this test
     */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /**
     * execute the test
     */
    virtual void run( util::logger &log ) override
    {
        static const int size = 64;
        static const int dims = 2;
        try
        {
            const uint32_t area = size * size;

            float4  data[ area ];
            float4 oData[ area ];

            for ( uint32_t i = 0; i < area; i++ )
            {
                // white image
                 data[i] = { 1.f, 1.f, 1.f, 1.f };
                // black image
                oData[i] = { 0.f, 0.f, 0.f, 1.f };
            }

            queue q;
            {
                range<dims> r( size, size );

                image<dims>  img(  data, CL_RGBA, CL_FLOAT, r );
                image<dims> oImg( oData, CL_RGBA, CL_FLOAT, r );

                q.submit( [&]( handler& cgh )
                {
                    cl::sycl::nd_range<dims> nd_r = nd_range<dims>( r, r );

                    accessor<float4, dims, cl::sycl::access::mode::read,  access::image>  in( img, cgh );
                    accessor<float4, dims, cl::sycl::access::mode::write, access::image> out( oImg, cgh );
                    accessor<float4, dims, cl::sycl::access::mode::write, access::image_array> outPtr_array( img_out, cgh );
                    accessor<float4, dims, cl::sycl::access::mode::read, access::image_array> inPtr_array( img_out, cgh );
                    accessor<float4, dims, cl::sycl::access::mode::read_write, access::image_array> inPtr_array_rw( img_out, cgh );
                    accessor<float4, dims, cl::sycl::access::mode::read_write, access::image> inPtr_rw( img_out, cgh );
                    sampler smpl( true, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST );

                    cgh.parallel_for<class TEST_NAME>( nd_r, [=] ( item<dims> item )
                    {
                        cl::sycl::id<dims> i( item );
                        auto datum = in[i];
                        out[i] = datum;
                    } );
                } );
                accessor<float4, dims, cl::sycl::access::mode::read_write, access::host_image> hostPtr( oImg );
            }
            for ( uint32_t i = 0; i < area; i++ )
            {
                
                CHECK_VALUE(log, oData[i], data[i], i );
            }

            q.wait_and_throw();

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
