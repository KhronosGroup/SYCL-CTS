/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME image_api

namespace image_api__
{
using namespace sycl_cts;
using namespace cl::sycl;

//this class tests images for CL_FLOAT only and CL_RGBA
template <int dims, int size>
class image_ctors
{
public:
    void operator()(util::logger &log, range<dims> r )
    {
        /* allocate host side buffer, too large for stack */
        util::UNIQUE_PTR<float> image_host( new float[4 * size] );

        //white image
        for(int i = 0; i < 4 * size; i++)
        {
            image_host.get()[i] = 1.0f;
        }

        ///TODO: check if for all types?
        //cl_channel_type
        {
            image<dims> img(
                        (void*) image_host.get(),
                        CL_RGBA,
                        CL_FLOAT,
                        r
                        );
            if(r.get(0) != img.get_range()[0] ||
               r.get(1) != img.get_range()[1] ||
               r.get(2) != img.get_range()[2])
            {
                FAIL( log, "Renges are not the same." );
            }

            if(size != img.get_size())
            {
                FAIL( log, "Sizes are not the same." );
            }
            queue myQueue;
            command_group(myQueue, [&]() {
                auto img_acc = img.template get_access<float4, dims, access::read_write, access::image>();
                auto myRange = nd_range<1>(range<1>(4*size), range<1>(4*size));
                auto myKernel = ([=](item<1> item) {
                    img_acc[item.get_global(0)] = 0.2;
                });
            });
        }

        //white image
        for(int i = 0; i < 4 * size; i++)
        {
           CHECK_VALUE(log, image_host.get()[i], 0.2f, i );
        }
    }

    void operator()(util::logger &log, range<dims> r, range<dims -1> p )
    {
        /* allocate host side buffer, too large for stack */
        util::UNIQUE_PTR<float> image_host( new float[4 * size] );

        //white image
        memset( image_host.get(), 1, sizeof( float ) * 4 * size );

        ///TODO: check if for all types?
        //cl_channel_type
        {
            image<dims> img(
                        (void*) image_host.get(),
                        CL_RGBA,
                        CL_FLOAT,
                        r,
                        p
                        );
            if(r.get(0) != img.get_range()[0] ||
               r.get(1) != img.get_range()[1] ||
               r.get(2) != img.get_range()[2])
            {
                FAIL( log, "Renges are not the same." );
            }

            if(p != img.get_pitch())
            {
                FAIL( log, "Pitchs are not the same." );
            }

            if(size * 4 != img.get_size())
            {
                FAIL( log, "Sizes are not the same." );
            }
        }
    }
};

/**
 * test cl::sycl::buffer initialization
 */
class TEST_NAME : public util::test_base_opencl
{
public:
    /** return information about this test
     * @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /**
     * execute the test
     * @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {
        try
        {
            const int size = 32;
            range<1> range1d(size);
            range<2> range2d(size, size);
            range<3> range3d(size, size, size);

            range<1> pitch1d(size);
            range<2> pitch2d(size, size);
            {
                image_ctors<1, size > img1d;
                image_ctors<2, size * size > img2d;
                image_ctors<3, size * size * size> img3d;
                img1d(log, range1d);
                img2d(log, range2d);
                img3d(log, range3d);
            }
            {
                image_ctors<2, size * size > img2d;
                image_ctors<3, size * size * size> img3d;
                img2d(log, range2d, pitch1d);
                img3d(log, range3d, pitch2d);
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

} /* namespace image_api__ */
