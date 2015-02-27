/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME image_constructors

namespace image_constructors__
{
using namespace sycl_cts;
using namespace cl::sycl;

cl_channel_order g_order[] =
{
    CL_R, CL_A, CL_RG, CL_RA, CL_RGB, CL_RGBA, CL_BGRA, CL_ARGB,
    CL_INTENSITY, CL_LUMINANCE, CL_Rx, CL_RGx, CL_RGBx,
    0
};

cl_channel_type g_type[] =
{
    CL_SNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT8, CL_UNORM_INT16,
    CL_UNORM_SHORT_565, CL_UNORM_SHORT_555, CL_UNORM_INT_101010,
    CL_SIGNED_INT8, CL_SIGNED_INT16, CL_SIGNED_INT32, CL_UNSIGNED_INT8,
    CL_UNSIGNED_INT16, CL_UNSIGNED_INT32, CL_HALF_FLOAT, CL_FLOAT,
    0
};

template <int dims, int size>
class image_ctors
{
public:

    void operator()( range<dims>& r , range<dims -1>* p = nullptr )
    {

        /* allocate host side buffer, too large for stack */
        util::UNIQUE_PTR<float> image_host( new float[4 * size] );

        float l_float = 1.0f;

        //white block
        memset( image_host.get(), (*(uint32_t*)&l_float), sizeof( float ) * 4 * size );

        
        /// We will have rubbish data on some of the types.
        /// But we are not checking correctness at this point.

        size_t l_type_itter, l_order_itter;
        //for each type
        for( l_type_itter = 0; g_type[ l_type_itter ] != 0 ; l_type_itter++ )
        {
            //for each order
            for( l_order_itter = 0; g_order[ l_order_itter ] != 0; l_order_itter++ )
            {
                //cl_channel_type
                {
                    image<dims> img(
                                (void*) image_host.get(),
                                g_order[ l_order_itter ],
                                g_type[ l_type_itter ],
                                r
                                );
                    image<dims> ref_img(img);
                }

                //constructor with pitch
                if(p)
                {
                    //using pith
                    image<dims> img(
                                (void*) image_host.get(),
                                g_order[ l_order_itter ],
                                g_type[ l_type_itter ],
                                r,
                                *p);
                    image<dims> ref_img(img);
                }
            }
        }
    }
};

/**
 * test cl::sycl::buffer initialization
 */
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
     */
    virtual void run( util::logger &log ) override
    {
        try
        {
            const int size = 32;
            range<1> range_1d(size);
            range<2> range_2d(size, size);
            range<3> range_3d(size, size, size);

            range<1> pitch_1d(size);
            range<2> pitch_2d(size, size);

            image_ctors<1, size > img_1d;
            image_ctors<2, size * size > img_2d;
            image_ctors<3, size * size * size> img_3d;

            img_1d(range_1d);
            img_2d(range_2d, &pitch_1d);
            img_3d(range_3d, &pitch_2d);

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

} /* namespace image_constructors__ */
