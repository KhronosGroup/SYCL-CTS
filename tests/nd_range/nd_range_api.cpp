/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_range_api

namespace nd_range_api__
{
using namespace sycl_cts;

/** test cl::sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base
{
public:
    /** return information about this test
     *  @param info, test_base::info structure as output
     */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /** execute the test
     *  @param log, test transcript logging class
     */
    virtual void run( util::logger &log )
    {
        try
        {
            //use accross all the dimentions
            size_t sizes[] = { 16, 32, 64 };

            //dim 1
            {
                const int dim = 1;
                //global size to be set to the size
                cl::sycl::range<dim> gs( sizes[0] );
                //local size to be set to 1/4 of the sizes
                cl::sycl::range<dim> ls( sizes[0]/4 );
                //offset to be set to 1/8 of the sizes
                cl::sycl::id<dim> offset( sizes[0]/8 );

                cl::sycl::nd_range<dim> noOffset( gs, ls );
                CHECK_VALUE(log, noOffset.get_global_range()[0], sizes[0], 0 );
                CHECK_VALUE(log, noOffset.get_local_range()[0], sizes[0]/4, 0 );

                CHECK_VALUE(log, noOffset.get_offset()[0], sizes[0]/8, 0 );
                CHECK_VALUE(log, noOffset.get_group_range()[0], sizes[0] /( sizes[0]/4 ), 0 );

                cl::sycl::nd_range<dim> deepCopy( noOffset );

                CHECK_VALUE(log, deepCopy.get_global_range()[0], sizes[0], 0 );
                CHECK_VALUE(log, deepCopy.get_local_range()[0], sizes[0]/4, 0 );

                CHECK_VALUE(log, deepCopy.get_offset()[0], sizes[0]/8, 0 );
                CHECK_VALUE(log, deepCopy.get_group_range()[0], sizes[0] /( sizes[0]/4 ), 0 );

                cl::sycl::nd_range<dim> withOffset( gs, ls, offset );
                CHECK_VALUE(log, withOffset.get_global_range()[0], sizes[0], 0 );
                CHECK_VALUE(log, withOffset.get_local_range()[0], sizes[0]/4, 0 );
                CHECK_VALUE(log, withOffset.get_offset()[0], sizes[0]/8, 0 );
                CHECK_VALUE(log, withOffset.get_group_range()[0], sizes[0] /( sizes[0]/4 ), 0 );
                cl::sycl::nd_range<dim> deepCopyOffset( withOffset );
                CHECK_VALUE(log, deepCopyOffset.get_global_range()[0], sizes[0], 0 );
                CHECK_VALUE(log, deepCopyOffset.get_local_range()[0], sizes[0]/4, 0 );
                CHECK_VALUE(log, deepCopyOffset.get_offset()[0], sizes[0]/8, 0 );
                CHECK_VALUE(log, deepCopyOffset.get_group_range()[0], sizes[0] /( sizes[0]/4 ), 0 );
            }

            //dim 2
            {
                const int dim = 2;
                //global size to be set to the size
                cl::sycl::range<dim> gs( sizes[0], sizes[1] );
                //local size to be set to 1/4 of the sizes
                cl::sycl::range<dim> ls( sizes[0]/4, sizes[1]/4 );
                //offset to be set to 1/8 of the sizes
                cl::sycl::id<dim> offset( sizes[0]/8, sizes[1]/8 );

                cl::sycl::nd_range<dim> noOffset( gs, ls );
                for(int i = 0; i < dim; i++)
                {
                    CHECK_VALUE(log, noOffset.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, noOffset.get_local_range()[i], sizes[i]/4, i );

                    CHECK_VALUE(log, noOffset.get_offset()[i], sizes[i]/8, i );
                    CHECK_VALUE(log, noOffset.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );

                    cl::sycl::nd_range<dim> deepCopy( noOffset );

                    CHECK_VALUE(log, deepCopy.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, deepCopy.get_local_range()[i], sizes[i]/4, i );

                    CHECK_VALUE(log, deepCopy.get_offset()[i], sizes[i]/8, i );
                    CHECK_VALUE(log, deepCopy.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );

                    cl::sycl::nd_range<dim> withOffset( gs, ls, offset );
                    CHECK_VALUE(log, withOffset.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, withOffset.get_local_range()[i], sizes[i]/4, i );
                    CHECK_VALUE(log, withOffset.get_offset()[i], sizes[0]/8, i );
                    CHECK_VALUE(log, withOffset.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );
                    cl::sycl::nd_range<dim> deepCopyOffset( withOffset );
                    CHECK_VALUE(log, deepCopyOffset.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, deepCopyOffset.get_local_range()[i], sizes[i]/4, i );
                    CHECK_VALUE(log, deepCopyOffset.get_offset()[i], sizes[i]/8, i );
                    CHECK_VALUE(log, deepCopyOffset.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );
                }
            }

            //dim 3
            {
                const int dim = 3;
                //global size to be set to the size
                cl::sycl::range<dim> gs( sizes[0], sizes[1], sizes[2] );
                //local size to be set to 1/4 of the sizes
                cl::sycl::range<dim> ls( sizes[0]/4, sizes[1]/4, sizes[2]/4 );
                //offset to be set to 1/8 of the sizes
                cl::sycl::id<dim> offset( sizes[0]/8, sizes[1]/8, sizes[2]/8 );

                cl::sycl::nd_range<dim> noOffset( gs, ls );
                for(int i = 0; i < dim; i++)
                {
                    CHECK_VALUE(log, noOffset.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, noOffset.get_local_range()[i], sizes[i]/4, i );

                    CHECK_VALUE(log, noOffset.get_offset()[i], sizes[i]/8, i );
                    CHECK_VALUE(log, noOffset.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );

                    cl::sycl::nd_range<dim> deepCopy( noOffset );

                    CHECK_VALUE(log, deepCopy.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, deepCopy.get_local_range()[i], sizes[i]/4, i );

                    CHECK_VALUE(log, deepCopy.get_offset()[i], sizes[i]/8, i );
                    CHECK_VALUE(log, deepCopy.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );

                    cl::sycl::nd_range<dim> withOffset( gs, ls, offset );
                    CHECK_VALUE(log, withOffset.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, withOffset.get_local_range()[i], sizes[i]/4, i );
                    CHECK_VALUE(log, withOffset.get_offset()[i], sizes[0]/8, i );
                    CHECK_VALUE(log, withOffset.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );
                    cl::sycl::nd_range<dim> deepCopyOffset( withOffset );
                    CHECK_VALUE(log, deepCopyOffset.get_global_range()[i], sizes[i], i );
                    CHECK_VALUE(log, deepCopyOffset.get_local_range()[i], sizes[i]/4, i );
                    CHECK_VALUE(log, deepCopyOffset.get_offset()[i], sizes[i]/8, i );
                    CHECK_VALUE(log, deepCopyOffset.get_group_range()[i], sizes[i] /( sizes[i]/4 ), i );
                }
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

} /* namespace nd_range_api__ */
