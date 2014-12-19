/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_range_constructors

namespace nd_range_constructors__
{
using namespace sycl_cts;

/** test cl::sycl::nd_range initialization
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
                cl::sycl::nd_range<dim> deepCopy( noOffset );

                cl::sycl::nd_range<dim> withOffset( gs, ls, offset );
                cl::sycl::nd_range<dim> deepCopyOffset( withOffset );
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
                cl::sycl::nd_range<dim> deepCopy( noOffset );

                cl::sycl::nd_range<dim> withOffset( gs, ls, offset );
                cl::sycl::nd_range<dim> deepCopyOffset( withOffset );
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
                cl::sycl::nd_range<dim> deepCopy( noOffset );

                cl::sycl::nd_range<dim> withOffset( gs, ls, offset );
                cl::sycl::nd_range<dim> deepCopyOffset( withOffset );
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

} /* namespace nd_range_constructors__ */
