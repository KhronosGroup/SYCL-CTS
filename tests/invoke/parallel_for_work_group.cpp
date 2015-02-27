/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME parallel_for_work_group

namespace parallel_for_work_group__
{
using namespace sycl_cts;

/** test cl::sycl::parallel_for_work_group
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
        static const uint32_t SIZE = 16;

        try
        {
            uint32_t f[3];
            {
                using namespace cl::sycl;
                default_selector sel;
                queue queue(sel);

                buffer<uint32_t, 1> buf( f, range<1>( 1 ) );

                queue.submit( [&]( handler& cgh )
                {
                    auto a_dev = buf.get_access<cl::sycl::access::mode::read_write>( cgh );

                    cgh.parallel_for_workgroup<class TEST_NAME>(
                        nd_range<3>(range<3>(SIZE, SIZE, SIZE), range<3>(1, 1, 1)),
                        [=] (group<3> myGroup)
                    {
                        //the local sizes should be
                        a_dev[0] = uint32_t( myGroup.get_local_range()[0] );
                        a_dev[1] = uint32_t( myGroup.get_local_range()[1] );
                        a_dev[2] = uint32_t( myGroup.get_local_range()[2] );
                    } );
                } );

                queue.wait_and_throw();
            }

            for ( uint32_t i = 0; i < 3; i++ )
            {
                if ( !CHECK_VALUE( log, f[i], SIZE, i ) )
                    return;
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

} /* namespace parallel_for_work_group__ */
