/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME parallel_for_workgroup

namespace parallel_for_workgroup__
{
using namespace sycl_cts;

#define SIZE 16

/** test cl::sycl::parallel_for_workgroup
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
            float f[3];
            {
                using namespace cl::sycl;
                default_selector sel;
                queue queue(sel);

                buffer<float,1> buf(&f[0], range<1>(1));
                command_group(queue, [&] () {
                    auto a_dev = buf.get_access<access::read_write>();

                    parallel_for_workgroup<class TEST_NAME>(
                        nd_range<3>(range<3>(SIZE, SIZE, SIZE), range<3>(1, 1, 1)),
                        [=] (group<3> myGroup)
                    {
                        //the local sizes should be
                        a_dev[0] = myGroup.get_local_range()[0];
                        a_dev[1] = myGroup.get_local_range()[1];
                        a_dev[2] = myGroup.get_local_range()[2];
                    });
                });
            }

            for(int i = 0; i < 3; i++)
            {
                if(f[i] != SIZE)
                {
                    CHECK_VALUE( log, f[i], static_cast<float>(SIZE), i );
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

} /* namespace parallel_for_workgroup__ */
