/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME exception_handling

namespace TEST_NAMESPACE
{
using namespace sycl_cts;
using namespace cl::sycl;

/**
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
        util::VECTOR<exception_ptr> excps;

        cl::sycl::function_class<void(cl::sycl::exception_list)> fn =
            [&]( cl::sycl::exception_list l )
        {
            for ( auto e : l )
                excps.push_back( e );
        };

        try
        {
            cl::sycl::queue q( fn );

            q.submit( [&](handler & cgh )
            {
                ;
            } );

            q.wait_and_throw();
        }
        catch( exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }

        for ( auto e : excps )
        {
            try
            {
                throw e;
            }
            catch ( cl_exception e )
            {
                util::STRING sc = e.get_description();
                context * c = e.get_context();
                cl_int ci = e.get_cl_code();
            }
            catch ( async_exception e )
            {
                sycl_cts::util::STRING sc = e.get_description();
                context * c = e.get_context();
            }
            catch ( exception e )
            {
                util::STRING sc = e.get_description();
                context * c = e.get_context();
                log_exception( log, e );
                FAIL( log, "sycl exception caught" );
            }
        }
    }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace explicit_pointer_constructors__ */
