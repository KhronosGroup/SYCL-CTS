/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_wait

namespace event_wait__
{
using namespace sycl_cts;
using namespace cl::sycl;

/**
 */
class TEST_NAME : public sycl_cts::util::test_base
{
public:

    /** return information about this test
     */
    virtual void get_info( test_base::info &out ) const override
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    /* enqueue an add command and return the complete event */
    event add_operation(sycl_cts::util::logger &log, queue &queue, buffer<float, 1> &d_data, const float operand)
    {
        return queue.submit( [&]( handler& cgh )
        {
            auto a_data = d_data.get_access<cl::sycl::access::mode::read_write>( cgh );

            cgh.single_task<class add_kernel>( [=]()
            {
                a_data[0] += operand;
            } );
        } ).get_complete();
    }

    /* enqueue a mul command and return the complete event */
    event mul_operation(sycl_cts::util::logger &log, queue &queue, buffer<float, 1> &d_data, const float operand)
    {
        return queue.submit( [&]( handler& cgh )
        {
            auto a_data = d_data.get_access<cl::sycl::access::mode::read_write>( cgh );

            cgh.single_task<class mul_kernel>( [=]()
            {
                a_data[0] *= operand;
            } );

        } ).get_complete();
    }

    /** Execute kernels, waiting in-between
     */
    bool wait_and_exec(sycl_cts::util::logger &log, queue &queueA, queue &queueB)
    {
        for (int i = 0; i < 4; ++i)
        {
            float h_data = 0.0;
            {   // Create a new scope so we can check the result of the buffer when it's written back to host

                buffer<float, 1> d_data(&h_data, range<1>( 1 ));

                event complete = mul_operation(log, queueA, d_data, 2.0);

                switch (i)
                {
                    case 0: 
                      {  // Test cl::sycl::event::wait()
                        complete.wait();
                        break;
                      }
                    case 1: 
                      { // Test cl::sycl::event::wait_and_throw()
                        complete.wait_and_throw();
                        break;
                      }
                    case 2:
                      {  // Test cl::sycl::event::wait(vector_class<event>)
                        vector_class<event> evt_list = complete.get_wait_list();
                        event::wait(evt_list);
                        break;
                      }
                    case 3:
                      {  // Test cl::sycl::event::wait_and_throw(vector_class<event>)
                        vector_class<event> evt_list = complete.get_wait_list();
                        event::wait_and_throw(evt_list);
                        break;
                      }
                }
                add_operation(log, queueB, d_data, -3.0);
            }
            if (h_data != -1.0)
            {
                return false;
            }
        }

        return true;
     }

    /** execute the test
     */
    virtual void run( sycl_cts::util::logger &log ) override
    {
        try
        {
           cts_selector selector;

           queue queueA( selector );
           queue queueB( selector );





           if (!wait_and_exec(log, queueA, queueB))
           {
              FAIL(log, "cl::sycl::event::wait() tests failed");
           }

           queueA.wait_and_throw();
           queueB.wait_and_throw();

        }
        catch ( exception e )
        {
            log_exception( log, e );
            FAIL( log, "sycl exception caught" );
        }
    }
};

// construction of this proxy will register the above test
sycl_cts::util::test_proxy<TEST_NAME> proxy;

} /* namespace event_wait__ */
