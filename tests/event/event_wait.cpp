/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
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
    *  @param info, test_base::info structure as output
    */
    virtual void get_info( test_base::info &out ) const
    {
        set_test_info( out, TOSTRING( TEST_NAME ), TEST_FILE );
    }

    event add_operation(sycl_cts::util::logger &log, queue &queue, buffer<float, 1> &d_data, const float operand)
    {

        /* add command to queue */
        command_group myCmdGroup(queue, [&]()
        {
            auto a_data = d_data.get_access<access::read_write>();

            single_task<class add_val>([=]() {
                a_data[0] += operand;
            });
        });

        return myCmdGroup.complete_event();
    }


    event mul_operation(sycl_cts::util::logger &log, queue &queue, buffer<float, 1> &d_data, const float operand)
    {
        /* add command to queue */
        command_group myCmdGroup(queue, [&]()
        {
            auto a_data = d_data.get_access<access::read_write>();

            single_task<class mul_val>([=]() {
                a_data[0] *= operand;
            });
        });

           return myCmdGroup.complete_event();
    }

    /** Execute kernels, waiting in-between
    */
    bool wait_and_exec(sycl_cts::util::logger &log, queue &queueA, queue &queueB)
    {

        for (int i = 0; i < 4; ++i)
        {
            float h_data = 0.0;
            {   // Create a new scope so we can check the result of the buffer when it's written back to host

                buffer<float, 1> d_data(&h_data, 1);

                add_operation(log, queueA, d_data, 1.0);
                event complete = mul_operation(log, queueA, d_data, 2.0);
                switch (i)
                {
                    case 0:  // Test cl::sycl::event::wait()
                        complete.wait();
                        break;
                    case 1:  // Test cl::sycl::event::wait_and_throw()
                        complete.wait_and_throw();
                        break;
                    case 2:  // Test cl::sycl::event::wait(vector_class<event>)
                        VECTOR_CLASS<event> evt_list = complete.get_wait_list();
                        event::wait(evt_list);
                        break;
                    case 3:  // Test cl::sycl::event::wait_and_throw(vector_class<event>)
                        VECTOR_CLASS<event> evt_list = complete.get_wait_list();
                        event::wait_and_throw(evt_list);
                        break;
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
    *  @param log, test transcript logging class
    */
    virtual void run( sycl_cts::util::logger &log )
    {
        try
        {
            /* create device selector */
           intel_selector l_selector;

           /* create command queue */
           queue queueA( l_selector);
           queue queueB( l_selector);

           if (!wait_and_exec(log, queueA, queueB))
           {
              FAIL(log, "cl::sycl::event::wait() tests failed");
           }

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

} /* namespace event_wait */
