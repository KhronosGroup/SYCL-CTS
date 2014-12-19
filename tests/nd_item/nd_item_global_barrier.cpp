/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2014 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_global_barrier

namespace nd_item_global_barrier__
{
    using namespace cl::sycl;
    using namespace sycl_cts;

    void test_barrier(util::logger & log, cl::sycl::queue &queue)
    {
        /* set workspace size */
        const int globalSize = 64;
        const int localSize = 2;

        /* allocate and assign host data */
        std::unique_ptr<int> data (new int[globalSize]);
        std::unique_ptr<int> scratch(new int[globalSize]);

        for (int i = 0; i < globalSize; ++i)
        {
            data.get()[i] = i;
            scratch.get()[i] = 0;
        }

        /* init ranges*/
        range<1> globalRange(globalSize);
        range<1> localRange(localSize);
        nd_range<1> NDRange(globalRange, localRange);

        /* run kernel to swap adjancent work item's global id*/
        {
            buffer<int, 1> buf(data.get(), globalRange);
            buffer<int, 1> scratchBuf(scratch.get(), globalRange);

            command_group(queue, [&]() {
                auto accGlobal = buf.get_access<access::read_write>();
                auto globalScratch = scratchBuf.get_access<access::read_write>();
                parallel_for<class global_barrier_kernel>(NDRange, [=](nd_item<1> item)
                {
                    int idx = (int)item.get_global(0);
                    int pos = idx & 1;
                    int opp = pos ^ 1;

                    globalScratch [pos] = accGlobal[idx];

                    item.barrier(access::fence_space::global);

                    accGlobal[idx] = globalScratch[opp];
                });
            });
        }

        /* check correct results returned*/
        bool passed = true;
        for (int i = 0; i < globalSize; ++i)
        {
            if (i % 2 == 0)
                passed &= data.get()[i] == (i + 1);
            else
                passed &= data.get()[i] == (i - 1);
        }

        if (!passed)
        {
            FAIL(log, "global barrier failed");
        }
    }

    /** test cl::sycl::nd_item global barrier
    */
    class TEST_NAME : public util::test_base
    {
    public:

        /** return information about this test
        *  @param info, test_base::info structure as output
        */
        virtual void get_info(test_base::info &out) const
        {
            set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
        }

        /** execute the test
        *  @param log, test transcript logging class
        */
        virtual void run(util::logger &log)
        {

            try
            {
                intel_selector selector;
                queue cmdQueue(selector);

                test_barrier(log, cmdQueue);
            }
            catch (cl::sycl::exception e)
            {
                log_exception(log, e);
                FAIL(log, "sycl exception caught");
            }
        }
    };

    namespace
    {
        // construction of this proxy will register the above test
        util::test_proxy<TEST_NAME> proxy;
    } /* namespace {} */

}  // sycl_cts
