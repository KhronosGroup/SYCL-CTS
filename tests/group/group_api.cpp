/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME group_api

namespace group_api__ {
using namespace sycl_cts;

static const int EXPECTED = 1;

class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      using namespace cl::sycl;
      default_selector sel;
      queue queue(sel);

      enum { eFAIL, ePASS };

      const uint32_t num_items = 5;

      int result[num_items];
      for (uint32_t i = 0; i < num_items; i++) result[i] = ePASS;

      {
        buffer<int, 1> buf(result, range<1>(num_items));
        queue.submit([&](handler &cgh) {
          auto a_dev = buf.get_access<cl::sycl::access::mode::read_write>(cgh);

          cgh.parallel_for_work_group<class TEST_NAME>(range<3>(1, 1, 1),
                                                       [=](group<3> my_group) {
            // get_group()
            id<3> m_get_group = my_group.get();
            if (m_get_group.get(0) > EXPECTED ||
                m_get_group.get(1) > EXPECTED || m_get_group.get(2) > EXPECTED)
              a_dev[0] = eFAIL;

            // get_local_range()
            range<3> m_get_local_range = my_group.get_local_range();
            if (m_get_local_range.get(0) > EXPECTED ||
                m_get_local_range.get(1) > EXPECTED ||
                m_get_local_range.get(2) > EXPECTED)
              a_dev[1] = eFAIL;

            // get_global_range()
            range<3> m_get_global_range = my_group.get_global_range();
            if (m_get_global_range.get(0) > EXPECTED ||
                m_get_global_range.get(1) > EXPECTED ||
                m_get_global_range.get(2) > EXPECTED)
              a_dev[2] = eFAIL;

            // get(int dimention)
            size_t m_get_x = my_group.get(0);
            size_t m_get_y = my_group.get(1);
            size_t m_get_z = my_group.get(2);
            if (m_get_x > EXPECTED || m_get_y > EXPECTED || m_get_z > EXPECTED)
              a_dev[3] = eFAIL;

            //[]
            size_t m_get_x_op = my_group[0];
            size_t m_get_y_op = my_group[1];
            size_t m_get_z_op = my_group[2];
            if (m_get_x_op > EXPECTED || m_get_y_op > EXPECTED ||
                m_get_z_op > EXPECTED)
              a_dev[4] = eFAIL;
          });
        });
      }

      for (uint32_t i = 0; i < num_items; i++)
        if (!CHECK_VALUE(log, result[i], ePASS, i)) return;

      queue.wait_and_throw();

    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace group_api__ */
