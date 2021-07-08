/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_pointers

namespace kernel_pointers__ {
using namespace sycl_cts;

/**
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      auto my_queue = util::get_cts_object::queue();

      typedef int res_type;

      res_type result = 0;
      {
        sycl::buffer<res_type, 1> buf_result(&result,
                                                 sycl::range<1>(1));

        uint32_t outer_index = 2;

        my_queue.submit([&](sycl::handler &cgh) {
          auto acc_result =
              buf_result.get_access<sycl::access::mode::read_write>(cgh);

          cgh.single_task<TEST_NAME>([acc_result, outer_index]() {
            uint8_t my_array[] = {0, 1, 2, 3, 4};

            uint8_t *ptr = my_array + outer_index;
            ptr++;

            acc_result[0] = *ptr;
          });
        });
      }
      if (result != 3) {
        FAIL(log, "Pointer in kernel not working correctly");
        log.note("wanted: 3, got:" + std::to_string(result));
      }

      my_queue.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_pointers__ */
