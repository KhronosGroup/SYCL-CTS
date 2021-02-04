/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
//  Test description:
//
//    This test makes sure that:
//      - Implementations handles correctly a non-uniform range passed to
//        group::parallel_for_work_item;
//      - Implementations handles correctly call to
//        group::parallel_for_work_item with a 0 logical local range.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_non_uniform_local_range

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test cl::sycl::range::get(int index) return size_t
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
      constexpr unsigned group_range = 4;
      constexpr unsigned local_range = 3;
      constexpr unsigned global_range = group_range * local_range;
      std::vector<int> data(global_range);

      // Set element of the vector with -1 to represent unset data.
      std::fill(data.begin(), data.end(), -1);

      auto myQueue = util::get_cts_object::queue();
      // using this scope we ensure that the buffer will update the host values
      // after the wait_and_throw
      {
        cl::sycl::buffer<int, 1> buf(data.data(),
                                     cl::sycl::range<1>(data.size()));

        myQueue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buf.template get_access<cl::sycl::access::mode::read_write>(cgh);

          cgh.parallel_for_work_group<class TEST_NAME>(
              cl::sycl::range<1>(group_range), cl::sycl::range<1>(local_range),
              [=](cl::sycl::group<1> group_pid) {
                group_pid.parallel_for_work_item(
                    cl::sycl::range<1>(group_pid.get_id()),
                    [&](cl::sycl::h_item<1> item_id) {
                      accessor[item_id.get_global()[0]] =
                          item_id.get_physical_local()[0];
                    });
              });
        });
        myQueue.wait_and_throw();
      }

      unsigned idx = 0;
      for (unsigned group_id = 0; group_id < group_range; group_id++) {
        for (unsigned local_id = 0; local_id < local_range; local_id++) {
          int expected = local_id < group_id ? local_id : -1;
          if (data[idx] != expected) {
            cl::sycl::string_class errorMessage =
                cl::sycl::string_class("Value for global id ") +
                std::to_string(idx) +
                cl::sycl::string_class(" was not correct (") +
                std::to_string(data[idx]) +
                cl::sycl::string_class(" instead of ") +
                std::to_string(expected) + ")";
            FAIL(log, errorMessage);
          }
          idx++;
        }
      }

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
