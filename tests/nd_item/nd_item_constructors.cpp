/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#include <array>

#define TEST_NAME nd_item_constructors

namespace {

template <int numDims>
struct nd_item_constructors_kernel;

enum class current_check {
  copy_constructor,
  move_constructor,
  copy_assignment,
  move_assignment,
  SIZE  // This should be last
};

}  // namespace

namespace TEST_NAME {
using namespace sycl_cts;

using success_array_t =
    std::array<bool, static_cast<size_t>(current_check::SIZE)>;

#define CHECK_EQUALITY_HELPER(success, actualValue, expectedValue) \
  {                                                                \
    if (actualValue != expectedValue) {                            \
      success = false;                                             \
    }                                                              \
  }

template <int index, int numDims, typename success_acc_t>
inline void check_equality_helper(success_acc_t& success,
                                  const cl::sycl::nd_item<numDims>& actual,
                                  const cl::sycl::nd_item<numDims>& expected) {
  CHECK_EQUALITY_HELPER(success, actual.get_global_id(index),
                        expected.get_global_id(index));
  CHECK_EQUALITY_HELPER(success, actual.get_local_id(index),
                        expected.get_local_id(index));
  CHECK_EQUALITY_HELPER(success, actual.get_group(index),
                        expected.get_group(index));
  CHECK_EQUALITY_HELPER(success, actual.get_group_range(index),
                        expected.get_group_range(index));
}

template <int numDims, typename success_acc_t>
inline void check_equality(success_acc_t& successAcc,
                           current_check currentCheck,
                           const cl::sycl::nd_item<numDims>& actual,
                           const cl::sycl::nd_item<numDims>& expected) {
  auto& success = successAcc[static_cast<size_t>(currentCheck)];
  if (numDims >= 1) {
    check_equality_helper<0>(success, actual, expected);
  }
  if (numDims >= 2) {
    check_equality_helper<1>(success, actual, expected);
  }
  if (numDims >= 3) {
    check_equality_helper<2>(success, actual, expected);
  }
  CHECK_EQUALITY_HELPER(success, actual.get_global_linear_id(),
                        expected.get_global_linear_id());
  CHECK_EQUALITY_HELPER(success, actual.get_local_linear_id(),
                        expected.get_local_linear_id());
  CHECK_EQUALITY_HELPER(success, actual.get_group_linear_id(),
                        expected.get_group_linear_id());
}

#undef CHECK_EQUALITY_HELPER

/** test cl::sycl::device initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const final {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <int numDims>
  void test_constructors(util::logger& log) {
    try {
      success_array_t success;
      std::fill(std::begin(success), std::end(success), true);

      {
        auto testQueue = util::get_cts_object::queue();

        const auto simpleRange = cl::sycl::range<numDims>();

        cl::sycl::buffer<bool> successBuf(success.data(),
                                          cl::sycl::range<1>(success.size()));

        testQueue.submit([&](cl::sycl::handler& cgh) {
          auto successAcc =
              successBuf.get_access<cl::sycl::access::mode::write>(cgh);

          cgh.parallel_for<nd_item_constructors_kernel<numDims>>(
              cl::sycl::nd_range<numDims>(simpleRange, simpleRange),
              [=](cl::sycl::nd_item<numDims> item) {
                // Check copy constructor
                cl::sycl::nd_item<numDims> copied(item);
                check_equality(successAcc, current_check::copy_constructor,
                               copied, item);

                // Check move constructor
                cl::sycl::nd_item<numDims> moved(std::move(copied));
                check_equality(successAcc, current_check::move_constructor,
                               moved, item);

                // Check copy assignment
                copied = moved;
                check_equality(successAcc, current_check::copy_assignment,
                               copied, item);

                // Check move assignment
                moved = std::move(copied);
                check_equality(successAcc, current_check::move_assignment,
                               moved, item);
              });
        });
      }

      CHECK_VALUE(log,
                  success[static_cast<size_t>(current_check::copy_constructor)],
                  true, numDims);
      CHECK_VALUE(log,
                  success[static_cast<size_t>(current_check::move_constructor)],
                  true, numDims);
      CHECK_VALUE(log,
                  success[static_cast<size_t>(current_check::copy_assignment)],
                  true, numDims);
      CHECK_VALUE(log,
                  success[static_cast<size_t>(current_check::move_assignment)],
                  true, numDims);
    } catch (const cl::sycl::exception& e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }

  /** execute the test
   */
  void run(util::logger& log) final {
    test_constructors<1>(log);
    test_constructors<2>(log);
    test_constructors<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAME
