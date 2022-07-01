/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/common.h"

#include <array>
#include <string>

#define TEST_NAME h_item_constructors

namespace {

template <int numDims>
struct h_item_constructors_kernel;

/**
 * @brief Provides a safe index for checking an operation
 */
enum class current_check {
  copy_constructor,
  move_constructor,
  copy_assignment,
  move_assignment,
  SIZE  // This should be last
};

}  // namespace

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/**
 * @brief Type used for storing individual operations
 */
using success_array_t =
    std::array<bool, static_cast<size_t>(current_check::SIZE)>;

/**
 * @brief Helper macro for calling the same function that expects an index
 *        on two different objects and comparing the results
 * @param index Current index to call the function with
 * @param success Reference to boolean values used to signal success
 * @param actual Object to retrieve the actual value from
 * @param expected Object to retrieve the expected value from
 * @param function The function to call on the actual and expected objects
 */
#define CHECK_EQUALITY_HELPER(index, success, actual, expected, function) \
  {                                                                       \
    const auto actualValue = actual.function(index);                      \
    const auto expectedValue = expected.function(index);                  \
    if (actualValue != expectedValue) {                                   \
      success = false;                                                    \
    }                                                                     \
  }

/**
 * @brief Helper function for comparing two h_item object by comparing
 *        the values of invoking the same member functions that expect an index
 * @tparam index Current index to call member functions with, required
 * @tparam numDims Number of dimensions of the h_item, deduced
 * @tparam success_acc_t Type of the accessor used for storing success, deduced
 * @param success Reference to boolean values used to signal success
 * @param actual Object to retrieve the actual values from
 * @param expected Object to retrieve the expected values from
 */
template <int index, int numDims, typename success_acc_t>
inline void check_equality_helper(success_acc_t& success,
                                  const sycl::h_item<numDims>& actual,
                                  const sycl::h_item<numDims>& expected) {
  CHECK_EQUALITY_HELPER(index, success, actual, expected, get_global_range);
  CHECK_EQUALITY_HELPER(index, success, actual, expected, get_global_id);
  CHECK_EQUALITY_HELPER(index, success, actual, expected, get_local_range);
  CHECK_EQUALITY_HELPER(index, success, actual, expected, get_local_id);
  CHECK_EQUALITY_HELPER(index, success, actual, expected,
                        get_logical_local_range);
  CHECK_EQUALITY_HELPER(index, success, actual, expected, get_logical_local_id);
  CHECK_EQUALITY_HELPER(index, success, actual, expected,
                        get_physical_local_range);
  CHECK_EQUALITY_HELPER(index, success, actual, expected,
                        get_physical_local_id);
}

#undef CHECK_EQUALITY_HELPER

/**
 * @brief Helper function for comparing two h_item object by comparing
 *        the values of invoking the same member functions that expect an index
 * @tparam numDims Number of dimensions of the h_item
 * @tparam success_acc_t Type of the accessor used for storing success
 * @param successAcc Reference to boolean values used to signal success
 * @param currentCheck Which check is currently being performed
 * @param actual Object to retrieve the actual values from
 * @param expected Object to retrieve the expected values from
 */
template <int numDims, typename success_acc_t>
inline void check_equality(success_acc_t& successAcc,
                           current_check currentCheck,
                           const sycl::h_item<numDims>& actual,
                           const sycl::h_item<numDims>& expected) {
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
}

/** test sycl::device initialization
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
    {
      success_array_t success;
      std::fill(std::begin(success), std::end(success), true);

      {
        auto testQueue = util::get_cts_object::queue();

        const auto simpleRange =
            util::get_cts_object::range<numDims>::get(1, 1, 1);

        sycl::buffer<bool> successBuf(success.data(),
                                          sycl::range<1>(success.size()));

        testQueue.submit([&](sycl::handler& cgh) {
          auto successAcc =
              successBuf.get_access<sycl::access_mode::write>(cgh);

          cgh.parallel_for_work_group<h_item_constructors_kernel<numDims>>(
              simpleRange, simpleRange, [=](sycl::group<numDims> group) {
                group.parallel_for_work_item(
                    simpleRange, [&](sycl::h_item<numDims> item) {
                      // Check copy constructor
                      sycl::h_item<numDims> copied(item);
                      check_equality(successAcc,
                                     current_check::copy_constructor, copied,
                                     item);

                      // Check move constructor
                      sycl::h_item<numDims> moved(std::move(copied));
                      check_equality(successAcc,
                                     current_check::move_constructor, moved,
                                     item);

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

}  // namespace TEST_NAMESPACE
