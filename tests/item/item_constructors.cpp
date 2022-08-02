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

#define TEST_NAME item_constructors

namespace {

template <int numDims>
struct item_constructors_kernel;

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
                                  const sycl::item<numDims>& actual,
                                  const sycl::item<numDims>& expected) {
  CHECK_EQUALITY_HELPER(success, actual.get_range(index),
                        expected.get_range(index));

  CHECK_EQUALITY_HELPER(success, actual.get_id(index), expected.get_id(index));
  CHECK_EQUALITY_HELPER(success, actual[index], expected[index]);
}

template <int numDims, typename success_acc_t>
inline void check_equality(success_acc_t& successAcc,
                           current_check currentCheck,
                           const sycl::item<numDims>& actual,
                           const sycl::item<numDims>& expected) {
  auto& success = successAcc[static_cast<size_t>(currentCheck)];
  if (actual.get_range() != expected.get_range()) {
    success = false;
  }
  if (numDims >= 1) {
    check_equality_helper<0>(success, actual, expected);
  }
  if (numDims >= 2) {
    check_equality_helper<1>(success, actual, expected);
  }
  if (numDims >= 3) {
    check_equality_helper<2>(success, actual, expected);
  }
  CHECK_EQUALITY_HELPER(success, actual.get_linear_id(),
                        expected.get_linear_id());
}

#undef CHECK_EQUALITY_HELPER

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

          cgh.parallel_for<item_constructors_kernel<numDims>>(
              simpleRange, [=](sycl::item<numDims> item) {
                // Check copy constructor
                sycl::item<numDims> copied(item);
                check_equality(successAcc, current_check::copy_constructor,
                               copied, item);

                // Check move constructor
                sycl::item<numDims> moved(std::move(copied));
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
