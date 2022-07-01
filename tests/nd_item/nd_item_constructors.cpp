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
#include "../common/common_semantics.h"
#include "../common/invoke.h"
#include "../../util/array.h"

#include <array>

#define TEST_NAME nd_item_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int numDims>
struct nd_item_setup_kernel;

template <int numDims>
struct nd_item_constructors_kernel;

template <int numDims>
struct nd_item_move_assignment_kernel;

enum class current_check: size_t {
  copy_constructor,
  move_constructor,
  copy_assignment,
  move_assignment,
  SIZE  // This should be last
};

using success_array_t =
    std::array<bool, to_integral(current_check::SIZE)>;

#define CHECK_EQUALITY_HELPER(success, actualValue, expectedValue) \
  {                                                                \
    if (actualValue != expectedValue) {                            \
      success = false;                                             \
    }                                                              \
  }

template <int numDims>
class state_storage {
private:
  size_t m_globalLinearId;
  size_t m_groupLinearId;
  size_t m_localLinearId;
  sycl_cts::util::array<size_t, numDims> m_globalId;
  sycl_cts::util::array<size_t, numDims> m_groupId;
  sycl_cts::util::array<size_t, numDims> m_localId;
  sycl_cts::util::array<size_t, numDims> m_globalRange;
  sycl_cts::util::array<size_t, numDims> m_groupRange;
  sycl_cts::util::array<size_t, numDims> m_localRange;
  sycl_cts::util::array<size_t, numDims> m_offset;
public:
  state_storage(const sycl::nd_item<numDims>& state)
  {
    m_globalLinearId = state.get_global_linear_id();
    m_groupLinearId = state.get_group_linear_id();
    m_localLinearId = state.get_local_linear_id();
    for (size_t dim = 0; dim < numDims; ++dim) {
      m_globalId[dim] = state.get_global_id(dim);
      m_groupId[dim] = state.get_group(dim);
      m_localId[dim] = state.get_local_id(dim);
      m_globalRange[dim] = state.get_global_range(dim);
      m_groupRange[dim] = state.get_group_range(dim);
      m_localRange[dim] = state.get_local_range(dim);
      m_offset[dim] = state.get_offset().get(dim);
    }
  }

  size_t get_global_linear_id() const {
    return m_globalLinearId;
  }

  size_t get_group_linear_id() const {
    return m_groupLinearId;
  }

  size_t get_local_linear_id() const {
    return m_localLinearId;
  }

  size_t get_global_id(int dim) const {
    return m_globalId[dim];
  }

  size_t get_group(int dim) const {
    return m_groupId[dim];
  }

  size_t get_local_id(int dim) const {
    return m_localId[dim];
  }

  size_t get_global_range(int dim) const {
    return m_globalRange[dim];
  }

  size_t get_group_range(int dim) const {
    return m_groupRange[dim];
  }

  size_t get_local_range(int dim) const {
    return m_localRange[dim];
  }

  size_t get_offset(int dim) const {
    return m_offset[dim];
  }
};

template <int index, int numDims, typename success_acc_t>
inline void check_equality_helper(success_acc_t& success,
                                  const sycl::nd_item<numDims>& actual,
                                  const state_storage<numDims>& expected) {
  CHECK_EQUALITY_HELPER(success, actual.get_global_id(index),
                        expected.get_global_id(index));
  CHECK_EQUALITY_HELPER(success, actual.get_local_id(index),
                        expected.get_local_id(index));
  CHECK_EQUALITY_HELPER(success, actual.get_group(index),
                        expected.get_group(index));
  CHECK_EQUALITY_HELPER(success, actual.get_global_range(index),
                        expected.get_global_range(index));
  CHECK_EQUALITY_HELPER(success, actual.get_group_range(index),
                        expected.get_group_range(index));
  CHECK_EQUALITY_HELPER(success, actual.get_local_range(index),
                        expected.get_local_range(index));
  // TODO: mark this check as testing deprecated functionality
  CHECK_EQUALITY_HELPER(success, actual.get_offset().get(index),
                        expected.get_offset(index));
}

template <int numDims, typename success_acc_t>
inline void check_equality(success_acc_t& successAcc,
                           current_check currentCheck,
                           const sycl::nd_item<numDims>& actual,
                           const state_storage<numDims>& expected) {
  auto& success = successAcc[to_integral(currentCheck)];
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
        const auto simpleRange =
            util::get_cts_object::range<numDims>::get(1, 1, 1);
        sycl::buffer<bool> successBuf(success.data(),
                                          sycl::range<1>(success.size()));

        auto testQueue = util::get_cts_object::queue();
        testQueue.submit([&](sycl::handler& cgh) {
          auto successAcc =
              successBuf.get_access<sycl::access_mode::write>(cgh);

          cgh.parallel_for<nd_item_constructors_kernel<numDims>>(
              sycl::nd_range<numDims>(simpleRange, simpleRange),
              [=](sycl::nd_item<numDims> item) {
                const auto& itemReadOnly = item;
                state_storage<numDims> expected(itemReadOnly);

                // Check copy constructor
                {
                  sycl::nd_item<numDims> copied(itemReadOnly);
                  check_equality(successAcc, current_check::copy_constructor,
                                 copied, expected);
                }
                // Check copy assignment
                {
                  auto copied = itemReadOnly;
                  check_equality(successAcc, current_check::copy_assignment,
                                 copied, expected);
                }
                // Check move constructor; invalidates item
                {
                  sycl::nd_item<numDims> moved(item);
                  check_equality(successAcc, current_check::move_constructor,
                                 moved, expected);
                }
              });
        });
        testQueue.submit([&](sycl::handler& cgh) {
          auto successAcc =
              successBuf.get_access<sycl::access_mode::write>(cgh);

          cgh.parallel_for<nd_item_move_assignment_kernel<numDims>>(
              sycl::nd_range<numDims>(simpleRange, simpleRange),
              [=](sycl::nd_item<numDims> item) {
                state_storage<numDims> expected(item);

                // Check move assignment; invalidates item
                auto moved = std::move(item);
                check_equality(successAcc, current_check::move_assignment,
                               moved, expected);
              });
        });
        testQueue.wait_and_throw();
      }

      // Check on the host side only if copy assignment works as expected
      if (success[to_integral(current_check::copy_assignment)]) {
        // nd_item is not default constructible, store two objects into the array
        static constexpr size_t numItems = 2;
        using setup_kernel_t = nd_item_setup_kernel<numDims>;
        auto items =
            store_instances<numItems, invoke_nd_item<numDims, setup_kernel_t>>();
        {
          const auto& item = items[0];
          const auto& itemReadOnly = item;
          state_storage<numDims> expected(itemReadOnly);

          // Check copy constructor
          {
            sycl::nd_item<numDims> copied(itemReadOnly);
            check_equality(success, current_check::copy_constructor,
                           copied, expected);
          }
          // Check copy assignment
          {
            auto copied = itemReadOnly;
            check_equality(success, current_check::copy_assignment,
                           copied, expected);
          }
          // Check move constructor; invalidates item
          {
            sycl::nd_item<numDims> moved(item);
            check_equality(success, current_check::move_constructor,
                           moved, expected);
          }
        }
        {
          const auto& item = items[1];
          state_storage<numDims> expected(item);

          // Check move assignment; invalidates item
          auto moved = std::move(item);
          check_equality(success, current_check::move_assignment,
                         moved, expected);
        }
      }

      CHECK_VALUE(log,
                  success[to_integral(current_check::copy_constructor)],
                  true, numDims);
      CHECK_VALUE(log,
                  success[to_integral(current_check::move_constructor)],
                  true, numDims);
      CHECK_VALUE(log,
                  success[to_integral(current_check::copy_assignment)],
                  true, numDims);
      CHECK_VALUE(log,
                  success[to_integral(current_check::move_assignment)],
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
