/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#include "../../util/array.h"
#include "../common/common.h"
#include "../common/common_semantics.h"
#include "../common/invoke.h"

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

enum class current_check : size_t {
  copy_constructor,
  move_constructor,
  copy_assignment,
  move_assignment,
  SIZE  // This should be last
};

using success_array_t = std::array<bool, to_integral(current_check::SIZE)>;

/**
 * @brief Stores all fields of a sycl::nd_item instance without copying the
 *        instance. */
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
  state_storage(const sycl::nd_item<numDims>& state) {
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
      // ensure deprecated feature does not get compiled if not enabled
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
      m_offset[dim] = state.get_offset().get(dim);
#endif
    }
  }

  /**
   * @brief Returns \p true if \p state is equal to the saved state.
   *        Does not take deprecated features into account. */
  bool check_equality(const sycl::nd_item<numDims>& state) {
    bool equal = true;
    equal &= m_globalLinearId == state.get_global_linear_id();
    equal &= m_groupLinearId == state.get_group_linear_id();
    equal &= m_localLinearId == state.get_local_linear_id();
    for (size_t dim = 0; dim < numDims; ++dim) {
      equal &= m_globalId[dim] == state.get_global_id(dim);
      equal &= m_groupId[dim] == state.get_group(dim);
      equal &= m_localId[dim] == state.get_local_id(dim);
      equal &= m_globalRange[dim] == state.get_global_range(dim);
      equal &= m_groupRange[dim] == state.get_group_range(dim);
      equal &= m_localRange[dim] == state.get_local_range(dim);
    }
    return equal;
  }

  /**
   * @brief Returns \p true if \p state is equal to the saved state.
   *        Takes deprecated features into account. */
  bool check_equality_deprecated(const sycl::nd_item<numDims>& state) {
    bool equal = true;
    for (size_t dim = 0; dim < numDims; ++dim) {
// ensure deprecated feature does not get compiled if not enabled
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
      equal &= m_offset[dim] == state.get_offset().get(dim);
#endif
    }
    return equal;
  }
};

template <int numDims>
void test_constructors(util::logger& log) {
  // contains the device test results for non-deprecated features
  success_array_t success;
  std::fill(std::begin(success), std::end(success), true);
  // contains the device test results for deprecated features
  success_array_t success_deprecated;
  std::fill(std::begin(success_deprecated), std::end(success_deprecated), true);

  {
    const auto simpleRange = util::get_cts_object::range<numDims>::get(5, 6, 7);
    sycl::buffer<bool> successBuf(success.data(),
                                  sycl::range<1>(success.size()));
    sycl::buffer<bool> successDeprecatedBuf(
        success_deprecated.data(), sycl::range<1>(success_deprecated.size()));

    auto testQueue = util::get_cts_object::queue();
    testQueue.submit([&](sycl::handler& cgh) {
      auto successAcc = successBuf.get_access<sycl::access_mode::write>(cgh);
      auto successDeprecatedAcc =
          successDeprecatedBuf.get_access<sycl::access_mode::write>(cgh);

      cgh.parallel_for<nd_item_constructors_kernel<numDims>>(
          sycl::nd_range<numDims>(simpleRange, simpleRange),
          [=](sycl::nd_item<numDims> item) {
            const auto& itemReadOnly = item;
            state_storage<numDims> expected(itemReadOnly);

            {  // Check copy constructor
              sycl::nd_item<numDims> copied(itemReadOnly);
              size_t idx = to_integral(current_check::copy_constructor);
              successAcc[idx] = expected.check_equality(copied);
              successDeprecatedAcc[idx] =
                  expected.check_equality_deprecated(copied);
            }
            {  // Check copy assignment
              auto copied = itemReadOnly;
              size_t idx = to_integral(current_check::copy_assignment);
              successAcc[idx] = expected.check_equality(copied);
              successDeprecatedAcc[idx] =
                  expected.check_equality_deprecated(copied);
            }
            {  // Check move constructor; invalidates item
              sycl::nd_item<numDims> moved(item);
              size_t idx = to_integral(current_check::move_constructor);
              successAcc[idx] = expected.check_equality(moved);
              successDeprecatedAcc[idx] =
                  expected.check_equality_deprecated(moved);
            }
          });
    });
    // submit function again as previous move construction may invalidate item
    testQueue.submit([&](sycl::handler& cgh) {
      auto successAcc = successBuf.get_access<sycl::access_mode::write>(cgh);
      auto successDeprecatedAcc =
          successDeprecatedBuf.get_access<sycl::access_mode::write>(cgh);

      cgh.parallel_for<nd_item_move_assignment_kernel<numDims>>(
          sycl::nd_range<numDims>(simpleRange, simpleRange),
          [=](sycl::nd_item<numDims> item) {
            state_storage<numDims> expected(item);

            // Check move assignment; invalidates item
            auto moved = std::move(item);
            size_t idx = to_integral(current_check::move_assignment);
            successAcc[idx] = expected.check_equality(moved);
            successDeprecatedAcc[idx] =
                expected.check_equality_deprecated(moved);
          });
    });
    testQueue.wait_and_throw();
  }

  // Continue host-side checks only if copy assignment works as expected
  REQUIRE(success[to_integral(current_check::copy_constructor)]);
  REQUIRE(success[to_integral(current_check::copy_assignment)]);
  CHECK(success[to_integral(current_check::move_constructor)]);
  CHECK(success[to_integral(current_check::move_assignment)]);
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  REQUIRE(success_deprecated[to_integral(current_check::copy_constructor)]);
  REQUIRE(success_deprecated[to_integral(current_check::copy_assignment)]);
  CHECK(success_deprecated[to_integral(current_check::move_constructor)]);
  CHECK(success_deprecated[to_integral(current_check::move_assignment)]);
#endif

  // nd_item is not default constructible, store two objects into the array
  static constexpr size_t numItems = 2;
  using setup_kernel_t = nd_item_setup_kernel<numDims>;
  auto items =
      store_instances<numItems, invoke_nd_item<numDims, setup_kernel_t>>();
  {
    const auto& item = items[0];
    const auto& itemReadOnly = item;
    state_storage<numDims> expected(itemReadOnly);

    {  // Check copy constructor
      sycl::nd_item<numDims> copied(itemReadOnly);
      CHECK(expected.check_equality(copied));
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
      CHECK(expected.check_equality_deprecated(copied));
#endif
    }
    {  // Check copy assignment
      auto copied = itemReadOnly;
      CHECK(expected.check_equality(copied));
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
      CHECK(expected.check_equality_deprecated(copied));
#endif
    }
    {  // Check move constructor; invalidates item
      sycl::nd_item<numDims> moved(item);
      CHECK(expected.check_equality(moved));
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
      CHECK(expected.check_equality_deprecated(moved));
#endif
    }
  }
  {
    const auto& item = items[1];
    state_storage<numDims> expected(item);

    // Check move assignment; invalidates item
    auto moved = std::move(item);
    CHECK(expected.check_equality(moved));
#ifdef SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    CHECK(expected.check_equality_deprecated(moved));
#endif
  }
}

class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const final {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
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
