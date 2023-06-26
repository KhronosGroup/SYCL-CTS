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

#include "../common/common.h"
#include "../common/once_per_unit.h"
#include "../common/semantics_by_value.h"

#include <array>

namespace nd_range_equality {
using namespace sycl_cts;

constexpr size_t error_count =
    to_integral(common_by_value_semantics::current_check::size);

template <int numDims>
std::array<sycl::nd_range<numDims>, 3> get_nd_ranges() {
  // Prepare ranges
  const auto range2 = util::get_cts_object::range<numDims>::get(2, 1, 1);
  const auto range4 = util::get_cts_object::range<numDims>::get(4, 1, 1);
  const auto range8 = util::get_cts_object::range<numDims>::get(8, 1, 1);

  // Prepare an array of nd_range
  // The nd_range objects are designed so that:
  //    0 and 1 have the same global range
  //    1 and 2 have the same local range
  //    0 and 2 are completely different
  return {sycl::nd_range<numDims>(range8, range4),
          sycl::nd_range<numDims>(range8, range2),
          sycl::nd_range<numDims>(range4, range2)};
}

template <int numDims>
void test_equality_on_host() {
  auto nd_ranges = get_nd_ranges<numDims>();
  // Perform comparisons on the stored nd_range objects
  const auto& object0 = nd_ranges[0];
  const auto& object1 = nd_ranges[1];
  const auto& object2 = nd_ranges[2];
  bool result[error_count];

  SECTION(std::string("Checking with same global and dim = ") +
          std::to_string(numDims)) {
    common_by_value_semantics::check_equality(object0, object1, result);
    for (int i = 0; i < error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result[i]);
    }
  }
  SECTION(std::string("Checking with same local and dim = ") +
          std::to_string(numDims)) {
    common_by_value_semantics::check_equality(object1, object2, result);
    for (int i = 0; i < error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result[i]);
    }
  }
  SECTION(std::string("Checking with different global and local and dim = ") +
          std::to_string(numDims)) {
    common_by_value_semantics::check_equality(object0, object2, result);
    for (int i = 0; i < error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result[i]);
    }
  }
}

template <int numDims>
class Kernel;

template <int numDims>
void check_equality_on_device(const sycl::nd_range<numDims>& a,
                              const sycl::nd_range<numDims>& other) {
  bool result[error_count];
  {
    sycl::buffer<bool, 1> res_buf(result, sycl::range(error_count));
    auto queue = once_per_unit::get_queue();
    queue
        .submit([&](sycl::handler& cgh) {
          auto res_acc = res_buf.get_access(cgh);
          cgh.single_task<Kernel<numDims>>([=] {
            common_by_value_semantics::check_equality(a, other, res_acc);
          });
        })
        .wait_and_throw();
  }
  for (int i = 0; i < error_count; ++i) {
    INFO(common_by_value_semantics::get_error_string(i));
    CHECK(result[i]);
  }
}

template <int numDims>
void test_equality_on_device() {
  auto nd_ranges = get_nd_ranges<numDims>();
  // Perform comparisons on the stored nd_range objects
  const auto& object0 = nd_ranges[0];
  const auto& object1 = nd_ranges[1];
  const auto& object2 = nd_ranges[2];
  SECTION(std::string("Checking with same global and dim = ") +
          std::to_string(numDims)) {
    check_equality_on_device<numDims>(object0, object1);
  }
  SECTION(std::string("Checking with same local and dim = ") +
          std::to_string(numDims)) {
    check_equality_on_device<numDims>(object1, object2);
  }
  SECTION(std::string("Checking with different global and local and dim = ") +
          std::to_string(numDims)) {
    check_equality_on_device<numDims>(object0, object2);
  }
}

TEST_CASE("Check sycl::nd_range equality check on host", "[nd_range]") {
  test_equality_on_host<1>();
  test_equality_on_host<2>();
  test_equality_on_host<3>();
}

TEST_CASE("Check sycl::nd_range equality check on device", "[nd_range]") {
  test_equality_on_device<1>();
  test_equality_on_device<2>();
  test_equality_on_device<3>();
}

}  // namespace nd_range_equality
