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

namespace nd_range_constructors {
enum class op_codes : size_t {
  ctor_range = 0,
  ctor_copy = 1,
  ctor_move = 2,
  assign_copy = 3,
  assign_move = 4,
  code_count
};

constexpr size_t error_count = to_integral(op_codes::code_count);

static constexpr size_t sizes[] = {16, 32, 64};

static const std::array<std::string, error_count> error_strings{
    "nd_range with range was not constructed correctly",
    "nd_range with nd_range was not constructed correctly",
    "nd_range with nd_range was not move constructed correctly",
    "nd_range with nd_range was not copy assigned correctly",
    "nd_range with nd_range was not move assigned correctly",
};

template <op_codes Code, typename ResultArray>
void set_success_operation(ResultArray& result, bool success) {
  int index = to_integral(Code);
  result[index] &= success;
}

std::string get_error_string(int code) { return error_strings[code]; }

template <int dim>
sycl::id<dim> get_offset() {
  if constexpr (1 == dim) {
    sycl::range<1> range(sizes[0] / 8);
    return sycl::id<1>(range);
  } else if constexpr (2 == dim) {
    sycl::range<2> range(sizes[0] / 8, sizes[1] / 8);
    return sycl::id<2>(range);
  } else if constexpr (3 == dim) {
    sycl::range<3> range(sizes[0] / 8, sizes[1] / 8, sizes[2] / 8);
    return sycl::id<3>(range);
  }
}

template <int dim>
sycl::range<dim> get_local_range() {
  if constexpr (1 == dim) {
    return sycl::range<1>(sizes[0] / 4);
  } else if constexpr (2 == dim) {
    return sycl::range<2>(sizes[0] / 4, sizes[1] / 4);
  } else if constexpr (3 == dim) {
    return sycl::range<3>(sizes[0] / 4, sizes[1] / 4, sizes[2] / 4);
  }
}

template <int dim>
sycl::range<dim> get_global_range() {
  if constexpr (1 == dim) {
    return sycl::range<1>(sizes[0]);
  } else if constexpr (2 == dim) {
    return sycl::range<2>(sizes[0], sizes[1]);
  } else if constexpr (3 == dim) {
    return sycl::range<3>(sizes[0], sizes[1], sizes[2]);
  }
}

template <int dim, bool with_offset>
sycl::nd_range<dim> get_nd_range(sycl::range<dim>& ls, sycl::range<dim>& gs) {
  if constexpr (with_offset) {
    return sycl::nd_range<dim>(gs, ls, get_offset<dim>());
  } else {
    return sycl::nd_range<dim>(gs, ls);
  }
}

template <int dim, bool with_offset, typename ResultArray>
void check_by_value_semantics(ResultArray& result, sycl::range<dim>& ls,
                              sycl::range<dim>& gs) {
  sycl::id<dim> offset = sycl_cts::util::get_cts_object::id<dim>::get(0, 0, 0);
  if constexpr (with_offset) {
    offset = get_offset<dim>();
  }
  sycl::nd_range<dim> nd_range = get_nd_range<dim, with_offset>(ls, gs);
  for (int i = 0; i < dim; i++) {
    set_success_operation<op_codes::ctor_range>(
        result, nd_range.get_global_range()[i] == gs[i]);
    set_success_operation<op_codes::ctor_range>(
        result, nd_range.get_local_range()[i] == ls[i]);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    set_success_operation<op_codes::ctor_range>(
        result, nd_range.get_offset()[i] == offset[i]);
#endif
    set_success_operation<op_codes::ctor_range>(
        result, nd_range.get_group_range()[i] == gs[i] / ls[i]);
  }
  sycl::nd_range<dim> copy(nd_range);
  for (int i = 0; i < dim; i++) {
    set_success_operation<op_codes::ctor_copy>(
        result, copy.get_global_range()[i] == gs[i]);
    set_success_operation<op_codes::ctor_copy>(
        result, copy.get_local_range()[i] == ls[i]);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    set_success_operation<op_codes::ctor_copy>(
        result, copy.get_offset()[i] == offset[i]);
#endif
    set_success_operation<op_codes::ctor_copy>(
        result, copy.get_group_range()[i] == gs[i] / ls[i]);
  }
  sycl::nd_range<dim> copy_assign(gs, ls);
  copy_assign = copy;
  for (int i = 0; i < dim; i++) {
    set_success_operation<op_codes::assign_copy>(
        result, copy_assign.get_global_range()[i] == gs[i]);
    set_success_operation<op_codes::assign_copy>(
        result, copy_assign.get_local_range()[i] == ls[i]);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    set_success_operation<op_codes::assign_copy>(
        result, copy_assign.get_offset()[i] == offset[i]);
#endif
    set_success_operation<op_codes::assign_copy>(
        result, copy_assign.get_group_range()[i] == gs[i] / ls[i]);
  }
  sycl::nd_range<dim> move(std::move(copy_assign));
  for (int i = 0; i < dim; i++) {
    set_success_operation<op_codes::ctor_move>(
        result, move.get_global_range()[i] == gs[i]);
    set_success_operation<op_codes::ctor_move>(
        result, move.get_local_range()[i] == ls[i]);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    set_success_operation<op_codes::ctor_move>(
        result, move.get_offset()[i] == offset[i]);
#endif
    set_success_operation<op_codes::ctor_move>(
        result, move.get_group_range()[i] == gs[i] / ls[i]);
  }
  sycl::nd_range<dim> move_assign(gs, ls);
  ;
  move_assign = std::move(move);
  for (int i = 0; i < dim; i++) {
    set_success_operation<op_codes::assign_move>(
        result, move_assign.get_global_range()[i] == gs[i]);
    set_success_operation<op_codes::assign_move>(
        result, move_assign.get_local_range()[i] == ls[i]);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    set_success_operation<op_codes::assign_move>(
        result, move_assign.get_offset()[i] == offset[i]);
#endif
    set_success_operation<op_codes::assign_move>(
        result, move_assign.get_group_range()[i] == gs[i] / ls[i]);
  }
}

template <int dim, bool with_offset>
void check_on_host() {
  bool result[error_count];
  std::fill(result, result + error_count, true);
  auto local_range = get_local_range<dim>();
  auto global_range = get_global_range<dim>();
  check_by_value_semantics<dim, with_offset>(result, local_range, global_range);
  for (int i = 0; i < error_count; ++i) {
    INFO(get_error_string(i));
    CHECK(result[i]);
  }
}

template <int dim, bool with_offset>
void check_on_device() {
  bool result[error_count];
  std::fill(result, result + error_count, true);
  {
    sycl::buffer<bool, 1> res_buf(result, sycl::range(error_count));
    auto queue = once_per_unit::get_queue();
    queue
        .submit([&](sycl::handler& cgh) {
          auto res_acc = res_buf.get_access(cgh);
          cgh.single_task([=] {
            auto local_range = get_local_range<dim>();
            auto global_range = get_global_range<dim>();
            check_by_value_semantics<dim, with_offset>(res_acc, local_range,
                                                       global_range);
          });
        })
        .wait_and_throw();
  }
  for (int i = 0; i < error_count; ++i) {
    INFO(get_error_string(i));
    CHECK(result[i]);
  }
}

TEST_CASE("sycl::nd_range constructors. copy by value semantics on host",
          "[nd_range]") {
  SECTION("Checking for dim 1 on host without offset") {
    check_on_host<1, false>();
  }
  SECTION("Checking for dim 2 on host without offset") {
    check_on_host<2, false>();
  }
  SECTION("Checking for dim 3 on host without offset") {
    check_on_host<3, false>();
  }
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  SECTION("Checking for dim 1 on host with offset") {
    check_on_host<1, true>();
  }
  SECTION("Checking for dim 2 on host with offset") {
    check_on_host<2, true>();
  }
  SECTION("Checking for dim 3 on host with offset") {
    check_on_host<3, true>();
  }
#endif
}

TEST_CASE(
    "sycl::nd_range constructors. copy by value semantics in kernel function",
    "[nd_range]") {
  SECTION("Checking for dim 1 in kernel function without offset") {
    check_on_device<1, false>();
  }
  SECTION("Checking for dim 2 in kernel function without offset") {
    check_on_device<1, false>();
  }
  SECTION("Checking for dim 3 in kernel function without offset") {
    check_on_device<1, false>();
  }
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  SECTION("Checking for dim 1 in kernel function with offset") {
    check_on_device<1, true>();
  }
  SECTION("Checking for dim 2 in kernel function with offset") {
    check_on_device<2, true>();
  }
  SECTION("Checking for dim 3 in kernel function with offset") {
    check_on_device<3, true>();
  }
#endif
}
}  // namespace nd_range_constructors
