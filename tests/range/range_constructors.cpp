/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
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

namespace range_constructors {

template <int dim>
class kernel;

enum class semantic_by_value_codes : size_t {
  ctor_size = 0,
  ctor_copy = 1,
  ctor_move = 2,
  assign_move = 3,
  code_count
};

constexpr size_t error_count = to_integral(semantic_by_value_codes::code_count);
constexpr size_t equality_error_count =
    to_integral(common_by_value_semantics::current_check::SIZE);

static constexpr size_t sizes[] = {16, 8, 4};

static const std::array<std::string, error_count> error_strings{
    "range with size_t was not constructed correctly",
    "range with range was not constructed correctly",
    "range with range was not move constructed correctly",
    "range with range was not move assigned correctly",
};

template <semantic_by_value_codes Code, typename ResultArray>
void set_success_operation(ResultArray& result) {
  int index = to_integral(Code);
  result[index] = true;
}

std::string get_error_string(int code) { return error_strings[code]; }

template <int dim>
struct tag {};

template <typename ResultArray>
void check_equality(ResultArray& result, tag<1>) {
  constexpr int dim = 1;
  sycl::range<dim> range_explicit(sizes[0]);
  sycl::range<dim> range_other(sizes[0] * 2);
  common_by_value_semantics::check_equality(range_explicit, range_other,
                                            result);
}

template <typename ResultArray>
void check_equality(ResultArray& result, tag<2>) {
  constexpr int dim = 2;
  sycl::range<dim> range_explicit(sizes[0], sizes[1]);
  sycl::range<dim> range_other(sizes[0] * 2, sizes[1] * 2);
  common_by_value_semantics::check_equality(range_explicit, range_other,
                                            result);
}

template <typename ResultArray>
void check_equality(ResultArray& result, tag<3>) {
  constexpr int dim = 3;
  sycl::range<dim> range_explicit(sizes[0], sizes[1], sizes[2]);
  sycl::range<dim> range_other(sizes[0] * 2, sizes[1] * 2, sizes[2] * 2);
  common_by_value_semantics::check_equality(range_explicit, range_other,
                                            result);
}

template <int dim, typename ResultArray>
void check_equality(ResultArray& result) {
  check_equality(result, tag<dim>{});
}

template <typename ResultArray>
void check_by_value_semantics(ResultArray& result, tag<1>) {
  constexpr int dim = 1;
  sycl::range<dim> range_explicit(sizes[0]);
  if ((range_explicit[0] == sizes[0]) && (range_explicit.get(0) == sizes[0])) {
    set_success_operation<semantic_by_value_codes::ctor_size>(result);
  }

  sycl::range<dim> range_deep(
      const_cast<const sycl::range<dim>&>(range_explicit));
  if ((range_deep[0] == sizes[0]) && (range_deep.get(0) == sizes[0])) {
    set_success_operation<semantic_by_value_codes::ctor_copy>(result);
  }

  sycl::range<dim> range_moved_constr(std::move(range_explicit));
  if ((range_moved_constr[0] == sizes[0]) &&
      (range_moved_constr.get(0) == sizes[0])) {
    set_success_operation<semantic_by_value_codes::ctor_move>(result);
  }

  sycl::range<dim> range_move_assign{0};
  range_move_assign = std::move(range_deep);
  if ((range_move_assign[0] == sizes[0]) &&
      (range_move_assign.get(0) == sizes[0])) {
    result[to_integral(semantic_by_value_codes::assign_move) * dim] = true;
    set_success_operation<semantic_by_value_codes::assign_move>(result);
  }
}

template <typename ResultArray>
void check_by_value_semantics(ResultArray& result, tag<2>) {
  constexpr int dim = 2;
  sycl::range<dim> range_explicit(sizes[0], sizes[1]);
  if ((range_explicit[0] == sizes[0]) && (range_explicit.get(0) == sizes[0]) &&
      (range_explicit[1] == sizes[1]) && (range_explicit.get(1) == sizes[1])) {
    set_success_operation<semantic_by_value_codes::ctor_size>(result);
  }

  sycl::range<dim> range_deep(
      const_cast<const sycl::range<dim>&>(range_explicit));
  if ((range_deep[0] == sizes[0]) && (range_deep.get(0) == sizes[0]) &&
      (range_deep[1] == sizes[1]) && (range_deep.get(1) == sizes[1])) {
    set_success_operation<semantic_by_value_codes::ctor_copy>(result);
  }

  sycl::range<dim> range_moved_constr(std::move(range_explicit));
  if ((range_moved_constr[0] == sizes[0]) &&
      (range_moved_constr.get(0) == sizes[0]) &&
      (range_moved_constr[1] == sizes[1]) &&
      (range_moved_constr.get(1) == sizes[1])) {
    set_success_operation<semantic_by_value_codes::ctor_move>(result);
  }

  sycl::range<dim> range_move_assign{0, 0};
  range_move_assign = std::move(range_deep);
  if ((range_move_assign[0] == sizes[0]) &&
      (range_move_assign.get(0) == sizes[0]) &&
      (range_move_assign[1] == sizes[1]) &&
      (range_move_assign.get(1) == sizes[1])) {
    set_success_operation<semantic_by_value_codes::assign_move>(result);
  }
}

template <typename ResultArray>
void check_by_value_semantics(ResultArray& result, tag<3>) {
  constexpr int dim = 3;
  sycl::range<dim> range_explicit(sizes[0], sizes[1], sizes[2]);
  if ((range_explicit[0] == sizes[0]) && (range_explicit.get(0) == sizes[0]) &&
      (range_explicit[1] == sizes[1]) && (range_explicit.get(1) == sizes[1]) &&
      (range_explicit[2] == sizes[2]) && (range_explicit.get(2) == sizes[2])) {
    set_success_operation<semantic_by_value_codes::ctor_size>(result);
  }

  sycl::range<dim> range_deep(
      const_cast<const sycl::range<dim>&>(range_explicit));
  if ((range_deep[0] == sizes[0]) && (range_deep.get(0) == sizes[0]) &&
      (range_deep[1] == sizes[1]) && (range_deep.get(1) == sizes[1]) &&
      (range_deep[2] == sizes[2]) && (range_deep.get(2) == sizes[2])) {
    set_success_operation<semantic_by_value_codes::ctor_copy>(result);
  }

  sycl::range<dim> range_moved_constr(std::move(range_explicit));
  if ((range_moved_constr[0] == sizes[0]) &&
      (range_moved_constr.get(0) == sizes[0]) &&
      (range_moved_constr[1] == sizes[1]) &&
      (range_moved_constr.get(1) == sizes[1]) &&
      (range_moved_constr[2] == sizes[2]) &&
      (range_moved_constr.get(2) == sizes[2])) {
    set_success_operation<semantic_by_value_codes::ctor_move>(result);
  }

  sycl::range<3> range_move_assign{0, 0, 0};
  range_move_assign = std::move(range_deep);
  if ((range_move_assign[0] == sizes[0]) &&
      (range_move_assign.get(0) == sizes[0]) &&
      (range_move_assign[1] == sizes[1]) &&
      (range_move_assign.get(1) == sizes[1]) &&
      (range_move_assign[2] == sizes[2]) &&
      (range_move_assign.get(2) == sizes[2])) {
    set_success_operation<semantic_by_value_codes::assign_move>(result);
  }
}

template <int dim, typename ResultArray>
void check_by_value_semantics(ResultArray& result) {
  check_by_value_semantics(result, tag<dim>{});
}

TEST_CASE(
    "sycl::range constructors. copy by value semantics in kernel function",
    "[range]") {
  auto queue = once_per_unit::get_queue();
  bool result[error_count];
  bool result_equalty[equality_error_count];
  SECTION("Checking for dim 1 in kernel function") {
    {
      sycl::buffer<bool, 1> res_buf(result, sycl::range(error_count));
      sycl::buffer<bool, 1> res_equality_buf(result_equalty,
                                             sycl::range(equality_error_count));
      queue.submit([&](sycl::handler& cgh) {
        sycl::accessor res_acc(res_buf, cgh);
        sycl::accessor res_equality_acc(res_equality_buf, cgh);
        cgh.single_task<kernel<1>>([=] {
          check_by_value_semantics<1>(res_acc);
          check_equality<1>(res_equality_acc);
        });
      });
    }
    for (int i = 0; i < error_count; ++i) {
      INFO(get_error_string(i));
      CHECK(result[i]);
    }
    for (int i = 0; i < equality_error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result_equalty[i]);
    }
  }
  SECTION("Checking for dim 2 in kernel function") {
    {
      sycl::buffer<bool, 1> res_buf(result, sycl::range(error_count));
      sycl::buffer<bool, 1> res_equality_buf(result_equalty,
                                             sycl::range(equality_error_count));
      queue.submit([&](sycl::handler& cgh) {
        sycl::accessor res_acc(res_buf, cgh);
        sycl::accessor res_equality_acc(res_equality_buf, cgh);
        cgh.single_task<kernel<2>>([=] {
          check_by_value_semantics<2>(res_acc);
          check_equality<2>(res_equality_acc);
        });
      });
    }
    for (int i = 0; i < error_count; ++i) {
      INFO(get_error_string(i));
      CHECK(result[i]);
    }
    for (int i = 0; i < equality_error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result_equalty[i]);
    }
  }
  SECTION("Checking for dim 3 in kernel function") {
    {
      sycl::buffer<bool, 1> res_buf(result, sycl::range(error_count));
      sycl::buffer<bool, 1> res_equality_buf(result_equalty,
                                             sycl::range(equality_error_count));
      queue.submit([&](sycl::handler& cgh) {
        sycl::accessor res_acc(res_buf, cgh);
        sycl::accessor res_equality_acc(res_equality_buf, cgh);
        cgh.single_task<kernel<3>>([=] {
          check_by_value_semantics<3>(res_acc);
          check_equality<3>(res_equality_acc);
        });
      });
    }
    for (int i = 0; i < error_count; ++i) {
      INFO(get_error_string(i));
      CHECK(result[i]);
    }
    for (int i = 0; i < equality_error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result_equalty[i]);
    }
  }
}

TEST_CASE("sycl::range constructors. copy by value semantics on host",
          "[range]") {
  bool result[error_count];
  bool result_equalty[equality_error_count];
  SECTION("Checking for dim 1 on host") {
    check_by_value_semantics<1>(result);
    for (int i = 0; i < error_count; ++i) {
      INFO(get_error_string(i));
      CHECK(result[i]);
    }
    check_equality<1>(result_equalty);
    for (int i = 0; i < equality_error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result_equalty[i]);
    }
  }
  SECTION("Checking for dim 2 on host") {
    check_by_value_semantics<2>(result);
    for (int i = 0; i < error_count; ++i) {
      INFO(get_error_string(i));
      CHECK(result[i]);
    }
    check_equality<2>(result_equalty);
    for (int i = 0; i < equality_error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result_equalty[i]);
    }
  }
  SECTION("Checking for dim 3 on host") {
    check_by_value_semantics<3>(result);
    for (int i = 0; i < error_count; ++i) {
      INFO(get_error_string(i));
      CHECK(result[i]);
    }
    check_equality<3>(result_equalty);
    for (int i = 0; i < equality_error_count; ++i) {
      INFO(common_by_value_semantics::get_error_string(i));
      CHECK(result_equalty[i]);
    }
  }
}

}  // namespace range_constructors
