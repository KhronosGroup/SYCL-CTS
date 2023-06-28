/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
#include "../common/invoke.h"
#include "../common/once_per_unit.h"
#include "../common/semantics_by_value.h"

namespace sub_group_semantics {

enum class op_codes : size_t {
  ctor_copy = 0,
  ctor_move,
  assign_copy,
  assign_move,
  code_count
};

constexpr size_t error_count = to_integral(op_codes::code_count);

static constexpr size_t sizes[] = {16, 32, 64};

static const std::array<std::string, error_count> error_strings{
    "sub_group with sub_group was not constructed correctly",
    "sub_group with sub_group was not move constructed correctly",
    "sub_group with sub_group was not copy assigned correctly",
    "sub_group with sub_group was not move assigned correctly",
};

template <op_codes Code, typename ResultArray>
void set_success_operation(ResultArray& result, bool success) {
  int index = to_integral(Code);
  result[index] = success;
}

std::string get_error_string(int code) { return error_strings[code]; }

struct sub_group_semantics_kernel;
struct sub_group_equality_kernel;
struct setup_kernel;

bool check_equality_by_id(const sycl::sub_group& actual,
                          sycl::id<1>* expected_ids) {
  return actual.get_group_id() == expected_ids[0] &&
         actual.get_local_id() == expected_ids[1];
}

template <typename ResultArray>
void check_by_value_semantics(sycl::sub_group& sub_group, ResultArray& result) {
  sycl::id<1> expected_ids[] = {sub_group.get_group_id(),
                                sub_group.get_local_id()};
  // Check copy constructor
  sycl::sub_group copied(sub_group);
  set_success_operation<op_codes::ctor_copy>(
      result, check_equality_by_id(copied, expected_ids));

  // Check copy assignment
  sycl::sub_group copy_assigned(sub_group);
  copy_assigned = sub_group;
  set_success_operation<op_codes::assign_copy>(
      result, check_equality_by_id(copy_assigned, expected_ids));

  // Check move constructor; invalidates sub_group
  sycl::sub_group moved(std::move(sub_group));
  set_success_operation<op_codes::ctor_move>(
      result, check_equality_by_id(moved, expected_ids));

  // Check move assignment
  sycl::sub_group move_assigned(copy_assigned);
  move_assigned = std::move(copy_assigned);
  set_success_operation<op_codes::assign_move>(
      result, check_equality_by_id(move_assigned, expected_ids));
}

TEST_CASE("sub_group by-value semantics", "[sub_group]") {
  bool result[error_count];
  std::fill(result, result + error_count, false);
  {
    sycl::buffer<bool, 1> res_buf(result, sycl::range(error_count));

    sycl::queue queue = once_per_unit::get_queue();
    const sycl::range<3> r{1, 1, 1};
    sycl::nd_range<3> nd_range(r, r);
    queue
        .submit([&](sycl::handler& cgh) {
          auto res_acc = res_buf.get_access<sycl::access_mode::read_write>(cgh);
          cgh.parallel_for<sub_group_semantics_kernel>(
              nd_range, [=](sycl::nd_item<3> nd_item) {
                sycl::sub_group sub_group = nd_item.get_sub_group();
                check_by_value_semantics(sub_group, res_acc);
              });
        })
        .wait_and_throw();
  }
  for (int i = 0; i < error_count; ++i) {
    INFO(get_error_string(i));
    CHECK(result[i]);
  }
}

TEST_CASE("Check sycl::sub_group equality", "[sub_group]") {
  size_t code_count =
      to_integral(common_by_value_semantics::current_check::size);
  bool result[code_count];
  std::fill(result, result + code_count, false);
  auto items = store_instances<2, invoke_sub_group<3, setup_kernel>>();
  {
    sycl::buffer<bool, 1> res_buf(result, sycl::range(code_count));
    auto queue = once_per_unit::get_queue();
    queue
        .submit([&](sycl::handler& cgh) {
          auto res_acc = res_buf.get_access(cgh);
          cgh.single_task<sub_group_equality_kernel>([=] {
            common_by_value_semantics::check_equality(items[0], items[1],
                                                      res_acc);
          });
        })
        .wait_and_throw();
  }
  for (int i = 0; i < code_count; ++i) {
    INFO(common_by_value_semantics::get_error_string(i));
    CHECK(result[i]);
  }
}

}  // namespace sub_group_semantics
