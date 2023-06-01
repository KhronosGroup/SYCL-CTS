/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
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
//  Provides multi_ptr deduction guides tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/get_cts_string.h"
#include "multi_ptr_common.h"

namespace multi_ptr_deduction_guides {
using namespace sycl;

template <typename T>
class check_multi_ptr_deduction {
 public:
  void operator()(const std::string& type) {
    type_name = type;

    check_for_space<global>();
    check_for_space<local>();
  }

 private:
  static constexpr access::address_space global =
      access::address_space::global_space;
  static constexpr access::address_space local =
      access::address_space::local_space;

  std::string type_name;

  template <access::address_space space>
  void check_for_space() {
    if constexpr (space == global) {
      check_for_mode<access::mode::read, space>();
      check_for_mode<access::mode::write, space>();
    }
    check_for_mode<access::mode::read_write, space>();
  }

  template <access::mode Mode, access::address_space accessor_space>
  void check_for_mode() {
    check_for_dims<1, Mode, accessor_space>();
    check_for_dims<2, Mode, accessor_space>();
    check_for_dims<3, Mode, accessor_space>();
  }

  template <int dims, access::mode Mode, access::address_space accessor_space>
  void check_for_dims() {
    using ElementType = std::conditional_t<
        Mode == access::mode::read && accessor_space == global, const T, T>;
    using acc_t = std::conditional_t<accessor_space == global,
                                     accessor<T, dims, Mode, target::device>,
                                     local_accessor<T, dims>>;
    bool res = false;
    T data{user_def_types::get_init_value_helper<T>(0)};
    auto r = sycl_cts::util::get_cts_object::range<dims>::get(1, 1, 1);
    {
      sycl::buffer<bool, 1> buf_res(&res, {1});
      sycl::buffer<T, dims> buf_data(&data, r);
      auto queue = once_per_unit::get_queue();
      queue
          .submit([&](sycl::handler& cgh) {
            acc_t _accessor;
            if constexpr (accessor_space == global)
              acc_t(buf_data, cgh);
            else
              acc_t(r, cgh);
            auto acc_res = buf_res.get_access(cgh);
            cgh.single_task([=] {
              auto mptr = multi_ptr(_accessor);
              acc_res[0] = std::is_same_v<decltype(mptr),
                                          multi_ptr<ElementType, accessor_space,
                                                    access::decorated::no>>;
            });
          })
          .wait_and_throw();
    }
    std::string mode_str{sycl_cts::get_cts_string::for_mode<Mode>()};
    std::string space_str{(accessor_space == global) ? "global" : "local"};
    std::string fail_str{"Incorrect deduction with type " + type_name + " in " +
                         std::to_string(dims) + " dimensions and " + mode_str +
                         " mode in " + space_str + " space "};
    INFO(fail_str);
    CHECK(res);
  }
};

// FIXME: re-enable when deduction guide for read is implemented
// Issue link https://github.com/intel/llvm/issues/9692
DISABLED_FOR_TEST_CASE(DPCPP)
("multi_ptr deduction guides", "[test_multi_ptr]")({
  for_all_types<check_multi_ptr_deduction>(deduction::vector_types);
  for_all_types<check_multi_ptr_deduction>(deduction::scalar_types);
});
}  // namespace multi_ptr_deduction_guides
