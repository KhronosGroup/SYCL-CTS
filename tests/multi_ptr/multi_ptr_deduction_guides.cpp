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
//  Provides multi_ptr deduction guides tests
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "multi_ptr_common.h"

namespace multi_ptr_deduction_guides {
using namespace sycl;

template <typename T>
class check_multi_ptr_deduction {
 public:
  void operator()(const std::string& type) {
    typeName = type;

    check_for_space<global>();
    check_for_space<local>();
  }

 private:
  static constexpr access::address_space global =
      access::address_space::global_space;
  static constexpr access::address_space local =
      access::address_space::local_space;

  std::string typeName;

  template <access::address_space space>
  void check_for_space() {
    check_for_mode<access::mode::read, space>();
    check_for_mode<access::mode::write, space>();
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
    using acc_t = std::conditional_t<accessor_space == global,
                                     accessor<T, dims, Mode, target::device>,
                                     local_accessor<T, dims>>;

    acc_t _accessor;
    multi_ptr mptr(_accessor);

    std::string mode_str{sycl_cts::get_cts_string::for_mode<Mode>()};
    std::string space_str{(accessor_space == global) ? "global" : "local"};
    std::string fail_str{"Incorrect deduction with type " + typeName + " in " +
                         std::to_string(dims) + " dimensions and " + mode_str +
                         " mode in " + space_str + " space "};

    INFO(fail_str);
    CHECK(std::is_same_v<decltype(mptr),
                         multi_ptr<T, accessor_space, access::decorated::no>>);
  }
};

TEST_CASE("multi_ptr deduction guides", "[test_multi_ptr]") {
  for_all_types<check_multi_ptr_deduction>(deduction::vector_types);
  for_all_types<check_multi_ptr_deduction>(deduction::scalar_types);
}
}  // namespace multi_ptr_deduction_guides
