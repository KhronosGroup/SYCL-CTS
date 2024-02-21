/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

#include "../../common/common.h"

namespace oneapi_ext = sycl::ext::oneapi::experimental;

// Helper class for working with non-uniform group of type GroupT. If the
// result is empty the work-item does not participate in the execution.
template <typename GroupT>
struct NonUniformGroupHelper;

template <>
struct NonUniformGroupHelper<oneapi_ext::ballot_group<sycl::sub_group>> {
  static constexpr size_t num_test_cases = 4;

  static bool is_supported(const sycl::device& dev) {
    return dev.has(sycl::aspect::ext_oneapi_ballot_group);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
    return true;
  }

  static oneapi_ext::ballot_group<sycl::sub_group> create(sycl::sub_group sg,
                                                          int test_case) {
    assert(test_case < num_test_cases);
    // Split it so that 1/3rd of the items are in the first "true" group and the
    // rest are in "false" group.
    switch (test_case) {
      case 0:
        return oneapi_ext::get_ballot_group(
            sg, sg.get_local_linear_id() < sg.get_local_range().size() / 3);
      case 1:
        return oneapi_ext::get_ballot_group(sg, sg.get_local_linear_id() & 1);
      case 2:
        return oneapi_ext::get_ballot_group(sg, true);
      case 3:
        return oneapi_ext::get_ballot_group(sg, false);
    }
    return oneapi_ext::get_ballot_group(sg, false);
  }

  static size_t preferred_single_worker_group_id(int test_case) {
    // Some work requires us to pick a single work-group to do work. Generally
    // we would pick group 0, but in case 2 it is empty so we pick 1 for that
    // instead.
    return test_case == 2;
  }

  static std::string get_name() { return "ballot_group<sycl::sub_group>"; }

  static std::string get_test_case_name(int test_case) {
    assert(test_case < num_test_cases);
    switch (test_case) {
      case 0:
        return "predicate is true for first N items.";
      case 1:
        return "predicate is true for all work-items with odd local id";
      case 2:
        return "predicate is true for all work-items";
      case 3:
        return "predicate is false for all work-items";
    }
    return "";
  }
};

template <size_t PartitionSize>
struct NonUniformGroupHelper<
    oneapi_ext::fixed_size_group<PartitionSize, sycl::sub_group>> {
  static constexpr size_t num_test_cases = 1;

  static bool is_supported(const sycl::device& dev) {
    return dev.has(sycl::aspect::ext_oneapi_fixed_size_group);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
    return true;
  }

  static oneapi_ext::fixed_size_group<PartitionSize, sycl::sub_group> create(
      sycl::sub_group sg, int test_case) {
    return oneapi_ext::get_fixed_size_group<PartitionSize>(sg);
  }

  static size_t preferred_single_worker_group_id(int) { return 0; }

  static std::string get_name() {
    return "fixed_size_group<" + std::to_string(PartitionSize) +
           ", sycl::sub_group>";
  }

  static std::string get_test_case_name(int) {
    return "testing fixed_size_group";
  }
};

template <>
struct NonUniformGroupHelper<oneapi_ext::tangle_group<sycl::sub_group>> {
  static constexpr size_t num_test_cases = 3;

  static bool is_supported(const sycl::device& dev) {
    return dev.has(sycl::aspect::ext_oneapi_tangle_group);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
    assert(test_case < num_test_cases);
    switch (test_case) {
      case 0:
        return sg.get_local_linear_id() < sg.get_local_range().size() / 3;
      case 1:
        return sg.get_local_linear_id() & 1;
      case 2:
        return true;
    }
    return false;
  }

  static oneapi_ext::tangle_group<sycl::sub_group> create(sycl::sub_group sg,
                                                          int test_case) {
    return oneapi_ext::get_tangle_group(sg);
  }

  static size_t preferred_single_worker_group_id(int) { return 0; }

  static std::string get_name() { return "tangle_group<sycl::sub_group>"; }

  static std::string get_test_case_name(int test_case) {
    assert(test_case < num_test_cases);
    switch (test_case) {
      case 0:
        return "predicate is true for first N items.";
      case 1:
        return "predicate is true for all work-items with odd local id";
      case 2:
        return "predicate is true for all work-items";
    }
    return "";
  }
};

template <>
struct NonUniformGroupHelper<oneapi_ext::opportunistic_group> {
  static constexpr size_t num_test_cases = 3;

  static bool is_supported(const sycl::device& dev) {
    return dev.has(sycl::aspect::ext_oneapi_opportunistic_group);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
    assert(test_case < num_test_cases);
    switch (test_case) {
      case 0:
        return sg.get_local_linear_id() < sg.get_local_range().size() / 3;
      case 1:
        return sg.get_local_linear_id() & 1;
      case 2:
        return true;
    }
    return false;
  }

  static oneapi_ext::opportunistic_group create(sycl::sub_group, int) {
    return oneapi_ext::this_kernel::get_opportunistic_group();
  }

  static size_t preferred_single_worker_group_id(int) { return 0; }

  static std::string get_name() { return "opportunistic_group"; }

  static std::string get_test_case_name(int test_case) {
    assert(test_case < num_test_cases);
    switch (test_case) {
      case 0:
        return "predicate is true for first N items.";
      case 1:
        return "predicate is true for all work-items with odd local id";
      case 2:
        return "predicate is true for all work-items";
    }
    return "";
  }
};
