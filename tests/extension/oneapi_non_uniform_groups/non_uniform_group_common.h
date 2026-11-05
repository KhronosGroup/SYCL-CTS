/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../common/common.h"

namespace oneapi_ext = sycl::ext::oneapi::experimental;

// A `fragment` can be created either by partitioning a group with
// `binary_partition()` or opportunistically with `get_opportunistic_group()`.
// Both produce the same type, so the group type alone cannot select the
// creation path. These tags key `NonUniformGroupHelper` on the path instead.
struct fragment_binary_partition {};
struct fragment_opportunistic {};

// Group packs to test for.
using GroupPackTypes =
    std::tuple<unnamed_type_pack<fragment_binary_partition>,
               unnamed_type_pack<oneapi_ext::chunk<1, sycl::sub_group>,
                                 oneapi_ext::chunk<2, sycl::sub_group>,
                                 oneapi_ext::chunk<4, sycl::sub_group>,
                                 oneapi_ext::chunk<8, sycl::sub_group>>,
#ifdef SYCL_EXT_ONEAPI_TANGLE
               unnamed_type_pack<oneapi_ext::tangle<sycl::sub_group>>,
#endif
               unnamed_type_pack<fragment_opportunistic>>;

// Helper class for working with the non-uniform group selected by TagT, which
// is either a group type or one of the fragment tags above. If the result is
// empty the work-item does not participate in the execution.
template <typename TagT>
struct NonUniformGroupHelper;

template <>
struct NonUniformGroupHelper<fragment_binary_partition> {
  using group_type = oneapi_ext::fragment<sycl::sub_group>;

  static constexpr size_t num_test_cases = 4;

  static bool is_supported(const sycl::device& dev) {
    return dev.has(sycl::aspect::ext_oneapi_fragment);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
    return true;
  }

  static group_type create(sycl::sub_group sg, int test_case) {
    // Split it so that 1/3rd of the items are in the first "true" group and the
    // rest are in "false" group.
    switch (test_case) {
      case 0:
        return oneapi_ext::binary_partition(
            sg, sg.get_local_linear_id() < sg.get_local_range().size() / 3);
      case 1:
        return oneapi_ext::binary_partition(sg, sg.get_local_linear_id() & 1);
      case 2:
        return oneapi_ext::binary_partition(sg, true);
      case 3:
        return oneapi_ext::binary_partition(sg, false);
    }
    return oneapi_ext::binary_partition(sg, false);
  }

  static size_t preferred_single_worker_group_id(int test_case) {
    // Some work requires us to pick a single work-group to do work. Generally
    // we would pick group 0, but in case 2 it is empty so we pick 1 for that
    // instead.
    return test_case == 2;
  }

  static std::string get_name() {
    return "fragment<sycl::sub_group> from binary_partition";
  }

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

template <size_t ChunkSize>
struct NonUniformGroupHelper<oneapi_ext::chunk<ChunkSize, sycl::sub_group>> {
  using group_type = oneapi_ext::chunk<ChunkSize, sycl::sub_group>;

  // Chunk testing has one case per size we test, but since the size is part of
  // the type it will have to be specified further out.
  static constexpr size_t num_test_cases = 1;

  static bool is_supported(const sycl::device& dev) {
    // For these tests we consider them unsupported if the smallest supported
    // sub-group size is less than the chunk size.
    std::vector<size_t> sg_sizes =
        dev.get_info<sycl::info::device::sub_group_sizes>();
    size_t min_sg_size = *std::min_element(sg_sizes.cbegin(), sg_sizes.cend());
    return min_sg_size >= ChunkSize && dev.has(sycl::aspect::ext_oneapi_chunk);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
    return true;
  }

  static group_type create(sycl::sub_group sg, int test_case) {
    return oneapi_ext::chunked_partition<ChunkSize>(sg);
  }

  static size_t preferred_single_worker_group_id(int) { return 0; }

  static std::string get_name() {
    return "chunk<" + std::to_string(ChunkSize) + ", sycl::sub_group>";
  }

  static std::string get_test_case_name(int) { return "testing chunk"; }
};

#ifdef SYCL_EXT_ONEAPI_TANGLE
template <>
struct NonUniformGroupHelper<oneapi_ext::tangle<sycl::sub_group>> {
  using group_type = oneapi_ext::tangle<sycl::sub_group>;

  static constexpr size_t num_test_cases = 3;

  static bool is_supported(const sycl::device& dev) {
    return dev.has(sycl::aspect::ext_oneapi_tangle);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
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

  static group_type create(sycl::sub_group sg, int test_case) {
    return oneapi_ext::entangle(sg);
  }

  static size_t preferred_single_worker_group_id(int) { return 0; }

  static std::string get_name() { return "tangle<sycl::sub_group>"; }

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
#endif

template <>
struct NonUniformGroupHelper<fragment_opportunistic> {
  using group_type = oneapi_ext::fragment<sycl::sub_group>;

  static constexpr size_t num_test_cases = 3;

  static bool is_supported(const sycl::device& dev) {
    return dev.has(sycl::aspect::ext_oneapi_fragment);
  }

  static bool should_participate(sycl::sub_group sg, int test_case) {
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

  static group_type create(sycl::sub_group, int) {
    return oneapi_ext::this_work_item::get_opportunistic_group();
  }

  static size_t preferred_single_worker_group_id(int) { return 0; }

  static std::string get_name() {
    return "fragment<sycl::sub_group> from get_opportunistic_group";
  }

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
