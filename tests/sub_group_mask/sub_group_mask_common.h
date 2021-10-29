/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides common methods for sub_group_mask tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SUB_GROUP_MASK_COMMON_H
#define __SYCLCTS_TESTS_SUB_GROUP_MASK_COMMON_H

#include "../common/common.h"
#include "../common/type_coverage.h"

#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
namespace {

static const auto types =
    named_type_pack<char, int, float>{"char", "int", "float"};

template <typename funT, typename PredT, typename T>
class test_kernel;

constexpr size_t globalSize = 128;
constexpr size_t localSize = 32;

struct even_predicate {
  bool operator()(const sycl::sub_group &sub_group) {
    return sub_group.get_local_id().get(0) % 2 == 0;
  }
};

struct mod3_predicate {
  bool operator()(const sycl::sub_group &sub_group) {
    return sub_group.get_local_id().get(0) % 3 == 0;
  }
};

struct true_predicate {
  bool operator()(const sycl::sub_group &sub_group) { return true; }
};

struct false_predicate {
  bool operator()(const sycl::sub_group &sub_group) { return false; }
};

struct first_half_predicate {
  bool operator()(const sycl::sub_group &sub_group) {
    return sub_group.get_local_id().get(0) <
           sub_group.get_local_range().get(0) / 2;
  }
};

struct second_half_predicate {
  bool operator()(const sycl::sub_group &sub_group) {
    return sub_group.get_local_id().get(0) >=
           sub_group.get_local_range().get(0) / 2;
  }
};

inline auto get_result_array() {
  std::array<bool, globalSize> resultArr;
  resultArr.fill(true);
  return resultArr;
}

inline bool if_check(const sycl::sub_group &sub_group) {
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return true;
#else
  return sub_group.leader();
#endif
}

template <typename funT, typename funTypeT, typename predT, typename maskT,
          typename T>
void check_mask_api(sycl_cts::util::logger &log) {
  sycl::range<1> globalRange(globalSize);
  sycl::range<1> localRange(localSize);
  sycl::nd_range<1> dataRange(globalRange, localRange);

  auto resultArr = get_result_array();
  auto resultTypeArr = get_result_array();

  auto testQueue = sycl_cts::util::get_cts_object::queue();
  {
    auto buffer = sycl::buffer(resultArr.data(), globalRange);
    auto bufferType = sycl::buffer(resultTypeArr.data(), globalRange);
    testQueue.submit([&](sycl::handler &h) {
      auto resultPtr =
          buffer.template get_access<sycl::access_mode::read_write>(h);
      auto resultTypePtr =
          bufferType.template get_access<sycl::access_mode::read_write>(h);
      h.parallel_for<test_kernel<funT, predT, T>>(
          dataRange, [=](sycl::nd_item<1> item) {
            auto sub_group = item.get_sub_group();
            maskT mask =
                sycl::ext::oneapi::group_ballot(sub_group, predT()(sub_group));
            if (if_check(sub_group))
              resultPtr[item.get_global_id(0)] = funT()(mask, sub_group);
            if (sub_group.leader())
              resultTypePtr[item.get_global_id(0)] = funTypeT()(mask);
          });
    });
  }
  for (int i = 0; i < globalSize; i++)
    if (!resultArr[i])
      FAIL(log, "Check result failed on element " + std::to_string(i));
  if (!resultTypeArr[0]) FAIL(log, "Check type failed");
}

template <typename funT, typename funTypeT, typename predT, typename T = void>
void check_const_api(sycl_cts::util::logger &log) {
  check_mask_api<funT, funTypeT, predT, const sycl::ext::oneapi::sub_group_mask,
                 T>(log);
}

template <typename funT, typename funTypeT, typename predT, typename T = void>
void check_non_const_api(sycl_cts::util::logger &log) {
  check_mask_api<funT, funTypeT, predT, sycl::ext::oneapi::sub_group_mask, T>(
      log);
}

}  // namespace
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

#endif  // __SYCLCTS_TESTS_SUB_GROUP_MASK_COMMON_H
