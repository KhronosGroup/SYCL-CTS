/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides common methods for sub_group_mask tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SUB_GROUP_MASK_COMMON_H
#define __SYCLCTS_TESTS_SUB_GROUP_MASK_COMMON_H

#include <array>

#include "../../common/common.h"
#include "../../common/type_coverage.h"

#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
namespace {

// Error types for sub_group_mask ctors
struct ctor_error {
  static constexpr size_t ctor_wrong =
      0;  // wrong sub_group_mask instance created
  static constexpr size_t copy_ctor_wrong =
      1;  // wrong sub_group_mask instance created with copy ctor
  static constexpr size_t assign_wrong =
      2;  // wrong result of sub_group_mask::operator=
  static constexpr size_t assign_type_wrong =
      3;  // wrong type of sub_group_mask::operator=

  static constexpr size_t error_types_count = 4;

  inline static const std::array<std::string, error_types_count> error_strings{
      std::string{"Error with constructed instance of sub_group_mask"},
      std::string{
          "Error with instance of sub_group_mask constructed with copy ctor"},
      std::string{"Error with result of sub_group_mask::operator="},
      std::string{"Error type of sub_group_mask::operator="}};
};

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE

static const auto types =
    named_type_pack<char, signed char, unsigned char, short, unsigned short,
                    int, unsigned int, long, unsigned long, long long,
                    unsigned long long>::generate(
        "char",           "signed char", "unsigned char",     "short",
        "unsigned short", "int",         "unsigned int",      "long",
        "unsigned long",  "long long",   "unsigned long long");
#else
static const auto types =
    named_type_pack<char, int>::generate("char", "int");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE

template <typename funT, typename PredT, typename T, size_t SGSize>
class test_kernel;

constexpr size_t globalSize = 32;
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

inline auto get_ctors_result_array() {
  std::array<bool, globalSize * ctor_error::error_types_count> resultArr;
  resultArr.fill(true);
  return resultArr;
}

inline bool if_check(const sycl::sub_group &sub_group) {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return true;
#else
  return sub_group.leader();
#endif
}
template <size_t SGSize, typename funT, typename funTypeT, typename predT,
          typename maskT, typename T = void>
struct check_mask_api {
  void operator()() {
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

        h.parallel_for<test_kernel<funT, predT, T, SGSize>>(
            dataRange, [=
        ](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SGSize)]] {
              auto sub_group = item.get_sub_group();
              maskT mask = sycl::ext::oneapi::group_ballot(sub_group,
                                                           predT()(sub_group));
              if (if_check(sub_group))
                resultPtr[item.get_global_id(0)] = funT()(mask, sub_group);
              if (sub_group.leader())
                resultTypePtr[item.get_global_id(0)] = funTypeT()(mask);
            });
      });
    }
    for (size_t i = 0; i < globalSize; i++) {
      {
        INFO("Check result failed on element " + std::to_string(i) +
             " with sub group size: " + std::to_string(SGSize));
        CHECK(true == resultArr[i]);
      }
    }
    {
      INFO("Check type failed");
      CHECK(true == resultTypeArr[0]);
    }
  }
};

using res_array = sycl::marray<bool, ctor_error::error_types_count>;

template <size_t SGSize, typename funT>
struct check_mask_ctors {
  void operator()() {
    sycl::range<1> globalRange(globalSize);
    sycl::range<1> localRange(localSize);
    sycl::nd_range<1> dataRange(globalRange, localRange);

    auto resultArr = get_ctors_result_array();

    auto testQueue = sycl_cts::util::get_cts_object::queue();
    {
      auto buffer = sycl::buffer(resultArr.data(), globalRange);

      testQueue.submit([&](sycl::handler &h) {
        auto resultPtr =
            buffer.template get_access<sycl::access_mode::read_write>(h);

        h.parallel_for(
            dataRange,
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SGSize)]] {
              auto sub_group = item.get_sub_group();
              if (if_check(sub_group)) {
                res_array result = funT()(item);
                for (size_t i = 0; i < ctor_error::error_types_count; ++i) {
                  resultPtr[item.get_global_id(0) + i] = result[i];
                }
              }
            });
      });
    }
    for (size_t i = 0; i < globalSize; i++) {
      for (size_t j = 0; j < ctor_error::error_types_count; ++j) {
        {
          INFO(ctor_error::error_strings[j] + " on element " +
               std::to_string(i) +
               " with sub group size: " + std::to_string(SGSize));
          CHECK(resultArr[i * ctor_error::error_types_count + j]);
        }
      }
    }
  }
};

template <template <size_t> class verification_func>
void check_diff_sub_group_sizes() {
  sycl::device device{sycl_cts::util::get_cts_object::device()};
  const auto available_sg_sizes =
      device.get_info<sycl::info::device::sub_group_sizes>();

  auto sb_begin = available_sg_sizes.begin();
  auto sb_end = available_sg_sizes.end();

  constexpr size_t eight_sg_size = 8;
  if (std::find(sb_begin, sb_end, eight_sg_size) != sb_end) {
    verification_func<eight_sg_size>{}();
  } else {
    WARN(
        "Test for 8 sub group size was skipped due to 8 is unsupported sub "
        "group size");
  }
  constexpr size_t sixteen_sg_size = 16;
  if (std::find(sb_begin, sb_end, sixteen_sg_size) != sb_end) {
    verification_func<sixteen_sg_size>{}();
  } else {
    WARN(
        "Test for 16 sub group size was skipped due to 16 is unsupported sub "
        "group size");
  }
  constexpr size_t thirty_two_sg_size = 32;
  if (std::find(sb_begin, sb_end, thirty_two_sg_size) != sb_end) {
    verification_func<thirty_two_sg_size>{}();
  } else {
    WARN(
        "Test for 32 sub group size was skipped due to 32 is unsupported sub "
        "group size");
  }
}

}  // namespace
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

#endif  // __SYCLCTS_TESTS_SUB_GROUP_MASK_COMMON_H
