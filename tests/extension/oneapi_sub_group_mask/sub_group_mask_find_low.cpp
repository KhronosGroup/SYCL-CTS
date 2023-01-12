/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides tests to check sub_group_mask find_low()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_find_low

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_find_low {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &sub_group) {
    return sub_group_mask.find_low() ==
           sycl::id(sub_group.get_local_range().get(0) / 2);
  }
};

struct check_result_find_low_no_bits_set {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &sub_group) {
    return sub_group_mask.find_low() ==
           sycl::id(sub_group.get_local_range().get(0));
  }
};

struct check_type_find_low {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<sycl::id<1>,
                        decltype(sub_group_mask.find_low())>::value;
  }
};

template <size_t SGSize>
using verification_func_for_second_half_predicate =
    check_mask_api<SGSize, check_result_find_low, check_type_find_low,
                   second_half_predicate,
                   const sycl::ext::oneapi::sub_group_mask>;
template <size_t SGSize>
using verification_func_for_false_predicate =
    check_mask_api<SGSize, check_result_find_low_no_bits_set,
                   check_type_find_low, false_predicate,
                   const sycl::ext::oneapi::sub_group_mask>;
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK

/** test sycl::oneapi::sub_group_mask interface
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK
    log.note("Check find_low() for mask with second half predicate");
    check_diff_sub_group_sizes<verification_func_for_second_half_predicate>(
        log);

    log.note("Check find_low() for mask with false predicate");
    check_diff_sub_group_sizes<verification_func_for_false_predicate>(log);
#else
    SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
