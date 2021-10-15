/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides tests to check sub_group_mask find_high()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_find_high

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_find_high {
  bool operator()(const sycl::ext::oneapi::sub_group_mask &sub_group_mask,
                  const sycl::sub_group &sub_group) {
    return sub_group_mask.find_high() ==
           sycl::id(sub_group.get_local_range().get(0) / 2 - 1);
  }
};

struct check_result_find_high_no_bits_set {
  bool operator()(const sycl::ext::oneapi::sub_group_mask &sub_group_mask,
                  const sycl::sub_group &sub_group) {
    return sub_group_mask.find_high() ==
           sycl::id(sub_group.get_local_range().get(0));
  }
};

struct check_type_find_high {
  bool operator()(const sycl::ext::oneapi::sub_group_mask &sub_group_mask) {
    return std::is_same<sycl::id<1>,
                        decltype(sub_group_mask.find_high())>::value;
  }
};
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
    log.note("Check find_high() for mask with first half predicate");
    check_const_api<check_result_find_high, check_type_find_high,
                    first_half_predicate>(log);

    log.note("Check find_high() for mask with false predicate");
    check_const_api<check_result_find_high_no_bits_set, check_type_find_high,
                    false_predicate>(log);
#else
    log.note("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined, test is skipped");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
