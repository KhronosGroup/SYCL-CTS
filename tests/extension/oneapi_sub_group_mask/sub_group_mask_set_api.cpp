/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides tests to check sub_group_mask set()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_set_api

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_set {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &sub_group) {
    // sub_group_mask's size must be in the range between 0 (excluded) and 32
    // (included) to rule out UB
    if (sub_group_mask.size() > 32 || sub_group_mask.size() == 0) return false;
    unsigned long after_set;
    sub_group_mask.set();
    sub_group_mask.extract_bits(after_set);
    // mask off irrelevant bits
    unsigned long mask =
        ULONG_MAX >> (CHAR_BIT * sizeof(unsigned long) - sub_group_mask.size());
    unsigned long all_set = ULONG_MAX & mask;
    after_set = after_set & mask;
    return after_set == all_set;
  }
};

struct check_type_set {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<void, decltype(sub_group_mask.set())>::value;
  }
};
template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_set, check_type_set, even_predicate,
                   sycl::ext::oneapi::sub_group_mask>;
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
    check_diff_sub_group_sizes<verification_func_for_even_predicate>(log);
#else
    SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
