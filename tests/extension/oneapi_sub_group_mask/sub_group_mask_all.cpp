/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides tests to check sub_group_mask all()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_all

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_all_false {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    return !sub_group_mask.all();
  }
};

struct check_result_all_true {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    return sub_group_mask.all();
  }
};

struct check_type_all {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<bool, decltype(sub_group_mask.all())>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_all_false, check_type_all,
                   even_predicate, const sycl::ext::oneapi::sub_group_mask>;
template <size_t SGSize>
using verification_func_for_true_predicate =
    check_mask_api<SGSize, check_result_all_true, check_type_all,
                   true_predicate, const sycl::ext::oneapi::sub_group_mask>;
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
    log.note("Check all() for mask with even predicate");
    check_diff_sub_group_sizes<verification_func_for_even_predicate>(log);

    log.note("Check all() for mask with true predicate");
    check_diff_sub_group_sizes<verification_func_for_true_predicate>(log);
#else
    SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
