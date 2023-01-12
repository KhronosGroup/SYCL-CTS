/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides tests to check sub_group_mask none()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_none

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_none_false {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    return !sub_group_mask.none();
  }
};

struct check_result_none_true {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    return sub_group_mask.none();
  }
};

struct check_type_none {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<bool, decltype(sub_group_mask.none())>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_none_false, check_type_none,
                   even_predicate, const sycl::ext::oneapi::sub_group_mask>;
template <size_t SGSize>
using verification_func_for_false_predicate =
    check_mask_api<SGSize, check_result_none_true, check_type_none,
                   false_predicate, const sycl::ext::oneapi::sub_group_mask>;
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
    log.note("Check none() for mask with even predicate");
    check_diff_sub_group_sizes<verification_func_for_even_predicate>(log);

    log.note("Check none() for mask with false predicate");
    check_diff_sub_group_sizes<verification_func_for_false_predicate>(log);
#else
    SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
