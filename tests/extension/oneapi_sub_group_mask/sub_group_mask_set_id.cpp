/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides tests to check sub_group_mask set(id)
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_set_id

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_set_id {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &sub_group) {
    for (size_t N = 0; N < sub_group_mask.size(); N++) {
      if (N % 3 == 0)
        sub_group_mask.set(sycl::id(N), true);
      else if (N % 3 == 1)
        sub_group_mask.set(sycl::id(N), false);
    }

    for (size_t N = 0; N < sub_group_mask.size(); N++) {
      switch (N % 3) {
        case 0:
          if (!sub_group_mask.test(sycl::id(N))) return false;
          break;
        case 1:
          if (sub_group_mask.test(sycl::id(N))) return false;
          break;
        case 2:
          if (sub_group_mask.test(sycl::id(N)) != (N % 2 == 0)) return false;
          break;
      }
    }
    return true;
  }
};

struct check_type_set_id {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<void, decltype(sub_group_mask.set(sycl::id()))>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_set_id, check_type_set_id,
                   even_predicate, sycl::ext::oneapi::sub_group_mask>;
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
