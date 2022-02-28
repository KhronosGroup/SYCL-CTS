/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides tests to check sub_group_mask operator[] const and reference
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_reference

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

struct check_result_reference {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    for (size_t N = 0; N < sub_group_mask.size(); N++) {
      sycl::ext::oneapi::sub_group_mask::reference ref_to_bit =
          sub_group_mask[sycl::id(N)];
      // check that reference to bit have correct value
      if (ref_to_bit != (N % 2 == 0)) return false;
      switch (N % 5) {
        case 0:
          // check reference operator=(bool x)
          // by assigning opposite value and checking corresponding bit in mask
          ref_to_bit = (N % 2 != 0);
          if (sub_group_mask[sycl::id(N)] != (N % 2 != 0)) return false;
          break;
        case 1:
          // check reference operator=(const reference& x)
          // by assigning reference for next bit and checking corresponding bit in mask
          if (N == sub_group_mask.size() - 1) break;
          ref_to_bit = sub_group_mask[sycl::id(N + 1)];
          if (sub_group_mask[sycl::id(N)] != ((N + 1) % 2 == 0)) return false;
          break;
        case 2:
          // check reference operator~()
          if (~ref_to_bit != (N % 2 != 0)) return false;
          break;
        case 3:
          // check reference operator bool()
          if (!!ref_to_bit != (N % 2 == 0)) return false;
          break;
        case 4:
          // check reference member function flip()
          if (!std::is_same<sycl::ext::oneapi::sub_group_mask::reference &,
                            decltype(ref_to_bit.flip())>::value)
            return false;
          if (ref_to_bit.flip() != (N % 2 != 0)) return false;
          break;
      }
    }
    return true;
  }
};

struct check_type_reference {
  bool operator()(sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    return std::is_same<sycl::ext::oneapi::sub_group_mask::reference,
                        decltype(sub_group_mask[sycl::id()])>::value;
  }
};

template <size_t SGSize>
using verification_func_for_even_predicate =
    check_mask_api<SGSize, check_result_reference, check_type_reference,
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
    log.note("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined, test is skipped");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
