/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests to check sub_group_mask extract_bits()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_extract_bits

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

// Since sub_group_mask with even predicate consists of 0101...01
// expected extracted bits are 0101..01 or 1010..10 depending on starting pos.
// Filling variable of type T with 01 or 10 to match size of the mask
// and the rest of starting bits are remaining 0.
template <typename T>
void get_expected_bits(T &out, uint32_t mask_size, int pos) {
  if (pos >= mask_size - 1) return;
  int init;
  if (pos % 2 == 0)
    init = 0b01;
  else
    init = 0b10;
  out = init;
  for (size_t i = 2; i + 2 <= sizeof(T) * CHAR_BIT && i + 2 <= mask_size - pos;
       i = i + 2) {
    out <<= 2;
    out += init;
  }
}

template <typename T>
struct check_result_extract_bits {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask,
                  const sycl::sub_group &) {
    for (size_t pos = 0; pos <= sub_group_mask.size(); pos++) {
      T bits;
      sub_group_mask.extract_bits(bits, sycl::id(pos));
      T expected(0);
      get_expected_bits(expected, sub_group_mask.size(), pos);
      if (bits != expected) return false;
    }
    return true;
  }
};

template <typename T>
struct check_type_extract_bits {
  bool operator()(const sycl::ext::oneapi::sub_group_mask sub_group_mask) {
    T bits;
    return std::is_same_v<void, decltype(sub_group_mask.extract_bits(bits))>;
  }
};

template <typename T>
struct check_for_type {
  template <size_t SGSize>
  using verification_func_for_even_predicate =
      check_mask_api<SGSize, check_result_extract_bits<T>,
                     check_type_extract_bits<T>, even_predicate,
                     const sycl::ext::oneapi::sub_group_mask>;

  void operator()(util::logger &log, const std::string &typeName) {
    log.note("testing: " + type_name_string<T>::get(typeName));
    check_diff_sub_group_sizes<verification_func_for_even_predicate>(log);
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
    for_all_types_and_marrays<check_for_type>(types, log);
#else
    SKIP("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
