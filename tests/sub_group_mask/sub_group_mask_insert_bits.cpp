/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests to check sub_group_mask insert_bits()
//
*******************************************************************************/

#include "sub_group_mask_common.h"

#define TEST_NAME sub_group_mask_insert_bits

namespace TEST_NAMESPACE {

using namespace sycl_cts;
#ifdef SYCL_EXT_ONEAPI_SUB_GROUP_MASK

// to get 0b0101.. to insert
template <typename T>
void get_bits(T &out) {
  out = 1;
  for (int i = 2; i + 2 <= sizeof(T) * CHAR_BIT; i = i + 2) {
    out <<= 2;
    out++;
  }
}

template <typename T, size_t nElements>
void get_bits(sycl::marray<T, nElements> &out) {
  T val;
  get_bits(val);
  std::fill(out.begin(), out.end(), val);
}

template <typename T>
struct check_result_insert_bits {
  bool operator()(sycl::ext::oneapi::sub_group_mask &sub_group_mask,
                  const sycl::sub_group &) {
    for (int pos = 0; pos < sub_group_mask.size(); pos++) {
      sycl::ext::oneapi::sub_group_mask mask = sub_group_mask;
      T bits;
      get_bits(bits);
      mask.insert_bits(bits, sycl::id(pos));
      for (int K = 0; K < mask.size(); K++)
        if (K >= pos && K < pos + CHAR_BIT * sizeof(T)) {
          if (mask.test(sycl::id(K)) != ((K - pos) % 2 == 0)) return false;
        } else {
          if (mask.test(sycl::id(K)) != (K % 3 == 0)) return false;
        }
    }
    return true;
  }
};

template <typename T>
struct check_type_insert_bits {
  bool operator()(sycl::ext::oneapi::sub_group_mask &sub_group_mask) {
    return std::is_same<void, decltype(sub_group_mask.insert_bits(T()))>::value;
  }
};

template <typename T>
struct check_for_type {
  void operator()(util::logger &log, const std::string &typeName) {
    log.note("testing: " + type_name_string<T>::get(typeName));
    check_non_const_api<check_result_insert_bits<T>, check_type_insert_bits<T>,
                        mod3_predicate, T>(log);
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
    log.note("SYCL_EXT_ONEAPI_SUB_GROUP_MASK is not defined, test is skipped");
#endif  // SYCL_EXT_ONEAPI_SUB_GROUP_MASK
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
