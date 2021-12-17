/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for equality and inequality operators for properties
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME property_list_prop_eq_op

namespace TEST_NAMESPACE {

using namespace sycl_cts;

template <typename T, typename U>
constexpr void check_equal(util::logger &log, T prop1, U prop2,
                           bool expected_equal) {
  if ((prop1 == prop2) != expected_equal) {
    FAIL(log, "wrong result for equality operator");
  }

  if ((prop1 != prop2) == expected_equal) {
    FAIL(log, "wrong result for inequality operator");
  }
}

/** test sycl::ext::oneapi::property_value equality operators
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
#if !defined(SYCL_EXT_ONEAPI_PROPERTY_LIST)
    log.skip("SYCL_EXT_ONEAPI_PROPERTY_LIST is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    log.skip("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    {
      using namespace sycl::ext::oneapi;
      property_list props{device_image_scope_v};
      constexpr auto prop_value_device_image_scope =
          props.get_property<device_image_scope>();

      property_list props2{implement_in_csr_v<true>};
      constexpr auto prop_value_implement_in_csr_true =
          props2.get_property<implement_in_csr>();

      property_list props3{implement_in_csr_v<false>};
      constexpr auto prop_value_implement_in_csr_false =
          props3.get_property<implement_in_csr>();

      check_equal(log, prop_value_device_image_scope,
                  prop_value_device_image_scope, true);
      check_equal(log, prop_value_implement_in_csr_true,
                  prop_value_implement_in_csr_true, true);
      check_equal(log, prop_value_implement_in_csr_true,
                  prop_value_implement_in_csr_false, false);
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
