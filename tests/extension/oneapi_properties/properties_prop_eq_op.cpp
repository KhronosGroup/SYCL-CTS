/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for equality and inequality operators for properties
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME properties_prop_eq_op

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test sycl::ext::oneapi::experimental::property_value equality operators
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
#if !defined(SYCL_EXT_ONEAPI_PROPERTIES)
    WARN("SYCL_EXT_ONEAPI_PROPERTIES is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    {
      using namespace sycl::ext::oneapi::experimental;
      properties props{device_image_scope};
      constexpr auto prop_value_device_image_scope =
          props.get_property<device_image_scope_key>();

      properties props2{implement_in_csr<true>};
      constexpr auto prop_value_implement_in_csr_true =
          props2.get_property<implement_in_csr_key>();

      properties props3{implement_in_csr<false>};
      constexpr auto prop_value_implement_in_csr_false =
          props3.get_property<implement_in_csr_key>();

      CHECK(prop_value_device_image_scope == prop_value_device_image_scope);
      CHECK_FALSE(prop_value_device_image_scope !=
                  prop_value_device_image_scope);

      CHECK(prop_value_implement_in_csr_true ==
            prop_value_implement_in_csr_true);
      CHECK_FALSE(prop_value_implement_in_csr_true !=
                  prop_value_implement_in_csr_true);

      CHECK_FALSE(prop_value_implement_in_csr_true ==
                  prop_value_implement_in_csr_false);
      CHECK(prop_value_implement_in_csr_true !=
            prop_value_implement_in_csr_false);
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
