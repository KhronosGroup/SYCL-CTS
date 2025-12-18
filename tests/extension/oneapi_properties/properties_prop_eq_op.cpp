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
    SKIP("SYCL_EXT_ONEAPI_PROPERTIES is not defined");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    SKIP("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined");
#else
    {
      using namespace sycl::ext::oneapi::experimental;
      properties props{device_image_scope};
      constexpr auto prop_value_device_image_scope =
          props.get_property<device_image_scope_key>();

      properties props2{host_access<host_access_enum::read>};
      constexpr auto prop_value_host_access_read =
          props2.get_property<host_access_key>();

      properties props3{host_access<host_access_enum::write>};
      constexpr auto prop_value_host_access_write =
          props3.get_property<host_access_key>();

      CHECK(prop_value_device_image_scope == prop_value_device_image_scope);
      CHECK_FALSE(prop_value_device_image_scope !=
                  prop_value_device_image_scope);

      CHECK(prop_value_host_access_read == prop_value_host_access_read);
      CHECK_FALSE(prop_value_host_access_read != prop_value_host_access_read);

      CHECK_FALSE(prop_value_host_access_read == prop_value_host_access_write);
      CHECK(prop_value_host_access_read != prop_value_host_access_write);
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
