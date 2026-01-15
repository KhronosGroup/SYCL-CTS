/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for compile-time properties::has_property
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME properties_has_property

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test compile-time properties::has_property
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
      properties prop_list{device_image_scope};

      if (!prop_list.has_property<device_image_scope_key>())
        FAIL(log, "properties should have device_image_scope property");
      if (prop_list.has_property<host_access_key>())
        FAIL(log, "properties shouldn't have host_access property");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
