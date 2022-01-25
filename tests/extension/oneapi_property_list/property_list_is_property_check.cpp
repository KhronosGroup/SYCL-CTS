/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for sycl::is_property for compile-time properties
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME property_list_is_property_check

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test sycl::is_property for compile-time properties
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
    WARN("SYCL_EXT_ONEAPI_PROPERTY_LIST is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    {
      using namespace sycl::ext::oneapi;

      // is_device_copyable for compile-time-constant properties
      if (!sycl::is_property_v<device_image_scope>)
        FAIL(log, "is_property_v for device_image_scope is not true");
      if (!sycl::is_property_v<host_access>)
        FAIL(log, "is_property_v for device_image_scope is not true");
      if (!sycl::is_property_v<init_mode>)
        FAIL(log, "is_property_v for device_image_scope is not true");
      if (!sycl::is_property_v<implement_in_csr>)
        FAIL(log, "is_property_v for device_image_scope is not true");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
