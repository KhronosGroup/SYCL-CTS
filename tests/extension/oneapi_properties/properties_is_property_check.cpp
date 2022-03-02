/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for sycl::is_property for compile-time properties
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME properties_is_property_check

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
#if !defined(SYCL_EXT_ONEAPI_PROPERTIES)
    WARN("SYCL_EXT_ONEAPI_PROPERTIES is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    {
      using namespace sycl::ext::oneapi::experimental;

      // is_property_key for compile-time-constant properties
      if (!is_property_key<device_image_scope_key>::value)
        FAIL(log,
             "is_property_key for device_image_scope_key is not true_type");
      if (!is_property_key<host_access_key>::value)
        FAIL(log, "is_property_key for host_access_key is not true_type");
      if (!is_property_key<init_mode_key>::value)
        FAIL(log, "is_property_key for init_mode_key is not true_type");
      if (!is_property_key<implement_in_csr_key>::value)
        FAIL(log, "is_property_key for implement_in_csr_key is not true_type");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
