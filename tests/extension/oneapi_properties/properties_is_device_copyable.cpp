/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for sycl::is_device_copyable for compile-time properties
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME properties_is_device_copyable

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test sycl::is_device_copyable for compile-time properties
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

      // is_device_copyable for compile-time-constant properties
      if (!sycl::is_device_copyable_v<decltype(device_image_scope)>)
        FAIL(log, "is_device_copyable_v for device_image_scope is not true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access<host_access_enum::read>)>)
        FAIL(log,
             "is_device_copyable_v for host_access<host_access_enum::read> is "
             "not true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access<host_access_enum::write>)>)
        FAIL(log,
             "is_device_copyable_v for host_access<host_access_enum::write> is "
             "not "
             "true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access<host_access_enum::read_write>)>)
        FAIL(log,
             "is_device_copyable_v for "
             "host_access<host_access_enum::read_write> is "
             "not true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access<host_access_enum::none>)>)
        FAIL(log,
             "is_device_copyable_v for host_access<host_access_enum::none> is "
             "not true");

      // is_device_copyable for empty properties
      properties prop_list1{};
      if (!sycl::is_device_copyable_v<decltype(prop_list1)>)
        FAIL(log, "is_device_copyable_v for empty properties is not true");

      // is_device_copyable for properties with only compile-time-constant
      // properties
      properties prop_list2{host_access<host_access_enum::read_write>,
                            device_image_scope};
      if (!sycl::is_device_copyable_v<decltype(prop_list2)>)
        FAIL(log,
             "is_device_copyable_v for properties with only "
             "compile-time-constant properties is not true");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
