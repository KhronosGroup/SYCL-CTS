/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for sycl::is_device_copyable for compile-time properties
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME property_list_is_device_copyable

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
#if !defined(SYCL_EXT_ONEAPI_PROPERTY_LIST)
    WARN("SYCL_EXT_ONEAPI_PROPERTY_LIST is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    {
      using namespace sycl::ext::oneapi;

      // is_device_copyable for compile-time-constant properties
      if (!sycl::is_device_copyable_v<decltype(device_image_scope_v)>)
        FAIL(log, "is_device_copyable_v for device_image_scope_v is not true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access_v<host_access::access::read>)>)
        FAIL(
            log,
            "is_device_copyable_v for host_access_v<access::read> is not true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access_v<host_access::access::write>)>)
        FAIL(log,
             "is_device_copyable_v for host_access_v<access::write> is not "
             "true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access_v<host_access::access::read_write>)>)
        FAIL(log,
             "is_device_copyable_v for host_access_v<access::read_write> is "
             "not true");
      if (!sycl::is_device_copyable_v<decltype(
              host_access_v<host_access::access::none>)>)
        FAIL(
            log,
            "is_device_copyable_v for host_access_v<access::none> is not true");
      if (!sycl::is_device_copyable_v<decltype(
              init_mode_v<init_mode::trigger::reprogram>)>)
        FAIL(log,
             "is_device_copyable_v for init_mode_v<trigger::reprogram is not "
             "true");
      if (!sycl::is_device_copyable_v<decltype(
              init_mode_v<init_mode::trigger::reset>)>)
        FAIL(log,
             "is_device_copyable_v for init_mode_v<trigger::reset is not true");
      if (!sycl::is_device_copyable_v<decltype(implement_in_csr_v<true>)>)
        FAIL(log,
             "is_device_copyable_v for implement_in_csr_v<true> is not true");
      if (!sycl::is_device_copyable_v<decltype(implement_in_csr_v<false>)>)
        FAIL(log,
             "is_device_copyable_v for implement_in_csr_v<false> is not true");

      // is_device_copyable for empty property_list
      property_list prop_list1{};
      if (!sycl::is_device_copyable_v<decltype(prop_list1)>)
        FAIL(log, "is_device_copyable_v for empty property_list is not true");

      // is_device_copyable for property_list with only compile-time-constant
      // properties
      property_list prop_list2{implement_in_csr_v<true>, device_image_scope_v};
      if (!sycl::is_device_copyable_v<decltype(prop_list2)>)
        FAIL(log,
             "is_device_copyable_v for property_list with only "
             "compile-time-constant properties is not true");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
