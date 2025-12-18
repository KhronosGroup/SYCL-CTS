/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for compile-time properties initialized with different
//  order
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME properties_different_order

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test compile-time properties initialized with different order
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
      properties P1{host_access<host_access_enum::read_write>,
                    device_image_scope};
      properties P2{device_image_scope,
                    host_access<host_access_enum::read_write>};
      if (!std::is_same_v<decltype(P1), decltype(P2)>)
        FAIL(log,
             "property lists initialized with different order are not the same "
             "type");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
