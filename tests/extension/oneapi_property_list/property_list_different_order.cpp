/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for compile-time property_list initialized with different
//  order
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME property_list_different_order

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test compile-time property_list initialized with different order
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
      property_list P1{implement_in_csr_v<true>, device_image_scope_v};
      property_list P2{device_image_scope_v, implement_in_csr_v<true>};
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
