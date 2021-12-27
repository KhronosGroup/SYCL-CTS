/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for sycl::ext::oneapi::is_property_list
//
*******************************************************************************/

#include "../../common/common.h"

#define TEST_NAME property_list_is_property_list

namespace TEST_NAMESPACE {

class A {};

using namespace sycl_cts;

/** test sycl::ext::oneapi::is_property_list
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
      property_list prop_list{device_image_scope_v, implement_in_csr_v<true>};

      if (!std::is_base_of_v<std::true_type,
                             is_property_list<decltype(prop_list)> >)
        FAIL(log,
             "is_property_list for property list is not derived from "
             "std::true_type");

      if (!is_property_list_v<decltype(prop_list)>)
        FAIL(log, "is_property_list_v for property list is not true");

      if (std::is_base_of_v<std::true_type, is_property_list<A> >)
        FAIL(log,
             "is_property_list for not property list is derived from "
             "std::true_type");

      if (is_property_list_v<A>)
        FAIL(log, "is_property_list_v for not property list is true");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
