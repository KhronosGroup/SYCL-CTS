/*******************************************************************************
//
//  SYCL 2020 Extension Conformance Test
//
//  Provides test for sycl::ext::oneapi::experimental::property_value
//
*******************************************************************************/

#include "../../common/common.h"
#include <type_traits>

#define TEST_NAME properties_property_value

namespace TEST_NAMESPACE {

using namespace sycl_cts;

// to check if class has member value
template <typename T, typename U = bool>
struct member_value : std::false_type {};

template <typename T>
struct member_value<T, decltype((void)T::value, true)> : std::true_type {};

// to check if class has member value_t
template <typename T, typename U = bool>
struct member_type_value_t : std::false_type {};

template <typename T>
struct member_type_value_t<T, decltype(typename T::value_t(), true)>
    : std::true_type {};

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)

using namespace sycl::ext::oneapi::experimental;

template <typename T>
void check(util::logger &log, T prop_value, host_access_enum access,
           std::string access_string) {
  if (prop_value.value != access) {
    FAIL(log, "member value for " + access_string + " is incorrect");
  }

  // since value_t = First and it's implementation defined, only check
  // availability
  if (!member_type_value_t<decltype(prop_value)>::value) {
    FAIL(log, "member type value_t for " + access_string + " is not available");
  }

  if (!std::is_same_v<typename decltype(prop_value)::key_t, host_access_key>) {
    FAIL(log, "member type key_t for " + access_string + " is incorrect");
  }
}
#endif
/** test sycl::ext::oneapi::experimental::property_value interface
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
      properties props{device_image_scope};
      auto prop_value = props.get_property<device_image_scope_key>();

      properties props_read{host_access<host_access_enum::read>};
      auto prop_value_read = props_read.get_property<host_access_key>();

      properties props_write{host_access<host_access_enum::write>};
      auto prop_value_write = props_write.get_property<host_access_key>();

      properties props_read_write{host_access<host_access_enum::read_write>};
      auto prop_value_read_write =
          props_read_write.get_property<host_access_key>();

      properties props_none{host_access<host_access_enum::none>};
      auto prop_value_none = props_none.get_property<host_access_key>();

      if (member_value<decltype(prop_value)>::value) {
        FAIL(log,
             "member value is available for property value without parameter");
      }

      if (member_type_value_t<decltype(prop_value)>::value) {
        FAIL(log,
             "member type value_t is available for property value without "
             "parameter");
      }

      if (!std::is_same_v<typename decltype(prop_value)::key_t,
                          device_image_scope_key>) {
        FAIL(log, "member type key_t for device_image_scope is incorrect");
      }

      check(log, prop_value_read, host_access_enum::read,
            "host_access_enum::read");
      check(log, prop_value_write, host_access_enum::write,
            "host_access_enum::write");
      check(log, prop_value_read_write, host_access_enum::read_write,
            "host_access_enum::read_write");
      check(log, prop_value_none, host_access_enum::none,
            "host_access_enum::none");
    }
#endif
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
