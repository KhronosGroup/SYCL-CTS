/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional test for device_global
//
//  The test creates device_global instance in a various ways:
//  1. In anonymous namespace
//  2. In namespace
//  3. As static member of a structure
//
//  The test tries to modify all device_global values inside the kernel and
//  verify that they changed as expected.
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_define_various_ways

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

namespace define_various_ways {

namespace {
template <typename T>
oneapi::experimental::device_global<T> dev_global;
}
namespace dum_namespace {
template <typename T>
oneapi::experimental::device_global<T> dev_global;
}
struct dum_struct {
  template <typename T>
  static inline oneapi::experimental::device_global<T> dev_global;
};

template <typename T>
struct kernel;

/**
 * @brief The function tests that the device_global instance can be correctly
 * defined in various ways
 * @tparam T Type of underlying device_global value
 */
template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  T def_value{};
  T new_val{};
  value_operations::assign(new_val, 1);

  auto queue = util::get_cts_object::queue();
  bool is_defined_correctly = false;
  bool is_default_values = false;
  {
    sycl::buffer<bool, 1> is_def_corr_buf(&is_defined_correctly,
                                          sycl::range<1>(1));
    sycl::buffer<bool, 1> is_default_buf(&is_default_values, sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
      auto is_def_corr_acc =
          is_def_corr_buf.template get_access<sycl::access_mode::write>(cgh);
      auto is_default_acc =
          is_default_buf.template get_access<sycl::access_mode::write>(cgh);
      cgh.single_task<kernel<T>>([=] {
        auto& dg1 = dev_global<T>.get();
        auto& dg2 = dum_namespace::dev_global<T>.get();
        auto& dg3 = dum_struct::dev_global<T>.get();

        // Check that contains default values
        is_default_acc[0] = value_operations::are_equal(dg1, def_value);
        is_default_acc[0] &= value_operations::are_equal(dg2, def_value);
        is_default_acc[0] &= value_operations::are_equal(dg3, def_value);

        value_operations::assign(dg1, new_val);
        value_operations::assign(dg2, new_val);
        value_operations::assign(dg3, new_val);

        is_def_corr_acc[0] =
            value_operations::are_equal(dev_global<T>, new_val);
        is_def_corr_acc[0] &=
            value_operations::are_equal(dum_namespace::dev_global<T>, new_val);
        is_def_corr_acc[0] &=
            value_operations::are_equal(dum_struct::dev_global<T>, new_val);
      });
    });
    queue.wait_and_throw();
  }
  if (!is_default_values) {
    std::string fail_msg = get_case_description(
        "device_global: Define various ways",
        "Instances were created with non-default values", type_name);
    FAIL(log, fail_msg);
  }
  if (!is_defined_correctly) {
    std::string fail_msg = get_case_description(
        "device_global: Define various ways",
        "Wrong value after change when defined device_global various ways",
        type_name);
    FAIL(log, fail_msg);
  }
}
}  // namespace define_various_ways

template <typename T>
class check_device_global_define_various_ways {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    define_various_ways::run_test<T>(log, type_name);
    define_various_ways::run_test<T[5]>(log, type_name);
  }
};
#endif

/** test device_global functional
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger& log) override {
#if !defined(SYCL_EXT_ONEAPI_PROPERTIES)
    WARN("SYCL_EXT_ONEAPI_PROPERTIES is not defined, test is skipped");
#elif !defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
    WARN("SYCL_EXT_ONEAPI_DEVICE_GLOBAL is not defined, test is skipped");
#else
    auto types = device_global_types::get_types();
    for_all_types<check_device_global_define_various_ways>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
