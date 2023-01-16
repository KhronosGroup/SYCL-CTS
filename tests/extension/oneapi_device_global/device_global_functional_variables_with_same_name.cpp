/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides functional test for device_global
//
//  Tests that device_global instance, created in unnamed namespace can be
//  shadowed by a variable with the same name.
//
*******************************************************************************/

#include "../../common/common.h"
#include "../../common/type_coverage.h"
#include "device_global_common.h"
#include "type_pack.h"

#define TEST_NAME device_global_functional_variables_with_same_name

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace device_global_common_functions;

#if defined(SYCL_EXT_ONEAPI_PROPERTIES) && \
    defined(SYCL_EXT_ONEAPI_DEVICE_GLOBAL)
namespace oneapi = sycl::ext::oneapi;

namespace variables_with_same_name {

template <typename T>
struct kernel;

template <typename T>
oneapi::experimental::device_global<T> dev_global;

namespace {
template <typename T>
oneapi::experimental::device_global<T> dev_global;

/**
 * @brief Get the device_global instance from unnamed namespace
 */
template <typename T>
oneapi::experimental::device_global<T>& get_dg_from_namespace() {
  return dev_global<T>;
}
}  // namespace

template <typename T>
void run_test(util::logger& log, const std::string& type_name) {
  bool is_changed_correctly{false};
  T def_val{};
  T new_val{};
  value_operations::assign(new_val, 42);
  auto queue = util::get_cts_object::queue();
  {
    sycl::buffer<bool, 1> is_changed_corr_buf(&is_changed_correctly,
                                              sycl::range<1>(1));
    queue.submit([&](sycl::handler& cgh) {
      auto is_changed_corr_acc =
          is_changed_corr_buf.template get_access<sycl::access_mode::write>(
              cgh);
      cgh.single_task<kernel<T>>([=] {
        // Getting access to instances
        auto& dg1 = get_dg_from_namespace<T>();
        auto& dg2 = variables_with_same_name::dev_global<T>;

        // Write different values to instances
        value_operations::assign(dg1, def_val);
        value_operations::assign(dg2, new_val);
        is_changed_corr_acc[0] = value_operations::are_equal(dg1, def_val);
        is_changed_corr_acc[0] &= value_operations::are_equal(dg2, new_val);

        // Write again but in different order
        value_operations::assign(dg1, new_val);
        value_operations::assign(dg2, def_val);
        is_changed_corr_acc[0] &= value_operations::are_equal(dg1, new_val);
        is_changed_corr_acc[0] &= value_operations::are_equal(dg2, def_val);
      });
    });
    queue.wait_and_throw();
  }
  if (!is_changed_correctly) {
    std::string fail_msg =
        get_case_description("device_global: Variables with same name",
                             "Wrong value after change when defined "
                             "device_global instances with same name",
                             type_name);
    FAIL(log, fail_msg);
  }
}
}  // namespace variables_with_same_name

template <typename T>
class check_device_global_kernel_bundle {
 public:
  void operator()(sycl_cts::util::logger& log, const std::string& type_name) {
    variables_with_same_name::run_test<T>(log, type_name);
    variables_with_same_name::run_test<T[5]>(log, type_name);
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
    for_all_types<check_device_global_kernel_bundle>(types, log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
